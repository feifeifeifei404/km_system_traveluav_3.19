# TravelUAV 大模型与模拟器交互数据分析

## 概述

本目录包含用于分析和可视化 TravelUAV 系统中大模型(LLM)与模拟器(AirSim)之间交互数据的工具。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        评估主流程 (eval.py)                       │
│                                                                   │
│  1. 加载数据集                                                     │
│  2. 初始化环境                                                     │
│  3. 开始轨迹导航循环                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    环境管理 (AirVLNENV)                          │
│                                                                   │
│  • 加载轨迹数据                                                    │
│  • 管理批次(batch)                                                │
│  • 设置目标物体                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
┌───────────────────────────┐   ┌──────────────────────────────┐
│  模拟器客户端              │   │  大模型推理                    │
│  (AirSimClientTool)       │   │  (TravelModelWrapper)        │
│                           │   │                              │
│  • 控制多个模拟器实例      │   │  • LLM 生成粗略航点           │
│  • 获取观测数据            │   │  • 轨迹模型精化航点           │
│  • 设置无人机位姿          │   │  • 多模态数据融合             │
│  • 移动无人机              │   │                              │
└───────────────────────────┘   └──────────────────────────────┘
         │                                   │
         ▼                                   ▼
┌───────────────────────────┐   ┌──────────────────────────────┐
│  模拟器服务端              │   │  模型输入输出                  │
│  (AirSimServerTool)       │   │                              │
│                           │   │  输入:                        │
│  • 启动UE4场景            │   │  • 历史轨迹图像               │
│  • 管理多个场景进程        │   │  • 目标描述文本               │
│  • 配置AirSim参数         │   │  • IMU/位置传感器数据         │
└───────────────────────────┘   │                              │
                                │  输出:                        │
                                │  • 3D航点坐标                 │
                                │  • 置信度分数                 │
                                └──────────────────────────────┘
```

## 数据流详解

### 1. 模拟器 → 大模型 (观测数据)

**文件**: `observation_data.py`

从AirSim获取的观测数据包括:

```python
observation = {
    # 图像数据 (5个相机视角)
    'rgb': [
        front_image,   # [256, 256, 3] uint8
        left_image,    # [256, 256, 3] uint8
        right_image,   # [256, 256, 3] uint8
        rear_image,    # [256, 256, 3] uint8
        down_image     # [256, 256, 3] uint8
    ],
    'depth': [
        front_depth,   # [256, 256] uint8
        left_depth,    # [256, 256] uint8
        right_depth,   # [256, 256] uint8
        rear_depth,    # [256, 256] uint8
        down_depth     # [256, 256] uint8
    ],
    
    # 传感器数据
    'sensors': {
        'state': {
            'position': [x, y, z],              # 3D位置 (float)
            'orientation': [qx, qy, qz, qw],    # 四元数姿态 (float)
            'linear_velocity': [vx, vy, vz],    # 线速度 (float)
            'angular_velocity': [wx, wy, wz],   # 角速度 (float)
            'collision': {
                'has_collided': False,           # 碰撞标志 (bool)
                'object_name': ''                # 碰撞物体名称 (str)
            }
        },
        'imu': {
            'rotation': [[...], [...], [...]],  # 3x3旋转矩阵 (float)
            'linear_acceleration': [ax, ay, az],# 线性加速度 (float)
            'angular_velocity': [wx, wy, wz]   # 角速度 (float)
        }
    },
    
    # 任务信息
    'instruction': "Navigate to the red building...",  # 导航指令 (str)
    'object_position': [x, y, z],                     # 目标位置 (float)
}
```

### 2. 大模型处理流程

**文件**: `model_processing.py`

#### 步骤1: 数据预处理 (`prepare_inputs`)

```python
model_input = {
    # 文本输入
    'input_ids': Tensor,              # [batch, seq_len] token索引
    'attention_mask': Tensor,         # [batch, seq_len] 注意力掩码
    'prompts': [                      # 原始提示文本列表
        "Navigate to...",
        "Find the..."
    ],
    
    # 图像输入
    'images': [                       # 处理后的图像张量列表
        Tensor,                       # [num_frames, C, H, W]
        Tensor,
        ...
    ],
    
    # 历史轨迹
    'historys': [                     # 历史轨迹编码
        Tensor,                       # [seq_len, hidden_dim]
        Tensor,
        ...
    ],
    
    # 位姿信息
    'orientations': Tensor,           # [batch, 4] 四元数
    
    # 控制标志
    'return_waypoints': True,         # 是否返回航点
    'use_cache': False                # 是否使用KV缓存
}
```

#### 步骤2: LLM推理 (`run_llm_model`)

```python
# LLM输出原始航点 (相对坐标)
waypoints_llm = [
    [dx, dy, dz, distance],  # 相对位移 + 距离
    [dx, dy, dz, distance],
    ...
]
# Shape: [batch_size, 4]

# 归一化为方向向量 + 距离
waypoints_normalized = [
    [dx/norm, dy/norm, dz/norm] * distance,
    ...
]
```

#### 步骤3: 轨迹模型精化 (`run_traj_model`)

```python
traj_input = {
    'img': Tensor,      # [batch, C, H, W] 当前前视图像
    'target': Tensor    # [batch, 3] 目标方向向量
}

# 轨迹模型输出 (相对坐标)
waypoints_refined = [
    [dx, dy, dz],
    [dx, dy, dz],
    ...
]
# Shape: [batch_size, 3]

# 转换到世界坐标系
waypoints_world = [
    rot @ waypoint + current_position,
    ...
]
```

### 3. 大模型 → 模拟器 (控制指令)

**文件**: `control_commands.py`

#### 航点控制

```python
control_command = {
    'type': 'moveOnPathAsync',
    'waypoints': [
        {'x': x1, 'y': y1, 'z': z1},
        {'x': x2, 'y': y2, 'z': z2},
        {'x': x3, 'y': y3, 'z': z3},
        ...
    ],
    'velocity': 1.0,              # 移动速度 (m/s)
    'drivetrain': 'ForwardOnly',  # 驱动模式
    'yaw_mode': {
        'is_rate': False
    },
    'lookahead': 3,               # 前瞻距离
    'adaptive_lookahead': 1       # 自适应前瞻
}
```

#### 位姿设置

```python
pose_command = {
    'type': 'simSetKinematics',
    'position': [x, y, z],
    'orientation': [qx, qy, qz, qw],
    'linear_velocity': [0, 0, 0],
    'angular_velocity': [0, 0, 0]
}
```

### 4. 完整交互循环

```
初始化
  │
  ├─→ 设置场景和目标物体
  │
导航循环开始
  │
  ├─→ [模拟器] 获取观测
  │     │
  │     ├─ RGB图像 (5视角)
  │     ├─ 深度图 (5视角)
  │     └─ 传感器数据 (位置/姿态/速度)
  │
  ├─→ [大模型] 处理观测
  │     │
  │     ├─ 步骤1: 数据预处理
  │     │   └─ 图像编码、文本嵌入、历史轨迹
  │     │
  │     ├─ 步骤2: LLM推理
  │     │   └─ 生成粗略航点方向
  │     │
  │     └─ 步骤3: 轨迹精化
  │         └─ 生成精确3D航点
  │
  ├─→ [模拟器] 执行动作
  │     └─ 沿航点移动无人机
  │
  ├─→ 检查终止条件
  │     ├─ 到达目标?
  │     ├─ 发生碰撞?
  │     └─ 超过最大步数?
  │
  └─→ 返回导航循环或结束
```

## 数据格式样例

### 样例1: 单步观测数据
参见: `sample_observation.json`

### 样例2: 模型输入数据
参见: `sample_model_input.json`

### 样例3: 模型输出数据
参见: `sample_model_output.json`

### 样例4: 控制指令
参见: `sample_control_command.json`

## 使用工具

### 1. 数据拦截和记录

运行 `data_interceptor.py` 可以在实际评估过程中拦截并记录所有交互数据:

```bash
python readapi/data_interceptor.py \
    --output_dir ./debug_data \
    --max_episodes 5
```

### 2. 数据可视化

```bash
python readapi/visualize_data.py \
    --data_file ./debug_data/episode_0.json
```

### 3. 数据统计分析

```bash
python readapi/analyze_statistics.py \
    --data_dir ./debug_data
```

## 关键发现

1. **数据流向**:
   - 模拟器 → 模型: 每步约 5MB (5个RGB + 5个Depth + 传感器)
   - 模型 → 模拟器: 每步约 1KB (航点坐标)

2. **坐标系转换**:
   - AirSim使用NED坐标系 (North-East-Down)
   - 模型输出相对坐标
   - 需要旋转矩阵转换到世界坐标

3. **关键瓶颈**:
   - 图像获取: ~50ms
   - LLM推理: ~200ms
   - 轨迹模型: ~50ms
   - 无人机移动: ~500ms

4. **数据类型**:
   - 图像: uint8 numpy数组
   - 传感器: float32 列表
   - 模型张量: torch.bfloat16

## 参考代码位置

- 模拟器客户端: `airsim_plugin/AirVLNSimulatorClientTool.py`
- 模拟器服务端: `airsim_plugin/AirVLNSimulatorServerTool.py`
- 环境管理: `src/vlnce_src/env_uav.py`
- 模型封装: `src/model_wrapper/travel_llm.py`
- 数据处理: `src/model_wrapper/utils/travel_util.py`
- 评估流程: `src/vlnce_src/eval.py`

