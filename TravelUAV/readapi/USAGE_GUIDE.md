# 数据记录功能使用指南

## ✅ 更新说明 (2025-12-03)

**新增完整数据保存功能**：

- ✅ **instruction**: 保存任务指令文本
- ✅ **prompts**: 保存完整的prompt文本
- ✅ **RGB图像**: 保存到PNG文件（5个视角）
- ✅ **深度图**: 保存到PNG + NPY文件（5个视角）

**只修改了一个文件**: `readapi/data_interceptor.py`

## 📖 如何使用

### 方式1: 在eval.py中使用（推荐）

在 `src/vlnce_src/eval.py` 中添加以下代码：

```python
import sys
from pathlib import Path

# 添加readapi到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'readapi'))

# 导入拦截器
from data_interceptor import DataInterceptor

# 初始化拦截器（新参数）
interceptor = DataInterceptor(
    output_dir='/mnt/data/TravelUAV/readapi/debug_data',  # 输出目录
    save_images=True,   # 保存RGB和深度图
    save_prompts=True   # 保存prompts文本
)
```

### 方式2: 命令行测试

```bash
cd /mnt/data/TravelUAV
python readapi/data_interceptor.py \
    --output_dir ./readapi/debug_data \
    --save_images \
    --save_prompts
```

## 🔧 新增参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `./debug_data` | 输出目录 |
| `save_images` | `True` | 是否保存RGB和深度图到文件 |
| `save_prompts` | `True` | 是否保存完整prompts文本 |

## 📁 输出文件结构

启用`save_images=True`后，输出结构如下：

```
/mnt/data/TravelUAV/readapi/debug_data/
├── episode_0000.json                # Episode数据（包含instruction、prompts等）
├── episode_0000/
│   └── images/
│       ├── step_0000_front_rgb.png  # 前视RGB图像
│       ├── step_0000_left_rgb.png   # 左视RGB图像
│       ├── step_0000_right_rgb.png  # 右视RGB图像
│       ├── step_0000_rear_rgb.png   # 后视RGB图像
│       ├── step_0000_down_rgb.png   # 俯视RGB图像
│       ├── step_0000_front_depth.png  # 前视深度图（归一化）
│       ├── step_0000_front_depth.npy  # 前视深度图（原始值）
│       ├── step_0000_left_depth.png
│       ├── step_0000_left_depth.npy
│       └── ...
├── episode_0001.json
├── episode_0001/
│   └── images/
│       └── ...
└── ...
```

## 📊 JSON文件内容

### Episode级别（新增字段标记为🆕）

```json
{
  "episode_id": 0,
  "map_name": "NewYorkCity",
  "seq_name": "NYC_seq_0042",
  "instruction": "Navigate to the red building...",  // 🆕 任务指令
  "target_position": [150.5, -95.2, -42.0],          // 🆕 目标位置
  "start_position": [100.0, -80.0, -40.0],           // 🆕 起始位置
  "total_steps": 12,
  "success": true,
  "distance_to_goal": 2.5,
  "collision": false
}
```

### Observation数据（新增字段标记为🆕）

```json
{
  "step": 0,
  "type": "observation",
  "sensors": {
    "position": [113.498, 69.027, -2.834],
    "orientation": [0.001, 0.002, 0.596, 0.803],
    "linear_velocity": [-0.0001, 0.0, 0.01],
    "angular_velocity": [0.0, 0.0, 0.0],           // 🆕
    "rotation": [[...], [...], [...]],
    "linear_acceleration": [0.0, 0.0, -9.8]        // 🆕
  },
  "rgb_info": {
    "num_images": 5,
    "shape": [256, 256, 3],
    "dtype": "uint8"
  },
  "rgb_files": [                                    // 🆕 RGB图像文件路径
    "/mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/step_0000_front_rgb.png",
    "/mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/step_0000_left_rgb.png",
    "/mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/step_0000_right_rgb.png",
    "/mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/step_0000_rear_rgb.png",
    "/mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/step_0000_down_rgb.png"
  ],
  "depth_info": {                                   // 🆕 深度图信息
    "num_images": 5,
    "shape": [256, 256],
    "dtype": "float32"
  },
  "depth_files": [                                  // 🆕 深度图文件路径
    "/mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/step_0000_front_depth.png",
    "..."
  ],
  "instruction": "Navigate to...",                  // 🆕 任务指令（如果在obs中）
  "object_position": [150.5, -95.2, -42.0],        // 🆕 目标位置
  "distance_to_goal": 28.73                         // 🆕 到目标距离
}
```

### Model Input数据（新增字段标记为🆕）

```json
{
  "step": 0,
  "type": "model_input",
  "input_ids": {
    "shape": [1, 250],
    "dtype": "torch.int64",
    "mean": 10682.62,
    "std": 12134.23
  },
  "prompts": [                                      // 🆕 完整prompt文本
    "You are an AI assistant helping navigate a drone.\n\nTask: Navigate to the red building...\n\nCurrent State:\n- Position: [113.498, 69.027, -2.834]\n..."
  ],
  "conversations": [...],                           // 🆕 对话历史（如果有）
  "images": {
    "type": "tensor_list",
    "shapes": [[5, 3, 224, 224]]
  },
  "historys": {
    "type": "tensor_list",
    "shapes": [[10, 4096]]
  }
}
```

### Control Command数据（新增字段标记为🆕）

```json
{
  "step": 0,
  "type": "control_command",
  "waypoints": [[113.476, 68.973, -3.760], ...],
  "velocity": 1.0,                                  // 🆕 速度参数
  "drivetrain": "ForwardOnly",                      // 🆕 驱动模式
  "lookahead": 3,                                   // 🆕 前瞻距离
  "adaptive_lookahead": 1                           // 🆕 自适应前瞻
}
```

## 🔍 查看保存的数据

### 查看JSON数据

```bash
# 查看episode元信息（包含instruction）
cat /mnt/data/TravelUAV/readapi/debug_data/episode_0000.json | jq '{episode_id, instruction, map_name}'

# 查看prompts文本
cat /mnt/data/TravelUAV/readapi/debug_data/episode_0000.json | jq '.steps[] | select(.type=="model_input") | .prompts'

# 查看RGB图像路径
cat /mnt/data/TravelUAV/readapi/debug_data/episode_0000.json | jq '.steps[] | select(.type=="observation") | .rgb_files'
```

### 查看图像

```bash
# 查看保存的图像
ls /mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/

# 用Python读取图像
python3 -c "
from PIL import Image
import numpy as np

# 读取RGB图像
rgb = Image.open('/mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/step_0000_front_rgb.png')
print(f'RGB shape: {np.array(rgb).shape}')

# 读取深度图（原始值）
depth = np.load('/mnt/data/TravelUAV/readapi/debug_data/episode_0000/images/step_0000_front_depth.npy')
print(f'Depth shape: {depth.shape}')
print(f'Depth range: [{depth.min():.2f}, {depth.max():.2f}]')
"
```

## 📊 数据分析

### 可视化轨迹

```bash
python readapi/visualize_data.py \
    --data_file /mnt/data/TravelUAV/readapi/debug_data/episode_0000.json
```

### 统计分析

```bash
python readapi/analyze_statistics.py \
    --data_dir /mnt/data/TravelUAV/readapi/debug_data
```

## ⚙️ 在eval.py中集成

### 完整集成代码

在 `src/vlnce_src/eval.py` 中添加：

```python
# === 在文件开头添加 ===
import sys
from pathlib import Path

# 添加readapi到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'readapi'))

try:
    from data_interceptor import DataInterceptor
    HAS_INTERCEPTOR = True
except ImportError:
    HAS_INTERCEPTOR = False
    print("[WARNING] data_interceptor未找到，跳过数据记录")


# === 在main函数或评估循环前添加 ===
# 初始化拦截器
interceptor = None
record_data = True  # 设置为True启用数据记录

if record_data and HAS_INTERCEPTOR:
    interceptor = DataInterceptor(
        output_dir='/mnt/data/TravelUAV/readapi/debug_data',
        save_images=True,
        save_prompts=True
    )
    print("✓ 数据拦截器已启用（保存图像和prompts）")


# === 在Episode开始时 ===
if interceptor:
    interceptor.start_episode({
        'map_name': episode_data.get('map_name'),
        'seq_name': episode_data.get('seq_name'),
        'instruction': episode_data.get('instruction'),  # 重要！
        'target_position': episode_data.get('target_position'),
        'start_position': episode_data.get('start_position')
    })


# === 在获取观测后 ===
observation = env.get_observation()
if interceptor:
    obs_data = interceptor.record_observation(observation)
    interceptor.add_step_data(obs_data)


# === 在模型推理前 ===
if interceptor:
    input_data = interceptor.record_model_input(model_inputs)
    interceptor.add_step_data(input_data)


# === 在模型推理后 ===
if interceptor:
    output_data = interceptor.record_model_output(model_output)
    interceptor.add_step_data(output_data)


# === 在执行动作后 ===
if interceptor:
    cmd_data = interceptor.record_control_command({
        'waypoints': waypoints,
        'velocity': 1.0,
        'drivetrain': 'ForwardOnly',
        'lookahead': 3
    })
    interceptor.add_step_data(cmd_data)
    interceptor.end_step()


# === 在Episode结束时 ===
if interceptor:
    interceptor.end_episode({
        'success': is_success,
        'distance_to_goal': final_distance,
        'collision': has_collision
    })
```

## 💡 提示

1. **磁盘空间**：启用`save_images=True`后，每个episode约10-20MB（主要是图像）
2. **性能影响**：图像保存会增加约10-20ms/步
3. **深度图精度**：PNG是归一化后的（0-255），NPY是原始浮点值
4. **图像格式**：使用PNG格式（无损压缩）

## 🐛 故障排查

### 问题1: ImportError: No module named 'PIL'

```bash
pip install Pillow
```

### 问题2: 图像未保存

检查：
- `save_images` 是否为 `True`
- 观测数据中是否有 `rgb` 和 `depth` 字段
- 输出目录是否有写入权限

### 问题3: instruction为空

检查：
- `episode_info` 中是否包含 `instruction` 字段
- 在 `start_episode()` 调用时是否传递了 `instruction`

## 📚 更多资源

- [README.md](README.md) - 系统架构说明
- [data_flow_diagram.md](data_flow_diagram.md) - 详细数据流图解
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - 完整集成指南
- [INDEX.md](INDEX.md) - 工具集索引

---

**现在可以保存完整的数据了！** 🎉

- ✅ instruction（任务指令）
- ✅ prompts（完整prompt文本）
- ✅ RGB图像（5视角PNG）
- ✅ 深度图（5视角PNG + NPY）
