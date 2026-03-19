# Launch文件对比说明

## 📊 **两个Launch文件的区别**

### 1️⃣ **原始版本** (`click_demo.launch.py`)

**位置**: `~/super_ws/src/SUPER/mission_planner/launch/click_demo.launch.py`

**启动的组件**:
```
┌──────────────────────────────┐
│  1. RViz (俯视图)             │  ← top_down.rviz
│  2. RViz (第一人称)           │  ← fpv.rviz
│  3. Perfect Drone仿真器       │  ← 虚拟无人机
│  4. SUPER规划器               │  ← 核心算法
└──────────────────────────────┘
```

**优点**:
- ✅ 一键启动完整的SUPER演示系统
- ✅ 自带虚拟仿真环境

**缺点（对AirSim集成）**:
- ❌ Perfect Drone与AirSim冲突（两个仿真器）
- ❌ 启动2个RViz窗口（资源浪费）
- ❌ 使用虚拟数据而非AirSim实时数据

---

### 2️⃣ **AirSim集成版** (`airsim_super.launch.py`) ⭐

**位置**: `/mnt/data/airsim_super_integration/launch/airsim_super.launch.py`

**启动的组件**:
```
┌──────────────────────────────┐
│  只启动SUPER规划器            │  ← 核心算法
└──────────────────────────────┘
```

**优点**:
- ✅ **纯净**：只启动必需的组件
- ✅ **无冲突**：不启动Perfect Drone
- ✅ **轻量**：不启动RViz（使用单独脚本）
- ✅ **专用**：为AirSim实时数据优化

**适用场景**:
- ✅ AirSim集成系统
- ✅ 使用实时LiDAR点云
- ✅ 需要自定义RViz配置
- ✅ 追求系统性能

---

## 📋 **详细对比表**

| 特性 | 原始版 (click_demo) | AirSim版 (airsim_super) |
|------|-------------------|----------------------|
| **SUPER规划器** | ✅ | ✅ |
| **Perfect Drone** | ✅ 启动 | ❌ 不启动 |
| **RViz窗口** | ✅ 2个 | ❌ 不启动 |
| **数据来源** | Perfect Drone虚拟数据 | **AirSim实时数据** |
| **点云来源** | .pcd文件或虚拟 | **AirSim LiDAR** |
| **里程计来源** | 虚拟无人机 | **AirSim真实状态** |
| **适用环境** | SUPER演示 | **AirSim集成** |
| **资源占用** | 高（4个节点） | **低（1个节点）** |
| **启动速度** | 慢 | **快** |

---

## 🎯 **使用场景**

### 使用原始版 (`click_demo.launch.py`)

**什么时候用**:
- 测试SUPER本身的功能
- 不使用AirSim
- 想要快速演示SUPER
- 使用SUPER自带的演示环境

**启动方法**:
```bash
cd ~/super_ws
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch mission_planner click_demo.launch.py
```

---

### 使用AirSim版 (`airsim_super.launch.py`) ⭐

**什么时候用** (推荐):
- ✅ **集成AirSim环境**
- ✅ **使用实时LiDAR点云**
- ✅ **需要TravelUAV场景**
- ✅ **自定义RViz配置**
- ✅ **追求系统性能**

**启动方法1** (推荐):
```bash
cd /mnt/data/airsim_super_integration/scripts
./start_super_airsim.sh
```

**启动方法2** (手动):
```bash
cd ~/super_ws
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch /mnt/data/airsim_super_integration/launch/airsim_super.launch.py
```

---

## 🔄 **数据流对比**

### 原始版数据流:

```
Perfect Drone (虚拟)
    ↓
虚拟传感器数据
    ↓
SUPER规划器
    ↓
虚拟控制 → Perfect Drone
```

### AirSim版数据流:

```
AirSim环境 (真实仿真)
    ↓
LiDAR实时点云 + 真实状态
    ↓
桥接节点 (数据转换)
    ↓
SUPER规划器
    ↓
控制指令 → 桥接节点 → AirSim
```

---

## 💡 **核心区别总结**

### 原始版 = **完整演示系统**
- 包含所有组件
- 自带仿真环境
- 适合演示和测试

### AirSim版 = **精简核心系统** ⭐
- 只保留SUPER规划器
- 依赖外部仿真（AirSim）
- **专门为实时集成优化**

---

## 🚀 **推荐的完整启动流程**

使用**AirSim集成版**的完整流程：

```bash
# 终端1: AirSim环境
cd /mnt/data/TravelUAV/envs/closeloop_envs
./ModularPark.sh -ResX=1280 -ResY=720 -windowed

# 终端2: SUPER规划器（使用新的AirSim版）
cd /mnt/data/airsim_super_integration/scripts
./start_super_airsim.sh

# 终端3: 桥接节点
cd /mnt/data/airsim_super_integration/scripts
./run_bridge.sh

# 终端4: RViz可视化
cd /mnt/data/airsim_super_integration/scripts
./start_rviz.sh

# 终端5: 发送目标点
cd /mnt/data/airsim_super_integration/scripts
./send_goal.sh 10 5 2
```

---

## ✅ **总结**

- **原始版**: 完整的SUPER演示系统，包含所有组件
- **AirSim版**: **专门为AirSim集成优化的精简版**，只启动核心

**对于AirSim集成，强烈推荐使用AirSim版！** ⭐

更轻量、更快、无冲突、专门优化！


