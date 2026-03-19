# TravelUAV + SUPER 无人机导航系统

## 项目简介

本项目将 **TravelUAV**（视觉语言导航大模型）与 **SUPER**（无人机局部规划器）集成，实现无人机在 AirSim 仿真环境中的端到端自主导航。

- **TravelUAV**：基于 LLaMA 的视觉语言模型，理解自然语言指令，输出导航目标点
- **SUPER**：基于 ROG-Map + MINCO 轨迹优化的局部避障规划器，输出高频控制指令
- **AirSim**：仿真环境（Carla Town05），提供物理仿真、相机图像、LiDAR 点云

---

## 系统架构

```
AirSim 仿真 (port 25000)
    │
    ├──→ TravelUAV（慢系统，~1Hz）
    │    直接 API：RGB图像、无人机状态、碰撞信息
    │    → 理解指令 → 发布 /goal_pose
    │
    └──→ 桥接节点（中间层）
         直接 API：LiDAR点云、无人机位姿
         → 坐标转换 NED→ENU
         → 发布 /cloud_registered + /lidar_slam/odom
              ↓
         SUPER（快系统，8Hz）
         → 建图 + 规划 → /planning/pos_cmd
              ↓
         桥接节点 → AirSim 控制无人机
```

---

## 主要文件

### 仿真环境
| 路径 | 说明 |
|------|------|
| `/mnt/data/TravelUAV/envs/carla_town_envs/Town05/` | Carla Town05 仿真场景 |
| `/mnt/data/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py` | AirSim 仿真服务器 |
| `/mnt/data/TravelUAV/airsim_plugin/AirVLNSimulatorClientTool.py` | AirSim 客户端工具 |

### TravelUAV 大模型
| 路径 | 说明 |
|------|------|
| `/mnt/data/TravelUAV/src/vlnce_src/eval.py` | 主评估入口 |
| `/mnt/data/TravelUAV/src/vlnce_src/env_uav.py` | 无人机环境接口 |
| `/mnt/data/TravelUAV/src/vlnce_src/super_ros2_client.py` | SUPER ROS2 客户端（发目标点、监控到达） |
| `/mnt/data/TravelUAV/scripts/eval_xiugai.sh` | 评估启动脚本 |
| `/mnt/data/TravelUAV/Dataset/` | 导航数据集 |

### SUPER 规划器
| 路径 | 说明 |
|------|------|
| `/home/wuyou/super_ws/src/SUPER/` | SUPER 源码 |
| `/home/wuyou/super_ws/src/SUPER/super_planner/config/click_smooth_ros2.yaml` | SUPER 配置文件（已针对 AirSim 调整） |
| `/home/wuyou/super_ws/src/SUPER/rog_map/` | ROG-Map 占据地图模块 |
| `/home/wuyou/super_ws/src/SUPER/mars_uav_sim/mars_quadrotor_msgs/` | 控制指令消息类型 |

### 桥接节点
| 路径 | 说明 |
|------|------|
| `/mnt/data/airsim_super_integration/scripts/airsim_super_bridge_ros2.py` | 核心桥接节点（AirSim↔ROS2） |
| `/mnt/data/airsim_super_integration/launch/airsim_super.launch.py` | SUPER 启动文件 |
| `/mnt/data/airsim_super_integration/scripts/verify_coordinate_detailed.py` | 坐标系验证工具 |

### ROS2 话题
| 话题 | 方向 | 说明 |
|------|------|------|
| `/cloud_registered` | 桥接→SUPER | LiDAR 点云（ENU world frame） |
| `/lidar_slam/odom` | 桥接→SUPER/TravelUAV | 无人机里程计（ENU） |
| `/goal_pose` | TravelUAV→SUPER | 导航目标点 |
| `/planning/pos_cmd` | SUPER→桥接 | 高频控制指令 |
| `/rog_map/occ` | SUPER→RViz | 占据地图可视化 |

---

## 启动流程（4个终端依次执行）

> **重要**：先启动仿真环境（向日葵），再按顺序执行以下终端命令。
> 等第3个终端（TravelUAV）显示 **connected** 之后，再启动第4个终端（桥接节点）。

### 前置：启动仿真环境（向日葵远程）观察仿真环境

```bash
cd /mnt/data/TravelUAV/envs/carla_town_envs/Town05/LinuxNoEditor
chmod +x CarlaUE4.sh
./CarlaUE4.sh -ResX=1280 -ResY=720 -windowed
```

---

### 终端 1：启动 AirSim 仿真服务器

```bash
conda activate llamanew
cd /mnt/data/TravelUAV/airsim_plugin
python AirVLNSimulatorServerTool.py --port 25000 --root_path /mnt/data/TravelUAV/envs
```

---

### 终端 2：启动 SUPER 规划器

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source ~/super_ws/install/local_setup.bash
cd /mnt/data/airsim_super_integration/launch
ros2 launch airsim_super.launch.py
```

---

### 终端 3：启动 TravelUAV 评估（等待 connected 后再启动终端4）

```bash
conda activate llamanew
export CUDA_VISIBLE_DEVICES=1
cd TravelUAV/
bash /mnt/data/TravelUAV/scripts/eval_xiugai.sh
```

---

### 终端 4：启动桥接节点（等终端3显示 connected 后执行）

```bash
conda deactivate
source /opt/ros/humble/setup.bash
source ~/super_ws/install/local_setup.bash
cd /mnt/data/airsim_super_integration/scripts
/usr/bin/python3 airsim_super_bridge_ros2.py
```

---

### 终端 5：查看点云（可选，RViz）

```bash
conda deactivate
cd /mnt/data/airsim_super_integration/scripts
./start_rviz.sh
```

---

## 坐标系说明

| 系统 | 坐标系 | X | Y | Z |
|------|--------|---|---|---|
| AirSim | NED body | 前 | 右 | 下 |
| ROS2 / SUPER | ENU world | 东 | 北 | 上 |
| 桥接转换 | NED→ENU | x←y | y←x | z←-z |

---

## 常见问题排查

### SUPER 没有收到点云
```bash
source /opt/ros/humble/setup.bash
ros2 topic list | grep cloud_registered
ros2 topic hz /cloud_registered
```
应该看到 `/cloud_registered` 以 ~5Hz 发布。

### SUPER 没有收到里程计
```bash
ros2 topic echo /lidar_slam/odom --no-arr | head -20
```
应该看到位置数据流动。

### SUPER 地图是否建立
```bash
ros2 topic hz /rog_map/occ
```
应该看到 ~2Hz 的地图更新。

### 无人机不动
检查 `mars_quadrotor_msgs` 是否编译安装：
```bash
source ~/super_ws/install/local_setup.bash
python3 -c "from mars_quadrotor_msgs.msg import PositionCommand; print('OK')"
```

---

## 重新编译 SUPER

修改源码后需要重新编译：

```bash
cd ~/super_ws
colcon build --symlink-install --packages-select mars_quadrotor_msgs rog_map super_planner mission_planner
source ~/super_ws/install/local_setup.bash
```
