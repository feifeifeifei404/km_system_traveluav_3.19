# AirSim-SUPER 集成系统 (ROS2 Humble版本)

## 📁 文件夹结构

```
airsim_super_integration/
├── README.md                    # 本文件
├── backup/                      # 原始文件备份
│   ├── launch_original/         # SUPER原始launch文件
│   ├── config_original/         # SUPER原始配置文件
│   └── airsim_settings_original.json  # AirSim原始配置
├── scripts/                     # Python脚本
│   ├── airsim_super_bridge_ros2.py    # AirSim-SUPER桥接节点 (ROS2)
│   ├── test_super_goal_ros2.py        # 目标点测试脚本 (ROS2)
│   └── test_airsim_connection.py      # AirSim连接测试
├── config/                      # 配置文件
│   └── airsim_config_ros2.yaml        # SUPER ROS2配置
├── launch/                      # ROS2 Launch文件
│   └── airsim_super_ros2.launch.py    # 集成系统launch文件
├── docs/                        # 文档
│   ├── QUICK_START_ROS2.md            # 快速开始指南
│   └── INTEGRATION_GUIDE_ROS2.md      # 详细集成指南
└── start_airsim_super_ros2.sh         # 一键启动脚本
```

## 🔧 使用方法

### 1. 一键启动
```bash
cd /mnt/data/airsim_super_integration
./start_airsim_super_ros2.sh
```

### 2. 手动启动
```bash
# 终端1: 启动AirSim
cd /mnt/data/TravelUAV/envs/closeloop_envs
./ModularPark.sh -ResX=1280 -ResY=720 -windowed

# 终端2: 启动SUPER (ROS2)
source /opt/ros/humble/setup.bash
source ~/super_ws/install/local_setup.bash
ros2 launch mission_planner click_demo.launch.py

# 终端3: 启动桥接节点
source /opt/ros/humble/setup.bash
cd /mnt/data/airsim_super_integration/scripts
python3 airsim_super_bridge_ros2.py

# 终端4: 发送测试目标点
source /opt/ros/humble/setup.bash
cd /mnt/data/airsim_super_integration/scripts
python3 test_super_goal_ros2.py 10 5 2
```

## 📝 注意事项

- **ROS版本**: ROS2 Humble (不是ROS1)
- **SUPER工作空间**: ~/super_ws (ROS2编译)
- **备份**: 所有原始文件已备份在 `backup/` 目录
- **不修改原文件**: 所有新文件都在此文件夹中

## 🔄 恢复原始配置

如需恢复SUPER原始配置：
```bash
# 恢复launch文件
cp -r backup/launch_original/* ~/super_ws/src/SUPER/mission_planner/launch/

# 恢复配置文件
cp -r backup/config_original/* ~/super_ws/src/SUPER/super_planner/config/

# 恢复AirSim配置
cp backup/airsim_settings_original.json ~/Documents/AirSim/settings.json
```

---

**创建日期**: 2026-01-20  
**ROS版本**: ROS2 Humble  
**Python版本**: Python 3.x


