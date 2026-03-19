#!/bin/bash

# 启动配置好的RViz（显示AirSim实时点云）

# 退出conda
conda deactivate 2>/dev/null || true

# 加载ROS2环境
source /opt/ros/humble/setup.bash

echo "启动RViz2..."
echo "配置文件: /mnt/data/airsim_super_integration/config/airsim_rviz.rviz"
echo ""
echo "显示内容:"
echo "  - AirSim实时点云: /cloud_registered"
echo "  - SUPER累积地图: /rog_map/occ"
echo "  - 规划路径: /fsm/path"
echo ""

# 启动RViz
rviz2 -d /mnt/data/airsim_super_integration/config/airsim_rviz.rviz


