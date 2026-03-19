#!/bin/bash

# AirSim-SUPER桥接节点启动脚本（自动设置环境）

# 退出conda环境
conda deactivate 2>/dev/null || true

# 加载ROS2环境
source /opt/ros/humble/setup.bash

# 加载SUPER工作空间（包含mars_quadrotor_msgs）
source ~/super_ws/install/local_setup.bash

echo "✓ 环境已加载"
echo "  - ROS2: Humble"
echo "  - SUPER工作空间: ~/super_ws"
echo ""
echo "启动AirSim-SUPER桥接节点..."
echo ""

# 运行桥接节点（使用系统Python，避免conda干扰）
cd "$(dirname "$0")"
/usr/bin/python3 airsim_super_bridge_ros2.py

