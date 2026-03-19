#!/bin/bash
# 快速重启脚本

echo "======================================================================="
echo "重启 AirSim-SUPER 桥接节点"
echo "======================================================================="

# 1. 杀死旧进程
echo "1️⃣ 杀死旧的桥接节点进程..."
pkill -f airsim_super_bridge_ros2
sleep 2

# 2. 启动新的桥接节点
echo "2️⃣ 启动新的桥接节点..."
cd /mnt/data/airsim_super_integration/scripts

# 设置 ROS 环境
source /opt/ros/humble/setup.bash

# 启动桥接节点
/usr/bin/python3 airsim_super_bridge_ros2.py

echo "======================================================================="
echo "桥接节点已启动"
echo "======================================================================="
echo ""
echo "在另一个终端中运行："
echo "  cd /mnt/data/airsim_super_integration/launch"
echo "  ros2 launch airsim_super.launch.py"
echo ""
