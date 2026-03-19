#!/bin/bash

# 启动双向Socket桥接服务（ROS2环境）

# 退出conda环境
conda deactivate 2>/dev/null || true

# 加载ROS2环境
source /opt/ros/humble/setup.bash

# 加载SUPER工作空间
source ~/super_ws/install/local_setup.bash

echo "======================================================================
启动双向Socket桥接服务
======================================================================

功能:
  - 接收TravelUAV目标点 → 发布到ROS2
  - 监听SUPER状态 → 返回给TravelUAV

监听端口: 127.0.0.1:65432
======================================================================
"

# 运行桥接服务
cd "$(dirname "$0")"
/usr/bin/python3 socket_to_ros2_bridge_bidirectional.py
