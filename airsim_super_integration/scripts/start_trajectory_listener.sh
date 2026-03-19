#!/bin/bash
# 启动SUPER轨迹监听器
# 用于接收并保存SUPER规划的轨迹

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 退出conda环境（避免冲突）
conda deactivate 2>/dev/null || true

# 加载ROS2环境
source /opt/ros/humble/setup.bash

# 加载SUPER工作空间（包含mars_quadrotor_msgs）
source ~/super_ws/install/local_setup.bash

echo "✓ 环境已加载"
echo "  - ROS2: Humble"
echo "  - SUPER工作空间: ~/super_ws"
echo ""
echo "========================================"
echo "启动 SUPER 轨迹监听器"
echo "========================================"
echo "订阅话题: /planning/pos_cmd"
echo "输出文件: /tmp/traveluav_super_bridge/latest_trajectory.json"
echo ""
echo "按 Ctrl+C 停止"
echo "========================================"

# 运行轨迹监听器（使用系统Python，避免conda干扰）
/usr/bin/python3 "$SCRIPT_DIR/trajectory_listener.py"
