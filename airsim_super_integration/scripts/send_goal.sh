#!/bin/bash

# 发送目标点脚本（自动设置环境）

# 退出conda环境
conda deactivate 2>/dev/null || true

# 加载ROS2环境
source /opt/ros/humble/setup.bash

# 加载SUPER工作空间
source ~/super_ws/install/local_setup.bash

# 发送目标点
cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    echo "使用默认目标点: (10, 5, 2)"
    /usr/bin/python3 test_super_goal_ros2.py 10 5 2
else
    echo "发送目标点: ($1, $2, $3)"
    /usr/bin/python3 test_super_goal_ros2.py "$@"
fi

