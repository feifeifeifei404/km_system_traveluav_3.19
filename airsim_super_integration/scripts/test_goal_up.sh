#!/bin/bash

# 自动获取当前位置并发送"往上1米"的目标

echo "获取当前无人机位置..."

# 退出conda
conda deactivate 2>/dev/null || true

# 加载ROS2环境
source /opt/ros/humble/setup.bash
source ~/super_ws/install/local_setup.bash

# 获取当前位置
ODOM=$(ros2 topic echo /lidar_slam/odom --once 2>/dev/null)

if [ -z "$ODOM" ]; then
    echo "❌ 无法获取位置，请检查桥接节点是否运行"
    exit 1
fi

# 解析 x, y, z
X=$(echo "$ODOM" | grep -A 3 "position:" | grep "x:" | head -1 | awk '{print $2}')
Y=$(echo "$ODOM" | grep -A 3 "position:" | grep "y:" | head -1 | awk '{print $2}')
Z=$(echo "$ODOM" | grep -A 3 "position:" | grep "z:" | head -1 | awk '{print $2}')

echo "当前位置:"
echo "  x: $X"
echo "  y: $Y"
echo "  z: $Z"

# 计算目标位置（往上1米）
TARGET_Z=$(echo "$Z + 1.0" | bc)

echo ""
echo "目标位置（往上1米）:"
echo "  x: $X"
echo "  y: $Y"
echo "  z: $TARGET_Z"
echo ""

# 发送目标
cd "$(dirname "$0")"
/usr/bin/python3 test_super_goal_ros2.py "$X" "$Y" "$TARGET_Z"
