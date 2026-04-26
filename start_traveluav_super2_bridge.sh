#!/bin/bash
set -e

# 1) 启动 SUPER
# source /opt/ros/humble/setup.bash
# source /mnt/super2/SUPER-master/install/setup.bash
# ros2 launch mission_planner click_demo.launch.py

# 2) 启动桥接
# source /opt/ros/humble/setup.bash
# source /mnt/super2/SUPER-master/install/setup.bash
# python3 /mnt/data/bridge_traveluav_super2_ros2.py

# 3) 启动 TravelUAV
# 在 TravelUAV 环境里运行 eval.py

echo "source /opt/ros/humble/setup.bash"
echo "source /mnt/super2/SUPER-master/install/setup.bash"
echo "ros2 launch mission_planner click_demo.launch.py"
echo ""
echo "另一个终端："
echo "source /opt/ros/humble/setup.bash"
echo "source /mnt/super2/SUPER-master/install/setup.bash"
echo "python3 /mnt/data/bridge_traveluav_super2_ros2.py"
