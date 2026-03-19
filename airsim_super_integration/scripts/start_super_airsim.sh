#!/bin/bash

# 启动AirSim集成版的SUPER规划器
#
# 这个脚本启动专门为AirSim定制的SUPER，不包含Perfect Drone仿真器
#
# 使用方法:
#   cd /mnt/data/airsim_super_integration/scripts
#   ./start_super_airsim.sh

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}        启动AirSim集成版SUPER规划器                                    ${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

# 退出conda环境
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

echo -e "${BLUE}[1/3] 检查SUPER工作空间...${NC}"

# 检查SUPER工作空间
if [ ! -d "/home/wuyou/super_ws" ]; then
    echo -e "${RED}✗ SUPER工作空间不存在${NC}"
    exit 1
fi

if [ ! -f "/home/wuyou/super_ws/install/local_setup.bash" ]; then
    echo -e "${RED}✗ SUPER工作空间未编译${NC}"
    exit 1
fi

echo -e "${GREEN}✓ SUPER工作空间就绪${NC}"

# 加载ROS2环境
echo ""
echo -e "${BLUE}[2/3] 加载ROS2环境...${NC}"
source /opt/ros/humble/setup.bash
source /home/wuyou/super_ws/install/local_setup.bash
echo -e "${GREEN}✓ ROS2环境已加载${NC}"

# 启动SUPER
echo ""
echo -e "${BLUE}[3/3] 启动SUPER规划器...${NC}"
echo ""
echo -e "${YELLOW}配置说明:${NC}"
echo "  - 不启动Perfect Drone仿真器（使用AirSim）"
echo "  - 不启动RViz窗口（使用单独的RViz脚本）"
echo "  - 只启动SUPER规划器核心节点"
echo ""
echo -e "${YELLOW}订阅的话题:${NC}"
echo "  - /cloud_registered  (AirSim实时点云)"
echo "  - /lidar_slam/odom   (AirSim里程计)"
echo "  - /goal              (目标点)"
echo ""
echo -e "${YELLOW}发布的话题:${NC}"
echo "  - /planning/pos_cmd  (控制指令)"
echo "  - /fsm/path          (规划路径)"
echo "  - /rog_map/occ       (占据地图)"
echo ""
echo -e "${GREEN}======================================================================${NC}"
echo ""

# 启动SUPER
ros2 launch /mnt/data/airsim_super_integration/launch/airsim_super.launch.py
