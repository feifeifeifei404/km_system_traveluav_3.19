#!/bin/bash

# SUPER + AirSim 统一启动脚本 (ROS2 Humble版本)
#
# 功能: 一键启动完整的TravelUAV+SUPER+AirSim集成系统
# 作者: Integration Team
# 日期: 2026-01-20
# ROS版本: ROS2 Humble

set -e  # 出错时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置路径
SUPER_WS="/home/wuyou/super_ws"
AIRSIM_ENV="/mnt/data/TravelUAV/envs/closeloop_envs"
INTEGRATION_DIR="/mnt/data/airsim_super_integration"
BRIDGE_SCRIPT="${INTEGRATION_DIR}/scripts/airsim_super_bridge_ros2.py"

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}    TravelUAV + SUPER + AirSim 统一集成系统启动 (ROS2 Humble)   ${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

# ====================
# 检查依赖
# ====================
echo -e "${BLUE}[1/5] 检查系统依赖...${NC}"

# 检查ROS2
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    echo -e "${RED}✗ ROS2 Humble未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ROS2 Humble已安装${NC}"

# 检查SUPER工作空间
if [ ! -d "$SUPER_WS" ]; then
    echo -e "${RED}✗ SUPER工作空间不存在: ${SUPER_WS}${NC}"
    exit 1
fi

if [ ! -f "$SUPER_WS/install/local_setup.bash" ]; then
    echo -e "${RED}✗ SUPER工作空间未编译${NC}"
    echo -e "${YELLOW}请先编译SUPER:${NC}"
    echo "  cd $SUPER_WS"
    echo "  colcon build"
    exit 1
fi
echo -e "${GREEN}✓ SUPER工作空间: ${SUPER_WS}${NC}"

# 检查AirSim环境
if [ ! -d "$AIRSIM_ENV" ]; then
    echo -e "${RED}✗ AirSim环境不存在: ${AIRSIM_ENV}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ AirSim环境: ${AIRSIM_ENV}${NC}"

# 检查Python依赖
if ! python3 -c "import airsim" 2>/dev/null; then
    echo -e "${RED}✗ airsim Python包未安装${NC}"
    echo -e "${YELLOW}请运行: pip install airsim${NC}"
    exit 1
fi
echo -e "${GREEN}✓ airsim Python包已安装${NC}"

echo ""

# ====================
# 启动模式选择
# ====================
echo -e "${CYAN}选择启动模式:${NC}"
echo ""
echo "  ${YELLOW}1.${NC} 完整启动 (AirSim + SUPER + 桥接节点)"
echo "  ${YELLOW}2.${NC} 仅启动SUPER + 桥接 (假设AirSim已运行)"
echo "  ${YELLOW}3.${NC} 仅启动桥接节点 (假设AirSim和SUPER都已运行)"
echo "  ${YELLOW}4.${NC} 测试AirSim连接"
echo ""
read -p "$(echo -e ${CYAN}请选择 [1-4]:${NC}) " mode

case $mode in
    1)
        echo ""
        echo -e "${BLUE}[2/5] 启动AirSim环境...${NC}"
        echo -e "${YELLOW}在新终端中启动ModularPark...${NC}"
        
        # 在新终端启动AirSim
        gnome-terminal --title="AirSim环境" -- bash -c "
            cd $AIRSIM_ENV
            echo -e '${GREEN}启动AirSim环境...${NC}'
            ./ModularPark.sh -ResX=1280 -ResY=720 -windowed
            exec bash
        " &
        
        echo -e "${GREEN}✓ AirSim环境已在新终端启动${NC}"
        echo -e "${YELLOW}等待15秒让AirSim完全启动...${NC}"
        sleep 15
        
        # 启动SUPER (ROS2)
        echo ""
        echo -e "${BLUE}[3/5] 启动SUPER规划器 (ROS2)...${NC}"
        
        gnome-terminal --title="SUPER规划器 (ROS2)" -- bash -c "
            conda deactivate 2>/dev/null || true
            source /opt/ros/humble/setup.bash
            source $SUPER_WS/install/local_setup.bash
            echo -e '${GREEN}启动SUPER规划节点 (ROS2)...${NC}'
            ros2 launch mission_planner click_demo.launch.py
            exec bash
        " &
        
        echo -e "${GREEN}✓ SUPER规划器已在新终端启动${NC}"
        echo -e "${YELLOW}等待5秒让SUPER初始化...${NC}"
        sleep 5
        
        # 启动桥接节点
        echo ""
        echo -e "${BLUE}[4/5] 启动AirSim-SUPER桥接节点 (ROS2)...${NC}"
        
        gnome-terminal --title="AirSim桥接节点 (ROS2)" -- bash -c "
            conda deactivate 2>/dev/null || true
            source /opt/ros/humble/setup.bash
            cd ${INTEGRATION_DIR}/scripts
            echo -e '${GREEN}启动桥接节点 (ROS2)...${NC}'
            python3 airsim_super_bridge_ros2.py
            exec bash
        " &
        
        echo -e "${GREEN}✓ 桥接节点已在新终端启动${NC}"
        sleep 2
        ;;
        
    2)
        echo ""
        echo -e "${BLUE}[2/5] 检查AirSim是否运行...${NC}"
        
        if ! pgrep -f "ModularPark" > /dev/null; then
            echo -e "${RED}✗ AirSim环境未运行${NC}"
            echo -e "${YELLOW}请先启动AirSim环境:${NC}"
            echo "  cd $AIRSIM_ENV"
            echo "  ./ModularPark.sh -ResX=1280 -ResY=720 -windowed"
            exit 1
        fi
        echo -e "${GREEN}✓ AirSim环境正在运行${NC}"
        
        # 启动SUPER (ROS2)
        echo ""
        echo -e "${BLUE}[3/5] 启动SUPER规划器 (ROS2)...${NC}"
        
        gnome-terminal --title="SUPER规划器 (ROS2)" -- bash -c "
            conda deactivate 2>/dev/null || true
            source /opt/ros/humble/setup.bash
            source $SUPER_WS/install/local_setup.bash
            echo -e '${GREEN}启动SUPER规划节点 (ROS2)...${NC}'
            ros2 launch mission_planner click_demo.launch.py
            exec bash
        " &
        
        echo -e "${GREEN}✓ SUPER规划器已在新终端启动${NC}"
        sleep 5
        
        # 启动桥接节点
        echo ""
        echo -e "${BLUE}[4/5] 启动AirSim-SUPER桥接节点 (ROS2)...${NC}"
        
        gnome-terminal --title="AirSim桥接节点 (ROS2)" -- bash -c "
            conda deactivate 2>/dev/null || true
            source /opt/ros/humble/setup.bash
            cd ${INTEGRATION_DIR}/scripts
            echo -e '${GREEN}启动桥接节点 (ROS2)...${NC}'
            python3 airsim_super_bridge_ros2.py
            exec bash
        " &
        
        echo -e "${GREEN}✓ 桥接节点已在新终端启动${NC}"
        sleep 2
        ;;
        
    3)
        echo ""
        echo -e "${BLUE}[2/5] 检查依赖服务...${NC}"
        
        # 检查AirSim
        if ! pgrep -f "ModularPark" > /dev/null; then
            echo -e "${YELLOW}⚠ AirSim环境未运行${NC}"
        else
            echo -e "${GREEN}✓ AirSim环境正在运行${NC}"
        fi
        
        # 检查ROS2节点
        if ! pgrep -f "ros2" > /dev/null; then
            echo -e "${RED}✗ ROS2节点未运行，请先启动SUPER${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ ROS2节点正在运行${NC}"
        
        # 启动桥接节点
        echo ""
        echo -e "${BLUE}[3/5] 启动AirSim-SUPER桥接节点 (ROS2)...${NC}"
        
        gnome-terminal --title="AirSim桥接节点 (ROS2)" -- bash -c "
            conda deactivate 2>/dev/null || true
            source /opt/ros/humble/setup.bash
            cd ${INTEGRATION_DIR}/scripts
            echo -e '${GREEN}启动桥接节点 (ROS2)...${NC}'
            python3 airsim_super_bridge_ros2.py
            exec bash
        " &
        
        echo -e "${GREEN}✓ 桥接节点已在新终端启动${NC}"
        sleep 2
        ;;
        
    4)
        echo ""
        echo -e "${BLUE}[2/5] 测试AirSim连接...${NC}"
        
        cd ${INTEGRATION_DIR}/scripts
        python3 test_airsim_connection.py
        exit 0
        ;;
        
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

# ====================
# 完成
# ====================
echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}                         系统启动完成!                               ${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo -e "${CYAN}📡 ROS2话题监控:${NC}"
echo "  ros2 topic echo /cloud_registered     # 查看点云数据"
echo "  ros2 topic echo /lidar_slam/odom      # 查看里程计数据"
echo "  ros2 topic echo /planning/pos_cmd     # 查看SUPER控制指令"
echo ""
echo -e "${CYAN}🎯 发送目标点:${NC}"
echo "  cd ${INTEGRATION_DIR}/scripts"
echo "  python3 test_super_goal_ros2.py 10 5 2"
echo ""
echo -e "${CYAN}或使用ros2命令:${NC}"
echo "  ros2 topic pub /goal geometry_msgs/PoseStamped \\"
echo "    \"{header: {frame_id: 'world'}, pose: {position: {x: 10.0, y: 5.0, z: 2.0}}}\""
echo ""
echo -e "${CYAN}📊 可视化:${NC}"
echo "  RViz2已自动启动（如果包含在launch文件中）"
echo "  或手动启动: ros2 run rviz2 rviz2"
echo ""
echo -e "${YELLOW}按Ctrl+C退出...${NC}"
echo ""

# 保持脚本运行
wait


