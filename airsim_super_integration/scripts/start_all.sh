#!/bin/bash

# 一键启动完整的 TravelUAV-SUPER 集成系统

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}        启动 TravelUAV-SUPER 完整集成系统                              ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# 检查 AirSim 是否运行
if ! pgrep -f "ModularPark" > /dev/null; then
    echo -e "${RED}✗ AirSim 未运行，请先启动 TravelUAV 环境${NC}"
    echo ""
    echo "在另一个终端运行："
    echo "  cd /mnt/data/TravelUAV/envs/closeloop_envs"
    echo "  ./ModularPark.sh -ResX=1280 -ResY=720 -windowed"
    exit 1
fi

echo -e "${GREEN}✓ AirSim 已运行${NC}"
echo ""

# 启动顺序
echo -e "${YELLOW}启动顺序：${NC}"
echo "  1. 桥接节点（AirSim ↔ ROS2）"
echo "  2. SUPER 规划器"
echo "  3. RViz 可视化"
echo ""

# 使用 tmux 或 gnome-terminal 启动多个终端
if command -v gnome-terminal &> /dev/null; then
    echo -e "${GREEN}使用 gnome-terminal 启动各个节点...${NC}"
    
    # 终端1: 桥接节点
    gnome-terminal --tab --title="Bridge" -- bash -c "
        conda deactivate 2>/dev/null || true
        cd /mnt/data/airsim_super_integration/scripts
        ./run_bridge.sh
        exec bash
    "
    
    sleep 3
    
    # 终端2: SUPER
    gnome-terminal --tab --title="SUPER" -- bash -c "
        conda deactivate 2>/dev/null || true
        cd /mnt/data/airsim_super_integration/scripts
        ./start_super_airsim.sh
        exec bash
    "
    
    sleep 3
    
    # 终端3: RViz
    gnome-terminal --tab --title="RViz" -- bash -c "
        conda deactivate 2>/dev/null || true
        cd /mnt/data/airsim_super_integration/scripts
        ./start_rviz.sh
        exec bash
    "
    
    echo ""
    echo -e "${GREEN}✓ 所有节点已启动！${NC}"
    echo ""
    echo -e "${YELLOW}等待 5 秒让系统初始化...${NC}"
    sleep 5
    
    echo ""
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}        系统已就绪！现在可以发送目标点                                ${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo ""
    echo "发送目标示例："
    echo "  ./send_goal.sh 0 5.0 -7.0   # 往前 5 米"
    echo "  ./send_goal.sh 3.0 0 -5.0   # 往右 3 米，往上 2 米"
    echo ""
    
else
    echo -e "${YELLOW}请手动按顺序启动：${NC}"
    echo ""
    echo "终端1: ./run_bridge.sh"
    echo "终端2: ./start_super_airsim.sh"
    echo "终端3: ./start_rviz.sh"
fi
