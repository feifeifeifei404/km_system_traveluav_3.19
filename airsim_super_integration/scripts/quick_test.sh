#!/bin/bash

# SUPER快速测试脚本
# 这个脚本会在新的终端窗口中启动所有必要的组件

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}              SUPER快速测试启动脚本                              ${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""

# 检查是否有显示
if [ -z "$DISPLAY" ]; then
    echo -e "${RED}✗ 未检测到显示设备，请先设置DISPLAY环境变量${NC}"
    exit 1
fi

echo -e "${BLUE}[1/5] 检查必要文件...${NC}"

# 检查AirSim环境
if [ ! -f "/mnt/data/TravelUAV/envs/closeloop_envs/ModularPark.sh" ]; then
    echo -e "${RED}✗ AirSim环境不存在${NC}"
    exit 1
fi

# 检查脚本
for script in start_super_airsim.sh run_bridge.sh send_goal.sh; do
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo -e "${RED}✗ 脚本不存在: $script${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✓ 所有必要文件就绪${NC}"
echo ""

# 函数: 在新终端中启动命令
launch_in_terminal() {
    local title=$1
    local cmd=$2
    
    # 尝试不同的终端模拟器
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="$title" -- bash -c "$cmd; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -T "$title" -e "$cmd; bash" &
    elif command -v konsole &> /dev/null; then
        konsole --new-tab -p tabtitle="$title" -e bash -c "$cmd; exec bash" &
    else
        echo -e "${RED}✗ 未找到终端模拟器（gnome-terminal, xterm, konsole）${NC}"
        exit 1
    fi
    
    sleep 2
}

echo -e "${BLUE}[2/5] 启动AirSim环境...${NC}"
echo "这将在新窗口中启动，请等待环境加载完成（看到无人机）"
echo ""

# 启动AirSim（在后台）
cd /mnt/data/TravelUAV/envs/closeloop_envs
./ModularPark.sh -ResX=1280 -ResY=720 -windowed &
AIRSIM_PID=$!

echo -e "${YELLOW}AirSim已在后台启动 (PID: $AIRSIM_PID)${NC}"
echo "等待30秒以便环境完全加载..."
sleep 30

echo ""
echo -e "${BLUE}[3/5] 启动SUPER规划器...${NC}"
launch_in_terminal "SUPER规划器" "cd $SCRIPT_DIR && ./start_super_airsim.sh"

echo ""
echo -e "${BLUE}[4/5] 启动桥接节点...${NC}"
launch_in_terminal "桥接节点" "cd $SCRIPT_DIR && ./run_bridge.sh"

echo ""
echo -e "${BLUE}[5/5] 准备发送目标点...${NC}"
echo "等待5秒让系统稳定..."
sleep 5

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}                    启动完成！                                  ${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo -e "${YELLOW}组件状态:${NC}"
echo "  ✓ AirSim环境 (PID: $AIRSIM_PID)"
echo "  ✓ SUPER规划器 (新终端)"
echo "  ✓ 桥接节点 (新终端)"
echo ""
echo -e "${YELLOW}下一步操作:${NC}"
echo ""
echo "1. 查看当前位置:"
echo "   ${BLUE}ros2 topic echo /lidar_slam/odom --once | grep -A 3 \"position:\"${NC}"
echo ""
echo "2. 发送测试目标点:"
echo "   ${BLUE}cd $SCRIPT_DIR${NC}"
echo "   ${BLUE}./send_goal.sh 3 7 -3${NC}"
echo ""
echo "3. 启动RViz可视化（可选）:"
echo "   ${BLUE}cd $SCRIPT_DIR${NC}"
echo "   ${BLUE}./start_rviz.sh${NC}"
echo ""
echo -e "${YELLOW}结束测试:${NC}"
echo "  - 关闭各个终端窗口"
echo "  - 或运行: ${RED}pkill -9 -f ModularPark${NC}"
echo ""
echo -e "${GREEN}================================================================${NC}"
echo ""

# 保存PID以便后续清理
echo $AIRSIM_PID > /tmp/airsim_test.pid

echo "按Enter键退出此脚本（组件将继续运行）..."
read

echo -e "${GREEN}脚本已退出，测试环境继续运行${NC}"
echo "要停止所有组件，请运行: pkill -9 -f ModularPark"
