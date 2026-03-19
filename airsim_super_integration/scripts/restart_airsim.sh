#!/bin/bash

# 停止AirSim环境脚本（远程服务器版本）

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}正在停止AirSim...${NC}"

# 强制结束AirSim
pkill -9 -f ModularPark

# 等待进程完全结束
sleep 3

# 检查是否还在运行
if ps aux | grep -i modularpark | grep -v grep > /dev/null; then
    echo -e "${RED}✗ AirSim进程仍在运行，请手动结束${NC}"
    exit 1
fi

echo -e "${GREEN}✓ AirSim已停止${NC}"
echo ""
echo -e "${YELLOW}下一步操作：${NC}"
echo "  1. 请在服务器本地（有显示器的机器）启动AirSim："
echo -e "     ${BLUE}cd /mnt/data/TravelUAV/envs/closeloop_envs${NC}"
echo -e "     ${BLUE}./ModularPark.sh -ResX=1280 -ResY=720 -windowed${NC}"
echo ""
echo "  2. 等待30-40秒，直到看到无人机和场景完全加载"
echo ""
echo "  3. 验证API服务器启动："
echo -e "     ${BLUE}netstat -tuln | grep 41451${NC}"
echo ""
echo "  4. 然后在远程终端重新运行桥接节点："
echo -e "     ${BLUE}cd /mnt/data/airsim_super_integration/scripts${NC}"
echo -e "     ${BLUE}./run_bridge.sh${NC}"
echo ""
