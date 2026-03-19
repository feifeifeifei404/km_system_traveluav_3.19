#!/bin/bash

# 停止AirSim环境脚本

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}正在停止AirSim...${NC}"

# 强制结束AirSim
pkill -9 -f ModularPark

# 等待进程完全结束
sleep 2

# 检查是否还在运行
if ps aux | grep -i modularpark | grep -v grep > /dev/null; then
    echo -e "${RED}✗ AirSim进程仍在运行${NC}"
    exit 1
else
    echo -e "${GREEN}✓ AirSim已停止${NC}"
fi
