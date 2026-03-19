#!/bin/bash

# 系统诊断脚本
# 检查AirSim-SUPER集成系统的所有组件状态

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}              AirSim-SUPER 系统诊断${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# 1. 检查ROS2环境
echo -e "${YELLOW}[1/7] 检查ROS2环境...${NC}"
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash 2>/dev/null
    echo -e "${GREEN}✓ ROS2 Humble 已安装${NC}"
else
    echo -e "${RED}✗ ROS2 Humble 未找到${NC}"
fi
echo ""

# 2. 检查ROS2节点
echo -e "${YELLOW}[2/7] 检查运行中的ROS2节点...${NC}"
NODES=$(ros2 node list 2>/dev/null)
if echo "$NODES" | grep -q "fsm_node"; then
    echo -e "${GREEN}✓ SUPER规划器正在运行${NC}"
else
    echo -e "${RED}✗ SUPER规划器未运行${NC}"
    echo -e "   启动: cd /mnt/data/airsim_super_integration/scripts && ./start_super_airsim.sh"
fi

if echo "$NODES" | grep -q "airsim_super_bridge"; then
    echo -e "${GREEN}✓ 桥接节点正在运行${NC}"
else
    echo -e "${RED}✗ 桥接节点未运行${NC}"
    echo -e "   启动: cd /mnt/data/airsim_super_integration/scripts && ./run_bridge.sh"
fi
echo ""

# 3. 检查关键话题
echo -e "${YELLOW}[3/7] 检查关键话题...${NC}"
TOPICS=$(ros2 topic list 2>/dev/null)

check_topic() {
    topic=$1
    name=$2
    if echo "$TOPICS" | grep -q "$topic"; then
        hz=$(timeout 2 ros2 topic hz "$topic" 2>&1 | grep "average rate" | awk '{print $3}')
        if [ -n "$hz" ]; then
            echo -e "${GREEN}✓ $name: ${hz} Hz${NC}"
        else
            echo -e "${YELLOW}⚠ $name: 已发布但无数据${NC}"
        fi
    else
        echo -e "${RED}✗ $name: 未发布${NC}"
    fi
}

check_topic "/cloud_registered" "点云数据"
check_topic "/lidar_slam/odom" "里程计数据"
check_topic "/planning/pos_cmd" "控制指令"
check_topic "/fsm/path" "规划路径"
echo ""

# 4. 检查AirSim连接
echo -e "${YELLOW}[4/7] 检查AirSim连接...${NC}"
python3 -c "
import airsim
import sys
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print('${GREEN}✓ AirSim连接正常${NC}')
    
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f'  位置: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})')
    print(f'  着陆状态: {state.landed_state}')
    
    if state.landed_state == airsim.LandedState.Landed:
        print('${YELLOW}⚠ 无人机在地面，可能需要起飞${NC}')
        print('  运行: python3 arm_and_takeoff.py')
    
except Exception as e:
    print('${RED}✗ AirSim连接失败${NC}')
    print(f'  错误: {e}')
    print('  请确保AirSim环境正在运行')
    sys.exit(1)
" 2>/dev/null
echo ""

# 5. 检查SUPER状态
echo -e "${YELLOW}[5/7] 检查SUPER规划器状态...${NC}"
if ros2 node list 2>/dev/null | grep -q "fsm_node"; then
    # 检查是否接收到目标点
    GOAL_TOPIC=$(ros2 topic list 2>/dev/null | grep "/goal")
    if [ -n "$GOAL_TOPIC" ]; then
        echo -e "${GREEN}✓ 目标点话题存在${NC}"
    else
        echo -e "${YELLOW}⚠ 未找到目标点话题${NC}"
    fi
    
    # 检查是否在发布控制指令
    CMD_HZ=$(timeout 2 ros2 topic hz /planning/pos_cmd 2>&1 | grep "average rate")
    if [ -n "$CMD_HZ" ]; then
        echo -e "${GREEN}✓ 正在发布控制指令${NC}"
    else
        echo -e "${YELLOW}⚠ 未发布控制指令（可能在等待目标点）${NC}"
        echo -e "   发送目标: ./send_goal.sh 10 5 2"
    fi
else
    echo -e "${RED}✗ SUPER未运行${NC}"
fi
echo ""

# 6. 检查数据流
echo -e "${YELLOW}[6/7] 检查数据流...${NC}"
echo -e "数据流应该是: ${BLUE}AirSim → 桥接 → SUPER → 桥接 → AirSim${NC}"

HAS_CLOUD=$(echo "$TOPICS" | grep -c "/cloud_registered")
HAS_ODOM=$(echo "$TOPICS" | grep -c "/lidar_slam/odom")
HAS_CMD=$(echo "$TOPICS" | grep -c "/planning/pos_cmd")

if [ $HAS_CLOUD -gt 0 ] && [ $HAS_ODOM -gt 0 ]; then
    echo -e "${GREEN}✓ AirSim → 桥接节点: 正常${NC}"
else
    echo -e "${RED}✗ AirSim → 桥接节点: 异常${NC}"
fi

if [ $HAS_CMD -gt 0 ]; then
    CMD_RATE=$(timeout 2 ros2 topic hz /planning/pos_cmd 2>&1 | grep "average rate" | wc -l)
    if [ $CMD_RATE -gt 0 ]; then
        echo -e "${GREEN}✓ SUPER → 桥接节点: 正常${NC}"
    else
        echo -e "${YELLOW}⚠ SUPER → 桥接节点: 等待目标点${NC}"
    fi
else
    echo -e "${RED}✗ SUPER → 桥接节点: 无数据${NC}"
fi
echo ""

# 7. 系统建议
echo -e "${YELLOW}[7/7] 系统建议...${NC}"

if ! ros2 node list 2>/dev/null | grep -q "airsim_super_bridge"; then
    echo -e "${RED}! 启动桥接节点: ./run_bridge.sh${NC}"
fi

if ! ros2 node list 2>/dev/null | grep -q "fsm_node"; then
    echo -e "${RED}! 启动SUPER: ./start_super_airsim.sh${NC}"
fi

python3 -c "
import airsim
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    state = client.getMultirotorState()
    if state.landed_state == airsim.LandedState.Landed:
        print('${YELLOW}! 无人机需要起飞: python3 arm_and_takeoff.py${NC}')
except:
    pass
" 2>/dev/null

CMD_RATE=$(timeout 2 ros2 topic hz /planning/pos_cmd 2>&1 | grep "average rate" | wc -l)
if [ $CMD_RATE -eq 0 ] && ros2 node list 2>/dev/null | grep -q "fsm_node"; then
    echo -e "${YELLOW}! 发送目标点: ./send_goal.sh 10 5 2${NC}"
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}              诊断完成${NC}"
echo -e "${BLUE}======================================================================${NC}"


