# 🚀 快速启动指南 (ROS2 Humble版本)

## ⚡ 3步启动系统

### 步骤1: 测试AirSim连接
```bash
cd /mnt/data/airsim_super_integration/scripts
python3 test_airsim_connection.py
```

**确保看到**:
```
✓✓✓ 所有测试通过！AirSim工作正常 ✓✓✓
```

### 步骤2: 一键启动集成系统
```bash
cd /mnt/data/airsim_super_integration
./start_airsim_super_ros2.sh
```

选择**模式2**（SUPER + 桥接，假设AirSim已运行）

**等待启动**:
- SUPER规划器启动（~5秒）
- 桥接节点启动（~2秒）

### 步骤3: 发送测试目标点
```bash
# 新终端
source /opt/ros/humble/setup.bash
cd /mnt/data/airsim_super_integration/scripts
python3 test_super_goal_ros2.py 10 5 2
```

## ✅ 验证系统运行

### 检查ROS2话题
```bash
# 查看所有话题
ros2 topic list

# 检查点云数据（应该~10 Hz）
ros2 topic hz /cloud_registered

# 检查里程计数据（应该~20 Hz）
ros2 topic hz /lidar_slam/odom

# 查看点云内容
ros2 topic echo /cloud_registered --once
```

**预期输出**:
```
/cloud_registered: ~10 Hz   # LiDAR点云
/lidar_slam/odom:  ~20 Hz   # 无人机状态
/goal: 按需          # 目标点
/planning/pos_cmd: ~15 Hz   # SUPER控制指令（有目标时）
```

## 🎯 常用操作

### 发送不同的目标点
```bash
# 近距离（5米）
python3 test_super_goal_ros2.py 5 0 2

# 中距离（10米）
python3 test_super_goal_ros2.py 10 5 2

# 远距离（20米）
python3 test_super_goal_ros2.py 20 10 3
```

### 使用ROS2命令发送目标
```bash
ros2 topic pub /goal geometry_msgs/PoseStamped \
  "{header: {frame_id: 'world'}, \
    pose: {position: {x: 10.0, y: 5.0, z: 2.0}}}" \
  --once
```

### 监控系统状态
```bash
# 查看节点
ros2 node list

# 查看话题详情
ros2 topic info /cloud_registered

# 实时查看点云
ros2 topic echo /cloud_registered

# 实时查看控制指令
ros2 topic echo /planning/pos_cmd
```

## 🐛 快速故障排查

### 问题1: 桥接节点报错 "No module named 'rospy'"
**原因**: 使用了ROS1的脚本  
**解决**: 确保使用ROS2版本的脚本：
```bash
cd /mnt/data/airsim_super_integration/scripts
python3 airsim_super_bridge_ros2.py  # 注意是 _ros2.py
```

### 问题2: 没有点云数据
```bash
# 检查话题
ros2 topic list | grep cloud

# 查看桥接节点日志（在桥接节点终端查看）
# 应该看到 "点云发布成功" 之类的信息
```

### 问题3: SUPER不规划路径
```bash
# 1. 确认收到目标点
ros2 topic echo /goal

# 2. 确认有地图数据
ros2 topic echo /cloud_registered --once

# 3. 重新发送目标
python3 test_super_goal_ros2.py 10 5 2
```

### 问题4: AirSim连接失败
```bash
# 测试连接
python3 test_airsim_connection.py

# 如果失败，检查AirSim是否运行
pgrep -f ModularPark

# 重启AirSim
cd /mnt/data/TravelUAV/envs/closeloop_envs
./ModularPark.sh -ResX=1280 -ResY=720 -windowed
```

## 📊 系统架构（简化）

```
AirSim (ModularPark)
    ↓ LiDAR + 状态
桥接节点 (airsim_super_bridge_ros2.py)
    ↓ ROS2话题
SUPER规划器 (fsm_node)
    ↓ 规划路径
桥接节点 (控制)
    ↓ AirSim API
无人机飞行
```

## 📝 重要文件位置

```
/mnt/data/airsim_super_integration/
├── start_airsim_super_ros2.sh          # 主启动脚本
├── scripts/
│   ├── airsim_super_bridge_ros2.py     # 桥接节点 ⭐
│   ├── test_super_goal_ros2.py         # 测试脚本
│   └── test_airsim_connection.py       # 连接测试
├── backup/                             # 原始文件备份
└── docs/
    ├── QUICK_START_ROS2.md             # 本文件
    └── INTEGRATION_GUIDE_ROS2.md       # 详细指南
```

## 🎓 下一步

1. **阅读详细文档**: `docs/INTEGRATION_GUIDE_ROS2.md`
2. **调整参数**: 修改SUPER配置文件
3. **集成TravelUAV**: 将视觉-语言规划接入
4. **实际飞行测试**: 在复杂场景中测试

---

**ROS版本**: ROS2 Humble  
**Python版本**: Python 3.x  
**更新日期**: 2026-01-20

**享受你的统一导航避障系统！** 🎉


