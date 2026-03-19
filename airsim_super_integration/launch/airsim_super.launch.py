"""
AirSim-SUPER 集成系统 Launch 文件

专门为AirSim实时点云定制的SUPER启动文件

特点:
- ✅ 启动SUPER规划器（核心）
- ❌ 不启动Perfect Drone仿真器（使用AirSim代替）
- ❌ 不启动RViz（使用单独的RViz脚本）
- ✅ 使用AirSim的实时点云数据
- ✅ 使用AirSim的实时里程计数据

使用方法:
    cd ~/super_ws
    source /opt/ros/humble/setup.bash
    source install/local_setup.bash
    ros2 launch /mnt/data/airsim_super_integration/launch/airsim_super.launch.py

作者: Integration Team
日期: 2026-01-20
ROS版本: ROS2 Humble
"""

import os.path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成launch描述"""
    
    # SUPER规划器的配置文件
    # 使用专门为AirSim定制的配置
    super_config_name = 'click_smooth_ros2.yaml'
    
    # 声明launch参数
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='false',
        description='Use simulation clock if true'
    )
    
    declare_super_config_cmd = DeclareLaunchArgument(
        'super_config', 
        default_value=super_config_name,
        description='SUPER planner config file name'
    )
    
    # ==============================================
    # SUPER规划器节点（核心！）
    # ==============================================
    super_planner_node = Node(
        package='super_planner',
        executable='fsm_node',
        name='fsm_node',
        output='screen',
        parameters=[{
            'config_name': LaunchConfiguration('super_config'),
        }]
    )
    
    # ==============================================
    # 构建Launch描述
    # ==============================================
    ld = LaunchDescription()
    
    # 添加参数声明
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_super_config_cmd)
    
    # 添加SUPER规划器（这是唯一需要的节点！）
    ld.add_action(super_planner_node)
    
    return ld


"""
================================================================================
                        启动后SUPER会做什么？
================================================================================

📥 订阅的话题（从AirSim桥接节点获取）:
    /cloud_registered       - AirSim实时LiDAR点云
    /lidar_slam/odom        - AirSim无人机状态（位置、速度、姿态）
    /goal                   - 目标点（从send_goal.sh发送）

📤 发布的话题（给桥接节点和RViz）:
    /planning/pos_cmd       - 控制指令（发给AirSim）
    /fsm/path               - 规划路径（显示在RViz）
    /rog_map/occ            - 占据栅格地图（显示在RViz）
    /rog_map/unk            - 未知区域地图（显示在RViz）

🔧 工作流程:
    1. 等待接收点云和里程计数据
    2. 建立环境地图（ROG-Map）
    3. 等待目标点
    4. 规划避障路径
    5. 发布控制指令
    6. 循环重复

================================================================================
"""
