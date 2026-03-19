#!/usr/bin/env python3
"""
AirSim-SUPER 桥接节点 (ROS2 Humble版本)

功能:
1. 从AirSim获取LiDAR点云，发布到 /cloud_registered
2. 从AirSim获取无人机状态，发布到 /lidar_slam/odom  
3. 订阅SUPER的控制指令并发送给AirSim

作者: Integration Team
日期: 2026-01-20
ROS版本: ROS2 Humble
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
import airsim
import numpy as np
import time
from threading import Lock

# ROS2消息类型
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import TransformBroadcaster

# SUPER的控制指令消息类型 (ROS2)
try:
    from mars_quadrotor_msgs.msg import PositionCommand
    HAVE_MARS_MSGS = True
except ImportError:
    HAVE_MARS_MSGS = False
    print("⚠ mars_quadrotor_msgs未安装，控制功能将被禁用")


class AirSimSuperBridgeROS2(Node):
    """AirSim-SUPER完整桥接节点 (ROS2)"""
    
    def __init__(self):
        super().__init__('airsim_super_bridge_ros2')
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("AirSim-SUPER 集成桥接节点启动 (ROS2 Humble)")
        self.get_logger().info("=" * 70)
        
        # ====================
        # 参数配置
        # ====================
        self.declare_parameter('vehicle_name', 'Drone_1')
        self.declare_parameter('lidar_name', 'Lidar1')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('world_frame_id', 'world')
        self.declare_parameter('body_frame_id', 'body')
        self.declare_parameter('use_enu', True)
        self.declare_parameter('control_z_up_positive', True)
        self.declare_parameter('airsim_ip', '127.0.0.1')
        self.declare_parameter('airsim_port', 0)
        self.declare_parameter('airsim_port_candidates', [25001, 25002, 25003])
        self.declare_parameter('airsim_rpc_timeout_sec', 3.0)
        self.declare_parameter('airsim_connect_timeout_sec', 12.0)
        self.declare_parameter('state_log_interval_sec', 2.0)
        
        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.lidar_name = self.get_parameter('lidar_name').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.world_frame_id = self.get_parameter('world_frame_id').value
        self.body_frame_id = self.get_parameter('body_frame_id').value
        self.use_enu = self.get_parameter('use_enu').value
        self.control_z_up_positive = self.get_parameter('control_z_up_positive').value
        self.airsim_ip = self.get_parameter('airsim_ip').value
        self.airsim_port = int(self.get_parameter('airsim_port').value)
        self.airsim_port_candidates = [int(p) for p in self.get_parameter('airsim_port_candidates').value]
        self.airsim_rpc_timeout_sec = float(self.get_parameter('airsim_rpc_timeout_sec').value)
        self.airsim_connect_timeout_sec = float(self.get_parameter('airsim_connect_timeout_sec').value)
        self.state_log_interval_sec = float(self.get_parameter('state_log_interval_sec').value)

        # ====================
        # 连接AirSim
        # ====================
        self.get_logger().info("连接到AirSim...")
        self.client = None
        ports_to_try = [self.airsim_port] if self.airsim_port > 0 else self.airsim_port_candidates
        last_error = None
        for port in ports_to_try:
            try:
                self.get_logger().info(f"尝试连接 AirSim: ip={self.airsim_ip}, port={port}")
                self.client = airsim.MultirotorClient(
                    ip=self.airsim_ip,
                    port=port,
                    timeout_value=self.airsim_rpc_timeout_sec
                )
                start_t = time.time()
                connected = False
                while time.time() - start_t < self.airsim_connect_timeout_sec:
                    if self.client.ping():
                        connected = True
                        break
                    time.sleep(0.2)
                if not connected:
                    raise RuntimeError(
                        f"ping timeout after {self.airsim_connect_timeout_sec:.1f}s"
                    )
                self.connected_airsim_port = port
                self.get_logger().info(f"✓ AirSim连接成功: ip={self.airsim_ip}, port={port}")
                break
            except Exception as e:
                last_error = e
                self.get_logger().warn(f"连接失败: ip={self.airsim_ip}, port={port}, error={e}")
                self.client = None

        if self.client is None:
            self.get_logger().error(
                f"✗ AirSim连接失败: ip={self.airsim_ip}, ports={ports_to_try}, last_error={last_error}"
            )
            raise RuntimeError("AirSim connection failed")
        
        # ====================
        # 状态变量
        # ====================
        self.lock = Lock()
        self.current_goal = None
        
        # ====================
        # QoS配置
        # ====================
        # QoS配置
        # ====================
        # 与 SUPER ROG Map 完全一致: best_effort, depth=1
        qos_profile_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # 订阅者使用 BEST_EFFORT QoS（与 SUPER 完全一致）
        qos_profile_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # ====================
        # ROS2发布者
        # ====================
        # 点云发布
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            '/cloud_registered',
            10  # 使用默认 QoS
        )
        
        # 里程计发布
        self.odom_pub = self.create_publisher(
            Odometry,
            '/lidar_slam/odom',
            10  # 使用默认 QoS
        )
        
        # 状态发布（用于调试）
        self.state_pub = self.create_publisher(
            Odometry,
            '/airsim_bridge/state',
            10
        )
        
        # ====================
        # ROS2订阅者
        # ====================
        # 订阅SUPER的控制指令
        if HAVE_MARS_MSGS:
            self.cmd_sub = self.create_subscription(
                PositionCommand,
                '/planning/pos_cmd',
                self.control_callback,
                qos_profile_sub
            )
            self.get_logger().info("✓ 订阅SUPER控制指令: /planning/pos_cmd")
        
        # 订阅目标点（与 TravelUAV super_ros2_client 发布的 /goal_pose 一致）
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # ====================
        # TF广播
        # ====================
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # ====================
        # 定时器（分开创建，避免点云拖慢里程计）
        # ====================
        # 里程计定时器：20Hz（快速，SUPER需要）
        odom_timer_period = 1.0 / self.publish_rate  # 秒
        self.odom_timer = self.create_timer(odom_timer_period, self.publish_odometry)
        
        # 点云定时器：5Hz（平衡实时性和处理负担）
        cloud_timer_period = 1.0 / 5.0  # 5Hz
        self.cloud_timer = self.create_timer(cloud_timer_period, self.publish_pointcloud)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("桥接节点初始化完成，开始数据传输...")
        self.get_logger().info(f"  - 点云话题: /cloud_registered")
        self.get_logger().info(f"  - 里程计话题: /lidar_slam/odom")
        self.get_logger().info(f"  - 目标点话题: /goal_pose")
        if HAVE_MARS_MSGS:
            self.get_logger().info(f"  - 控制指令话题: /planning/pos_cmd")
        self.get_logger().info(f"  - 里程计频率: {self.publish_rate} Hz")
        self.get_logger().info(f"  - 点云频率: 5.0 Hz")
        self.get_logger().info(f"  - 坐标系: {'ENU (ROS标准)' if self.use_enu else 'NED (AirSim)'}")
        self.get_logger().info("=" * 70)
    
    # 注释掉：已改用分开的定时器，不再需要publish_loop
    # def publish_loop(self):
    #     """主循环：发布点云和里程计"""
    #     try:
    #         # 1. 获取并发布点云
    #         self.publish_pointcloud()
    #         
    #         # 2. 获取并发布里程计
    #         self.publish_odometry()
    #         
    #     except Exception as e:
    #         self.get_logger().warn(f"数据发布出错: {e}", throttle_duration_sec=5.0)
    
    # def publish_pointcloud(self):
    #     """从AirSim获取LiDAR点云并发布"""
    #     try:
    #         # 获取LiDAR数据
    #         lidar_data = self.client.getLidarData(
    #             lidar_name=self.lidar_name,
    #             vehicle_name=self.vehicle_name
    #         )
            
    #         if not lidar_data.point_cloud or len(lidar_data.point_cloud) < 3:
    #             return
            
    #         # 转换为numpy数组 (N, 3)
    #         points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            
    #         # 坐标系转换: NED -> ENU
    #         if self.use_enu:
    #             # AirSim: X-前 Y-右 Z-下 (NED)
    #             # ROS:    X-东 Y-北 Z-上 (ENU)
    #             # 转换: [x_enu, y_enu, z_enu] = [y_ned, x_ned, -z_ned]
    #             points_enu = np.zeros_like(points)
    #             points_enu[:, 0] = points[:, 1]   # x_enu = y_ned
    #             points_enu[:, 1] = points[:, 0]   # y_enu = x_ned
    #             points_enu[:, 2] = -points[:, 2]  # z_enu = -z_ned
    #             points = points_enu
            
    #         # 创建PointCloud2消息
    #         header = Header()
    #         header.stamp = self.get_clock().now().to_msg()
    #         header.frame_id = self.world_frame_id

    #         # 创建点云消息
    #         cloud_msg = pc2.create_cloud_xyz32(header, points.tolist())
            
    #         # 发布
    #         self.cloud_pub.publish(cloud_msg)
            
    #     except Exception as e:
    #         self.get_logger().warn(f"点云发布失败: {e}", throttle_duration_sec=10.0)
    def publish_pointcloud(self):
        """从AirSim获取LiDAR点云并发布"""
        try:
            # 获取LiDAR数据
            lidar_data = self.client.getLidarData(
                lidar_name=self.lidar_name,
                vehicle_name=self.vehicle_name
            )

            if not lidar_data.point_cloud or len(lidar_data.point_cloud) < 3:
                return

            # 转换为numpy数组 (N, 3)
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            # ========================================
            # 点云下采样（减少计算量）
            # ========================================
            # 每 2 个点取 1 个（减少 50%）
            downsample_rate = 2
            points = points[::downsample_rate]

            # 坐标系转换: NED body -> ENU body
            if self.use_enu:
                points_enu = np.zeros_like(points)
                points_enu[:, 0] = points[:, 1]   # x_enu = y_ned
                points_enu[:, 1] = points[:, 0]   # y_enu = x_ned
                points_enu[:, 2] = -points[:, 2]  # z_enu = -z_ned
                points = points_enu

            # 获取无人机位置
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            
            if self.use_enu:
                drone_pos = np.array([pos.y_val, pos.x_val, -pos.z_val])
            else:
                drone_pos = np.array([pos.x_val, pos.y_val, pos.z_val])

            # 获取无人机姿态
            q = state.kinematics_estimated.orientation
            quat_ned = [q.x_val, q.y_val, q.z_val, q.w_val]
            
            if self.use_enu:
                quat = self.quat_ned_to_enu_correct(quat_ned)
            else:
                quat = quat_ned

            # 四元数共轭（world -> body 变成 body -> world）
            quat_conj = [quat[0], quat[1], quat[2], -quat[3]]
            R = self.quat_to_rot_matrix(quat_conj)
            
            # 点云转到 world frame
            points_world = (R @ points.T).T + drone_pos

            # 发布
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.world_frame_id

            cloud_msg = pc2.create_cloud_xyz32(header, points_world.tolist())
            self.cloud_pub.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"点云发布异常: {type(e).__name__}: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())


    def quat_to_rot_matrix(self, quat):
        """四元数转旋转矩阵"""
        x, y, z, w = quat
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        return R

        
    def publish_odometry(self):
        """从AirSim获取无人机状态并发布里程计"""
        try:
            # 获取无人机状态（必须指定 vehicle_name，与 TravelUAV 一致）
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            
            # 位置
            pos_airsim = state.kinematics_estimated.position
            pos = [pos_airsim.x_val, pos_airsim.y_val, pos_airsim.z_val]
            
            # 速度
            vel_airsim = state.kinematics_estimated.linear_velocity
            vel = [vel_airsim.x_val, vel_airsim.y_val, vel_airsim.z_val]
            
            # 姿态（四元数）
            q_airsim = state.kinematics_estimated.orientation
            quat = [q_airsim.x_val, q_airsim.y_val, q_airsim.z_val, q_airsim.w_val]
            
            # 坐标系转换: NED -> ENU
            if self.use_enu:
                # 位置转换
                pos_enu = [pos[1], pos[0], -pos[2]]
                vel_enu = [vel[1], vel[0], -vel[2]]
                
                # 四元数转换 (NED -> ENU) - 使用正确的转换
                quat_enu = self.quat_ned_to_enu_correct(quat)
                
                pos = pos_enu
                vel = vel_enu
                quat = quat_enu
            
            # 创建Odometry消息
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = self.world_frame_id
            odom.child_frame_id = self.body_frame_id
            
            # 位置
            odom.pose.pose.position.x = pos[0]
            odom.pose.pose.position.y = pos[1]
            odom.pose.pose.position.z = pos[2]
            
            # 姿态
            odom.pose.pose.orientation.x = quat[0]
            odom.pose.pose.orientation.y = quat[1]
            odom.pose.pose.orientation.z = quat[2]
            odom.pose.pose.orientation.w = quat[3]
            
            # 速度
            odom.twist.twist.linear.x = vel[0]
            odom.twist.twist.linear.y = vel[1]
            odom.twist.twist.linear.z = vel[2]
            
            # 发布里程计
            self.odom_pub.publish(odom)
            
            # 按配置频率打印位置
            self.get_logger().info(
                f"无人机位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                f"速度: ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})",
                throttle_duration_sec=self.state_log_interval_sec
            )

            # 发布TF
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.world_frame_id
            t.child_frame_id = self.body_frame_id
            t.transform.translation.x = pos[0]
            t.transform.translation.y = pos[1]
            t.transform.translation.z = pos[2]
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            self.get_logger().warn(f"里程计发布失败: {e}", throttle_duration_sec=10.0)
    
    def quat_ned_to_enu_correct(self, quat_ned):
        """正确的NED四元数转ENU四元数
        
        NED坐标系: X-前, Y-右, Z-下
        ENU坐标系: X-东, Y-北, Z-上
        
        转换方法：
        1. 将NED四元数转换为旋转矩阵
        2. 应用坐标系转换矩阵
        3. 转换回四元数
        """
        qx, qy, qz, qw = quat_ned
        
        # NED -> ENU 的旋转矩阵转换
        # 坐标变换: [x_enu, y_enu, z_enu] = [y_ned, x_ned, -z_ned]
        # 对应的旋转矩阵是：
        # T = [[0, 1, 0],
        #      [1, 0, 0],
        #      [0, 0, -1]]
        
        # 四元数 NED -> ENU 的转换
        # 通过旋转矩阵中间转换得到
        qx_enu = qy
        qy_enu = qx
        qz_enu = -qz
        qw_enu = qw
        
        return [qx_enu, qy_enu, qz_enu, qw_enu]
    
    def goal_callback(self, msg):
        """接收目标点"""
        with self.lock:
            self.current_goal = msg
            
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"收到新目标点:")
        self.get_logger().info(f"  位置: ({msg.pose.position.x:.2f}, "
                             f"{msg.pose.position.y:.2f}, "
                             f"{msg.pose.position.z:.2f})")
        self.get_logger().info("=" * 70)

    # def control_callback(self, msg):
    #     """接收SUPER的控制指令并发送给AirSim"""
    #     if not HAVE_MARS_MSGS:
    #         return
        
    #     try:
    #         # 提取位置指令 (ENU)
    #         pos = [msg.position.x, msg.position.y, msg.position.z]
    #         vel = [msg.velocity.x, msg.velocity.y, msg.velocity.z]
    #         yaw = msg.yaw
            
    #         # 坐标系转换: ENU -> NED
    #         if self.use_enu:
    #             # ENU: (x, y, z) -> NED: (y, x, -z)
    #             pos_ned = [pos[1], pos[0], -pos[2]]
    #             # 速度也要转
    #             vel_ned = [vel[1], vel[0], -vel[2]] 
    #         else:
    #             pos_ned = pos
    #             vel_ned = vel
            
    #         # [新增] 强制夺取控制权
    #         # 防止 TravelUAV (Python端) 抢占后导致 ROS 指令失效
    #         self.client.enableApiControl(True)
    #         self.client.armDisarm(True)

    #         # --- 发送控制指令 ---
    #         # SUPER 发送的是 20Hz 的高频轨迹点。
    #         # 这种情况下，使用 moveByPositionAsync (增量/瞬时控制) 或 moveOnPath 更合适
    #         # 但为了简单且兼容之前的逻辑，我们用 moveToPositionAsync 并设置很短的超时
            
    #         target_speed = np.linalg.norm(vel_ned)
    #         # 确保速度不为0，否则 AirSim 会报错
    #         target_speed = max(target_speed, 0.5) 

    #         # 注意：moveToPositionAsync 是去绝对坐标
    #         # 最后一个参数 timeout_sec 设置短一点，让它能快速响应下一个指令
    #         self.client.moveToPositionAsync(
    #             pos_ned[0], pos_ned[1], pos_ned[2],
    #             velocity=target_speed * 1.5, # 稍微给大一点速度上限，保证能跟上
    #             timeout_sec=0.1,  # 0.1秒超时，强制刷新指令
    #             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
    #             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=np.rad2deg(yaw))
    #         )
            
    #         self.get_logger().info(
    #             f"控制指令: 位置=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.}), "
    #             f"速度={np.linalg.norm(vel):.2f}",
    #             throttle_duration_sec=1.0
    #         )
            
    #     except Exception as e:
    #         self.get_logger().warn(f"控制指令执行失败: {e}", throttle_duration_sec=5.0)

    
    # def control_callback(self, msg):
    #     """接收SUPER的控制指令并发送给AirSim"""
    #     if not HAVE_MARS_MSGS:
    #         return
        
    #     try:
    #         # 提取位置指令
    #         pos = [msg.position.x, msg.position.y, msg.position.z]
    #         vel = [msg.velocity.x, msg.velocity.y, msg.velocity.z]
    #         acc = [msg.acceleration.x, msg.acceleration.y, msg.acceleration.z]
    #         yaw = msg.yaw
            
    #         # 坐标系转换: ENU -> NED
    #         if self.use_enu:
    #             pos_ned = [pos[1], pos[0], -pos[2]]
    #             vel_ned = [vel[1], vel[0], -vel[2]]
    #             acc_ned = [acc[1], acc[0], -acc[2]]
    #         else:
    #             pos_ned = pos
    #             vel_ned = vel
    #             acc_ned = acc
    #         # 控制 Z：标准 AirSim NED 为 负Z=上。若飞机仍往反方向走，可能是 (1)本机 API 为 Z-up 需改发 pos[2]
    #         # 或 (2)SUPER 规划出的 pos_cmd 本身 z 就偏小（目标/odom 坐标系或话题不一致）
    #         # 确保仿真运行且拥有控制权
    #         self.client.simPause(False)  # 确保仿真不暂停（TravelUAV可能会暂停）
    #         self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
    #         self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            
    #         # 使用位置+速度混合控制
    #         vel_norm = np.linalg.norm(vel_ned)
            
    #         # 提取目标速度作为最大速度限制
    #         target_speed = max(vel_norm, 0.5)  # 至少 0.5 m/s
            
    #         # 使用位置控制，但设置较高的速度限制
    #         self.client.moveToPositionAsync(
    #             pos_ned[0], pos_ned[1], pos_ned[2],
    #             velocity=target_speed * 3.0,  # 放大3倍确保快速响应
    #             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=np.rad2deg(yaw)),
    #             vehicle_name=self.vehicle_name
    #         )
            
    #         self.get_logger().info(
    #             f"控制指令ROS(ENU): ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
    #             f"NED: ({pos_ned[0]:.2f}, {pos_ned[1]:.2f}, {pos_ned[2]:.2f}), "
    #             f"速度={np.linalg.norm(vel):.2f}",
    #             throttle_duration_sec=1.0
    #         )
            
    #     except Exception as e:
    #         self.get_logger().warn(f"控制指令执行失败: {e}", throttle_duration_sec=5.0)
    def control_callback(self, msg):
        """接收SUPER的控制指令并发送给AirSim，限制最大步长和速度"""
        if not HAVE_MARS_MSGS:
            return

        try:
            # -----------------------------
            # 提取位置指令
            # -----------------------------
            target_enu = [msg.position.x, msg.position.y, msg.position.z]
            vel_cmd = [msg.velocity.x, msg.velocity.y, msg.velocity.z]
            yaw = msg.yaw

            # -----------------------------
            # 获取当前无人机 ENU 位置
            # -----------------------------
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            if self.use_enu:
                current_enu = [pos.y_val, pos.x_val, -pos.z_val]  # NED -> ENU
            else:
                current_enu = [pos.x_val, pos.y_val, pos.z_val]

            # -----------------------------
            # 限制最大步长
            # -----------------------------
            max_xy_step = 5.0  # X/Y 最大移动 5 m
            max_z_step = 3.0   # Z 最大移动 3 m

            delta_xy = np.array(target_enu[:2]) - np.array(current_enu[:2])
            dist_xy = np.linalg.norm(delta_xy)
            if dist_xy > max_xy_step:
                delta_xy = delta_xy / dist_xy * max_xy_step  # 缩放到最大步长

            delta_z = target_enu[2] - current_enu[2]
            if abs(delta_z) > max_z_step:
                delta_z = np.sign(delta_z) * max_z_step

            safe_target_enu = [
                current_enu[0] + delta_xy[0],
                current_enu[1] + delta_xy[1],
                current_enu[2] + delta_z
            ]

            # -----------------------------
            # ENU -> NED 转换
            # -----------------------------
            if self.use_enu:
                pos_ned = [safe_target_enu[1], safe_target_enu[0], -safe_target_enu[2]]
            else:
                pos_ned = safe_target_enu

            # -----------------------------
            # 确保控制权
            # -----------------------------
            self.client.simPause(False)
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)

            # -----------------------------
            # 设置速度
            # -----------------------------
            vel_norm = np.linalg.norm(vel_cmd)
            target_speed = max(vel_norm, 0.5) * 3.0  # 原始速度 * 3
            max_speed = 5.0  # 最大速度限制
            velocity = min(target_speed, max_speed)

            # -----------------------------
            # 发送 moveToPositionAsync
            # -----------------------------

            
            self.client.moveToPositionAsync(
                pos_ned[0], pos_ned[1], pos_ned[2],
                velocity=velocity,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=np.rad2deg(yaw)),
                vehicle_name=self.vehicle_name
            )

            # self.get_logger().info(
            #     f"当前 ENU: {current_enu}, "
            #     f"目标 ROS ENU: {target_enu}, "
            #     f"安全目标 ENU: {safe_target_enu}, "
            #     f"NED: {pos_ned}, "
            #     f"速度: {velocity:.2f} m/s"
            # )

        except Exception as e:
            self.get_logger().warn(f"控制指令执行失败: {e}", throttle_duration_sec=5.0)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        bridge = AirSimSuperBridgeROS2()
        # 多线程执行器：避免点云回调阻塞里程计回调，降低 SUPER 的 No odom 概率
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(bridge)
        executor.spin()
    except KeyboardInterrupt:
        print("\n桥接节点关闭")
    except Exception as e:
        print(f"桥接节点异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
