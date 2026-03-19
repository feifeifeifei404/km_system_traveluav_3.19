#!/usr/bin/env python3
"""
SUPER ROS2客户端 - 使用ROS2话题通信（替代Socket）
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import time
import threading
# import debugpy


class SUPERRos2Client(Node):
    """SUPER ROS2客户端 - 发布目标点，监控状态"""
    
    def __init__(self):
        super().__init__('super_ros2_client')
        
        # 发布目标点到SUPER
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # 订阅odometry获取当前位置（监控是否到达）
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/lidar_slam/odom',
            self.odom_callback,
            qos_profile
        )
        
        # 状态变量（用于轨迹记录：与 SimState.trajectory 格式一致）
        self.current_pos = None
        self.current_orientation = None   # [x, y, z, w]
        self.current_linear_velocity = None
        self.current_angular_velocity = None
        self.target_pos = None
        self.lock = threading.Lock()
        
        self.get_logger().info("[SUPER ROS2] 客户端启动")
    
    def odom_callback(self, msg):
        """更新当前位置、姿态、速度（用于到达判断与轨迹记录）"""
        with self.lock:
            self.current_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])
            self.current_orientation = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
            self.current_linear_velocity = [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]
            self.current_angular_velocity = [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ]
    
    def send_goal(self, x, y, z):
        """
        发送目标点给SUPER
        
        参数:
            x, y, z: 目标点（自动识别 AirSim NED 或 ROS ENU）
        
        返回:
            bool: 是否成功发送
        """
        try:
            goal_input = np.array([float(x), float(y), float(z)], dtype=np.float64)
            goal_ros_from_airsim = np.array([goal_input[1], goal_input[0], -goal_input[2]], dtype=np.float64)
            goal_ros_direct = goal_input.copy()

            selected_frame = "airsim_ned"
            goal_ros = goal_ros_from_airsim
            d_air = None
            d_ros = None

            with self.lock:
                current_pos = self.current_pos.copy() if self.current_pos is not None else None

            if current_pos is not None:
                d_air = float(np.linalg.norm(current_pos - goal_ros_from_airsim))
                d_ros = float(np.linalg.norm(current_pos - goal_ros_direct))
                if d_ros < d_air:
                    selected_frame = "ros_enu"
                    goal_ros = goal_ros_direct
            
            ros_x, ros_y, ros_z = goal_ros.tolist()
            
            # 创建目标点消息
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            msg.pose.position.x = float(ros_x)
            msg.pose.position.y = float(ros_y)
            msg.pose.position.z = float(ros_z)
            msg.pose.orientation.w = 1.0

            # debugpy.breakpoint()  # 断点5: publish 前，可看 ros_x/ros_y/ros_z、msg
            # 发布目标点
            self.goal_pub.publish(msg)
            
            with self.lock:
                self.target_pos = np.array([ros_x, ros_y, ros_z])
            
            if d_air is None or d_ros is None:
                self.get_logger().info(
                    f"[SUPER] 发送目标点: 输入({x:.2f}, {y:.2f}, {z:.2f}) "
                    f"-> ROS({ros_x:.2f}, {ros_y:.2f}, {ros_z:.2f}), frame={selected_frame}"
                )
            else:
                self.get_logger().info(
                    f"[SUPER] 发送目标点: 输入({x:.2f}, {y:.2f}, {z:.2f}) "
                    f"-> ROS({ros_x:.2f}, {ros_y:.2f}, {ros_z:.2f}), frame={selected_frame}, "
                    f"d_air={d_air:.2f}, d_ros={d_ros:.2f}"
                )
            return True
        
        except Exception as e:
            self.get_logger().error(f"[SUPER] 发送目标点失败: {e}")
            return False
    
    def get_current_position(self):
        """获取当前位置"""
        with self.lock:
            return self.current_pos.copy() if self.current_pos is not None else None

    @staticmethod
    def quat_to_rot_matrix(quat):
        x, y, z, w = quat
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]

    def get_current_state_as_trajectory_point(self):
        """
        获取当前状态，格式与 SimState.trajectory 元素一致，供 eval 使用。
        trajectory[-1]['sensors']['state'] 需包含 position, orientation, linear_velocity, angular_velocity。
        """
        with self.lock:
            if self.current_pos is None or self.current_orientation is None:
                return None
            rot = self.quat_to_rot_matrix(self.current_orientation)
            return {
                'sensors': {
                    'state': {
                        'position': self.current_pos.tolist(),
                        'orientation': list(self.current_orientation),
                        'linear_velocity': list(self.current_linear_velocity) if self.current_linear_velocity else [0.0, 0.0, 0.0],
                        'angular_velocity': list(self.current_angular_velocity) if self.current_angular_velocity else [0.0, 0.0, 0.0],
                    },
                    'imu': {
                        'rotation': rot,
                        'angular_velocity': list(self.current_angular_velocity) if self.current_angular_velocity else [0.0, 0.0, 0.0],
                    },
                }
            }
    
    def wait_for_arrival(self, target_pos_airsim, threshold=2.0, timeout=120.0, height_threshold=1.0, check_interval=0.5, record_interval=0.5):
        """
        等待无人机到达目标点，并按规定间隔记录轨迹（与 SimState.trajectory 格式一致）。
        
        参数:
            target_pos_airsim: AirSim坐标的目标点 [x, y, z]
            threshold: 3D 到达阈值（米）
            height_threshold: 高度差阈值（米），垂直方向需单独满足此要求
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）
            record_interval: 轨迹记录间隔（秒），与 eval 的 record_interval 一致
        
        返回:
            (success, status, trajectory): (是否成功, 状态信息, 轨迹点列表)
        """
        # 转换目标点到ROS坐标
        target_ros = np.array([
            target_pos_airsim[1],   # y -> x
            target_pos_airsim[0],   # x -> y
            -target_pos_airsim[2]   # -z -> z
        ])
        
        start_time = time.time()
        last_pos = None
        last_record_time = 0.0
        stationary_count = 0
        trajectory = []
        
        self.get_logger().info(
            f"[SUPER] 等待到达目标点: ROS({target_ros[0]:.2f}, {target_ros[1]:.2f}, {target_ros[2]:.2f}), "
            f"3D阈值={threshold:.2f}m, 高度阈值={height_threshold:.2f}m, 超时={timeout:.0f}s"
        )
        
        while True:
            elapsed = time.time() - start_time
            
            # 检查超时
            if elapsed > timeout:
                self.get_logger().warn(f"[SUPER] 超时 ({timeout:.0f}秒)")
                return False, "TIMEOUT", trajectory
            
            # 获取当前位置
            current_pos = self.get_current_position()
            if current_pos is None:
                time.sleep(check_interval)
                continue
            
            # 按 record_interval 记录轨迹点（与 SimState.trajectory 格式一致）
            if elapsed - last_record_time >= record_interval:
                pt = self.get_current_state_as_trajectory_point()
                if pt is not None:

                    trajectory.append(pt)
                    last_record_time = elapsed
            
            # 计算 3D 距离与高度差
            distance = np.linalg.norm(current_pos - target_ros)
            height_diff = abs(current_pos[2] - target_ros[2])
            
            # 到达条件：3D 距离 < threshold 且 高度差 < height_threshold
            if distance < threshold and height_diff < height_threshold:
                self.get_logger().info(
                    f"[SUPER] ✓ 到达目标！3D距离={distance:.2f}m, 高度差={height_diff:.2f}m, 耗时={elapsed:.1f}s"
                )
                # 到达时再记录终点状态
                pt = self.get_current_state_as_trajectory_point()
                if pt is not None:
# ROS -> AirSim
                    pos_ros = pt['sensors']['state']['position']
                    pt['sensors']['state']['position'] = [
                        pos_ros[1],   # y_ros -> x_airsim
                        pos_ros[0],   # x_ros -> y_airsim
                        -pos_ros[2]   # -z_ros -> z_airsim
                    ]
                    
                    trajectory.append(pt)
                return True, "ARRIVED", trajectory
            
            # 检查是否静止（可能卡住了）
            if last_pos is not None:
                movement = np.linalg.norm(current_pos - last_pos)
                if movement < 0.05:  # 移动小于5cm
                    stationary_count += 1
                    if stationary_count > 200:  # 静止超过10秒
                        self.get_logger().warn(
                            f"[SUPER] 无人机静止超过10秒，距离目标还有{distance:.2f}m，高度差{height_diff:.2f}m"
                        )
                        return False, "STUCK", trajectory
                else:
                    stationary_count = 0
            
            last_pos = current_pos.copy()
            
            # 定期打印进度（含高度差）
            if int(elapsed) % 5 == 0 and elapsed > 0:
                self.get_logger().info(
                    f"[SUPER] 飞行中... 3D距离={distance:.2f}m, 高度差={height_diff:.2f}m, 已用时={elapsed:.0f}s"
                )
            
            time.sleep(check_interval)


# 全局ROS2客户端实例
_ros2_client = None
_ros2_thread = None
_rclpy_initialized = False


def init_ros2_client():
    """初始化ROS2客户端（单例模式）"""
    global _ros2_client, _ros2_thread, _rclpy_initialized
    
    if _ros2_client is not None:
        return _ros2_client
    
    # 初始化rclpy
    if not _rclpy_initialized:
        rclpy.init()
        _rclpy_initialized = True
    
    # 创建ROS2节点
    _ros2_client = SUPERRos2Client()
    
    # 在后台线程中运行spin
    _ros2_thread = threading.Thread(
        target=lambda: rclpy.spin(_ros2_client),
        daemon=True
    )
    _ros2_thread.start()
    
    # 等待节点启动
    time.sleep(0.5)
    
    return _ros2_client


def get_super_ros2_client():
    """获取SUPER ROS2客户端（单例）"""
    global _ros2_client
    
    if _ros2_client is None:
        return init_ros2_client()
    
    return _ros2_client


def shutdown_ros2_client():
    """关闭ROS2客户端"""
    global _ros2_client, _rclpy_initialized
    
    if _ros2_client is not None:
        _ros2_client.destroy_node()
        _ros2_client = None
    
    if _rclpy_initialized:
        rclpy.shutdown()
        _rclpy_initialized = False
