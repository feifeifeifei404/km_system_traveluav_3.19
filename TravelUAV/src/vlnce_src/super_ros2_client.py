#!/usr/bin/env python3
"""
SUPER ROS2客户端 - 使用ROS2话题通信（回退到快系统 odom 版）
- 当前位置订阅快系统自身 /lidar_slam/odom
- 不再订阅 AirSim 真值 /airsim/truth_odom
- 不再使用 goal_offset
- 发送目标时仅使用已经确认的坐标轴关系：
    x_ros = y_sim
    y_ros = x_sim
    z_ros = -z_sim
"""

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from perfect_drone_sim.srv import SetInitialPose
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, UInt16


class SUPERRos2Client(Node):
    """SUPER ROS2客户端 - 订阅快系统 /lidar_slam/odom，负责发目标与读取当前位姿"""

    def __init__(self):
        super().__init__('super_ros2_client')

        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.bridge_execution_pub = self.create_publisher(Bool, '/bridge/execution_enabled', 10)
        self.set_initial_pose_client = self.create_client(
            SetInitialPose,
            '/perfect_drone/set_initial_pose',
        )

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/lidar_slam/odom',
            self.odom_callback,
            qos_profile,
        )

        self.port_sub = self.create_subscription(
            UInt16,
            '/bridge/connected_airsim_port',
            self.connected_port_callback,
            10,
        )

        self.current_pos = None
        self.current_orientation = None
        self.current_linear_velocity = None
        self.current_angular_velocity = None
        self.last_odom_wall_time = None
        self.target_pos = None
        self.connected_airsim_port = None
        self.lock = threading.Lock()

        self.get_logger().info(
            '[SUPER ROS2] 客户端启动（订阅 /lidar_slam/odom，不使用 AirSim 真值，不使用 goal_offset）'
        )

    def connected_port_callback(self, msg):
        with self.lock:
            self.connected_airsim_port = int(msg.data)

    def odom_callback(self, msg):
        """这里收到的是快系统自身 /lidar_slam/odom。"""
        with self.lock:
            self.current_pos = np.array(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ],
                dtype=np.float64,
            )
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
            self.last_odom_wall_time = time.time()

    def wait_for_first_odom(self, timeout=5.0):
        """等待快系统首帧 /lidar_slam/odom，避免 current=None 时就发目标。"""
        start = time.time()
        while time.time() - start < timeout:
            current_pos = self.get_current_position()
            if current_pos is not None:
                self.get_logger().info(
                    f'[SUPER] 已收到首帧 /lidar_slam/odom: {np.round(current_pos, 3).tolist()}'
                )
                return True
            time.sleep(0.05)

        self.get_logger().error('[SUPER] 超时：未收到 /lidar_slam/odom 首帧')
        return False

    def set_goal_offset(self, offset):
        self.get_logger().warn('[SUPER] 当前模式不再使用 goal_offset，忽略 set_goal_offset() 调用')

    def reset_initial_pose(self, x, y, z, yaw=0.0, clear_path=True, timeout=5.0):
        if not self.set_initial_pose_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error('[SUPER] /perfect_drone/set_initial_pose 服务不可用')
            return False

        request = SetInitialPose.Request()
        request.x = float(x)
        request.y = float(y)
        request.z = float(z)
        request.yaw = float(yaw)
        request.clear_path = bool(clear_path)

        future = self.set_initial_pose_client.call_async(request)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if future.done():
                try:
                    response = future.result()
                except Exception as e:
                    self.get_logger().error(f'[SUPER] set_initial_pose 调用失败: {e}')
                    return False
                if response is None:
                    self.get_logger().error('[SUPER] set_initial_pose 返回空响应')
                    return False
                if response.success:
                    self.get_logger().info(
                        f'[SUPER] set_initial_pose 成功: pos=({x:.2f}, {y:.2f}, {z:.2f}), yaw={yaw:.2f}, clear_path={clear_path}'
                    )
                    return True
                self.get_logger().error(f'[SUPER] set_initial_pose 失败: {response.message}')
                return False
            time.sleep(0.05)

        self.get_logger().error('[SUPER] set_initial_pose 调用超时')
        return False

    def clear_goal_offset(self):
        pass

    def set_bridge_execution(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        for _ in range(3):
            self.bridge_execution_pub.publish(msg)
            time.sleep(0.02)
        self.get_logger().info(f'[SUPER] set_bridge_execution({bool(enabled)}) 已发布到 /bridge/execution_enabled')
        return True

    def send_goal(self, x, y, z):
        """
        参数:
            x, y, z: 仿真环境中的目标坐标（sim系）
        仅按已经确认的坐标关系转换到 ROS 目标:
            x_ros = y_sim
            y_ros = x_sim
            z_ros = -z_sim
        不再叠加任何 offset。
        """
        try:
            current_pos = self.get_current_position()
            if current_pos is None:
                self.get_logger().error('[SUPER] 当前 /lidar_slam/odom 为空，拒绝发送目标')
                return False

            goal_input = np.array([float(x), float(y), float(z)], dtype=np.float64)
            goal_ros = np.array([goal_input[1], goal_input[0], -goal_input[2]], dtype=np.float64)
            ros_x, ros_y, ros_z = goal_ros.tolist()

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.pose.position.x = float(ros_x)
            msg.pose.position.y = float(ros_y)
            msg.pose.position.z = float(ros_z)
            msg.pose.orientation.w = 1.0

            self.goal_pub.publish(msg)

            with self.lock:
                self.target_pos = np.array([ros_x, ros_y, ros_z], dtype=np.float64)

            delta = self.target_pos - current_pos

            self.get_logger().info(
                f'[SUPER] 发送目标点: SIM({x:.2f}, {y:.2f}, {z:.2f}) -> ROS({ros_x:.2f}, {ros_y:.2f}, {ros_z:.2f})'
            )
            self.get_logger().info(
                f'[SUPER DEBUG] current={np.round(current_pos, 3).tolist()} '
                f'target={np.round(self.target_pos, 3).tolist()} '
                f'delta={np.round(delta, 3).tolist()}'
            )
            return True

        except Exception as e:
            self.get_logger().error(f'[SUPER] 发送目标点失败: {e}')
            return False

    def get_current_position(self):
        with self.lock:
            return self.current_pos.copy() if self.current_pos is not None else None

    def get_connected_airsim_port(self):
        with self.lock:
            return int(self.connected_airsim_port) if self.connected_airsim_port else None

    def is_fast_system_alive(self, odom_timeout=3.0):
        if not rclpy.ok():
            return False
        with self.lock:
            last_odom_wall_time = self.last_odom_wall_time
        if last_odom_wall_time is None:
            return False
        return (time.time() - last_odom_wall_time) <= odom_timeout

    def is_fsm_node_present(self):
        try:
            node_names = {name for name, _namespace in self.get_node_names_and_namespaces()}
        except Exception as e:
            self.get_logger().error(f'[SUPER] 查询 ROS 节点失败: {e}')
            return False
        return 'fsm_node' in node_names


_ros2_client = None
_ros2_thread = None
_rclpy_initialized = False
_fast_system_fatal_flag = False
_fast_system_fatal_reason = None
_fast_system_monitor_thread = None
_fast_system_monitor_started = False


def init_ros2_client():
    global _ros2_client, _ros2_thread, _rclpy_initialized, _fast_system_fatal_flag, _fast_system_fatal_reason, _fast_system_monitor_thread, _fast_system_monitor_started

    if _ros2_client is not None:
        return _ros2_client

    _fast_system_fatal_flag = False
    _fast_system_fatal_reason = None
    _fast_system_monitor_started = False

    if not _rclpy_initialized:
        rclpy.init()
        _rclpy_initialized = True

    _ros2_client = SUPERRos2Client()

    _ros2_thread = threading.Thread(
        target=lambda: rclpy.spin(_ros2_client),
        daemon=True,
    )
    _ros2_thread.start()

    def _monitor_fsm_node_exit():
        global _fast_system_fatal_flag, _fast_system_fatal_reason, _fast_system_monitor_started
        seen_fsm_node = False
        _fast_system_monitor_started = True
        if _ros2_client is not None:
            _ros2_client.get_logger().info('[SUPER] fsm_node monitor thread started')
        monitor_start_time = time.time()
        initial_presence_grace_seconds = 10.0
        while _ros2_client is not None and rclpy.ok():
            try:
                is_present = _ros2_client.is_fsm_node_present()
                if is_present:
                    seen_fsm_node = True
                elif seen_fsm_node:
                    _fast_system_fatal_flag = True
                    _fast_system_fatal_reason = 'fsm_node process died'
                    _ros2_client.get_logger().error('[FATAL][SUPER] 检测到 fsm_node 已退出')
                    return
                elif time.time() - monitor_start_time > initial_presence_grace_seconds:
                    _fast_system_fatal_flag = True
                    _fast_system_fatal_reason = 'fsm_node never appeared in ROS graph'
                    _ros2_client.get_logger().error('[FATAL][SUPER] 启动后在 ROS graph 中始终未发现 fsm_node')
                    return
            except Exception as e:
                _fast_system_fatal_flag = True
                _fast_system_fatal_reason = f'fsm_node monitor failed: {e}'
                if _ros2_client is not None:
                    _ros2_client.get_logger().error(f'[FATAL][SUPER] fsm_node 监听失败: {e}')
                return
            time.sleep(0.2)

    _fast_system_monitor_thread = threading.Thread(
        target=_monitor_fsm_node_exit,
        daemon=True,
    )
    _fast_system_monitor_thread.start()

    if not _ros2_client.wait_for_first_odom(timeout=5.0):
        _ros2_client.get_logger().warn('[SUPER] 启动后尚未拿到快系统 odom，后续 send_goal 会继续检查')

    return _ros2_client


def get_super_ros2_client():
    global _ros2_client
    if _ros2_client is None:
        return init_ros2_client()
    return _ros2_client


def has_fast_system_fatal_error():
    return bool(_fast_system_fatal_flag)


def is_fast_system_monitor_started():
    return bool(_fast_system_monitor_started)


def get_fast_system_fatal_reason():
    return _fast_system_fatal_reason


def raise_if_fast_system_fatal():
    if _fast_system_fatal_flag:
        raise RuntimeError(f'[FATAL][SUPER] {_fast_system_fatal_reason or "fast system fatal error"}')


def compute_super_goal_offset(sim_state, super_client):
    """当前模式下不再使用 offset，保留函数仅兼容旧调用。"""
    return None


def close_ros2_client():
    global _ros2_client, _rclpy_initialized, _fast_system_fatal_flag, _fast_system_fatal_reason, _fast_system_monitor_started

    if _ros2_client is not None:
        _ros2_client.destroy_node()
        _ros2_client = None

    if _rclpy_initialized:
        rclpy.shutdown()
        _rclpy_initialized = False

    _fast_system_fatal_flag = False
    _fast_system_fatal_reason = None
    _fast_system_monitor_started = False