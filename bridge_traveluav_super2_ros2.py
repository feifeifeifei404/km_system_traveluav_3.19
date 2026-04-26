#!/usr/bin/env python3
"""
最小侵入 AirSim <-> SUPER ROS2 桥接
- 不改 TravelUAV 逻辑
- 不改 /mnt/super2/SUPER-master 逻辑
- 只提供 SUPER 所需 ROS2 话题，并执行 SUPER 的 /planning/pos_cmd

当前定位策略说明：
- 快系统当前位姿来源：/lidar_slam/odom（由快系统自身发布）
- 本桥接节点仍可发布 /airsim/truth_odom 作为调试/对照真值，但快系统当前不订阅它
- 坐标轴关系保持为：x_ros = y_sim, y_ros = x_sim, z_ros = -z_sim

话题:
  发布:
    /airsim/truth_odom     Odometry   (AirSim 真值，已转 ROS 坐标，仅调试/对照)
  订阅:
    /planning/pos_cmd      mars_quadrotor_msgs/PositionCommand
    /goal_pose             PoseStamped
"""

import json
import time
import traceback
from pathlib import Path
from threading import Lock

import airsim
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

from mars_quadrotor_msgs.msg import PositionCommand


T_NED_TO_ENU = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=np.float64)


def quat_xyzw_to_rotmat(quat_xyzw):
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def rotmat_to_quat_xyzw(rot):
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def yaw_from_rotmat_enu(rot):
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


class AirSimSuperBridge(Node):
    def __init__(self):
        super().__init__('airsim_super_bridge_minimal')

        self.declare_parameter('vehicle_name', 'Drone_1')
        self.declare_parameter('lidar_name', 'Lidar1')
        self.declare_parameter('world_frame_id', 'world')
        self.declare_parameter('body_frame_id', 'body')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('airsim_ip', '127.0.0.1')
        self.declare_parameter('airsim_port', 0)
        self.declare_parameter('airsim_port_candidates', [25001, 25002, 25003])
        self.declare_parameter('airsim_settings_root', '/mnt/data/TravelUAV/airsim_plugin/settings')

        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.lidar_name = self.get_parameter('lidar_name').value
        self.world_frame_id = self.get_parameter('world_frame_id').value
        self.body_frame_id = self.get_parameter('body_frame_id').value
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.airsim_ip = self.get_parameter('airsim_ip').value
        self.airsim_port = int(self.get_parameter('airsim_port').value)
        self.airsim_port_candidates = [int(p) for p in self.get_parameter('airsim_port_candidates').value]
        self.airsim_settings_root = self.get_parameter('airsim_settings_root').value

        self.lock = Lock()
        self.current_goal = None

        self.client = self._connect_airsim()

        # AirSim 真值仅作为调试/对照输出，不作为快系统当前位姿来源
        self.odom_pub = self.create_publisher(Odometry, '/airsim/truth_odom', 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.cmd_sub = self.create_subscription(
            PositionCommand,
            '/planning/pos_cmd',
            self.control_callback,
            cmd_qos,
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_timer(1.0 / self.publish_rate, self.publish_odometry)

        self.get_logger().info('=' * 60)
        self.get_logger().info('AirSimSuperBridge 已启动')
        self.get_logger().info('快系统当前位置来源: /lidar_slam/odom（非本节点发布）')
        self.get_logger().info('本节点发布调试真值: /airsim/truth_odom')
        self.get_logger().info('控制输入订阅: /goal_pose, /planning/pos_cmd')
        self.get_logger().info('使用坐标关系: x_ros=y_sim, y_ros=x_sim, z_ros=-z_sim')
        self.get_logger().info('=' * 60)

    def _scan_ports_from_settings(self):
        ports = []
        settings_root = Path(self.airsim_settings_root)
        if not settings_root.exists():
            self.get_logger().warn(f'settings root not found: {settings_root}')
            return ports

        candidates = sorted(
            settings_root.glob('*/settings.json'),
            key=lambda p: p.parent.name if p.parent.name.isdigit() else p.parent.name,
        )
        for settings_path in candidates:
            try:
                with settings_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                port_value = data.get('ApiServerPort')
                if port_value is None:
                    continue
                port = int(port_value)
                ports.append(port)
            except Exception as e:
                self.get_logger().warn(f'failed reading {settings_path}: {e}')

        ports = sorted(set(ports))
        if ports:
            self.get_logger().info(f'ports from settings: {ports}')
        return ports

    def _connect_airsim(self):
        if self.airsim_port > 0:
            ports = [self.airsim_port]
            self.get_logger().info(f'use explicit airsim_port={self.airsim_port}')
        else:
            scanned_ports = self._scan_ports_from_settings()
            ports = scanned_ports if scanned_ports else self.airsim_port_candidates
            self.get_logger().info(f'candidate airsim ports: {ports}')

        last_error = None
        for port in ports:
            try:
                self.get_logger().info(f'trying airsim {self.airsim_ip}:{port}')
                client = airsim.MultirotorClient(ip=self.airsim_ip, port=port, timeout_value=3.0)
                start = time.time()
                while time.time() - start < 10.0:
                    if client.ping():
                        self.get_logger().info(f'connected airsim {self.airsim_ip}:{port}')
                        return client
                    time.sleep(0.2)
                raise RuntimeError('ping timeout')
            except Exception as e:
                last_error = e
                self.get_logger().warn(f'airsim connect failed on {port}: {e}')
        raise RuntimeError(f'AirSim connect failed: {last_error}')

    def goal_callback(self, msg: PoseStamped):
        with self.lock:
            self.current_goal = msg
        self.get_logger().info(
            f'goal received ROS=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})',
            throttle_duration_sec=1.0,
        )

    def publish_odometry(self):
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            q = state.kinematics_estimated.orientation

            pos_ned = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)
            vel_ned = np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float64)
            quat_ned = np.array([q.x_val, q.y_val, q.z_val, q.w_val], dtype=np.float64)

            pos_out = T_NED_TO_ENU @ pos_ned
            vel_out = T_NED_TO_ENU @ vel_ned
            rot_ned = quat_xyzw_to_rotmat(quat_ned)
            rot_enu = T_NED_TO_ENU @ rot_ned @ T_NED_TO_ENU.T
            quat_out = rotmat_to_quat_xyzw(rot_enu)

            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = self.world_frame_id
            odom.child_frame_id = self.body_frame_id
            odom.pose.pose.position.x = float(pos_out[0])
            odom.pose.pose.position.y = float(pos_out[1])
            odom.pose.pose.position.z = float(pos_out[2])
            odom.pose.pose.orientation.x = float(quat_out[0])
            odom.pose.pose.orientation.y = float(quat_out[1])
            odom.pose.pose.orientation.z = float(quat_out[2])
            odom.pose.pose.orientation.w = float(quat_out[3])
            odom.twist.twist.linear.x = float(vel_out[0])
            odom.twist.twist.linear.y = float(vel_out[1])
            odom.twist.twist.linear.z = float(vel_out[2])

            self.odom_pub.publish(odom)

            if not hasattr(self, '_odom_pub_counter'):
                self._odom_pub_counter = 0
            self._odom_pub_counter += 1
            if self._odom_pub_counter % 100 == 0:
                self.get_logger().info(
                    f'[Heartbeat] /airsim/truth_odom 已发布 {self._odom_pub_counter} 次, '
                    f'当前位置: ({pos_out[0]:.2f}, {pos_out[1]:.2f}, {pos_out[2]:.2f})'
                )

            t = TransformStamped()
            t.header.stamp = odom.header.stamp
            t.header.frame_id = self.world_frame_id
            t.child_frame_id = self.body_frame_id
            t.transform.translation.x = float(pos_out[0])
            t.transform.translation.y = float(pos_out[1])
            t.transform.translation.z = float(pos_out[2])
            t.transform.rotation.x = float(quat_out[0])
            t.transform.rotation.y = float(quat_out[1])
            t.transform.rotation.z = float(quat_out[2])
            t.transform.rotation.w = float(quat_out[3])
            self.tf_broadcaster.sendTransform(t)

        except Exception as e:
            self.get_logger().error(f'publish_odometry 异常: {e}')
            self.get_logger().error(traceback.format_exc())

    def control_callback(self, msg: PositionCommand):
        try:
            target_enu = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float64)
            vel_cmd_enu = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z], dtype=np.float64)

            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            q = state.kinematics_estimated.orientation

            current_ned = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)
            current_enu = T_NED_TO_ENU @ current_ned
            quat_ned = np.array([q.x_val, q.y_val, q.z_val, q.w_val], dtype=np.float64)
            rot_ned = quat_xyzw_to_rotmat(quat_ned)
            rot_enu = T_NED_TO_ENU @ rot_ned @ T_NED_TO_ENU.T
            yaw_enu = yaw_from_rotmat_enu(rot_enu)

            max_xy_step = 5.0
            max_z_step = 3.0
            delta_xy = target_enu[:2] - current_enu[:2]
            dist_xy = np.linalg.norm(delta_xy)
            if dist_xy > max_xy_step and dist_xy > 1e-6:
                delta_xy = delta_xy / dist_xy * max_xy_step
            delta_z = float(target_enu[2] - current_enu[2])
            if abs(delta_z) > max_z_step:
                delta_z = np.sign(delta_z) * max_z_step

            safe_target_enu = np.array([
                current_enu[0] + delta_xy[0],
                current_enu[1] + delta_xy[1],
                current_enu[2] + delta_z,
            ], dtype=np.float64)

            pos_ned = T_NED_TO_ENU.T @ safe_target_enu
            vel_cmd_ned = T_NED_TO_ENU.T @ vel_cmd_enu
            vel_norm = float(np.linalg.norm(vel_cmd_ned))
            velocity = min(max(vel_norm, 0.5) * 3.0, 5.0)

            yaw_ned_deg = float(np.rad2deg(np.pi / 2.0 - yaw_enu))

            self.client.simPause(False)
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)

            self.client.moveToPositionAsync(
                float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]),
                velocity=velocity,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_ned_deg),
                vehicle_name=self.vehicle_name,
            )
        except Exception as e:
            self.get_logger().warn(f'control_callback failed: {e}', throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = AirSimSuperBridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()