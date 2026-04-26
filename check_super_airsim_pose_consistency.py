#!/usr/bin/env python3
"""
校验 AirSim 实际位置 与 ROS /lidar_slam/odom 是否数值一致。

用途：
- 不改现有 bridge / SUPER / TravelUAV 逻辑
- 直接读取 AirSim 当前状态
- 同时订阅 /lidar_slam/odom
- 将 AirSim NED 转成 ROS ENU 后，与 odom 数值逐次比较

注意：
- /lidar_slam/odom 往往是局部里程计坐标，起点通常接近 0
- AirSim ENU 更接近仿真世界下的全局坐标
- 因此不能直接生比，需要先做一次初始偏移对齐

结论解释：
- 如果 AirSim(转ENU) ≈ 对齐后的 /lidar_slam/odom，则两边运动一致
- TravelUAV 的 SUPER ROS2 client 也是订阅这路 /lidar_slam/odom，所以慢系统看到的位姿也一致
"""

import json
from pathlib import Path

import airsim
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class PoseConsistencyChecker(Node):
    def __init__(self):
        super().__init__('pose_consistency_checker')

        self.declare_parameter('vehicle_name', 'Drone_1')
        self.declare_parameter('airsim_ip', '127.0.0.1')
        self.declare_parameter('airsim_port', 0)
        self.declare_parameter('airsim_port_candidates', [25001, 25002, 25003])
        self.declare_parameter('airsim_settings_root', '/mnt/data/TravelUAV/airsim_plugin/settings')
        self.declare_parameter('odom_topic', '/lidar_slam/odom')
        self.declare_parameter('use_enu', True)
        self.declare_parameter('print_rate_hz', 2.0)
        self.declare_parameter('warn_threshold_m', 0.30)
        self.declare_parameter('align_once_on_start', True)
        self.declare_parameter('align_z', True)

        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.airsim_ip = self.get_parameter('airsim_ip').value
        self.airsim_port = int(self.get_parameter('airsim_port').value)
        self.airsim_port_candidates = [int(p) for p in self.get_parameter('airsim_port_candidates').value]
        self.airsim_settings_root = self.get_parameter('airsim_settings_root').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.use_enu = bool(self.get_parameter('use_enu').value)
        self.print_rate_hz = float(self.get_parameter('print_rate_hz').value)
        self.warn_threshold_m = float(self.get_parameter('warn_threshold_m').value)
        self.align_once_on_start = bool(self.get_parameter('align_once_on_start').value)
        self.align_z = bool(self.get_parameter('align_z').value)

        self.last_odom_pos = None
        self.last_odom_stamp = None

        # 新增：记录 AirSim ENU 与 odom 的初始偏移
        self.offset_inited = False
        self.offset = np.zeros(3, dtype=np.float64)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)

        self.client = self._connect_airsim()
        self.create_timer(1.0 / max(self.print_rate_hz, 0.1), self.check_once)

        self.get_logger().info('pose consistency checker ready')
        self.get_logger().info(f'compare AirSim vs {self.odom_topic}')

    def odom_callback(self, msg: Odometry):
        self.last_odom_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=np.float64)
        self.last_odom_stamp = msg.header.stamp

    def _scan_ports_from_settings(self):
        ports = []
        settings_root = Path(self.airsim_settings_root)
        if not settings_root.exists():
            return ports

        for settings_path in sorted(settings_root.glob('*/settings.json')):
            try:
                with settings_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                port_value = data.get('ApiServerPort')
                if port_value is not None:
                    ports.append(int(port_value))
            except Exception:
                pass
        return sorted(set(ports))

    def _connect_airsim(self):
        ports = [self.airsim_port] if self.airsim_port > 0 else self._scan_ports_from_settings()
        if not ports:
            ports = self.airsim_port_candidates

        last_error = None
        for port in ports:
            try:
                client = airsim.MultirotorClient(ip=self.airsim_ip, port=port, timeout_value=3.0)
                if client.ping():
                    self.get_logger().info(f'connected airsim {self.airsim_ip}:{port}')
                    return client
            except Exception as e:
                last_error = e
        raise RuntimeError(f'AirSim connect failed: {last_error}')

    def _get_airsim_pos_enu(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position

        # AirSim 默认常见为 NED，这里转 ENU:
        # x_enu = y_ned
        # y_enu = x_ned
        # z_enu = -z_ned
        if self.use_enu:
            return np.array([pos.y_val, pos.x_val, -pos.z_val], dtype=np.float64)

        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)

    def _init_offset_if_needed(self, airsim_pos):
        if self.offset_inited:
            return True

        if self.last_odom_pos is None:
            return False

        self.offset = airsim_pos - self.last_odom_pos
        if not self.align_z:
            self.offset[2] = 0.0

        self.offset_inited = True
        self.get_logger().info(
            f'initialized offset={np.round(self.offset, 3).tolist()} '
            f'(align_z={self.align_z})'
        )
        return True

    def check_once(self):
        try:
            airsim_pos = self._get_airsim_pos_enu()

            if self.last_odom_pos is None:
                self.get_logger().info(
                    f'AirSim ENU={np.round(airsim_pos, 3).tolist()} | waiting odom...',
                    throttle_duration_sec=2.0,
                )
                return

            if self.align_once_on_start and not self.offset_inited:
                ok = self._init_offset_if_needed(airsim_pos)
                if not ok:
                    return
                # 初始化偏移这一帧不做误差判断，下一帧开始比
                return

            odom_aligned = self.last_odom_pos + self.offset
            delta = odom_aligned - airsim_pos
            err = float(np.linalg.norm(delta))
            xy_err = float(np.linalg.norm(delta[:2]))
            z_err = float(abs(delta[2]))

            msg = (
                f'AirSim ENU={np.round(airsim_pos, 3).tolist()} | '
                f'OdomRaw={np.round(self.last_odom_pos, 3).tolist()} | '
                f'OdomAligned={np.round(odom_aligned, 3).tolist()} | '
                f'Delta={np.round(delta, 3).tolist()} | '
                f'xy_err={xy_err:.3f} m | z_err={z_err:.3f} m | err={err:.3f} m'
            )

            if err > self.warn_threshold_m:
                self.get_logger().warn(msg)
            else:
                self.get_logger().info(msg)

        except Exception as e:
            self.get_logger().warn(f'check failed: {e}', throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = PoseConsistencyChecker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()