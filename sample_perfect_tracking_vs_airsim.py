#!/usr/bin/env python3
"""
连续采样 perfect_tracking world 与 AirSim ENU，并保存 CSV。

功能：
- 订阅 /lidar_slam/odom
- 连接 AirSim RPC
- 将 AirSim NED 自动转成 ENU
- 计算 delta = perfect_tracking_world - AirSim_ENU
- 按固定频率把结果保存到 CSV

输出列：
wall_time,ros_stamp,
pt_x,pt_y,pt_z,
pt_qx,pt_qy,pt_qz,pt_qw,
as_ned_x,as_ned_y,as_ned_z,
as_enu_x,as_enu_y,as_enu_z,
dx,dy,dz,delta_norm
"""

import csv
import json
import time
from pathlib import Path

import airsim
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class PtAirSimSampler(Node):
    def __init__(self):
        super().__init__('pt_airsim_sampler')

        self.declare_parameter('vehicle_name', 'Drone_1')
        self.declare_parameter('airsim_ip', '127.0.0.1')
        self.declare_parameter('airsim_port', 0)
        self.declare_parameter('airsim_port_candidates', [25001, 25002, 25003])
        self.declare_parameter('airsim_settings_root', '/mnt/data/TravelUAV/airsim_plugin/settings')
        self.declare_parameter('odom_topic', '/lidar_slam/odom')
        self.declare_parameter('sample_rate_hz', 2.0)
        self.declare_parameter('max_samples', 120)
        self.declare_parameter('output_csv', '/mnt/data/verify_pt_vs_airsim/pt_vs_airsim_samples.csv')

        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.airsim_ip = self.get_parameter('airsim_ip').value
        self.airsim_port = int(self.get_parameter('airsim_port').value)
        self.airsim_port_candidates = [int(p) for p in self.get_parameter('airsim_port_candidates').value]
        self.airsim_settings_root = self.get_parameter('airsim_settings_root').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.sample_rate_hz = float(self.get_parameter('sample_rate_hz').value)
        self.max_samples = int(self.get_parameter('max_samples').value)
        self.output_csv = Path(self.get_parameter('output_csv').value)

        self.last_odom_pos = None
        self.last_odom_quat = None
        self.last_odom_stamp = None
        self.sample_count = 0

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.output_csv.open('w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'wall_time', 'ros_stamp',
            'pt_x', 'pt_y', 'pt_z',
            'pt_qx', 'pt_qy', 'pt_qz', 'pt_qw',
            'as_ned_x', 'as_ned_y', 'as_ned_z',
            'as_enu_x', 'as_enu_y', 'as_enu_z',
            'dx', 'dy', 'dz', 'delta_norm',
        ])
        self.csv_file.flush()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)

        self.client = self._connect_airsim()
        self.create_timer(1.0 / max(self.sample_rate_hz, 0.1), self.sample_once)

        self.get_logger().info(f'sampling {self.odom_topic} vs AirSim into {self.output_csv}')

    def destroy_node(self):
        try:
            if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
                self.csv_file.close()
        finally:
            super().destroy_node()

    def odom_callback(self, msg: Odometry):
        self.last_odom_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=np.float64)
        self.last_odom_quat = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=np.float64)
        self.last_odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

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

    def _get_airsim_state(self):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        pos = state.kinematics_estimated.position
        pos_ned = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)
        pos_enu = np.array([pos.y_val, pos.x_val, -pos.z_val], dtype=np.float64)
        return pos_ned, pos_enu

    def sample_once(self):
        try:
            if self.last_odom_pos is None:
                self.get_logger().info('waiting for /lidar_slam/odom...', throttle_duration_sec=2.0)
                return

            pos_ned, pos_enu = self._get_airsim_state()
            delta = self.last_odom_pos - pos_enu
            delta_norm = float(np.linalg.norm(delta))
            wall_time = time.time()

            self.csv_writer.writerow([
                f'{wall_time:.9f}',
                f'{self.last_odom_stamp:.9f}',
                f'{self.last_odom_pos[0]:.9f}', f'{self.last_odom_pos[1]:.9f}', f'{self.last_odom_pos[2]:.9f}',
                f'{self.last_odom_quat[0]:.9f}', f'{self.last_odom_quat[1]:.9f}', f'{self.last_odom_quat[2]:.9f}', f'{self.last_odom_quat[3]:.9f}',
                f'{pos_ned[0]:.9f}', f'{pos_ned[1]:.9f}', f'{pos_ned[2]:.9f}',
                f'{pos_enu[0]:.9f}', f'{pos_enu[1]:.9f}', f'{pos_enu[2]:.9f}',
                f'{delta[0]:.9f}', f'{delta[1]:.9f}', f'{delta[2]:.9f}', f'{delta_norm:.9f}',
            ])
            self.csv_file.flush()

            self.sample_count += 1
            self.get_logger().info(
                f'#{self.sample_count} PT={np.round(self.last_odom_pos, 3).tolist()} | '
                f'AirSim ENU={np.round(pos_enu, 3).tolist()} | '
                f'Delta={np.round(delta, 3).tolist()} | norm={delta_norm:.3f} m'
            )

            if self.max_samples > 0 and self.sample_count >= self.max_samples:
                self.get_logger().info(f'reached max_samples={self.max_samples}, exiting')
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().warn(f'sample failed: {e}', throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = PtAirSimSampler()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        else:
            node.destroy_node()


if __name__ == '__main__':
    main()
