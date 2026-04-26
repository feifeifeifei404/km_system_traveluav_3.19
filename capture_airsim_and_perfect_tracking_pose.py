#!/usr/bin/env python3
import json
import time
import airsim
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry


class PoseCapture(Node):
    def __init__(self):
        super().__init__('pose_capture')

        self.declare_parameter('airsim_ip', '127.0.0.1')
        self.declare_parameter('airsim_port', 25001)
        self.declare_parameter('vehicle_name', 'Drone_1')
        self.declare_parameter('odom_topic', '/lidar_slam/odom')

        self.airsim_ip = self.get_parameter('airsim_ip').value
        self.airsim_port = int(self.get_parameter('airsim_port').value)
        self.vehicle_name = self.get_parameter('vehicle_name').value
        self.odom_topic = self.get_parameter('odom_topic').value

        self.last_odom = None
        self.last_odom_walltime = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos
        )

        self.client = airsim.MultirotorClient(
            ip=self.airsim_ip,
            port=self.airsim_port,
            timeout_value=3.0
        )
        self.client.confirmConnection()
        self.get_logger().info(
            f'connected airsim {self.airsim_ip}:{self.airsim_port}'
        )

    def odom_callback(self, msg):
        self.last_odom = msg
        self.last_odom_walltime = time.monotonic()

    def wait_for_odom(self, timeout_sec=5.0):
        start = time.monotonic()
        while self.last_odom is None and (time.monotonic() - start < timeout_sec):
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_odom is not None

    def capture_once(self):
        # 先确保拿到一帧最新 odom
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)

        if self.last_odom is None:
            raise RuntimeError('No /lidar_slam/odom received')

        t_before = time.monotonic()
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        t_after = time.monotonic()

        # AirSim 原始 NED
        p = state.kinematics_estimated.position
        q = state.kinematics_estimated.orientation

        airsim_pos_ned = [p.x_val, p.y_val, p.z_val]
        airsim_quat_xyzw = [q.x_val, q.y_val, q.z_val, q.w_val]

        # Odom / perfect_tracking
        odom = self.last_odom
        odom_pos = [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ]
        odom_quat_xyzw = [
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ]

        odom_stamp = {
            'sec': int(odom.header.stamp.sec),
            'nanosec': int(odom.header.stamp.nanosec),
        }

        result = {
            'capture_walltime': time.time(),
            'airsim_query_monotonic_before': t_before,
            'airsim_query_monotonic_after': t_after,
            'odom_receive_monotonic': self.last_odom_walltime,
            'odom_header_frame_id': odom.header.frame_id,
            'odom_child_frame_id': odom.child_frame_id,
            'airsim_position_ned': airsim_pos_ned,
            'airsim_orientation_xyzw': airsim_quat_xyzw,
            'perfect_tracking_position': odom_pos,
            'perfect_tracking_orientation_xyzw': odom_quat_xyzw,
            'odom_stamp': odom_stamp,
        }
        return result


def main():
    rclpy.init()
    node = PoseCapture()

    if not node.wait_for_odom(timeout_sec=5.0):
        print('ERROR: no /lidar_slam/odom received')
        node.destroy_node()
        rclpy.shutdown()
        return

    data = node.capture_once()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
