#!/usr/bin/env python3
"""简单的坐标系验证"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import numpy as np

class SimpleVerifier(Node):
    def __init__(self):
        super().__init__('simple_verifier')
        self.cloud_sub = self.create_subscription(PointCloud2, '/cloud_registered', self.cloud_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, '/lidar_slam/odom', self.odom_cb, 10)
        self.count = 0
    
    def cloud_cb(self, msg):
        self.count += 1
        if self.count % 5 == 0:  # 每 5 次打印一次
            self.get_logger().info(f"\n📊 点云: frame_id={msg.header.frame_id}, width={msg.width}, height={msg.height}")
    
    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        self.get_logger().info(f"🚁 里程计: frame_id={msg.header.frame_id}, pos=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")

rclpy.init()
verifier = SimpleVerifier()
try:
    rclpy.spin(verifier)
except KeyboardInterrupt:
    pass
finally:
    verifier.destroy_node()
    rclpy.shutdown()
