#!/usr/bin/env python3
"""
坐标系转换验证脚本

用途：验证点云和里程计的坐标系是否正确对齐
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np


class CoordinateSystemVerifier(Node):
    """验证坐标系转换"""
    
    def __init__(self):
        super().__init__('coordinate_verifier')
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("坐标系验证工具启动")
        self.get_logger().info("=" * 70)
        
        # 订阅话题
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/cloud_registered',
            self.cloud_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/lidar_slam/odom',
            self.odom_callback,
            10
        )
        
        self.drone_pos = None
        self.cloud_frame_id = None
        self.odom_frame_id = None
        
    def cloud_callback(self, msg):
        """处理点云消息"""
        self.cloud_frame_id = msg.header.frame_id
        
        # 提取点云数据
        points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        
        if len(points) > 0:
            points = np.array([(p[0], p[1], p[2]) for p in points])
            
            self.get_logger().info(
                f"\n📊 点云信息:\n"
                f"  Frame ID: {self.cloud_frame_id}\n"
                f"  点数: {len(points)}\n"
                f"  X 范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]\n"
                f"  Y 范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]\n"
                f"  Z 范围: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]",
                throttle_duration_sec=2.0
            )
            
            # 验证点云是否围绕无人机
            if self.drone_pos is not None:
                self.verify_alignment(points)
    
    def odom_callback(self, msg):
        """处理里程计消息"""
        self.odom_frame_id = msg.header.frame_id
        
        pos = msg.pose.pose.position
        self.drone_pos = np.array([pos.x, pos.y, pos.z])
        
        quat = msg.pose.pose.orientation
        vel = msg.twist.twist.linear
        
        self.get_logger().info(
            f"\n🚁 无人机状态:\n"
            f"  Frame ID: {self.odom_frame_id}\n"
            f"  位置: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})\n"
            f"  速度: ({vel.x:.2f}, {vel.y:.2f}, {vel.z:.2f})\n"
            f"  姿态: ({quat.x:.3f}, {quat.y:.3f}, {quat.z:.3f}, {quat.w:.3f})",
            throttle_duration_sec=2.0
        )
    
    def verify_alignment(self, points):
        """验证点云是否与无人机位置对齐"""
        if self.drone_pos is None:
            return
        
        # 计算点云相对于无人机的距离
        distances = np.linalg.norm(points - self.drone_pos, axis=1)
        
        mean_dist = distances.mean()
        max_dist = distances.max()
        min_dist = distances.min()
        
        # 检查点云是否围绕无人机
        if mean_dist < 50 and max_dist < 100:  # 合理范围
            status = "✅ 对齐正确"
        else:
            status = "❌ 可能有问题"
        
        self.get_logger().info(
            f"\n🔍 对齐验证 {status}:\n"
            f"  平均距离: {mean_dist:.2f} m\n"
            f"  最小距离: {min_dist:.2f} m\n"
            f"  最大距离: {max_dist:.2f} m",
            throttle_duration_sec=2.0
        )


def main(args=None):
    rclpy.init(args=args)
    
    try:
        verifier = CoordinateSystemVerifier()
        rclpy.spin(verifier)
    except KeyboardInterrupt:
        print("\n验证工具关闭")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
