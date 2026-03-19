#!/usr/bin/env python3
"""
详细的坐标系验证工具
验证 AirSim → 桥接节点 → SUPER 的完整坐标转换链
"""

import sys
sys.path.insert(0, '/opt/ros/humble/local/lib/python3.10/dist-packages')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2 as pc2
import numpy as np
from collections import deque

class CoordinateVerifier(Node):
    def __init__(self):
        super().__init__('coordinate_verifier')
        
        # 订阅点云和里程计
        self.cloud_sub = self.create_subscription(
            PointCloud2, '/cloud_registered', self.cloud_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/lidar_slam/odom', self.odom_callback, 10)
        
        # 数据缓存
        self.cloud_data = deque(maxlen=5)
        self.odom_data = deque(maxlen=5)
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("坐标系详细验证工具启动")
        self.get_logger().info("=" * 80)
    
    def cloud_callback(self, msg):
        """处理点云消息"""
        points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        
        if len(points) > 0:
            points = np.array([(p[0], p[1], p[2]) for p in points])
            
            self.cloud_data.append({
                'frame_id': msg.header.frame_id,
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
                'num_points': len(points),
                'x_range': (points[:, 0].min(), points[:, 0].max()),
                'y_range': (points[:, 1].min(), points[:, 1].max()),
                'z_range': (points[:, 2].min(), points[:, 2].max()),
                'center': points.mean(axis=0),
            })
    
    def odom_callback(self, msg):
        """处理里程计消息"""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        self.odom_data.append({
            'frame_id': msg.header.frame_id,
            'child_frame_id': msg.child_frame_id,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            'position': (pos.x, pos.y, pos.z),
            'orientation': (ori.x, ori.y, ori.z, ori.w),
        })
    
    def print_analysis(self):
        """打印详细分析"""
        if not self.cloud_data or not self.odom_data:
            self.get_logger().warn("还没有收到数据")
            return
        
        cloud = self.cloud_data[-1]
        odom = self.odom_data[-1]
        
        self.get_logger().info("\n" + "=" * 80)
        self.get_logger().info("📊 点云信息")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Frame ID: {cloud['frame_id']}")
        self.get_logger().info(f"点数: {cloud['num_points']}")
        self.get_logger().info(f"X 范围: [{cloud['x_range'][0]:.2f}, {cloud['x_range'][1]:.2f}]")
        self.get_logger().info(f"Y 范围: [{cloud['y_range'][0]:.2f}, {cloud['y_range'][1]:.2f}]")
        self.get_logger().info(f"Z 范围: [{cloud['z_range'][0]:.2f}, {cloud['z_range'][1]:.2f}]")
        self.get_logger().info(f"点云中心: ({cloud['center'][0]:.2f}, {cloud['center'][1]:.2f}, {cloud['center'][2]:.2f})")
        
        self.get_logger().info("\n" + "=" * 80)
        self.get_logger().info("🚁 无人机里程计")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Frame ID: {odom['frame_id']}")
        self.get_logger().info(f"Child Frame ID: {odom['child_frame_id']}")
        pos = odom['position']
        self.get_logger().info(f"位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        ori = odom['orientation']
        self.get_logger().info(f"姿态 (四元数): ({ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f})")
        
        # 验证点云是否围绕无人机
        self.get_logger().info("\n" + "=" * 80)
        self.get_logger().info("🔍 坐标系验证")
        self.get_logger().info("=" * 80)
        
        # 检查点云中心是否接近无人机位置
        cloud_center = cloud['center']
        drone_pos = np.array(pos)
        distance = np.linalg.norm(cloud_center - drone_pos)
        
        self.get_logger().info(f"点云中心到无人机的距离: {distance:.2f} m")
        
        # 检查坐标系一致性
        if cloud['frame_id'] == odom['frame_id'] == 'world':
            self.get_logger().info("✅ Frame ID 一致: 都是 'world'")
        else:
            self.get_logger().warn(f"⚠️ Frame ID 不一致: 点云={cloud['frame_id']}, 里程计={odom['frame_id']}")
        
        # 检查点云范围
        if cloud['z_range'][0] >= 0:
            self.get_logger().info(f"✅ Z 坐标范围正确: [{cloud['z_range'][0]:.2f}, {cloud['z_range'][1]:.2f}] (都 >= 0)")
        else:
            self.get_logger().warn(f"⚠️ Z 坐标范围异常: [{cloud['z_range'][0]:.2f}, {cloud['z_range'][1]:.2f}] (有负值)")
        
        # 检查点云是否围绕无人机
        if distance < 50:  # 合理范围
            self.get_logger().info(f"✅ 点云围绕无人机: 距离 {distance:.2f} m (合理)")
        else:
            self.get_logger().warn(f"⚠️ 点云距离无人机太远: {distance:.2f} m")
        
        self.get_logger().info("=" * 80 + "\n")

def main():
    rclpy.init()
    verifier = CoordinateVerifier()
    
    try:
        # 运行 10 秒，每 2 秒打印一次
        for i in range(5):
            rclpy.spin_once(verifier, timeout_sec=2)
            verifier.print_analysis()
    except KeyboardInterrupt:
        pass
    finally:
        verifier.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
