#!/usr/bin/env python3
"""
实时轨迹可视化工具
实时显示无人机在 XY 平面上的移动轨迹
"""
import sys
sys.path.insert(0, '/opt/ros/humble/local/lib/python3.10/dist-packages')

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import numpy as np

class TrajectoryPlotter(Node):
    def __init__(self):
        super().__init__('trajectory_plotter')
        self.xs = deque(maxlen=2000)
        self.ys = deque(maxlen=2000)
        self.zs = deque(maxlen=2000)
        self.current = [0, 0, 0]
        self.lock = threading.Lock()

        self.create_subscription(Odometry, '/lidar_slam/odom', self.odom_cb, 10)
        self.get_logger().info("轨迹可视化启动，等待里程计数据...")

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        with self.lock:
            self.xs.append(x)
            self.ys.append(y)
            self.zs.append(z)
            self.current = [x, y, z]


def main():
    rclpy.init()
    node = TrajectoryPlotter()

    # ROS spin 在后台线程
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # matplotlib 实时绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('无人机实时轨迹', fontsize=14)

    def update(frame):
        with node.lock:
            xs = list(node.xs)
            ys = list(node.ys)
            zs = list(node.zs)
            cur = node.current.copy()

        # XY 平面轨迹
        ax1.cla()
        if xs:
            ax1.plot(xs, ys, 'b-', linewidth=1, alpha=0.7, label='轨迹')
            ax1.plot(xs[0], ys[0], 'go', markersize=8, label='起点')
            ax1.plot(cur[0], cur[1], 'r^', markersize=10, label='当前位置')
        ax1.set_xlabel('X (东, m)')
        ax1.set_ylabel('Y (北, m)')
        ax1.set_title(f'XY 平面  当前:({cur[0]:.1f}, {cur[1]:.1f})')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # 高度变化
        ax2.cla()
        if zs:
            ax2.plot(range(len(zs)), zs, 'r-', linewidth=1.5)
            ax2.axhline(y=cur[2], color='orange', linestyle='--', alpha=0.7)
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('Z 高度 (m)')
        ax2.set_title(f'高度变化  当前:{cur[2]:.2f}m')
        ax2.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
