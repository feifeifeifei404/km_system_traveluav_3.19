#!/usr/bin/env python3.10
# 必须用 ROS Humble 对应的 Python 3.10 运行；若默认 python3 是 conda 的 3.12 会报 rclpy 找不到
"""
直接向 SUPER 发送一个目标点，不经过 TravelUAV。用于单独测试 SUPER 断点/调试。

用法（先启动 SUPER 并附加 GDB，再在另一个终端运行）:
  source /mnt/data/super_ws/install/setup.bash
  python3.10 /mnt/data/super_ws/send_goal_to_super.py [x] [y] [z]
  若没有 python3.10，可先 conda deactivate 再用 python3

默认发送 (5, 5, -5)，也可只改其中一个，例如:
  python3.10 send_goal_to_super.py 10 0 -3
"""
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


def main():
    x = float(sys.argv[1]) if len(sys.argv) > 1 else -23.0
    y = float(sys.argv[2]) if len(sys.argv) > 2 else -101.0
    z = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    rclpy.init()
    node = Node("send_goal_once")
    pub = node.create_publisher(PoseStamped, "/goal_pose", 1)

    msg = PoseStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = 1.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0

    # 给订阅者一点时间
    import time
    time.sleep(0.5)
    pub.publish(msg)
    node.get_logger().info(f"已发送目标点 /goal_pose: x={x}, y={y}, z={z}")
    time.sleep(0.2)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
