#!/usr/bin/env python3
"""
测试SUPER目标点发送 (ROS2版本)

发送一个目标点给SUPER，测试完整的集成系统

使用方法:
python3 test_super_goal_ros2.py [x] [y] [z]

示例:
python3 test_super_goal_ros2.py 10 5 2
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import sys
import time


class GoalSender(Node):
    """目标点发送节点"""
    
    def __init__(self, x, y, z):
        super().__init__('test_goal_sender')
        
        self.x = x
        self.y = y
        self.z = z
        
        # 创建发布者（SUPER订阅/goal_pose话题）
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # 等待发布者准备好
        time.sleep(1.0)
        
        # 发送目标点
        self.send_goal()
    
    def send_goal(self):
        """发送目标点给SUPER"""
        
        # 创建目标点消息
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "world"
        
        goal_msg.pose.position.x = self.x
        goal_msg.pose.position.y = self.y
        goal_msg.pose.position.z = self.z
        
        # 朝向（默认朝向）
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        # 发送目标点
        print("=" * 70)
        print("发送目标点给SUPER:")
        print(f"  位置: ({self.x}, {self.y}, {self.z})")
        print("=" * 70)
        
        # 多次发送确保收到
        for i in range(5):
            self.goal_pub.publish(goal_msg)
            time.sleep(0.2)
        
        print("✓ 目标点已发送!")
        print("\n查看RViz中的路径规划结果")
        print("或运行: ros2 topic echo /planning/pos_cmd")


def main(args=None):
    """主函数"""
    
    # 解析命令行参数
    if len(sys.argv) >= 4:
        try:
            x = float(sys.argv[1])
            y = float(sys.argv[2])
            z = float(sys.argv[3])
        except ValueError:
            print("错误: 参数必须是数字")
            sys.exit(1)
    else:
        # 默认目标点
        print("未提供目标点，使用默认值")
        x, y, z = 10.0, 5.0, 2.0
    
    rclpy.init(args=args)
    
    try:
        sender = GoalSender(x, y, z)
        # 保持节点运行一小段时间以确保消息发送
        rclpy.spin_once(sender, timeout_sec=2.0)
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


