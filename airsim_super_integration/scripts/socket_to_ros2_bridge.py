#!/usr/bin/env python3
"""
Socket 转 ROS2 桥接服务 (增强版)
功能: 接收 TravelUAV (JSON) -> 发布 SUPER 目标点 (/goal)
"""
import socket
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import sys

# 配置
HOST = '127.0.0.1' # 如果 TravelUAV 和 ROS 在同一台机器/容器
PORT = 65432       # 必须与 TravelUAV eval.py 中的端口一致

class SocketGoalSender(Node):
    def __init__(self):
        super().__init__('socket_goal_sender')
        # SUPER 监听 /goal_pose 话题
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.get_logger().info(f"[*] Socket-to-ROS2 Bridge 启动")
        self.get_logger().info(f"[*] 监听端口: {HOST}:{PORT}")
        self.get_logger().info(f"[*] 等待 TravelUAV 发送目标点...")

    def process_and_publish(self, data_str):
        try:
            # 1. 解析 JSON
            goal_data = json.loads(data_str)
            
            # TravelUAV (AirSim) 原始坐标
            as_x = goal_data['x']
            as_y = goal_data['y']
            as_z = goal_data['z']

            # 2. 坐标系转换 (必须与 airsim_super_bridge_ros2.py 保持严格一致)
            # 规则: AirSim NED -> ROS ENU
            # ROS_X (东) = AirSim_Y (右)
            # ROS_Y (北) = AirSim_X (前)
            # ROS_Z (上) = -AirSim_Z (下)
            ros_x = as_y
            ros_y = as_x
            ros_z = -as_z

            # 3. 构造 ROS 消息
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "world" # 确保 SUPER 的 map frame 是 world
            
            msg.pose.position.x = float(ros_x)
            msg.pose.position.y = float(ros_y)
            msg.pose.position.z = float(ros_z)
            
            # 方向默认朝向前方 (w=1.0)
            msg.pose.orientation.w = 1.0
            
            # 4. 发布
            self.goal_pub.publish(msg)
            
            # 5. 打印详细日志 (用于调试)
            log_msg = (
                f"\n>>> 收到指令:\n"
                f"    AirSim原始: (x={as_x:.2f}, y={as_y:.2f}, z={as_z:.2f})\n"
                f"    ROS转换后 : (x={ros_x:.2f}, y={ros_y:.2f}, z={ros_z:.2f}) <发布成功>"
            )
            self.get_logger().info(log_msg)
            return True

        except Exception as e:
            self.get_logger().error(f"数据解析失败: {e}")
            return False

def main():
    rclpy.init()
    node = SocketGoalSender()
    
    # 建立 Socket Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 允许端口复用 (防止重启脚本时报 Address already in use)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        
        # 设置为非阻塞模式，以便让 rclpy 也能有机会运行 (如果需要)
        # 这里为了简单，使用阻塞 accept，因为这是慢系统，频率很低
        
        while rclpy.ok():
            # 等待连接
            conn, addr = server.accept()
            with conn:
                node.get_logger().info(f"[+] TravelUAV 已连接: {addr}")
                
                # 接收数据
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    
                    # 处理数据
                    if node.process_and_publish(data.decode('utf-8')):
                        conn.sendall(b"Goal Received")
                    else:
                        conn.sendall(b"Error Parsing")
                        
                node.get_logger().info(f"[-] TravelUAV 断开连接 (等待下一次连接)")
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Server Error: {e}")
    finally:
        server.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()