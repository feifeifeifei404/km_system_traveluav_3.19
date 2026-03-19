#!/usr/bin/env python3
"""
Socket双向通信桥接服务
- 接收TravelUAV目标点 → 发布到ROS2
- 监听SUPER状态 → 返回给TravelUAV
"""
import socket
import json
import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

HOST = '127.0.0.1'
PORT = 65432

class BidirectionalBridge(Node):
    def __init__(self):
        super().__init__('bidirectional_bridge')
        
        # 发布目标点给SUPER
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # ⭐ 配置QoS为BEST_EFFORT以匹配AirSim桥接
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 订阅里程计获取当前位置
        self.odom_sub = self.create_subscription(
            Odometry,
            '/lidar_slam/odom',
            self.odom_callback,
            qos_profile
        )
        
        # 状态变量
        self.current_pos = None
        self.target_pos = None
        self.status = "IDLE"  # IDLE/PLANNING/FLYING/ARRIVED/FAILED/TIMEOUT
        self.arrival_threshold = 0.5
        self.lock = threading.Lock()
        
        # 状态检查定时器
        self.status_timer = self.create_timer(0.2, self.check_status)
        
        self.get_logger().info(f"[*] 双向Socket桥接服务启动")
        self.get_logger().info(f"[*] 监听端口: {HOST}:{PORT}")
    
    def odom_callback(self, msg):
        """更新当前位置"""
        with self.lock:
            self.current_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])
    
    def check_status(self):
        """定期检查是否到达目标"""
        with self.lock:
            if self.target_pos is None or self.current_pos is None:
                return
            
            if self.status == "FLYING":
                dist = np.linalg.norm(self.current_pos - self.target_pos)
                if dist < self.arrival_threshold:
                    self.status = "ARRIVED"
                    self.get_logger().info(f"✓ 到达目标！距离: {dist:.3f}m")
    
    def send_goal(self, x, y, z):
        """发布目标点到ROS2，带高度和距离限制"""
        try:
            # 坐标转换: AirSim NED -> ROS ENU
            ros_x = y
            ros_y = x
            ros_z = -z
            
            self.get_logger().info(
                f"📍 收到目标点 (AirSim): ({x:.2f}, {y:.2f}, {z:.2f}) -> "
                f"(ROS): ({ros_x:.2f}, {ros_y:.2f}, {ros_z:.2f})"
            )
            
            # 📝 取消过滤：SUPER可以处理50-100米范围，不需要限制距离
            # 仅记录目标点信息，不做修改
            if self.current_pos is not None:
                horizontal_dist = np.sqrt((ros_x - self.current_pos[0])**2 + 
                                         (ros_y - self.current_pos[1])**2)
                vertical_dist = abs(ros_z - self.current_pos[2])
                total_dist = np.sqrt(horizontal_dist**2 + vertical_dist**2)
                
                self.get_logger().info(
                    f"📊 目标点分析: 水平={horizontal_dist:.2f}m, 垂直={vertical_dist:.2f}m, 总距离={total_dist:.2f}m"
                )
            
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            msg.pose.position.x = float(ros_x)
            msg.pose.position.y = float(ros_y)
            msg.pose.position.z = float(ros_z)
            msg.pose.orientation.w = 1.0
            
            self.goal_pub.publish(msg)
            
            self.get_logger().info(
                f"✓ 发送给SUPER (ROS): ({ros_x:.2f}, {ros_y:.2f}, {ros_z:.2f})"
            )
            
            with self.lock:
                self.target_pos = np.array([ros_x, ros_y, ros_z])
                self.status = "FLYING"
            
            log_msg = (
                f"\n>>> 收到目标点:\n"
                f"    AirSim原始: (x={x:.2f}, y={y:.2f}, z={z:.2f})\n"
                f"    ROS转换后 : (x={ros_x:.2f}, y={ros_y:.2f}, z={ros_z:.2f})"
            )
            self.get_logger().info(log_msg)
            return True
        except Exception as e:
            self.get_logger().error(f"发布目标点失败: {e}")
            return False
    
    def get_status(self):
        """获取当前状态"""
        with self.lock:
            return {
                "status": self.status,
                "current_pos": self.current_pos.tolist() if self.current_pos is not None else None,
                "target_pos": self.target_pos.tolist() if self.target_pos is not None else None
            }

def handle_client(conn, node):
    """处理单个客户端连接"""
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            
            try:
                request = json.loads(data.decode('utf-8'))
                cmd = request.get('cmd')
                
                if cmd == 'send_goal':
                    # 发送目标点
                    x = request['x']
                    y = request['y']
                    z = request['z']
                    
                    if node.send_goal(x, y, z):
                        response = {"status": "ok", "message": "Goal sent"}
                    else:
                        response = {"status": "error", "message": "Failed to send goal"}
                
                elif cmd == 'get_status':
                    # 查询状态
                    status_info = node.get_status()
                    response = {"status": "ok", "data": status_info}
                
                else:
                    response = {"status": "error", "message": f"Unknown command: {cmd}"}
                
                conn.sendall(json.dumps(response).encode('utf-8'))
                
            except json.JSONDecodeError:
                error_response = {"status": "error", "message": "Invalid JSON"}
                conn.sendall(json.dumps(error_response).encode('utf-8'))
    
    except Exception as e:
        node.get_logger().error(f"处理客户端出错: {e}")

def main():
    rclpy.init()
    node = BidirectionalBridge()
    
    # 在后台线程运行ROS2
    def spin_ros():
        rclpy.spin(node)
    
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    
    # Socket服务器
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)
        
        node.get_logger().info("[*] 等待TravelUAV连接...")
        
        while rclpy.ok():
            conn, addr = server.accept()
            node.get_logger().info(f"[+] TravelUAV连接: {addr}")
            
            # 每个连接在新线程处理
            client_thread = threading.Thread(
                target=handle_client,
                args=(conn, node),
                daemon=True
            )
            client_thread.start()
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"服务器错误: {e}")
    finally:
        server.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
