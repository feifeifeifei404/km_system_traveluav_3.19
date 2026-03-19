"""
SUPER Socket通信客户端
在TravelUAV的conda环境(llama)中使用
不依赖ROS2
"""
import socket
import json
import time
import numpy as np

class SUPERSocketClient:
    """通过Socket与SUPER通信的客户端"""
    
    def __init__(self, host='127.0.0.1', port=65432, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connection_timeout = 5.0
    
    def _send_request(self, request):
        """发送请求并接收响应"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.connection_timeout)
                s.connect((self.host, self.port))
                s.sendall(json.dumps(request).encode('utf-8'))
                
                # 接收响应
                data = s.recv(4096)
                response = json.loads(data.decode('utf-8'))
                return response
        except socket.timeout:
            return {"status": "error", "message": "Connection timeout"}
        except ConnectionRefusedError:
            return {"status": "error", "message": "SUPER Bridge not running"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def send_goal(self, x, y, z):
        """
        发送目标点给SUPER
        
        参数:
            x, y, z: AirSim坐标系下的目标点
        
        返回:
            bool: 是否成功发送
        """
        request = {
            'cmd': 'send_goal',
            'x': float(x),
            'y': float(y),
            'z': float(z)
        }
        
        response = self._send_request(request)
        
        if response.get('status') == 'ok':
            print(f"[SUPER] 目标点已发送: ({x:.2f}, {y:.2f}, {z:.2f})")
            return True
        else:
            print(f"[SUPER] 发送失败: {response.get('message')}")
            return False
    
    def get_status(self):
        """
        获取SUPER当前状态
        
        返回:
            dict: {
                'status': 'IDLE'/'FLYING'/'ARRIVED'/'FAILED',
                'current_pos': [x, y, z],
                'target_pos': [x, y, z]
            }
            或None（如果失败）
        """
        request = {'cmd': 'get_status'}
        response = self._send_request(request)
        
        if response.get('status') == 'ok':
            return response.get('data')
        else:
            return None
    
    def wait_for_arrival(self, timeout=60.0, check_interval=0.5):
        """
        等待SUPER飞行到目标点
        
        参数:
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）
        
        返回:
            tuple: (success, final_status)
                - success (bool): 是否成功到达
                - final_status (str): 最终状态
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_info = self.get_status()
            
            if status_info is None:
                print("[SUPER] 无法获取状态，SUPER Bridge可能断开")
                return False, "CONNECTION_LOST"
            
            status = status_info.get('status')
            
            if status == "ARRIVED":
                print(f"[SUPER] ✓ 已到达目标点")
                return True, status
            elif status == "FAILED":
                print(f"[SUPER] ✗ 规划失败")
                return False, status
            
            time.sleep(check_interval)
        
        print(f"[SUPER] ✗ 超时（{timeout}秒）")
        return False, "TIMEOUT"
    
    def wait_for_arrival_with_position(self, env, target_pos, threshold=2.0, 
                                      timeout=120.0, record_interval=0.5):
        """
        等待SUPER飞行并记录轨迹（兼容TravelUAV现有代码）
        
        参数:
            env: AirVLNENV实例
            target_pos: [x, y, z] AirSim坐标系目标点
            threshold: 到达阈值（米）
            timeout: 超时时间（秒）
            record_interval: 轨迹记录间隔（秒）
        
        返回:
            tuple: (success, trajectory, collision_detected)
        """
        import airsim
        
        start_time = time.time()
        trajectory = []
        last_record_time = time.time()
        collision_detected = False
        
        # 创建临时客户端查询位置
        try:
            temp_client = airsim.MultirotorClient(port=25001)
            temp_client.confirmConnection()
        except:
            print("[ERROR] 无法连接到AirSim（端口25001）")
            return False, [], False
        
        # 计算目标距离，动态调整超时时间
        try:
            state = temp_client.getMultirotorState()
            pos = state.kinematics_estimated.position
            curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
            distance = np.linalg.norm(curr_pos - np.array(target_pos))
            # 根据距离调整超时：假设平均速度1m/s，加50%余量
            dynamic_timeout = max(timeout, distance * 1.5)
            print(f"[SUPER] 当前位置: {np.round(curr_pos, 2)}")
            print(f"[SUPER] 目标位置: {np.round(target_pos, 2)}")
            print(f"[SUPER] 距离: {distance:.2f}m, 超时: {dynamic_timeout:.1f}s")
            timeout = dynamic_timeout
        except:
            pass
        
        print(f"[SUPER] 等待飞行到目标点...")
        
        while time.time() - start_time < timeout:
            try:
                # 获取当前状态
                state = temp_client.getMultirotorState()
                pos = state.kinematics_estimated.position
                curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
                
                # 检查碰撞
                if state.collision.has_collided:
                    print("[SUPER] ✗ 检测到碰撞！")
                    collision_detected = True
                    break
                
                # 记录轨迹
                current_time = time.time()
                if current_time - last_record_time >= record_interval:
                    orientation = state.kinematics_estimated.orientation
                    trajectory.append({
                        'position': curr_pos.tolist(),
                        'orientation': [
                            orientation.x_val,
                            orientation.y_val,
                            orientation.z_val,
                            orientation.w_val
                        ],
                        'timestamp': current_time
                    })
                    last_record_time = current_time
                
                # 检查是否到达
                dist = np.linalg.norm(curr_pos - np.array(target_pos))
                if dist < threshold:
                    print(f"[SUPER] ✓ 到达目标！距离: {dist:.2f}m")
                    # 记录最后一个位置
                    trajectory.append({
                        'position': curr_pos.tolist(),
                        'orientation': [
                            orientation.x_val,
                            orientation.y_val,
                            orientation.z_val,
                            orientation.w_val
                        ],
                        'timestamp': current_time
                    })
                    return True, trajectory, False
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[ERROR] 获取位置失败: {e}")
                time.sleep(0.5)
        
        print(f"[SUPER] ✗ 超时（{timeout}秒）")
        return False, trajectory, collision_detected


# 全局单例
_super_client = None

def get_super_client(host='127.0.0.1', port=65432):
    """获取全局SUPER客户端单例"""
    global _super_client
    if _super_client is None:
        _super_client = SUPERSocketClient(host=host, port=port)
    return _super_client
