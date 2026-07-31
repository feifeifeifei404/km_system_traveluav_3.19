from collections import deque
import multiprocessing
import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import os,sys

import tqdm

cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

from utils.logger import logger


IMAGE_TIMING_LOG_ENABLED = False


def image_timing_log(message):
    if IMAGE_TIMING_LOG_ENABLED:
        logger.info(message)


def image_timing_warn(message):
    if IMAGE_TIMING_LOG_ENABLED:
        logger.warning(message)


class BaseSensor:
    def __init__(self) -> None:
        pass

    def retrieve(self):
        raise NotImplementedError()

class State(BaseSensor):
    def __init__(self, client, drone_name=''):
        self.data = {'position': None, 'linear_velocity': None, 'linear_acceleration':None,
                     'orientation':None, 'angular_velocity':None, 'angular_acceleration':None}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name

    def retrieve(self):
        data = self.client.getMultirotorState(vehicle_name=self.drone_name)
        collision = {}
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        collision['has_collided'] = collision_info.has_collided
        collision['object_name'] = data.collision.object_name
        gps_location = [data.gps_location.latitude,data.gps_location.longitude,data.gps_location.altitude]
        timestamp = data.timestamp
        position = list(data.kinematics_estimated.position)
        linear_velocity = list(data.kinematics_estimated.linear_velocity)
        linear_acceleration = list(data.kinematics_estimated.linear_acceleration)
        orientation = list(data.kinematics_estimated.orientation)
        angular_velocity = list(data.kinematics_estimated.angular_velocity)
        angular_acceleration = list(data.kinematics_estimated.angular_acceleration)

        self.data.update({'collision': collision, 
                          'gps_location': gps_location,
                          'timestamp': timestamp, 
                          'position': position,
                          'linear_velocity': linear_velocity,
                          'linear_acceleration': linear_acceleration,
                          'orientation': orientation,
                          'angular_velocity': angular_velocity,
                          'angular_acceleration': angular_acceleration
                          })
        return self.data
        
        
class Imu(BaseSensor):
    def __init__(self, client, drone_name='', imu_name=''):
        self.data = {}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name
        self.imu_name = imu_name

    def retrieve(self):
        data = self.client.getImuData(imu_name=self.imu_name,vehicle_name=self.drone_name)
        time_stamp = data.time_stamp
        orientation = data.orientation
        angular_velocity = list(data.angular_velocity)
        linear_acceleration = list(data.linear_acceleration)
        q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                      [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                      [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)])).tolist()
        self.data.update({'time_stamp': time_stamp, 'rotation': rotation_matrix, 'orientation': list(data.orientation),
                          'linear_acceleration': linear_acceleration, 'angular_velocity': angular_velocity})
        return self.data


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False
        self.started_at = None
        self.finished_at = None

    def run(self):
        self.started_at = time.perf_counter()
        self.result = self.func(*self.args)
        self.finished_at = time.perf_counter()
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    @staticmethod
    def _join_threads_with_monitor(threads, label, warn_interval=10.0):
        join_start = time.perf_counter()
        last_warn = join_start
        while True:
            alive = []
            for index_1, row in enumerate(threads):
                for index_2, thread in enumerate(row):
                    if thread.is_alive():
                        elapsed = time.perf_counter() - (thread.started_at or join_start)
                        stage = getattr(thread, 'debug_stage', 'unknown')
                        detail = getattr(thread, 'debug_detail', '')
                        alive.append((index_1, index_2, elapsed, stage, detail))
            if not alive:
                break
            now = time.perf_counter()
            if now - last_warn >= warn_interval:
                alive_msg = ', '.join([
                    f"machine={i} scene={j} alive_for={elapsed:.1f}s stage={stage} detail={detail}"
                    for i, j, elapsed, stage, detail in alive
                ])
                image_timing_warn(f"[TIMING][{label}] waiting for image threads: {alive_msg}")
                last_warn = now
            time.sleep(0.5)
        for row in threads:
            for thread in row:
                thread.join(timeout=0)
        return time.perf_counter() - join_start

    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]
        self.airsim_ports = []
        self.airsim_ip = '127.0.0.1'
        self._init_check()
        self.objects_name_cnt = [[0 for _ in list(item['open_scenes'])] for item in machines_info ]

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _confirmConnection(self) -> bool:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    confirmed = False
                    count = 0
                    while not confirmed and count < 30:
                        try:
                            self.airsim_clients[index_1][index_2].confirmConnection()
                            confirmed = True
                        except Exception as e:
                            time.sleep(1)
                            print('failed', e)
                            count += 1
                            pass
        
        return confirmed

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def run_call(self, airsim_timeout: int=300) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients


        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f'开始打开场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            logger.info(f'gpus: {self.machines_info[index]}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])))
            if result[0] == False:
                logger.error(f'打开场景失败，机器: {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('打开场景失败')
            assert len(result[1]) == 2, '打开场景失败'
            print('waiting for airsim connection...')
            time.sleep(3 * len(self.machines_info[index]['open_scenes']) + 35)
            ip = result[1][0]
            # 如果IP是bytes类型，需要解码为字符串
            if isinstance(ip, bytes):
                ip = ip.decode('utf-8')
            ports = result[1][1]
            self.airsim_ip = ip
            self.airsim_ports = ports
            # assert str(ip) == str(socket_client.address._host), '打开场景失败'
            assert str(ip) == str(socket_client.address._host), f'打开场景失败: IP不匹配 {ip} != {socket_client.address._host}'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), '打开场景失败'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.MultirotorClient(ip=ip, port=port, timeout_value=airsim_timeout)
                    setattr(self.airsim_clients[index][i], '_traveluav_debug_ip', ip)
                    setattr(self.airsim_clients[index][i], '_traveluav_debug_port', port)
                    setattr(self.airsim_clients[index][i], '_traveluav_debug_timeout', airsim_timeout)
                    setattr(self.airsim_clients[index][i], '_traveluav_debug_scene', self.machines_info[index]['open_scenes'][i])
                    logger.info(
                        f"[AirSimClient] created machine={index} scene={i} map={self.machines_info[index]['open_scenes'][i]} "
                        f"ip={ip} port={port} timeout={airsim_timeout}s"
                    )
                    print(port)

            logger.info(f'打开场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            return ports

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        threads = []
        
        if not (np.array(thread_results) == True).all():
            raise Exception('打开场景失败')

        after = time.time()
        diff = after - before
        logger.info(f"启动时间：{diff}")

        assert self._confirmConnection(), 'server connect failed'
        self._closeSocketConnection()
    
    def collect_DDP(self, data_dir, workers):
        def init_worker(index, lock):
            with lock:
                multiprocessing.current_process().client_port = self.machines_info[0]['SOCKET_PORT']
                multiprocessing.current_process().machine_ip = self.machines_info[0]['MACHINE_IP']
                multiprocessing.current_process().client = self.airsim_clients[0][index.value]
                multiprocessing.current_process().port = self.airsim_ports[index.value]
                index.value += 1
        index = multiprocessing.Value('i', 0)
        lock = multiprocessing.Lock()
        with multiprocessing.Pool(workers, initializer=init_worker, initargs=(index, lock)) as p:
            r = list(tqdm.tqdm(p.imap_unordered(collect, data_dir), total=len(data_dir)))

    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
                )

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients

            self._closeConnection()

            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'开始关闭所有场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'关闭所有场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)

    def move_path_by_waypoints(self, waypoints_list, start_states):
        velocity = 2
        drivetrain = airsim.DrivetrainType.ForwardOnly
        yaw_mode=airsim.YawMode(is_rate=False)
        lookahead=3
        adaptive_lookahead=1
        def move_path(airsim_client: airsim.VehicleClient, waypoints, start_state):
            results = []
            state_sensor = State(airsim_client, )
            imu_sensor = Imu(airsim_client, imu_name='Imu')
            path = [airsim.Vector3r(*waypoint[0:3]) for waypoint in waypoints]
            # ⚠️ 集成SUPER后：不能在这里抢夺控制权！
            # 控制权应该由SUPER的桥接节点统一管理
            # airsim_client.enableApiControl(True)
            # airsim_client.armDisarm(True)
            airsim_client.simPause(False)
            # ⚠️ 不能用simSetKinematics重置状态，会干扰SUPER
            # airsim_client.simSetKinematics(start_state, ignore_collision=False)
            state_info = state_sensor.retrieve()
            airsim_client.moveOnPathAsync(path=path, 
                                velocity=velocity, 
                                drivetrain=drivetrain, 
                                yaw_mode=yaw_mode, 
                                lookahead=lookahead, 
                                adaptive_lookahead=adaptive_lookahead)
            target_idx = len(path)  # 使用实际航点数量，而不是硬编码的5
            current_idx = 0
            pos_queue = deque(maxlen=20)
            start_time = time.perf_counter()
            collision = False
            distance = 10000
            while True:
                time.sleep(0.005)
                # 动态超时：每个航点给 10 秒
                timeout = max(10, len(path) * 10)
                if time.perf_counter() - start_time > timeout:
                    logger.warning(f'Timeout after {timeout}s, path length: {len(path)}')
                    return None
                target = path[current_idx]
                state_info = copy.deepcopy(state_sensor.retrieve())
                imu_info = copy.deepcopy(imu_sensor.retrieve())
                position = np.array(state_info['position'])
                pos_queue.append(position)
                if len(pos_queue) == pos_queue.maxlen:
                    recent_loc = position
                    history_loc = pos_queue.popleft()
                    delta_distance = np.linalg.norm(history_loc -recent_loc)
                    if delta_distance < 0.1:
                        print('move on path api: stuck max len')
                        collision = True
                        break
                new_distance = np.linalg.norm(position - np.array([target.x_val, target.y_val, target.z_val]))
                if new_distance > distance:
                    results.append({'sensors': {'state': state_info, 'imu': imu_info}})
                    current_idx += 1
                    if current_idx == target_idx:
                        # ⚠️ 集成SUPER后：不能暂停仿真
                        # airsim_client.simPause(True)
                        break
                    else:
                        distance = 10000
                else:
                    distance = new_distance
            return {'states': results, 'collision': collision}
        
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(move_path, (self.airsim_clients[index_1][index_2], waypoints_list[index_1][index_2], start_states[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        result_poses_list = []
        error_flag = False
        for index_1, _ in enumerate(threads):
            result_poses_list.append([])
            for index_2, _ in enumerate(threads[index_1]):
                result =  threads[index_1][index_2].get_result()
                result_poses_list[index_1].append(
                    result
                )
                if result is None:
                    error_flag = True
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('move path by waypoints failed.')
            return None
        if error_flag:
            return None
        return result_poses_list
    
    def setPoses(self, poses: list) -> bool:
        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose) -> None:
            if airsim_client is None:
                raise Exception('error')
                return

            airsim_client.simSetKinematics(
                state=pose,
                ignore_collision=True,
            )
            # ⚠️ 集成SUPER后：不能暂停仿真！SUPER需要连续运行
            # airsim_client.simContinueForFrames(1)
            # airsim_client.simPause(True)
            
            # 确保仿真继续运行
            airsim_client.simPause(False)

            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses失败')
            return False

        return True
    
    def setObjects(self, object_list: list):
        def _setObject(airsim_client: airsim.VehicleClient, object_info: dict) -> None:
            if airsim_client is None:
                raise Exception('error')
                return
            asset_name = object_info['asset_name']
            pose = object_info['pose']
            scale = object_info['scale']
            object_cnt = object_info['object_cnt']
            if object_cnt > 0:
                airsim_client.simDestroyObject('my_object_' + str(object_cnt - 1))
            success = airsim_client.simSpawnObject(
                    'my_object_' + str(object_cnt), asset_name, pose, scale, physics_enabled=False, is_blueprint=False)
            # ⚠️ 集成SUPER后：不能暂停仿真
            # airsim_client.simContinueForFrames(1)
            # airsim_client.simPause(True)
            
            # 确保仿真继续运行
            airsim_client.simPause(False)
            return success

        threads = []
        thread_results = []
        cnt = 0
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                object_list[cnt]['object_cnt'] = self.objects_name_cnt[index_1][index_2]
                threads[index_1].append(
                    MyThread(_setObject, (self.airsim_clients[index_1][index_2], object_list[cnt]))
                )
                self.objects_name_cnt[index_1][index_2] += 1
                cnt += 1

        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('set Object失败')
            return False
        return True
    
    def getImageResponses(self, cameras=['FrontCamera', 'LeftCamera', 'RightCamera', 'RearCamera', 'DownCamera'], poses=None):
        def _getImages(airsim_client: airsim.VehicleClient, machine_idx=None, scene_idx=None):
            if airsim_client is None:
                raise Exception('client is None.')
                return None, None
            time_sleep_cnt = 0
            total_start = time.perf_counter()
            while True:
                try:
                    images, depth_images = [], []
                    split_rpc_total_start = time.perf_counter()
                    for camera_name in cameras:
                        camera_total_start = time.perf_counter()

                        rgb_request = airsim.ImageRequest(
                            camera_name,
                            airsim.ImageType.Scene,
                            pixels_as_float=False,
                            compress=False,
                        )
                        depth_request = airsim.ImageRequest(
                            camera_name,
                            airsim.ImageType.DepthPlanar,
                            pixels_as_float=True,
                            compress=False,
                        )

                        rgb_rpc_start = time.perf_counter()
                        image_timing_log(
                            f"[TIMING][getImageResponses] single simGetImages START camera={camera_name} "
                            f"type=Scene machine={machine_idx} scene={scene_idx}"
                        )
                        rgb_resp = airsim_client.simGetImages(requests=[rgb_request])[0]
                        image_timing_log(
                            f"[TIMING][getImageResponses] single simGetImages DONE camera={camera_name} "
                            f"type=Scene machine={machine_idx} scene={scene_idx}: "
                            f"{time.perf_counter() - rgb_rpc_start:.3f}s "
                            f"width={rgb_resp.width} height={rgb_resp.height}"
                        )

                        depth_rpc_start = time.perf_counter()
                        image_timing_log(
                            f"[TIMING][getImageResponses] single simGetImages START camera={camera_name} "
                            f"type=DepthPlanar machine={machine_idx} scene={scene_idx}"
                        )
                        depth_resp = airsim_client.simGetImages(requests=[depth_request])[0]
                        image_timing_log(
                            f"[TIMING][getImageResponses] single simGetImages DONE camera={camera_name} "
                            f"type=DepthPlanar machine={machine_idx} scene={scene_idx}: "
                            f"{time.perf_counter() - depth_rpc_start:.3f}s "
                            f"width={depth_resp.width} height={depth_resp.height} "
                            f"float_len={len(depth_resp.image_data_float)}"
                        )

                        camera_decode_start = time.perf_counter()
                        image = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(rgb_resp.height, rgb_resp.width, 3)
                        depth_img_in_meters = airsim.list_to_2d_float_array(
                            depth_resp.image_data_float,
                            depth_resp.width,
                            depth_resp.height,
                        )
                        depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
                        images.append(image)
                        depth_images.append(depth_image)
                        image_timing_log(
                            f"[TIMING][getImageResponses] decode camera={camera_name} machine={machine_idx} "
                            f"scene={scene_idx}: {time.perf_counter() - camera_decode_start:.3f}s "
                            f"rgb_shape={image.shape} depth_shape={depth_image.shape} "
                            f"camera_total={time.perf_counter() - camera_total_start:.3f}s"
                        )

                    image_timing_log(
                        f"[TIMING][getImageResponses] split rpc all machine={machine_idx} scene={scene_idx}: "
                        f"{time.perf_counter() - split_rpc_total_start:.3f}s cameras={cameras}"
                    )
                    break
                except Exception as e:
                    time_sleep_cnt += 1
                    logger.error(
                        f"[TIMING][getImageResponses] simGetImages ERROR machine={machine_idx} scene={scene_idx} "
                        f"elapsed={time.perf_counter() - rpc_start if 'rpc_start' in locals() else -1:.3f}s error={e}"
                    )
                    logger.error("图片获取错误: " + str(e))
                    logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                    time.sleep(1)
                if time_sleep_cnt > 10:
                    raise Exception('图片获取失败')
            image_timing_log(
                f"[TIMING][getImageResponses] _getImages total machine={machine_idx} scene={scene_idx}: "
                f"{time.perf_counter() - total_start:.3f}s retries={time_sleep_cnt}"
            )
            return images, depth_images

        total_start = time.perf_counter()
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], index_1, index_2))
                )
        launch_start = time.perf_counter()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        image_timing_log(f"[TIMING][getImageResponses] launch threads: {time.perf_counter() - launch_start:.3f}s")
        join_start = time.perf_counter()
        join_elapsed = self._join_threads_with_monitor(threads, 'getImageResponses')
        image_timing_log(f"[TIMING][getImageResponses] join threads: {join_elapsed:.3f}s")
        collect_start = time.perf_counter()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        image_timing_log(f"[TIMING][getImageResponses] collect results: {time.perf_counter() - collect_start:.3f}s")
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None

        image_timing_log(f"[TIMING][getImageResponses] total: {time.perf_counter() - total_start:.3f}s")
        return responses
    
    
    def getImageResponsesForRecord(self, cameras=['FrontCameraRecord', 'DownCameraRecord'], poses=None):
        def _log_response(response, camera_name, image_type, machine_idx, scene_idx):
            if not IMAGE_TIMING_LOG_ENABLED:
                return
            uint8_len = len(getattr(response, 'image_data_uint8', []) or [])
            float_len = len(getattr(response, 'image_data_float', []) or [])
            logger.info(
                f"[DIAG][getImageResponsesForRecord] response camera={camera_name} type={image_type} "
                f"machine={machine_idx} scene={scene_idx} width={getattr(response, 'width', None)} "
                f"height={getattr(response, 'height', None)} uint8_len={uint8_len} float_len={float_len} "
                f"pixels_as_float={getattr(response, 'pixels_as_float', None)} compress={getattr(response, 'compress', None)}"
            )

        def _get_single_response(airsim_client, request, camera_name, image_type, machine_idx, scene_idx, current_thread):
            setattr(current_thread, 'debug_stage', 'record_single_rpc')
            setattr(current_thread, 'debug_detail', f"camera={camera_name} type={image_type}")
            rpc_start = time.perf_counter()
            image_timing_log(
                f"[TIMING][getImageResponsesForRecord] single simGetImages START camera={camera_name} "
                f"type={image_type} machine={machine_idx} scene={scene_idx}"
            )
            responses = airsim_client.simGetImages(requests=[request])
            rpc_elapsed = time.perf_counter() - rpc_start
            image_timing_log(
                f"[TIMING][getImageResponsesForRecord] single simGetImages DONE camera={camera_name} "
                f"type={image_type} machine={machine_idx} scene={scene_idx}: {rpc_elapsed:.3f}s "
                f"response_count={len(responses) if responses is not None else 'None'}"
            )
            if responses is None or len(responses) != 1:
                raise Exception(
                    f"single simGetImages returned invalid response_count={len(responses) if responses is not None else 'None'} "
                    f"camera={camera_name} type={image_type}"
                )
            response = responses[0]
            _log_response(response, camera_name, image_type, machine_idx, scene_idx)
            return response

        def _getImages(airsim_client: airsim.VehicleClient, machine_idx=None, scene_idx=None):
            if airsim_client is None:
                raise Exception('client is None.')
                return None, None
            time_sleep_cnt = 0
            total_start = time.perf_counter()
            while True:
                try:
                    client_ip = getattr(airsim_client, '_traveluav_debug_ip', 'unknown')
                    client_port = getattr(airsim_client, '_traveluav_debug_port', getattr(airsim_client, 'port', getattr(airsim_client, '_port', 'unknown')))
                    client_timeout = getattr(airsim_client, '_traveluav_debug_timeout', getattr(airsim_client, 'timeout_value', 'unknown'))
                    client_scene = getattr(airsim_client, '_traveluav_debug_scene', 'unknown')
                    current_thread = threading.current_thread()
                    image_timing_log(
                        f"[DIAG][getImageResponsesForRecord] begin split-request mode machine={machine_idx} scene={scene_idx} "
                        f"map={client_scene} ip={client_ip} port={client_port} timeout={client_timeout}s cameras={cameras}"
                    )

                    setattr(current_thread, 'debug_stage', 'record_camera_info')
                    setattr(current_thread, 'debug_detail', ','.join(cameras))
                    camera_info_total_start = time.perf_counter()
                    for camera_name in cameras:
                        camera_info_start = time.perf_counter()
                        try:
                            camera_info = airsim_client.simGetCameraInfo(camera_name)
                            pose = getattr(camera_info, 'pose', None)
                            fov = getattr(camera_info, 'fov', None)
                            image_timing_log(
                                f"[DIAG][getImageResponsesForRecord] simGetCameraInfo OK camera={camera_name} "
                                f"machine={machine_idx} scene={scene_idx}: {time.perf_counter() - camera_info_start:.3f}s "
                                f"fov={fov} pose={pose}"
                            )
                        except Exception as camera_info_error:
                            logger.error(
                                f"[DIAG][getImageResponsesForRecord] simGetCameraInfo FAILED camera={camera_name} "
                                f"machine={machine_idx} scene={scene_idx}: {time.perf_counter() - camera_info_start:.3f}s "
                                f"error={camera_info_error}"
                            )
                    image_timing_log(
                        f"[TIMING][getImageResponsesForRecord] camera info all machine={machine_idx} scene={scene_idx}: "
                        f"{time.perf_counter() - camera_info_total_start:.3f}s"
                    )

                    images, depth_images = [], []
                    split_rpc_total_start = time.perf_counter()
                    for camera_name in cameras:
                        camera_total_start = time.perf_counter()
                        rgb_request = airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                        depth_request = airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)

                        rgb_resp = _get_single_response(
                            airsim_client, rgb_request, camera_name, 'Scene', machine_idx, scene_idx, current_thread
                        )
                        depth_resp = _get_single_response(
                            airsim_client, depth_request, camera_name, 'DepthPerspective', machine_idx, scene_idx, current_thread
                        )

                        setattr(current_thread, 'debug_stage', 'record_decode')
                        setattr(current_thread, 'debug_detail', f"camera={camera_name}")
                        camera_decode_start = time.perf_counter()
                        rgb_uint8_len = len(getattr(rgb_resp, 'image_data_uint8', []) or [])
                        depth_float_len = len(getattr(depth_resp, 'image_data_float', []) or [])
                        expected_rgb_len = getattr(rgb_resp, 'height', 0) * getattr(rgb_resp, 'width', 0) * 3
                        expected_depth_len = getattr(depth_resp, 'height', 0) * getattr(depth_resp, 'width', 0)
                        if rgb_uint8_len != expected_rgb_len:
                            logger.error(
                                f"[DIAG][getImageResponsesForRecord] RGB data length mismatch camera={camera_name} "
                                f"machine={machine_idx} scene={scene_idx}: got={rgb_uint8_len} expected={expected_rgb_len} "
                                f"shape=({getattr(rgb_resp, 'height', None)}, {getattr(rgb_resp, 'width', None)}, 3)"
                            )
                        if depth_float_len != expected_depth_len:
                            logger.error(
                                f"[DIAG][getImageResponsesForRecord] depth data length mismatch camera={camera_name} "
                                f"machine={machine_idx} scene={scene_idx}: got={depth_float_len} expected={expected_depth_len} "
                                f"shape=({getattr(depth_resp, 'height', None)}, {getattr(depth_resp, 'width', None)})"
                            )

                        rgb_decode_start = time.perf_counter()
                        image = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(rgb_resp.height, rgb_resp.width, 3)
                        image_timing_log(
                            f"[TIMING][getImageResponsesForRecord] decode rgb camera={camera_name} machine={machine_idx} "
                            f"scene={scene_idx}: {time.perf_counter() - rgb_decode_start:.3f}s"
                        )
                        depth_decode_start = time.perf_counter()
                        depth_img_in_meters = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)
                        depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
                        image_timing_log(
                            f"[TIMING][getImageResponsesForRecord] decode depth camera={camera_name} machine={machine_idx} "
                            f"scene={scene_idx}: {time.perf_counter() - depth_decode_start:.3f}s"
                        )
                        images.append(image)
                        depth_images.append(depth_image)
                        image_timing_log(
                            f"[TIMING][getImageResponsesForRecord] camera total camera={camera_name} machine={machine_idx} "
                            f"scene={scene_idx}: {time.perf_counter() - camera_total_start:.3f}s "
                            f"decode={time.perf_counter() - camera_decode_start:.3f}s rgb_shape={image.shape} depth_shape={depth_image.shape}"
                        )
                        logger.info(
                            f"[RECORD_RESOLUTION] camera={camera_name} rgb={rgb_resp.width}x{rgb_resp.height} "
                            f"depth={depth_resp.width}x{depth_resp.height}"
                        )

                    image_timing_log(
                        f"[TIMING][getImageResponsesForRecord] split rpc all machine={machine_idx} scene={scene_idx}: "
                        f"{time.perf_counter() - split_rpc_total_start:.3f}s cameras={cameras}"
                    )
                    setattr(current_thread, 'debug_stage', 'record_done')
                    setattr(current_thread, 'debug_detail', f"images={len(images)} depths={len(depth_images)}")
                    break
                except Exception as e:
                    setattr(threading.current_thread(), 'debug_stage', 'record_error_retry')
                    setattr(threading.current_thread(), 'debug_detail', str(e))
                    time_sleep_cnt += 1
                    logger.error(
                        f"[TIMING][getImageResponsesForRecord] split request ERROR machine={machine_idx} scene={scene_idx} "
                        f"elapsed={time.perf_counter() - total_start:.3f}s error={e}"
                    )
                    logger.error("图片获取错误: " + str(e))
                    logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                    time.sleep(1)
                if time_sleep_cnt > 10:
                    raise Exception('图片获取失败')
            image_timing_log(
                f"[TIMING][getImageResponsesForRecord] _getImages total machine={machine_idx} scene={scene_idx}: "
                f"{time.perf_counter() - total_start:.3f}s retries={time_sleep_cnt} mode=split"
            )
            return images, depth_images

        total_start = time.perf_counter()
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], index_1, index_2))
                )
        launch_start = time.perf_counter()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        image_timing_log(f"[TIMING][getImageResponsesForRecord] launch threads: {time.perf_counter() - launch_start:.3f}s")
        join_elapsed = self._join_threads_with_monitor(threads, 'getImageResponsesForRecord')
        image_timing_log(f"[TIMING][getImageResponsesForRecord] join threads: {join_elapsed:.3f}s")
        collect_start = time.perf_counter()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        image_timing_log(f"[TIMING][getImageResponsesForRecord] collect results: {time.perf_counter() - collect_start:.3f}s")
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponsesForRecord失败')
            return None

        image_timing_log(f"[TIMING][getImageResponsesForRecord] total: {time.perf_counter() - total_start:.3f}s mode=split")
        return responses

    def getSensorInfo(self, ):
        def get_sensor_info(airsim_client: airsim.VehicleClient, ):
            state_sensor = State(airsim_client, )
            imu_sensor = Imu(airsim_client)
            state_info = state_sensor.retrieve()
            imu_info = imu_sensor.retrieve()
            return {'sensors': {'state':state_info, 'imu': imu_info}}
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(get_sensor_info, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        results = []
        for index_1, _ in enumerate(threads):
            results.append([])
            for index_2, _ in enumerate(threads[index_1]):
                results[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getSensorInfo failed.')
            return None
        return results 