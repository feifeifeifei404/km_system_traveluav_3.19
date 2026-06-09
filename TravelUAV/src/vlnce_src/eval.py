import os
from pathlib import Path
import sys
import time
import json
import shutil
import random
import socket  # <--- 新增
import math    # <--- 新增

# import debugpy

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

import airsim # 确保导入


def _now_perf():
    return time.perf_counter()


def _duration_seconds(start_time):
    return round(time.perf_counter() - start_time, 6)


def _to_builtin_number(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _to_builtin_list(values):
    if values is None:
        return None
    if hasattr(values, 'tolist'):
        values = values.tolist()
    return [_to_builtin_number(v) for v in values]


def _get_eval_scene_dir_name():
    eval_save_path = getattr(args, 'eval_save_path', None)
    if not eval_save_path:
        return None
    return os.path.basename(os.path.normpath(eval_save_path)) or None


def _get_step_timing_dir(ori_data_dir):
    episode_id = os.path.basename(ori_data_dir.rstrip('/'))
    eval_scene_dir = _get_eval_scene_dir_name()
    if eval_scene_dir:
        return os.path.join('/mnt/data/TravelUAV/result/timing', eval_scene_dir, episode_id, 'step_timing')
    return os.path.join('/mnt/data/TravelUAV/result/timing', episode_id, 'step_timing')


def _get_step_timing_json_path(ori_data_dir, step_index):
    return os.path.join(_get_step_timing_dir(ori_data_dir), f'step{step_index + 1}.json')


def _write_step_timing_json(ori_data_dir, step_index, step_payload):
    timing_dir = _get_step_timing_dir(ori_data_dir)
    os.makedirs(timing_dir, exist_ok=True)
    timing_json_path = _get_step_timing_json_path(ori_data_dir, step_index)
    with open(timing_json_path, 'w', encoding='utf-8') as f:
        json.dump(step_payload, f, indent=4, ensure_ascii=False)
    return timing_json_path

sys.path.append(str(Path(str(os.getcwd())).resolve()))

# === 添加数据拦截器导入 ===
readapi_path = str(Path(__file__).resolve().parents[2] / 'readapi')
if readapi_path not in sys.path:
    sys.path.insert(0, readapi_path)

try:
    from data_interceptor import DataInterceptor
    HAS_INTERCEPTOR = True
    print("[INFO] ✓ 数据拦截器已成功导入")
except ImportError as e:
    HAS_INTERCEPTOR = False
    print(f"[WARNING] ✗ 数据拦截器导入失败: {e}")
    print("[WARNING] 将继续运行，但不会记录交互数据")

from utils.logger import logger
from utils.env_utils_uav import env_timing_log
from utils.utils import *
from src.model_wrapper.travel_llm import TravelModelWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized
from src.vlnce_src.scoring_util import score_and_select_best_waypoint
from src.model_wrapper.utils.travel_util import transform_to_world

# =========================================================================
#  [新增模块] SUPER 集成通信模块：仅负责给 SUPER2 发局部目标
# =========================================================================
from src.vlnce_src.super_ros2_client import (
    get_super_ros2_client,
    compute_super_goal_offset,
    raise_if_fast_system_fatal,
    has_fast_system_fatal_error,
    get_fast_system_fatal_reason,
)

# def wait_for_arrival_in_airsim(env, target_pos, threshold=2.0, timeout=60.0):
#     """
#     在 AirSim 中轮询，直到无人机接近目标点。
#     env: AirVLNENV 实例
#     target_pos: [x, y, z] 目标位置
#     """
#     start_time = time.time()
#     logger.info(f"[Wait] Waiting for SUPER to fly to {np.round(target_pos, 2)}...")
    
#     while time.time() - start_time < timeout:
#         # 获取当前位置 (直接调用底层 client 获取最快)
#         # 假设 env.uav.client 是 airsim.MultirotorClient
#         try:
#             # 注意：TravelUAV 的 env 封装层级较多，这里尝试获取真实位置
#             # 如果 env.uav.client 不可直接访问，可以使用 env.get_obs() 但那样效率低且会触发渲染
#             # 这里假设 env.uav.client 可用
#             state = env.uav.client.getMultirotorState()
#             pos = state.kinematics_estimated.position
#             curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
            
#             # 计算距离 (只计算 XY 平面距离，忽略高度微小差异，或者计算 3D 距离)
#             dist = np.linalg.norm(curr_pos - np.array(target_pos))
            
#             if dist < threshold:
#                 logger.info(f"[Wait] Arrived! Final Dist: {dist:.2f}m")
#                 # 到达后悬停一下，确保稳定
#                 time.sleep(1.0) 
#                 return True
                
#         except Exception as e:
#             # 如果无法获取 client，降级使用 time.sleep 估算
#             print(f"[WARNING] 无法获取实时位置 ({e})，使用硬延时...")
#             time.sleep(5.0) 
#             return True
            
#         time.sleep(0.5) # 降低轮询频率
        
#     logger.warning("[Wait] Timeout! SUPER might be stuck or path is too long.")
#     return False

def wait_for_arrival_in_airsim(env, target_pos, threshold=2.0, timeout=60.0, record_interval=0.1):
    """
    等待 SUPER 执行，但轨迹记录/碰撞判断/到达判断全部基于 AirSim 真值。

    到达判定：
        3D 距离 dist < threshold（默认 2.0m）

    返回：
        success (bool): 是否成功到达
        trajectory (list): 基于 AirSim 真值记录的轨迹
        collision_detected (bool): 是否检测到碰撞/卡住
        final_stable_state (dict|None): wait 结束后额外稳定采样得到的最终状态
    """

    def _connect_wait_client():
        super_client = get_super_ros2_client()
        selected_port = super_client.get_connected_airsim_port()
        if not selected_port:
            raise RuntimeError('Bridge has not published connected AirSim port yet')

        logger.info(f'[Wait] Using bridge-selected AirSim port {selected_port} for arrival monitoring')
        client = airsim.MultirotorClient(port=selected_port)
        client.confirmConnection()
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)
        dist = np.linalg.norm(curr_pos - np.array(target_pos, dtype=np.float64))
        logger.info(
            f'[Wait] Probe bridge-selected port={selected_port} current={np.round(curr_pos, 2)} dist_to_target={dist:.2f}m'
        )
        return client, selected_port

    def _build_trajectory_point(state):
        pos = state.kinematics_estimated.position
        orient = state.kinematics_estimated.orientation
        return {
            'sensors': {
                'state': {
                    'position': [pos.x_val, pos.y_val, pos.z_val],
                    'orientation': [orient.x_val, orient.y_val, orient.z_val, orient.w_val],
                    'linear_velocity': [
                        state.kinematics_estimated.linear_velocity.x_val,
                        state.kinematics_estimated.linear_velocity.y_val,
                        state.kinematics_estimated.linear_velocity.z_val,
                    ],
                    'angular_velocity': [
                        state.kinematics_estimated.angular_velocity.x_val,
                        state.kinematics_estimated.angular_velocity.y_val,
                        state.kinematics_estimated.angular_velocity.z_val,
                    ],
                    'collision': {
                        'has_collided': bool(state.collision.has_collided),
                        'object_name': str(state.collision.object_name),
                    },
                }
            }
        }

    def _sample_final_stable_state(client, selected_port, settle_seconds=1.0, sample_interval=0.1):
        deadline = time.time() + settle_seconds
        latest_point = None
        while time.time() < deadline:
            state = client.getMultirotorState()
            latest_point = _build_trajectory_point(state)
            time.sleep(sample_interval)

        if latest_point is not None:
            final_pos = latest_point['sensors']['state']['position']
            logger.info(
                f"[Wait] Final stabilized state on port {selected_port}: {np.round(final_pos, 2)}"
            )
        return latest_point

    start_time = time.time()
    logger.info(f"[Wait] Waiting for SUPER to fly to {np.round(target_pos, 2)}...")

    temp_client, selected_port = _connect_wait_client()

    trajectory = []
    last_record_time = 0.0
    collision_detected = False

    last_check_pos = None
    last_check_time = time.time()
    stuck_timeout = 15.0
    last_progress_log_time = 0.0

    while time.time() - start_time < timeout:
        raise_if_fast_system_fatal()
        try:
            state = temp_client.getMultirotorState()
            pos = state.kinematics_estimated.position
            curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)

            if time.time() - last_record_time >= record_interval:
                trajectory.append(_build_trajectory_point(state))
                last_record_time = time.time()

            if state.collision.has_collided:
                collision_detected = True
                logger.warning(
                    f'[Wait] Collision detected during flight on port {selected_port}! Abort immediately.'
                )
                collision_point = _build_trajectory_point(state)
                trajectory.append(collision_point)
                return False, trajectory, True, collision_point

            target_pos_np = np.array(target_pos, dtype=np.float64)
            diff = curr_pos - target_pos_np
            xy_dist = float(np.linalg.norm(diff[:2]))
            z_dist = float(abs(diff[2]))
            dist = float(np.linalg.norm(diff))
            if time.time() - last_progress_log_time >= 1.0:
                logger.info(
                    f'[Wait] port={selected_port} current={np.round(curr_pos, 2)} '
                    f'dist={dist:.2f}m xy_dist={xy_dist:.2f}m z_dist={z_dist:.2f}m '
                    f'collision={bool(state.collision.has_collided)}'
                )
                last_progress_log_time = time.time()

            if time.time() - last_check_time > stuck_timeout:
                if last_check_pos is not None:
                    movement = np.linalg.norm(curr_pos - last_check_pos)
                    if movement < 0.5:
                        logger.error(
                            f'[Wait] CRITICAL: Drone stuck on port {selected_port}! '
                            f'Moved {movement:.2f}m in {stuck_timeout}s, dist={dist:.2f}m'
                        )
                        final_stable_state = _sample_final_stable_state(temp_client, selected_port)
                        return False, trajectory, False, final_stable_state
                last_check_pos = curr_pos.copy()
                last_check_time = time.time()

            if dist < threshold:
                logger.info(f'[Wait] Arrived on port {selected_port}! Final Dist: {dist:.2f}m')
                trajectory.append(_build_trajectory_point(state))
                final_stable_state = _sample_final_stable_state(temp_client, selected_port)
                if final_stable_state is not None:
                    trajectory.append(final_stable_state)
                return True, trajectory, collision_detected, final_stable_state

        except Exception as e:
            logger.warning(f'[Wait] Temp client failed on port {selected_port}: {e}')
            time.sleep(1.0)

        time.sleep(0.2)

    final_stable_state = _sample_final_stable_state(temp_client, selected_port)
    if final_stable_state is not None:
        trajectory.append(final_stable_state)
    logger.warning(f'[Wait] Timeout on port {selected_port}!')
    return False, trajectory, collision_detected, final_stable_state


def wait_for_arrival_in_airsim(env, target_pos, threshold=2.0, timeout=60.0, record_interval=0.1):
    """
    等待 SUPER 执行，但轨迹记录/碰撞判断/到达判断全部基于 AirSim 真值。

    到达判定：
        3D 距离 dist < threshold（默认 2.0m）

    返回：
        success (bool): 是否成功到达
        trajectory (list): 基于 AirSim 真值记录的轨迹
        collision_detected (bool): 是否检测到碰撞/卡住
        final_stable_state (dict|None): wait 结束后额外稳定采样得到的最终状态
    """

    def _connect_wait_client():
        super_client = get_super_ros2_client()
        selected_port = super_client.get_connected_airsim_port()
        if not selected_port:
            raise RuntimeError('Bridge has not published connected AirSim port yet')

        logger.info(f'[Wait] Using bridge-selected AirSim port {selected_port} for arrival monitoring')
        client = airsim.MultirotorClient(port=selected_port)
        client.confirmConnection()
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)
        dist = np.linalg.norm(curr_pos - np.array(target_pos, dtype=np.float64))
        logger.info(
            f'[Wait] Probe bridge-selected port={selected_port} current={np.round(curr_pos, 2)} dist_to_target={dist:.2f}m'
        )
        return client, selected_port

    def _build_trajectory_point(state):
        pos = state.kinematics_estimated.position
        orient = state.kinematics_estimated.orientation
        return {
            'sensors': {
                'state': {
                    'position': [pos.x_val, pos.y_val, pos.z_val],
                    'orientation': [orient.x_val, orient.y_val, orient.z_val, orient.w_val],
                    'linear_velocity': [
                        state.kinematics_estimated.linear_velocity.x_val,
                        state.kinematics_estimated.linear_velocity.y_val,
                        state.kinematics_estimated.linear_velocity.z_val,
                    ],
                    'angular_velocity': [
                        state.kinematics_estimated.angular_velocity.x_val,
                        state.kinematics_estimated.angular_velocity.y_val,
                        state.kinematics_estimated.angular_velocity.z_val,
                    ],
                    'collision': {
                        'has_collided': bool(state.collision.has_collided),
                        'object_name': str(state.collision.object_name),
                    },
                }
            }
        }

    def _sample_final_stable_state(client, selected_port, settle_seconds=1.0, sample_interval=0.1):
        deadline = time.time() + settle_seconds
        latest_point = None
        while time.time() < deadline:
            state = client.getMultirotorState()
            latest_point = _build_trajectory_point(state)
            time.sleep(sample_interval)

        if latest_point is not None:
            final_pos = latest_point['sensors']['state']['position']
            logger.info(
                f"[Wait] Final stabilized state on port {selected_port}: {np.round(final_pos, 2)}"
            )
        return latest_point

    start_time = time.time()
    logger.info(f"[Wait] Waiting for SUPER to fly to {np.round(target_pos, 2)}...")

    temp_client, selected_port = _connect_wait_client()

    trajectory = []
    last_record_time = 0.0
    collision_detected = False

    last_check_pos = None
    last_check_time = time.time()
    stuck_timeout = 15.0
    last_progress_log_time = 0.0

    while time.time() - start_time < timeout:
        raise_if_fast_system_fatal()
        try:
            state = temp_client.getMultirotorState()
            pos = state.kinematics_estimated.position
            curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)

            if time.time() - last_record_time >= record_interval:
                trajectory.append(_build_trajectory_point(state))
                last_record_time = time.time()

            if state.collision.has_collided:
                collision_detected = True
                logger.warning(
                    f'[Wait] Collision detected during flight on port {selected_port}! Abort immediately.'
                )
                collision_point = _build_trajectory_point(state)
                trajectory.append(collision_point)
                return False, trajectory, True, collision_point

            target_pos_np = np.array(target_pos, dtype=np.float64)
            diff = curr_pos - target_pos_np
            xy_dist = float(np.linalg.norm(diff[:2]))
            z_dist = float(abs(diff[2]))
            dist = float(np.linalg.norm(diff))
            if time.time() - last_progress_log_time >= 1.0:
                logger.info(
                    f'[Wait] port={selected_port} current={np.round(curr_pos, 2)} '
                    f'dist={dist:.2f}m xy_dist={xy_dist:.2f}m z_dist={z_dist:.2f}m '
                    f'collision={bool(state.collision.has_collided)}'
                )
                last_progress_log_time = time.time()

            if time.time() - last_check_time > stuck_timeout:
                if last_check_pos is not None:
                    movement = np.linalg.norm(curr_pos - last_check_pos)
                    if movement < 0.5:
                        logger.error(
                            f'[Wait] CRITICAL: Drone stuck on port {selected_port}! '
                            f'Moved {movement:.2f}m in {stuck_timeout}s, dist={dist:.2f}m'
                        )
                        final_stable_state = _sample_final_stable_state(temp_client, selected_port)
                        return False, trajectory, False, final_stable_state
                last_check_pos = curr_pos.copy()
                last_check_time = time.time()

            if dist < threshold:
                logger.info(f'[Wait] Arrived on port {selected_port}! Final Dist: {dist:.2f}m')
                trajectory.append(_build_trajectory_point(state))
                final_stable_state = _sample_final_stable_state(temp_client, selected_port)
                if final_stable_state is not None:
                    trajectory.append(final_stable_state)
                return True, trajectory, collision_detected, final_stable_state

        except Exception as e:
            logger.warning(f'[Wait] Temp client failed on port {selected_port}: {e}')
            time.sleep(1.0)

        time.sleep(0.2)

    final_stable_state = _sample_final_stable_state(temp_client, selected_port)
    if final_stable_state is not None:
        trajectory.append(final_stable_state)
    logger.warning(f'[Wait] Timeout on port {selected_port}!')
    return False, trajectory, collision_detected, final_stable_state


def apply_gt_corridor_assist(local_goal, current_pos, gt_trajectory, logger=None):
    local_goal = np.array(local_goal, dtype=np.float64)

    has_current_pos = current_pos is not None
    has_gt_trajectory = gt_trajectory is not None and len(gt_trajectory) > 0
    if logger is not None:
        logger.info(
            '[GT Corridor Assist] '
            f'has_current_pos={has_current_pos} has_gt_trajectory={has_gt_trajectory} '
            f'gt_len={len(gt_trajectory) if gt_trajectory is not None else 0}'
        )

    if not has_current_pos:
        return local_goal
    current_pos = np.array(current_pos, dtype=np.float64)
    if not has_gt_trajectory:
        return local_goal

    gt_points = []
    for item in gt_trajectory:
        if isinstance(item, dict):
            pos = item.get('position')
            if pos is None and 'sensors' in item and 'state' in item['sensors']:
                pos = item['sensors']['state'].get('position')
        else:
            pos = item
        if pos is None or len(pos) < 3:
            continue
        gt_points.append(np.array(pos[:3], dtype=np.float64))

    if len(gt_points) < 2:
        return local_goal

    gt_points_np = np.stack(gt_points, axis=0)
    dists_to_current = np.linalg.norm(gt_points_np - current_pos[None, :], axis=1)
    nearest_idx = int(np.argmin(dists_to_current))

    forward_ref_idx = nearest_idx
    accum_dist = 0.0
    for idx in range(nearest_idx + 1, len(gt_points_np)):
        accum_dist += float(np.linalg.norm(gt_points_np[idx] - gt_points_np[idx - 1]))
        forward_ref_idx = idx
        if accum_dist >= 4.0:
            break

    gt_ref = gt_points_np[forward_ref_idx]
    xy_error = float(np.linalg.norm(local_goal[:2] - gt_ref[:2]))
    z_error = float(abs(local_goal[2] - gt_ref[2]))

    if not (xy_error > 2.5 or z_error > 1.2):
        return local_goal

    strong_correction = xy_error > 4.0 or z_error > 2.0
    model_weight = 0.5 if strong_correction else 0.8
    gt_weight = 1.0 - model_weight
    corrected_goal = model_weight * local_goal + gt_weight * gt_ref

    if logger is not None:
        logger.info(
            '[GT Corridor Assist] '
            f'nearest_idx={nearest_idx} ref_idx={forward_ref_idx} '
            f'xy_error={xy_error:.2f} z_error={z_error:.2f} '
            f'strong_correction={strong_correction} '
            f'blend={model_weight:.1f}*model+{gt_weight:.1f}*gt_ref '
            f'goal_before={np.round(local_goal, 2)} goal_after={np.round(corrected_goal, 2)}'
        )

    return corrected_goal

# =========================================================================

# ========= 小工具：递归找 key + 安全转型 (保持原样) =========
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def _find_in(obj, candidates):
    from collections.abc import Mapping, Sequence
    if isinstance(obj, Mapping):
        for k in candidates:
            if k in obj:
                return obj[k]
        for v in obj.values():
            r = _find_in(v, candidates)
            if r is not None:
                return r
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for v in obj:
            r = _find_in(v, candidates)
            if r is not None:
                return r
    return None

def _to_first_list(x):
    try:
        if x is None:
            return None
        x0 = x[0]
        if hasattr(x0, "tolist"):
            return x0.tolist()
        if isinstance(x0, (list, tuple)):
            return list(x0)
        return [float(x0)]
    except Exception:
        return None

def _to_first_float(x):
    try:
        import numpy as np, torch as _torch
        if x is None:
            return None
        if isinstance(x, (list, tuple, np.ndarray)):
            return float(x[0])
        if isinstance(x, _torch.Tensor):
            return float(x[0].item() if x.ndim > 0 else x.item())
        return float(x)
    except Exception:
        return None

def _to_vector(x):
    try:
        import numpy as np, torch as _torch
        if x is None:
            return None
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().reshape(-1).tolist()
        if isinstance(x, np.ndarray):
            return x.reshape(-1).tolist()
        if isinstance(x, (list, tuple)):
            if len(x) == 1 and isinstance(x[0], (list, tuple)):
                x = x[0]
            flat = []
            for v in x:
                if isinstance(v, (list, tuple)):
                    flat.extend(v)
                else:
                    flat.append(v)
            return [float(v) for v in flat]
        return [float(x)]
    except Exception:
        return None

def _slice_batch(x, i):
    try:
        import numpy as np, torch as _torch
        if x is None:
            return None
        if isinstance(x, _torch.Tensor):
            return x[i] if x.ndim >= 1 and x.shape[0] > i else x
        if isinstance(x, np.ndarray):
            return x[i] if x.ndim >= 1 and x.shape[0] > i else x
        if isinstance(x, (list, tuple)):
            return x[i] if len(x) > i else x
        return x
    except Exception:
        return x

def _shape_of(x):
    try:
        import numpy as np, torch as _torch
        if isinstance(x, _torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, np.ndarray):
            return tuple(x.shape)
        if isinstance(x, (list, tuple)):
            return (len(x),)
        return None
    except Exception:
        return None

def _extract_pos_dist(sample, i, bs_hint=None):
    import numpy as np
    import torch as _torch
    from collections.abc import Mapping, Sequence

    def _maybe_slice(v):
        try:
            if v is None or bs_hint is None:
                return v
            if isinstance(v, _torch.Tensor) and v.ndim >= 1 and v.shape[0] == bs_hint:
                return v[i]
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == bs_hint:
                return v[i]
            if isinstance(v, (list, tuple)) and len(v) == bs_hint:
                return v[i]
            return v
        except Exception:
            return v

    if isinstance(sample, Mapping):
        pos_val = None
        for k in ["positions","position","agent_positions","agent_position",
                  "curr_positions","xyz","pose","poses","state","states","loc","locs","coords","coord"]:
            if k in sample:
                pos_val = sample[k]
                break
        dist_val = None
        for k in ["remain_dists","distance","dist","dist_to_goals","goal_distance",
                  "remain_distance","remaining_distance"]:
            if k in sample:
                dist_val = sample[k]
                break
        pos_val = _maybe_slice(pos_val)
        dist_val = _maybe_slice(dist_val)
        return _to_vector(pos_val), _to_first_float(dist_val)

    if isinstance(sample, (list, tuple)):
        if len(sample) == 4:
            obs, r, d, info = sample
            for cand in (obs, info):
                pos, dist = _extract_pos_dist(cand, i, bs_hint)
                if pos is not None or dist is not None:
                    return pos, dist
            for cand in sample:
                pos, dist = _extract_pos_dist(cand, i, bs_hint)
                if pos is not None or dist is not None:
                    return pos, dist

        if any(isinstance(e, Mapping) for e in sample):
            for e in sample:
                pos, dist = _extract_pos_dist(e, i, bs_hint)
                if pos is not None or dist is not None:
                    return pos, dist

        if len(sample) > 0 and any(isinstance(sample[0], t) for t in (list, tuple, np.ndarray, _torch.Tensor)):
            elem = sample[i] if (bs_hint is not None and len(sample) == bs_hint and i < len(sample)) else sample[0]
            if isinstance(elem, (np.ndarray, _torch.Tensor)) and getattr(elem, "ndim", 0) == 2 and elem.shape[1] >= 3:
                return _to_vector(elem[0]), None
            return _to_vector(elem), None

        try:
            return _to_vector(sample), None
        except Exception:
            return None, None

    if isinstance(sample, (_torch.Tensor, np.ndarray)):
        arr = sample
        if bs_hint is not None and getattr(arr, "ndim", 0) >= 2 and arr.shape[0] == bs_hint:
            arr = arr[i]
        if getattr(arr, "ndim", 0) == 1:
            return _to_vector(arr), None
        if getattr(arr, "ndim", 0) == 2 and arr.shape[1] >= 3:
            return _to_vector(arr[0]), None
        return None, None

    try:
        return [float(sample)], None
    except Exception:
        return None, None

def format_for_json(data):
    if isinstance(data, dict):
        return {k: format_for_json(v) for k, v in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        return [format_for_json(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().float().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    else:
        return str(data)

def save_numpy_as_image(numpy_array, file_path):
    try:
        image_bgr = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image_bgr)
    except Exception as e:
        print(f"Error saving numpy array as image to {file_path}: {e}")

# 循环地执行模型推理 → 环境交互 → 结果记录
def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir, interceptor=None):
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach (port 5678)...")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    # debugpy.listen(("127.0.0.1", 5678))
    # print("Waiting for debugger attach (127.0.0.1:5678)...")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()

    model_wrapper.eval() 

    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        episode_idx = 0 
        
        while True: 
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            raise_if_fast_system_fatal()
            
            # === 记录点1: Episode开始 ===
            if interceptor and env_batchs:
                try:
                    interceptor.start_episode({
                        'map_name': env_batchs[0].get('map_name', 'unknown'),
                        'seq_name': env_batchs[0].get('seq_name', 'unknown'),
                        'instruction': env_batchs[0].get('instruction', '')
                    })
                except Exception as e:
                    print(f"[WARNING] Episode开始记录失败: {e}")
            
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)
            pbar.update(n=eval_env.batch_size)
            episode_timing_path = None

            super_client = get_super_ros2_client()
            super_client.set_bridge_execution(False)
            logger.info("[Bridge Execution] disabled at episode reset")
            time.sleep(1.2)
            super_client.reset_fsm(timeout=15.0)
            time.sleep(0.8)
            logger.info("[SUPER] episode reset completed")

            for t in range(int(args.maxWaypoints) + 1):
                raise_if_fast_system_fatal()
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))
                step_timing = {
                    'step_index': t,
                    'started_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'durations': {},
                    'values': {},
                    'status': 'running'
                }
                total_step_start = _now_perf()

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
                
                # === 记录点2: 获取当前观测数据 ===
                if interceptor and batch_state.episodes:
                    try:
                        current_obs = batch_state.episodes[0][-1] if len(batch_state.episodes[0]) > 0 else {}
                        obs_record = {
                            'sensors': current_obs.get('sensors', {}),
                            'rgb': current_obs.get('rgb', []), 
                            'depth': current_obs.get('depth', []), 
                            'instruction': current_obs.get('instruction', ''),
                            'object_position': current_obs.get('object_position', None)
                        }
                        obs_data = interceptor.record_observation(obs_record)
                        interceptor.add_step_data(obs_data)
                    except Exception as e:
                        print(f"[WARNING] 观测数据记录失败: {e}")


                # =================== BUDGET FORCING 逻辑 ===================
                
                final_refined_waypoints = [] # 初始化

                if args.use_budget_forcing:
                    # 1. 获取当前状态的助理提示
                    assist_start = time.perf_counter()
                    assist_notices = batch_state.get_assist_notices()
                    logger.info(f"[TIMING][Step {t}] batch_state.get_assist_notices: {time.perf_counter() - assist_start:.3f}s")

                    # === 使用命令行传入的参数 ===
                    num_parallel_thoughts = args.num_parallel_thoughts

                    # --- 1. 生成初始候选 (并行思考) ---
                    logger.info(f"Step: {t}, Stage 1: Generating {num_parallel_thoughts} initial candidates via Dropout...")
                    print(f"\n{'='*20} Step [{t}]: Stage 1 - Parallel Thinking {'='*20}")

                    prepare_start = time.perf_counter()
                    initial_inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(
                        batch_state.episodes, batch_state.target_positions, assist_notices,
                        refinement_step=0, intermediate_waypoint=None
                    )
                    logger.info(f"[TIMING][Step {t}] model_wrapper.prepare_inputs(initial): {time.perf_counter() - prepare_start:.3f}s")
                    
                    # === 记录点3: 记录模型输入 ===
                    if interceptor:
                        try:
                            input_data = interceptor.record_model_input(initial_inputs)
                            interceptor.add_step_data(input_data)
                        except Exception as e:
                            print(f"[WARNING] 模型输入记录失败: {e}")
                    
                    token_count_initial = initial_inputs['input_ids'].shape[1]
                    batch_state.tokens_per_step[0].append({"initial_candidates": token_count_initial})
                    batch_state.llm_calls_per_step[0].append({"initial_candidates": int(num_parallel_thoughts)})

                    model_wrapper.model.train()
                    initial_candidates = []
                    for i in range(num_parallel_thoughts):
                        run_start = time.perf_counter()
                        _, intermediate_outputs = model_wrapper.run(
                            inputs=initial_inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets
                        )
                        logger.info(f"[TIMING][Step {t}] model_wrapper.run(parallel #{i+1}): {time.perf_counter() - run_start:.3f}s")
                        if intermediate_outputs.get("waypoints_llm_new") is not None and len(intermediate_outputs.get("waypoints_llm_new")) > 0:
                            new_candidate = intermediate_outputs.get("waypoints_llm_new")[0]
                            initial_candidates.append(new_candidate)
                            
                            formatted_coords = np.round(new_candidate, 2)
                            print(f"    [Parallel Thought #{i+1}] Predicted Coords (World): {formatted_coords}")

                    model_wrapper.model.eval()

                    # --- 2. 直接对并行候选择优，不再做串行 refinement ---
                    logger.info(f"Step: {t}, Stage 2: Scoring and selecting the best parallel candidate...")
                    print(f"\n{'-'*20} Step [{t}]: Stage 2 - Final Selection {'-'*20}")

                    best_waypoint = None
                    if initial_candidates:
                        best_waypoint = score_and_select_best_waypoint(
                            candidates=initial_candidates,
                            current_episode=batch_state.episodes[0],
                            target_position=batch_state.target_positions[0]
                        )
                    else:
                        logger.error(f"Step: {t}, All candidate generation failed. Terminating episode.")
                        batch_state.dones[0] = True
                        continue

                    formatted_best_coords = np.round(best_waypoint, 2)
                    print(f"    Final Selected Coords: {formatted_best_coords}")
                    print(f"{'='*60}\n")
                        
                    final_refined_waypoints = [best_waypoint]
                    
                    # === 记录点4: 记录模型输出 ===
                    if interceptor:
                        try:
                            output_record = {
                                'waypoints_llm_new': initial_candidates,
                                'refined_waypoints': [],
                                'waypoints_world': final_refined_waypoints
                            }
                            output_data = interceptor.record_model_output(output_record)
                            interceptor.add_step_data(output_data)
                        except Exception as e:
                            print(f"[WARNING] 模型输出记录失败: {e}")

                else:
                    # 标准推理模式
                    prepare_start = _now_perf()
                    inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions)
                    step_timing['durations']['prepare_inputs'] = _duration_seconds(prepare_start)
                    env_timing_log(f"[TIMING][Step {t}] model_wrapper.prepare_inputs(standard): {step_timing['durations']['prepare_inputs']:.3f}s")
                    
                    # 已禁用：慢系统输入调试日志落盘
                    
                    if interceptor:
                        interceptor.add_step_data(interceptor.record_model_input(inputs))

                    token_count = inputs['input_ids'].shape[1]
                    batch_state.tokens_per_step[0].append({"initial_candidates": int(token_count)})
                    batch_state.llm_calls_per_step[0].append({"initial_candidates": 1})
                    
                    run_start = _now_perf()
                    final_refined_waypoints, _ = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)
                    step_timing['durations']['model_run'] = _duration_seconds(run_start)
                    env_timing_log(f"[TIMING][Step {t}] model_wrapper.run(standard): {step_timing['durations']['model_run']:.3f}s")
                    
                    if interceptor:
                        interceptor.add_step_data(interceptor.record_model_output({'waypoints_world': final_refined_waypoints}))
                    
                    if final_refined_waypoints is not None and len(final_refined_waypoints) > 0:
                        formatted_coords = np.round(final_refined_waypoints[0], 2)
                        print(f"\n{'='*20} Step [{t}]: Standard Inference {'='*20}")
                        print(f"    Predicted Coords: {formatted_coords}")
                        print(f"{'='*60}\n")
                
                # ======================================================================================
                #  <<< 核心替换区域 START: 替换掉原本的小模型 refine 和 env.makeActions >>>
                # ======================================================================================
                
                # 0. 检查episode是否已经结束（与原makeActions保持一致）
                batch_idx = 0  # 当前只处理第一个batch
                safety_super_client = get_super_ros2_client()
                safety_super_client.set_bridge_execution(False)
                logger.info("[Bridge Execution] disabled at slow-system decision boundary")
                if eval_env.sim_states[batch_idx].is_end:
                    logger.info(f"[Bridge] Episode already ended, skipping movement")
                    step_timing['values']['skip_reason'] = 'episode_already_ended'
                    safety_super_client.set_bridge_execution(False)
                    eval_env.pause_sim()
                    logger.info("[AirSim Sync] AirSim paused for already-ended get_obs")
                    get_obs_start = _now_perf()
                    outputs = eval_env.get_obs()
                    step_timing['durations']['get_obs'] = _duration_seconds(get_obs_start)
                    logger.info(f"[TIMING][Step {t}] eval_env.get_obs(already ended): {step_timing['durations']['get_obs']:.3f}s")
                    # 直接跳到后续更新逻辑
                
                # 1. 获取最终决策的子目标点 (Sub-goal)
                if final_refined_waypoints is not None and len(final_refined_waypoints) > 0:
                    # 直接把大模型选出的最优点发给快系统，不再从长轨迹中取中间点
                    raw_sub_goal = np.array(final_refined_waypoints[0], dtype=np.float64)
                    sub_goal = raw_sub_goal.copy()
                    step_timing['values']['raw_model_goal'] = _to_builtin_list(raw_sub_goal)

                    # GT 轨迹走廊辅助：仅在发送给 SUPER 之前，对明显偏离走廊的局部目标做轻微拉回
                    current_episode = batch_state.episodes[batch_idx] if batch_state.episodes and len(batch_state.episodes) > batch_idx else None
                    current_frame = current_episode[-1] if current_episode and len(current_episode) > 0 else None
                    current_pos = None
                    if current_frame and 'sensors' in current_frame and 'state' in current_frame['sensors']:
                        current_pos = current_frame['sensors']['state'].get('position')
                    gt_trajectory = eval_env.batch[batch_idx].get('trajectory', None)
                    if current_pos is not None and gt_trajectory is not None:
                        sub_goal = apply_gt_corridor_assist(
                            local_goal=sub_goal,
                            current_pos=current_pos,
                            gt_trajectory=gt_trajectory,
                            logger=logger,
                        )

                    # 2. 将子目标点发送给 SUPER (Fast System)
                    if current_pos is not None:
                        current_pos_np = np.array(current_pos, dtype=np.float64)
                        xy_dist = float(np.linalg.norm((sub_goal - current_pos_np)[:2]))
                        z_delta = float(sub_goal[2] - current_pos_np[2])
                        step_timing['values']['current_position_before_send'] = _to_builtin_list(current_pos_np)
                        step_timing['values']['xy_dist'] = round(xy_dist, 6)
                        step_timing['values']['z_delta'] = round(z_delta, 6)
                        if abs(z_delta) > 5.0:
                            original_z = float(sub_goal[2])
                            sub_goal[2] = float(current_pos_np[2] + np.sign(z_delta) * 5.0)
                            z_delta = float(sub_goal[2] - current_pos_np[2])
                            step_timing['values']['z_delta_after_clamp'] = round(z_delta, 6)
                            logger.info(
                                f"[Bridge] Clamp sub-goal Z before SUPER: current_z={current_pos_np[2]:.2f} "
                                f"target_z_before={original_z:.2f} target_z_after={sub_goal[2]:.2f}"
                            )
                    step_timing['values']['final_sub_goal'] = _to_builtin_list(sub_goal)
                    print(f"[Bridge] Sending Goal to SUPER: {sub_goal}")
                    # debugpy.breakpoint()  # 断点4: 即将发送的 sub_goal
                    # 确保 sub_goal 是 [x, y, z] 格式
                    try:
                        # 使用Socket客户端发送目标点
                        super_client = get_super_ros2_client()
                        goal_offset = compute_super_goal_offset(eval_env.sim_states[batch_idx], super_client)
                        if goal_offset is not None:
                            super_client.set_goal_offset(goal_offset)
                            step_timing['values']['goal_offset'] = _to_builtin_list(goal_offset)
                        else:
                            super_client.clear_goal_offset()
                            logger.warning("[Bridge] Failed to compute SUPER goal offset, fallback to raw frame conversion")
                        eval_env.resume_sim()
                        logger.info("[AirSim Sync] AirSim resumed before enabling Bridge execution")
                        super_client.set_bridge_execution(True)
                        logger.info("[Bridge Execution] enabled before sending SUPER goal")
                        send_goal_start = _now_perf()
                        success = super_client.send_goal(sub_goal[0], sub_goal[1], sub_goal[2])
                        step_timing['durations']['send_goal'] = _duration_seconds(send_goal_start)
                        
                        if success:
                            fsm_present_check_start = _now_perf()
                            fast_system_alive_after_send = super_client.is_fast_system_alive()
                            step_timing['durations']['fsm_node_present_after_send_goal_check'] = _duration_seconds(fsm_present_check_start)
                            step_timing['values']['fsm_node_present_after_send_goal'] = bool(fast_system_alive_after_send)
                            # 3. 阻塞等待并获取轨迹
                            wait_start = _now_perf()
                            arrival_success, super_trajectory, collision_detected, final_stable_state = wait_for_arrival_in_airsim(
                                eval_env, sub_goal, threshold=2.0, timeout=15.0
                            )
                            step_timing['durations']['wait_for_arrival'] = _duration_seconds(wait_start)
                            env_timing_log(f"[TIMING][Step {t}] wait_for_arrival_in_airsim: {step_timing['durations']['wait_for_arrival']:.3f}s")
                            super_client.set_bridge_execution(False)
                            logger.info("[Bridge Execution] disabled after SUPER arrival/failure before AirSim pause")
                            eval_env.pause_sim()
                            logger.info("[AirSim Sync] AirSim paused after disabling Bridge execution before get_obs")

                            # 快系统死亡检测：子目标超时但 fsm_node 仍存活 -> 轻微失败；否则致命失败直接退出整个 eval
                            if not arrival_success:
                                fast_system_alive = False
                                try:
                                    fsm_present_check_start = _now_perf()
                                    fast_system_alive = super_client.is_fast_system_alive()
                                    step_timing['durations']['fsm_node_present_after_wait_check'] = _duration_seconds(fsm_present_check_start)
                                    step_timing['values']['fsm_node_present_after_wait'] = bool(fast_system_alive)
                                except Exception as alive_error:
                                    logger.error(f"[SUPER] fast system alive check failed: {alive_error}")
                                if fast_system_alive:
                                    logger.error("[SUPER] 子目标超时，但 fsm_node 仍存活；按轻微失败处理，当前 episode 结束")
                                    eval_env.sim_states[batch_idx].is_end = True
                                else:
                                    raise RuntimeError("[FATAL][SUPER] 子目标超时且 fsm_node 已崩溃/失联，退出整个 eval")
                            
                            step_timing['values']['arrival_success'] = bool(arrival_success)
                            step_timing['values']['collision_detected'] = bool(collision_detected)
                            step_timing['values']['super_trajectory_points'] = len(super_trajectory) if super_trajectory else 0
                            if final_stable_state is not None:
                                step_timing['values']['final_stable_position'] = _to_builtin_list(final_stable_state['sensors']['state']['position'])

                            # 4. 先更新 sim_states（必须在 get_obs 之前！）
                            #    原因：get_obs() 内部通过 multiprocessing 将 sim_states 序列化到子进程，
                            #    子进程会用 state.pose (即 trajectory[-1]) 计算 predict_start_index。
                            #    如果不先更新 trajectory，子进程拿到的是旧位置，计算结果会错误。
                            #    这与原始 makeActions 的顺序一致：先 makeActions 更新状态，再 get_obs。
                            
                            # 4.1 更新轨迹信息
                            if super_trajectory and len(super_trajectory) > 0:
                                if final_stable_state is not None:
                                    super_trajectory[-1] = final_stable_state
                                eval_env.sim_states[batch_idx].trajectory.extend(super_trajectory)
                            else:
                                logger.warning(f"[Bridge] No trajectory returned from SUPER, recording current state only")
                                current_state = eval_env.sim_states[batch_idx].trajectory[-1] if eval_env.sim_states[batch_idx].trajectory else None
                                if final_stable_state is not None:
                                    eval_env.sim_states[batch_idx].trajectory.append(final_stable_state)
                                elif current_state:
                                    eval_env.sim_states[batch_idx].trajectory.append(current_state)
                            
                            # 4.2 更新步数
                            eval_env.sim_states[batch_idx].step += 1
                            
                            # 4.3 更新碰撞状态
                            eval_env.sim_states[batch_idx].is_collisioned = collision_detected
                            
                            # 4.4 更新 pre_waypoints (记录本次发送给SUPER的目标点)
                            if hasattr(sub_goal, 'tolist'):
                                waypoint_list = [sub_goal.tolist()]
                            else:
                                waypoint_list = [list(sub_goal)]
                            eval_env.sim_states[batch_idx].pre_waypoints = waypoint_list
                            
                            # 4.5 检查是否到达目标（成功条件）
                            target_position = eval_env.batch[batch_idx]['object_position']
                            if final_stable_state is not None:
                                current_position = final_stable_state['sensors']['state']['position']
                            else:
                                current_position = eval_env.sim_states[batch_idx].pose[0:3]
                            dist_to_target = np.linalg.norm(np.array(current_position) - np.array(target_position))
                            
                            if dist_to_target < eval_env.sim_states[batch_idx].SUCCESS_DISTANCE:
                                eval_env.sim_states[batch_idx].oracle_success = True
                                
                                logger.info(f"[Bridge] SUCCESS! Reached target at distance {dist_to_target:.2f}m")
                            
                            # 4.6 检查是否超过最大步数（终止条件）
                            if eval_env.sim_states[batch_idx].step >= int(args.maxWaypoints):
                                eval_env.sim_states[batch_idx].is_end = True
                                logger.info(f"[Bridge] Reached max waypoints ({args.maxWaypoints}), ending episode")
                            
                            # 4.7 如果碰撞或到达失败，也可以标记为终止
                            if not arrival_success:
                                logger.warning(f"[Bridge] Failed to reach sub-goal, may need to end episode")
                                # 可选：设置 is_end = True，根据你的策略决定
                                # eval_env.sim_states[batch_idx].is_end = True
                            
                            # 4.8 更新距离测量（与原makeActions保持一致）
                            eval_env.update_measurements()
                            
                            # 5. 最后才获取观测（此时 sim_states 已更新完毕）
                            #    get_obs 会把更新后的 sim_states 发给子进程，
                            #    子进程基于正确的新位置计算 predict_start_index 和 teacher_action
                            get_obs_start = _now_perf()
                            outputs = eval_env.get_obs()
                            step_timing['durations']['get_obs'] = _duration_seconds(get_obs_start)
                            logger.info(f"[TIMING][Step {t}] eval_env.get_obs(after SUPER): {step_timing['durations']['get_obs']:.3f}s")
                            
                        else:
                            print("[ERROR] Failed to send goal to SUPER. Skipping movement.")
                            super_client.set_bridge_execution(False)
                            logger.info("[Bridge Execution] disabled after SUPER send_goal failure")
                            eval_env.pause_sim()
                            logger.info("[AirSim Sync] AirSim paused after SUPER send_goal failure before get_obs")
                            step_timing['values']['arrival_success'] = False
                            fast_system_alive = False
                            try:
                                fsm_present_check_start = _now_perf()
                                fast_system_alive = super_client.is_fast_system_alive()
                                step_timing['durations']['fsm_node_present_after_send_goal_check'] = _duration_seconds(fsm_present_check_start)
                                step_timing['values']['fsm_node_present_after_send_goal'] = bool(fast_system_alive)
                            except Exception as alive_error:
                                logger.error(f"[SUPER] fast system alive check failed after send_goal failure: {alive_error}")
                            if not fast_system_alive:
                                raise RuntimeError("[FATAL][SUPER] send_goal 失败且 fsm_node 已崩溃/失联，退出整个 eval")
                            # SUPER发送失败，但快系统仍活着：仅当前 episode 失败
                            eval_env.sim_states[batch_idx].is_end = True
                            get_obs_start = _now_perf()
                            outputs = eval_env.get_obs()
                            step_timing['durations']['get_obs'] = _duration_seconds(get_obs_start)
                            logger.info(f"[TIMING][Step {t}] eval_env.get_obs(send goal failed): {step_timing['durations']['get_obs']:.3f}s")
                            
                    except Exception as e:
                        step_timing['values']['exception'] = str(e)
                        fatal_fast_system_failure = isinstance(e, RuntimeError) and ("[FATAL][SUPER]" in str(e) or "AirSim RPC timeout" in str(e))
                        if not fatal_fast_system_failure:
                            try:
                                if not super_client.is_fast_system_alive():
                                    fatal_fast_system_failure = True
                                    e = RuntimeError(f"[FATAL][SUPER] fast system dead during eval exception: {e}")
                            except Exception as alive_error:
                                logger.error(f"[SUPER] fast system alive check failed during exception handling: {alive_error}")
                                fatal_fast_system_failure = True
                                e = RuntimeError(f"[FATAL][SUPER] fast system alive check failed during exception handling: {alive_error}; original_error={e}")
                        try:
                            super_client.set_bridge_execution(False)
                            logger.info("[Bridge Execution] disabled after SUPER exception")
                        except Exception as disable_error:
                            logger.warning(f"[Bridge Execution] failed to disable after SUPER exception: {disable_error}")
                        eval_env.pause_sim()
                        logger.info("[AirSim Sync] AirSim paused after SUPER exception before get_obs")
                        print(f"[ERROR] SUPER integration error: {e}")
                        print(f"DEBUG: sub_goal type: {type(sub_goal)}, value: {sub_goal}")
                        if fatal_fast_system_failure:
                            raise
                        # SUPER异常，标记为失败
                        eval_env.sim_states[batch_idx].is_end = True
                        get_obs_start = _now_perf()
                        outputs = eval_env.get_obs()
                        step_timing['durations']['get_obs'] = _duration_seconds(get_obs_start)
                        logger.info(f"[TIMING][Step {t}] eval_env.get_obs(SUPER exception): {step_timing['durations']['get_obs']:.3f}s")
                else:
                    # 如果没有waypoints，只更新观测
                    logger.warning("[Bridge] No final_refined_waypoints, skipping movement")
                    step_timing['values']['skip_reason'] = 'no_final_refined_waypoints'
                    safety_super_client.set_bridge_execution(False)
                    eval_env.pause_sim()
                    logger.info("[AirSim Sync] AirSim paused for no-waypoint get_obs")
                    get_obs_start = _now_perf()
                    outputs = eval_env.get_obs()
                    step_timing['durations']['get_obs'] = _duration_seconds(get_obs_start)
                    logger.info(f"[TIMING][Step {t}] eval_env.get_obs(no waypoint): {step_timing['durations']['get_obs']:.3f}s")

                # ======================================================================================
                #  <<< 核心替换区域 END >>>
                # ======================================================================================

                # 更新状态：观测 + done 预测 + 评估指标
                update_start = _now_perf()
                batch_state.update_from_env_output(outputs)
                step_timing['durations']['update_from_env_output'] = _duration_seconds(update_start)
                env_timing_log(f"[TIMING][Step {t}] batch_state.update_from_env_output: {step_timing['durations']['update_from_env_output']:.3f}s")
                predict_done_start = time.perf_counter()
                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                env_timing_log(f"[TIMING][Step {t}] model_wrapper.predict_done: {time.perf_counter() - predict_done_start:.3f}s")
                metric_start = time.perf_counter()
                batch_state.update_metric()
                env_timing_log(f"[TIMING][Step {t}] batch_state.update_metric: {time.perf_counter() - metric_start:.3f}s")
                step_timing['durations']['total_step_time'] = _duration_seconds(total_step_start)
                step_timing['status'] = 'completed'
                batch_state.step_timings[batch_idx].append(step_timing)
                _write_step_timing_json(batch_state.ori_data_dirs[batch_idx], t, step_timing)
                
                # === 记录点: 结束步骤 ===
                if interceptor:
                    try:
                        interceptor.end_step()
                    except Exception as e:
                        print(f"[WARNING] 步骤结束记录失败: {e}")
            
            # === 记录点6: Episode结束 ===
            if interceptor:
                try:
                    final_metrics = batch_state.get_metrics() if hasattr(batch_state, 'get_metrics') else {}
                    episode_result = {
                        'success': bool(eval_env.sim_states[0].oracle_success) if hasattr(eval_env, 'sim_states') and len(eval_env.sim_states) > 0 else False,
                        'distance_to_goal': batch_state.remain_dists[0] if hasattr(batch_state, 'remain_dists') and len(batch_state.remain_dists) > 0 else None,
                        'metrics': final_metrics
                    }
                    interceptor.end_episode(episode_result)
                except Exception as e:
                    print(f"[WARNING] Episode结束记录失败: {e}")
            
            episode_idx += 1

        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)

    interceptor = None
    record_data = getattr(args, 'record_data', True) # 默认为 True 或者从 args 读取
    if HAS_INTERCEPTOR and record_data:
        try:
            record_dir = getattr(args, 'record_dir', './debug_data')
            interceptor = DataInterceptor(output_dir=record_dir)
            print(f"[INFO] ✓ 数据拦截器已启用，数据将保存到: {record_dir}")
        except Exception as e:
            print(f"[ERROR] 数据拦截器初始化失败: {e}")
            interceptor = None

    setup()
    assert CheckPort(), 'error port'

    print("***************************************************")
    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)
    print("***************************************************")
    try:
        eval(model_wrapper=model_wrapper,
             assist=assist,
             eval_env=eval_env,
             eval_save_dir=eval_save_path,
             interceptor=interceptor)
    finally:
        try:
            shutdown_super_client = get_super_ros2_client()
            shutdown_super_client.set_bridge_execution(False)
            print("[Bridge Execution] disabled during eval shutdown")
        except Exception as e:
            print(f"[WARNING] eval shutdown disable Bridge execution failed: {e}")
        try:
            eval_env.pause_sim()
            print("[AirSim Sync] AirSim paused during eval shutdown")
        except Exception as e:
            print(f"[WARNING] eval shutdown pause AirSim failed: {e}")
        eval_env.delete_VectorEnvUtil()