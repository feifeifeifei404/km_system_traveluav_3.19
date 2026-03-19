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
#  [新增模块] SUPER 集成通信与控制模块
# =========================================================================
from src.vlnce_src.super_ros2_client import get_super_ros2_client

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

def wait_for_arrival_in_airsim(env, target_pos, threshold=2.0, timeout=60.0, record_interval=0.5):
    """
    等待SUPER飞行到目标点（使用ROS2客户端）
    
    返回：
        success (bool): 是否成功到达
        trajectory (list): 飞行轨迹（ROS2版本返回空列表）
        collision_detected (bool): 是否检测到碰撞/卡住
    """
    super_client = get_super_ros2_client()
    
    # 调用ROS2客户端的wait_for_arrival（内部按 record_interval 记录轨迹）
    success, status, trajectory = super_client.wait_for_arrival(
        target_pos, threshold, timeout, check_interval=record_interval, record_interval=record_interval
    )

    collision_detected = (status == "STUCK")
    return success, trajectory, collision_detected


def wait_for_arrival_in_airsim_old(env, target_pos, threshold=2.0, timeout=60.0, record_interval=0.5):
    """
    旧版本（备份）
    """
    start_time = time.time()
    logger.info(f"[Wait] Waiting for SUPER to fly to {np.round(target_pos, 2)}...")
    
    temp_client = airsim.MultirotorClient(port=25001) 
    temp_client.confirmConnection()

    state = temp_client.getMultirotorState()
    print(f"DEBUG: Landed State: {state.landed_state}, Collision: {state.collision.has_collided}")

    trajectory = []
    last_record_time = time.time()
    collision_detected = False
    
    last_check_pos = None
    last_check_time = time.time()
    stuck_timeout = 15.0 
    
    while time.time() - start_time < timeout:
        try:
            # 使用临时 client 获取真实位置
            state = temp_client.getMultirotorState()
            pos = state.kinematics_estimated.position
            orient = state.kinematics_estimated.orientation
            curr_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
            
            # 记录轨迹点（按时间间隔采样）
            # 重要：使用与原makeActions相同的数据结构格式
            if time.time() - last_record_time >= record_interval:
                trajectory_point = {
                    'sensors': {
                        'state': {
                            'position': [pos.x_val, pos.y_val, pos.z_val],
                            'orientation': [orient.x_val, orient.y_val, orient.z_val, orient.w_val],
                            'linear_velocity': [
                                state.kinematics_estimated.linear_velocity.x_val,
                                state.kinematics_estimated.linear_velocity.y_val,
                                state.kinematics_estimated.linear_velocity.z_val
                            ],
                            'angular_velocity': [
                                state.kinematics_estimated.angular_velocity.x_val,
                                state.kinematics_estimated.angular_velocity.y_val,
                                state.kinematics_estimated.angular_velocity.z_val
                            ]
                        }
                    }
                }
                trajectory.append(trajectory_point)
                last_record_time = time.time()
            
            # 检测碰撞
            if state.collision.has_collided:
                collision_detected = True
                logger.warning(f"[Wait] Collision detected during flight!")
            
            # --- 僵死检测 ---
            if time.time() - last_check_time > stuck_timeout:
                if last_check_pos is not None:
                    movement = np.linalg.norm(curr_pos - last_check_pos)
                    if movement < 0.5: 
                        logger.error(f"[Wait] CRITICAL: Drone stuck! Moved {movement:.2f}m in {stuck_timeout}s")
                        return False, trajectory, True  # 视为碰撞/卡住 
                last_check_pos = curr_pos
                last_check_time = time.time()

            # --- 距离检测 ---
            dist = np.linalg.norm(curr_pos - np.array(target_pos))
            # print(f"Dist: {dist:.2f} | Cur: {curr_pos} | Tgt: {target_pos}", end='\r')
            
            if dist < threshold:
                logger.info(f"[Wait] Arrived! Final Dist: {dist:.2f}m")
                # 记录最终状态（使用与原makeActions相同的格式）
                final_point = {
                    'sensors': {
                        'state': {
                            'position': [pos.x_val, pos.y_val, pos.z_val],
                            'orientation': [orient.x_val, orient.y_val, orient.z_val, orient.w_val],
                            'linear_velocity': [
                                state.kinematics_estimated.linear_velocity.x_val,
                                state.kinematics_estimated.linear_velocity.y_val,
                                state.kinematics_estimated.linear_velocity.z_val
                            ],
                            'angular_velocity': [
                                state.kinematics_estimated.angular_velocity.x_val,
                                state.kinematics_estimated.angular_velocity.y_val,
                                state.kinematics_estimated.angular_velocity.z_val
                            ]
                        }
                    }
                }
                trajectory.append(final_point)
                time.sleep(1.0) 
                return True, trajectory, collision_detected
                
        except Exception as e:
            logger.warning(f"[Wait] Temp client failed: {e}")
            time.sleep(1.0)
            
        time.sleep(0.2) 
        
    logger.warning("[Wait] Timeout!")
    return False, trajectory, collision_detected

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

            for t in range(int(args.maxWaypoints) + 1):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))

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
                    assist_notices = batch_state.get_assist_notices()

                    # === 使用命令行传入的参数 ===
                    num_parallel_thoughts = args.num_parallel_thoughts
                    num_serial_refinements = args.num_refinement_steps

                    # --- 1. 生成初始候选 (并行思考) ---
                    logger.info(f"Step: {t}, Stage 1: Generating {num_parallel_thoughts} initial candidates via Dropout...")
                    print(f"\n{'='*20} Step [{t}]: Stage 1 - Parallel Thinking {'='*20}")

                    initial_inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(
                        batch_state.episodes, batch_state.target_positions, assist_notices,
                        refinement_step=0, intermediate_waypoint=None
                    )
                    
                    # === 记录点3: 记录模型输入 ===
                    if interceptor:
                        try:
                            input_data = interceptor.record_model_input(initial_inputs)
                            interceptor.add_step_data(input_data)
                        except Exception as e:
                            print(f"[WARNING] 模型输入记录失败: {e}")
                    
                    token_count_initial = initial_inputs['input_ids'].shape[1]
                    batch_state.tokens_per_step[0].append({"initial_candidates": token_count_initial})

                    model_wrapper.model.train()
                    initial_candidates = []
                    for i in range(num_parallel_thoughts):
                        _, intermediate_outputs = model_wrapper.run(
                            inputs=initial_inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets
                        )
                        if intermediate_outputs.get("waypoints_llm_new") is not None and len(intermediate_outputs.get("waypoints_llm_new")) > 0:
                            new_candidate = intermediate_outputs.get("waypoints_llm_new")[0]
                            initial_candidates.append(new_candidate)
                            
                            formatted_coords = np.round(new_candidate, 2)
                            print(f"    [Parallel Thought #{i+1}] Predicted Coords (World): {formatted_coords}")

                    model_wrapper.model.eval()

                    # --- 2. 深化改进候选 (串行思考) ---
                    final_candidates = []
                    if initial_candidates:
                        logger.info(f"Step: {t}, Stage 2: Generating refined candidates with {num_serial_refinements} refinement steps...")
                        print(f"\n{'*'*20} Step [{t}]: Stage 2 - Serial Refinement {'*'*20}")
                    
                        step_refinement_tokens_all_candidates = []

                        for initial_wp in initial_candidates:
                            current_wp = initial_wp
                            tokens_for_this_candidate = []
                            
                            for i in range(num_serial_refinements):
                                flat_current_wp = np.array(current_wp).flatten()
                                rethink_inputs, _, _, _ = model_wrapper.prepare_inputs(
                                    batch_state.episodes, batch_state.target_positions, assist_notices,
                                    refinement_step=i + 1, intermediate_waypoint=flat_current_wp
                                )
                                token_count_rethink = rethink_inputs['input_ids'].shape[1]
                                tokens_for_this_candidate.append(token_count_rethink)
                                
                                refined_wp_batch, _ = model_wrapper.run(
                                    inputs=rethink_inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets
                                )
                                
                                if refined_wp_batch is not None and len(refined_wp_batch) > 0:
                                    current_wp = refined_wp_batch[0]
                                    formatted_refined_coords = np.round(current_wp, 2)
                                    print(f"        -> Refinement Step #{i+1} New Coords: {formatted_refined_coords}")
                                else:
                                    print(f"        -> Refinement Step #{i+1} FAILED.")
                                    break
                            
                            step_refinement_tokens_all_candidates.append(tokens_for_this_candidate)
                            final_candidates.append(current_wp)

                        batch_state.tokens_per_step[0][-1]["refinement_steps"] = step_refinement_tokens_all_candidates
                    
                    # --- 3. 择优选取最终答案 ---
                    logger.info(f"Step: {t}, Stage 3: Scoring and selecting the best candidate...")
                    print(f"\n{'-'*20} Step [{t}]: Stage 3 - Final Selection {'-'*20}")
                
                    best_waypoint = None
                    if final_candidates:
                        best_waypoint = score_and_select_best_waypoint(
                            candidates=final_candidates,
                            current_episode=batch_state.episodes[0],
                            target_position=batch_state.target_positions[0]
                        )
                    elif initial_candidates:
                        logger.warning(f"Step: {t}, Refined candidates list is empty. Falling back to the first initial candidate.")
                        # 这里原本是调用小模型 run_traj_model，现在我们仍然让 LLM 决定点，交给 SUPER 去飞
                        # 所以我们只需要这个坐标点
                        best_waypoint = initial_candidates[0] # 直接使用初始点作为 fallback
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
                                'refined_waypoints': final_candidates,
                                'waypoints_world': final_refined_waypoints
                            }
                            output_data = interceptor.record_model_output(output_record)
                            interceptor.add_step_data(output_data)
                        except Exception as e:
                            print(f"[WARNING] 模型输出记录失败: {e}")

                else:
                    # 标准推理模式
                    inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions)
                    
                    if interceptor:
                        interceptor.add_step_data(interceptor.record_model_input(inputs))
                    
                    final_refined_waypoints, _ = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)
                    
                    if interceptor:
                        interceptor.add_step_data(interceptor.record_model_output({'waypoints_world': final_refined_waypoints}))
                    
                    if final_refined_waypoints:
                        formatted_coords = np.round(final_refined_waypoints[0], 2)
                        print(f"\n{'='*20} Step [{t}]: Standard Inference {'='*20}")
                        print(f"    Predicted Coords: {formatted_coords}")
                        print(f"{'='*60}\n")
                
                # ======================================================================================
                #  <<< 核心替换区域 START: 替换掉原本的小模型 refine 和 env.makeActions >>>
                # ======================================================================================
                
                # 0. 检查episode是否已经结束（与原makeActions保持一致）
                batch_idx = 0  # 当前只处理第一个batch
                if eval_env.sim_states[batch_idx].is_end:
                    logger.info(f"[Bridge] Episode already ended, skipping movement")
                    outputs = eval_env.get_obs()
                    # 直接跳到后续更新逻辑
                    batch_state.update_from_env_output(outputs)
                    batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                    batch_state.update_metric()
                    continue
                
                # 1. 获取最终决策的子目标点 (Sub-goal)
                if final_refined_waypoints and len(final_refined_waypoints) > 0:
                    # final_refined_waypoints 是一个 list，里面可能包含一个 trajectory array
                    trajectory = final_refined_waypoints[0] 
                    
                    # 检查 trajectory 是否是多点轨迹 (例如 shape (N, 3))
                    if hasattr(trajectory, 'shape') and len(trajectory.shape) > 1:
                        # ⚠️ 集成SUPER后：取第2个点作为近期子目标（第1个点太近，第2个点合适）
                        # 原因：TravelUAV生成的是完整轨迹，如果取最后一个点，目标太远太陡，SUPER无法规划
                        sub_goal = trajectory[min(4, len(trajectory)-1)]  # 取第2个点（或最后一个如果轨迹只有1个点） 
                    else:
                        # 如果本身就是单个点
                        sub_goal = trajectory

                    # 2. 将子目标点发送给 SUPER (Fast System)
                    print(f"[Bridge] Sending Goal to SUPER: {sub_goal}")
                    # debugpy.breakpoint()  # 断点4: 即将发送的 sub_goal
                    # 确保 sub_goal 是 [x, y, z] 格式
                    try:
                        # 使用Socket客户端发送目标点
                        super_client = get_super_ros2_client()
                        success = super_client.send_goal(sub_goal[0], sub_goal[1], sub_goal[2])
                        
                        if success:
                            # 3. 阻塞等待并获取轨迹
                            arrival_success, super_trajectory, collision_detected = wait_for_arrival_in_airsim(
                                eval_env, sub_goal, threshold=2.0, timeout=120.0
                            )
                            
                            # 检查SUPER是否成功到达
                            if not arrival_success:
                                logger.error(f"[SUPER] 未能到达目标点，超时或失败")
                                # 标记失败但继续（让TravelUAV判断是否结束）
                                eval_env.sim_states[batch_idx].is_collisioned = True
                            
                            # 4. 先更新 sim_states（必须在 get_obs 之前！）
                            #    原因：get_obs() 内部通过 multiprocessing 将 sim_states 序列化到子进程，
                            #    子进程会用 state.pose (即 trajectory[-1]) 计算 predict_start_index。
                            #    如果不先更新 trajectory，子进程拿到的是旧位置，计算结果会错误。
                            #    这与原始 makeActions 的顺序一致：先 makeActions 更新状态，再 get_obs。
                            
                            # 4.1 更新轨迹信息
                            if super_trajectory and len(super_trajectory) > 0:
                                eval_env.sim_states[batch_idx].trajectory.extend(super_trajectory)
                            else:
                                logger.warning(f"[Bridge] No trajectory returned from SUPER, recording current state only")
                                current_state = eval_env.sim_states[batch_idx].trajectory[-1] if eval_env.sim_states[batch_idx].trajectory else None
                                if current_state:
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
                            outputs = eval_env.get_obs()
                            
                        else:
                            print("[ERROR] Failed to send goal to SUPER. Skipping movement.")
                            # SUPER发送失败，标记为失败
                            eval_env.sim_states[batch_idx].is_end = True
                            outputs = eval_env.get_obs()
                            
                    except Exception as e:
                        print(f"[ERROR] SUPER integration error: {e}")
                        print(f"DEBUG: sub_goal type: {type(sub_goal)}, value: {sub_goal}")
                        # SUPER异常，标记为失败
                        eval_env.sim_states[batch_idx].is_end = True
                        outputs = eval_env.get_obs()
                else:
                    # 如果没有waypoints，只更新观测
                    logger.warning("[Bridge] No final_refined_waypoints, skipping movement")
                    outputs = eval_env.get_obs()

                # ======================================================================================
                #  <<< 核心替换区域 END >>>
                # ======================================================================================

                # 更新状态：观测 + done 预测 + 评估指标
                batch_state.update_from_env_output(outputs)
                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                batch_state.update_metric()
                
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
                        'success': batch_state.dones[0] if hasattr(batch_state, 'dones') and len(batch_state.dones) > 0 else False,
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
    eval(model_wrapper=model_wrapper,
         assist=assist,
         eval_env=eval_env,
         eval_save_dir=eval_save_path,
         interceptor=interceptor)

    eval_env.delete_VectorEnvUtil()