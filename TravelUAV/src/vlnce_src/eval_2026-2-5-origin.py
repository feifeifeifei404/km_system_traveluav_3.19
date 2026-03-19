import os
from pathlib import Path
import sys
import time
import json
import shutil
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

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
from src.model_wrapper.utils.travel_util import transform_to_world  # <--- 添加这一行


# ========= 小工具：递归找 key + 安全转型 =========
def set_seed(seed: int = 42):
    """
    固定所有相关的随机数种子，确保实验的可复现性。
    """
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
    """在嵌套 dict/list 里递归找第一个匹配键"""
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
    """batch 常见是 [N,3] / list of list；只取第 0 个元素为 list"""
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
    """把任意 torch/np/list 位置向量转成一维 float list，保留全部维度"""
    try:
        import numpy as np, torch as _torch
        if x is None:
            return None
        # torch 张量
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().reshape(-1).tolist()
        # numpy
        if isinstance(x, np.ndarray):
            return x.reshape(-1).tolist()
        # list/tuple：可能是 [[x,y,z]]、[x,y,z]、[[x],[y],[z]] 等
        if isinstance(x, (list, tuple)):
            # 如果是 list of list 并且长度为1，解一层
            if len(x) == 1 and isinstance(x[0], (list, tuple)):
                x = x[0]
            # 如果是列向量式 [[x],[y],[z]]，拍扁
            flat = []
            for v in x:
                if isinstance(v, (list, tuple)):
                    flat.extend(v)
                else:
                    flat.append(v)
            # 统一转 float
            return [float(v) for v in flat]
        # 标量兜底
        return [float(x)]
    except Exception:
        return None


def _slice_batch(x, i):
    """从批量张量/数组/列表里取第 i 个样本；不是批量就原样返回"""
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
    """
    尽量从 sample 里提取 position(全维向量) 和 distance(标量)。
    能处理：dict / np / torch / list / tuple，特别是 (obs, reward, done, info) 这种 4 元组。
    """
    import numpy as np
    import torch as _torch
    from collections.abc import Mapping, Sequence

    def _maybe_slice(v):
        # 如果 v 是批量 (第 0 维 == bs_hint)，切第 i 个；否则原样
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

    # ---------- 先处理 dict ----------
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

    # ---------- 再处理 tuple/list ----------
    if isinstance(sample, (list, tuple)):
        # 典型 Gym： (obs, reward, done, info)
        if len(sample) == 4:
            obs, r, d, info = sample
            # 优先从 obs 抽，再从 info 抽
            for cand in (obs, info):
                pos, dist = _extract_pos_dist(cand, i, bs_hint)
                if pos is not None or dist is not None:
                    return pos, dist
            # 如果 obs/info 都没拿到，再尝试其他元素（以防你们自定义了顺序）
            for cand in sample:
                pos, dist = _extract_pos_dist(cand, i, bs_hint)
                if pos is not None or dist is not None:
                    return pos, dist

        # 如果 tuple/list 里含有 dict，就递归每个元素
        if any(isinstance(e, Mapping) for e in sample):
            for e in sample:
                pos, dist = _extract_pos_dist(e, i, bs_hint)
                if pos is not None or dist is not None:
                    return pos, dist

        # 数组/张量的情况：例如 (B,D) 或 (N,3) 集合等
        if len(sample) > 0 and any(isinstance(sample[0], t) for t in (list, tuple, np.ndarray, _torch.Tensor)):
            elem = sample[i] if (bs_hint is not None and len(sample) == bs_hint and i < len(sample)) else sample[0]
            if isinstance(elem, (np.ndarray, _torch.Tensor)) and getattr(elem, "ndim", 0) == 2 and elem.shape[1] >= 3:
                return _to_vector(elem[0]), None
            return _to_vector(elem), None

        # 否则把整个 list 作为向量兜底（可能会失败，失败就返回 None）
        try:
            return _to_vector(sample), None
        except Exception:
            return None, None

    # ---------- 张量/数组 ----------
    if isinstance(sample, (_torch.Tensor, np.ndarray)):
        arr = sample
        if bs_hint is not None and getattr(arr, "ndim", 0) >= 2 and arr.shape[0] == bs_hint:
            arr = arr[i]
        if getattr(arr, "ndim", 0) == 1:
            return _to_vector(arr), None
        if getattr(arr, "ndim", 0) == 2 and arr.shape[1] >= 3:
            return _to_vector(arr[0]), None
        return None, None

    # ---------- 标量兜底 ----------
    try:
        return [float(sample)], None
    except Exception:
        return None, None


def format_for_json(data):
    """
    递归地将数据转换为JSON兼容的格式。
    可以处理字典、列表、Tensor、Numpy Array等。
    """
    if isinstance(data, dict):
        # 如果是字典，递归处理它的每一个值
        return {k: format_for_json(v) for k, v in data.items()}

    elif isinstance(data, list) or isinstance(data, tuple):
        # 如果是列表或元组，递归处理它的每一个元素
        return [format_for_json(item) for item in data]

    elif isinstance(data, torch.Tensor):
        # 如果是PyTorch张量，将其转换为列表
        # 新增 .float()，将 BFloat16 等特殊类型转换为标准的 Float32
        return data.detach().cpu().float().numpy().tolist()

    elif isinstance(data, np.ndarray):
        # 如果是Numpy数组，直接转换为列表
        return data.tolist()

    elif isinstance(data, (str, int, float, bool)) or data is None:
        # 如果是JSON原生支持的类型，直接返回
        return data

    else:
        # 对于其他所有无法识别的类型（如自定义类的实例），将其转换为字符串
        # 这是一个保底策略，确保程序不会因无法序列化而崩溃
        return str(data)


def save_numpy_as_image(numpy_array, file_path):
    """
    将一个NumPy数组（通常是RGB格式）保存为图片文件。
    """
    try:
        # 仿真环境通常输出RGB格式的NumPy数组,
        # 而 cv2.imwrite 需要BGR格式，所以我们转换一下颜色通道。
        image_bgr = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image_bgr)
    except Exception as e:
        print(f"Error saving numpy array as image to {file_path}: {e}")


# 循环地执行模型推理 → 环境交互 → 结果记录
def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir, interceptor=None):
    model_wrapper.eval()  # 模型进入评估模式，关闭 dropout/BN 等训练特性

    with torch.no_grad():
        dataset = BatchIterator(eval_env)      # 初始化数据批次的迭代器，并设置进度条显示
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        episode_idx = 0  # 添加episode计数器
        
        while True:         # 不断获取环境的下一批数据，直到没有数据可用为止
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
            
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)    # 初始化当前 batch 的状态对象
            pbar.update(n=eval_env.batch_size)

            # inputs, rot_to_targets, processed_conversations, unbatched_inputs = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions)    # 准备模型输入,追踪每个任务实例episode 的状态、位置信息、动作结果

            for t in range(int(args.maxWaypoints) + 1):    # 进入轨迹推理循环（最多 maxWaypoints 步）,每次迭代代表 UAV 执行一步
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))    # 日志记录当前步数,记录当前步数和完成的进度

                is_terminate = batch_state.check_batch_termination(t)    # 判断是否全部 episode 已结束
                if is_terminate:
                    break
                
                # === 记录点2: 获取当前观测数据 ===
                if interceptor and batch_state.episodes:
                    try:
                        # 构建观测数据记录
                        current_obs = batch_state.episodes[0][-1] if len(batch_state.episodes[0]) > 0 else {}
                        obs_record = {
                            'sensors': current_obs.get('sensors', {}),
                            'rgb': current_obs.get('rgb', []),      # 传递实际的RGB图像数据
                            'depth': current_obs.get('depth', []),  # 传递实际的深度图数据
                            'instruction': current_obs.get('instruction', ''),
                            'object_position': current_obs.get('object_position', None)
                        }
                        obs_data = interceptor.record_observation(obs_record)
                        interceptor.add_step_data(obs_data)
                    except Exception as e:
                        print(f"[WARNING] 观测数据记录失败: {e}")

                # refined_waypoints, intermediate_outputs = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)    # 模型推理，输出经过模型优化后的路径点

                # =================== BUDGET FORCING 核心改动区域 ===================
                # # 1. 获取当前状态的助理提示
                # assist_notices = batch_state.get_assist_notices()

                # # --- 1. 生成3个初始候选 (通过启用 Dropout) ---
                # logger.info(f"Step: {t}, Stage 1: Generating 3 initial candidates via Dropout...")
                # initial_inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(
                #     batch_state.episodes,
                #     batch_state.target_positions,
                #     assist_notices,
                #     refinement_step=0,
                #     intermediate_waypoint=None,
                # )

                # # === 新增代码开始 ===
                # # 获取当前批次中每个任务的token数 (由于batch size为1，直接取即可)
                # # initial_inputs['input_ids'] 的形状是 [batch_size, sequence_length]
                # token_count_initial = initial_inputs['input_ids'].shape[1]
                # # 记录到batch_state中 (假设batch_size为1)
                # batch_state.tokens_per_step[0].append({"initial_candidates": token_count_initial})
                # # === 新增代码结束 ===

                # model_wrapper.model.train()
                # initial_candidates = []
                # for _ in range(3):
                #     _, intermediate_outputs = model_wrapper.run(
                #         inputs=initial_inputs,
                #         episodes=batch_state.episodes,
                #         rot_to_targets=rot_to_targets,
                #     )
                #     if intermediate_outputs.get("waypoints_llm_new") is not None and len(intermediate_outputs.get("waypoints_llm_new")) > 0:
                #         initial_waypoint = intermediate_outputs.get("waypoints_llm_new")[0]
                #         initial_candidates.append(initial_waypoint)
                # model_wrapper.model.eval()
                
                # # --- 2. 深化3个改进候选 ---
                # final_candidates = []
                # if initial_candidates: # 仅当成功生成初始候选时才进行深化
                #     logger.info(f"Step: {t}, Stage 2: Generating refined candidates...")
                #     # 记录每次深化思考的token数
                #     step_refinement_tokens = []
                #     for initial_wp in initial_candidates:
                #         rethink_inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(
                #             batch_state.episodes,
                #             batch_state.target_positions,
                #             assist_notices,
                #             refinement_step=1,
                #             intermediate_waypoint=initial_wp,
                #         )
                #         # === 新增代码开始 ===
                #         token_count_rethink = rethink_inputs['input_ids'].shape[1]
                #         step_refinement_tokens.append(token_count_rethink)
                #         # === 新增代码结束 ===

                #         refined_wp_batch, _ = model_wrapper.run(
                #             inputs=rethink_inputs,
                #             episodes=batch_state.episodes,
                #             rot_to_targets=rot_to_targets,
                #         )
                #         if refined_wp_batch is not None and len(refined_wp_batch) > 0:
                #             final_candidates.append(refined_wp_batch[0])

                #     # 将这次深化的所有token数记录下来
                #     batch_state.tokens_per_step[0][-1]["refinement_steps"] = step_refinement_tokens    

                # # --- 3. 择优选取最终答案 (包含防御性检查和备用方案) ---
                # logger.info(f"Step: {t}, Stage 3: Scoring and selecting the best candidate...")
                
                # # --- 核心改动 ---
                # if final_candidates:
                #     # 如果深化成功，有最终候选，则进行评分
                #     best_waypoint = score_and_select_best_waypoint(
                #         candidates=final_candidates,
                #         current_episode=batch_state.episodes[0],
                #         target_position=batch_state.target_positions[0]
                #     )
                # elif initial_candidates:
                #     # 如果深化失败，但有初始候选，则使用第一个初始候选作为备用方案
                #     logger.warning(f"Step: {t}, Refined candidates list is empty. Falling back to the first initial candidate.")
                #     # 我们需要将这个粗略的航点（只有方向和距离）通过轨迹模型转化为可执行路径
                #     refined_fallback_wp, _ = model_wrapper.run_traj_model(
                #         batch_state.episodes, 
                #         np.array([initial_candidates[0]]), # 需要包装成numpy array
                #         rot_to_targets
                #     )
                #     best_waypoint = refined_fallback_wp[0]
                # else:
                #     # 如果连初始候选都没有生成，这是一个严重问题，终止当前任务
                #     logger.error(f"Step: {t}, All candidate generation failed. Terminating episode.")
                #     batch_state.dones[0] = True # 标记任务结束
                #     continue # 跳过这一步的后续所有操作

                # final_refined_waypoints = [best_waypoint]


                if args.use_budget_forcing:
                    # 1. 获取当前状态的助理提示
                    assist_notices = batch_state.get_assist_notices()

                    # === 使用命令行传入的参数 ===
                    num_parallel_thoughts = args.num_parallel_thoughts
                    num_serial_refinements = args.num_refinement_steps

                    #**********************************上次

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
                            
                            # === 核心改动：在打印前，将相对坐标转换为世界坐标 ===
                            # transform_to_world 需要一个列表作为输入，所以我们把 new_candidate 放入列表中
                            # 函数返回的也是一个列表，我们取第一个元素 [0]
                            # world_candidate = transform_to_world([new_candidate], batch_state.episodes)[0]                              
                            # === 打印 ===
                            # 为了可读性，使用np.round将坐标格式化为两位小数
                            formatted_coords = np.round(new_candidate, 2)
                            print(f"    [Parallel Thought #{i+1}] Predicted Coords (World): {formatted_coords}")

                    model_wrapper.model.eval()

                    # --- 2. 深化改进候选 (串行思考) ---
                    final_candidates = []
                    if initial_candidates:
                        logger.info(f"Step: {t}, Stage 2: Generating refined candidates with {num_serial_refinements} refinement steps...")
                        # === 新增打印 ===
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
                                    # === 新增打印 ===
                                    formatted_refined_coords = np.round(current_wp, 2)
                                    print(f"        -> Refinement Step #{i+1} New Coords: {formatted_refined_coords}")
                                else:
                                    # === 新增打印 ===
                                    print(f"        -> Refinement Step #{i+1} FAILED.")
                                    break
                                
                            
                            step_refinement_tokens_all_candidates.append(tokens_for_this_candidate)
                            final_candidates.append(current_wp)

                        batch_state.tokens_per_step[0][-1]["refinement_steps"] = step_refinement_tokens_all_candidates
                    
                    # --- 3. 择优选取最终答案 ---
                    logger.info(f"Step: {t}, Stage 3: Scoring and selecting the best candidate...")
                     # === 新增打印 ===
                    print(f"\n{'-'*20} Step [{t}]: Stage 3 - Final Selection {'-'*20}")
                
                    if final_candidates:
                        best_waypoint = score_and_select_best_waypoint(
                            candidates=final_candidates,
                            current_episode=batch_state.episodes[0],
                            target_position=batch_state.target_positions[0]
                        )
                    elif initial_candidates:
                        logger.warning(f"Step: {t}, Refined candidates list is empty. Falling back to the first initial candidate.")
                        refined_fallback_wp, _ = model_wrapper.run_traj_model(
                            batch_state.episodes, np.array([initial_candidates[0]]), rot_to_targets
                        )
                        best_waypoint = refined_fallback_wp[0]
                    else:
                        logger.error(f"Step: {t}, All candidate generation failed. Terminating episode.")
                        batch_state.dones[0] = True
                        continue

                    # === 新增打印 ===
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
                    #**********************************上次

#    # ======================== 新代码开始 ========================

#                     # --- Stage 1: 并行思考 (生成100个初始候选) ---
#                     logger.info(f"Step: {t}, Stage 1: Generating {num_parallel_thoughts} initial candidates...")
                      
#                     initial_inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(
#                         batch_state.episodes, batch_state.target_positions, assist_notices,
#                         refinement_step=0, intermediate_waypoint=None
#                     )
#                     token_count_initial = initial_inputs['input_ids'].shape[1]
#                     batch_state.tokens_per_step[0].append({"initial_candidates": token_count_initial})

#                     model_wrapper.model.train()
#                     initial_candidates_relative = [] # 存放相对坐标，用于后续计算
#                     initial_candidates_world = []    # 存放世界坐标，仅用于打印
#                     for _ in range(num_parallel_thoughts):
#                         _, intermediate_outputs = model_wrapper.run(
#                             inputs=initial_inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets
#                         )
#                         if intermediate_outputs.get("waypoints_llm_new") is not None and len(intermediate_outputs.get("waypoints_llm_new")) > 0:
#                             # 获取LLM输出的原始相对坐标
#                             relative_candidate = intermediate_outputs.get("waypoints_llm_new")[0]
#                             initial_candidates_relative.append(relative_candidate)
                              
#                             # 转换为世界坐标并存储，仅为打印
#                             world_candidate = transform_to_world([relative_candidate], batch_state.episodes)[0]
#                             initial_candidates_world.append(world_candidate)
#                     model_wrapper.model.eval()

#                     # --- Stage 2: 串行提炼 (精炼100个候选) ---
#                     final_candidates = [] # 存放100个精炼后的点
#                     if initial_candidates_relative:
#                         logger.info(f"Step: {t}, Stage 2: Refining {len(initial_candidates_relative)} candidates...")
#                         step_refinement_tokens_all_candidates = []
#                         for initial_wp in initial_candidates_relative:
#                             current_wp = initial_wp
#                             tokens_for_this_candidate = []
#                             # 只有一个精炼步骤
#                             rethink_inputs, _, _, _ = model_wrapper.prepare_inputs(
#                                 batch_state.episodes, batch_state.target_positions, assist_notices,
#                                 refinement_step=1, intermediate_waypoint=np.array(current_wp).flatten()
#                             )
                              
#                             # 注意：这里的 model_wrapper.run() 输出的已经是世界坐标
#                             refined_wp_batch, _ = model_wrapper.run(
#                                 inputs=rethink_inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets
#                             )
                              
#                             if refined_wp_batch is not None and len(refined_wp_batch) > 0:
#                                 final_candidates.append(refined_wp_batch[0])
                      
#                     # --- Stage 3: 最终选择 ---
#                     best_waypoint = None
#                     if final_candidates:
#                         logger.info(f"Step: {t}, Stage 3: Scoring and selecting the best candidate...")
#                         best_waypoint = score_and_select_best_waypoint(
#                             candidates=final_candidates,
#                             current_episode=batch_state.episodes[0],
#                             target_position=batch_state.target_positions[0]
#                         )
#                     # 防御性代码：如果精炼失败，则退回到使用第一个初始想法
#                     elif initial_candidates_relative:
#                         logger.warning(f"Step: {t}, Refinement failed. Falling back to the first initial candidate.")
#                         # 注意：这里的 model_wrapper.run_traj_model 输出的已经是世界坐标
#                         refined_fallback_wp, _ = model_wrapper.run_traj_model(
#                             batch_state.episodes, np.array([initial_candidates_relative[0]]), rot_to_targets
#                         )
#                         best_waypoint = refined_fallback_wp[0]
#                     else:
#                         logger.error(f"Step: {t}, All candidate generation failed. Terminating episode.")
#                         batch_state.dones[0] = True
#                         continue

#                     # --- 统一打印所有结果 ---
#                     print(f"\n\n{'='*25} STEP {t} DECISION SUMMARY {'='*25}")
#                     # 1. 打印第一步的100个初始预测点 (LLM的草图)
#                     print(f"\n--- Stage 1: Initial Parallel Thoughts (100 plans) ---")
#                     for i, wp_plan in enumerate(initial_candidates_world):
#                         # wp_plan 是一个3点草图 [W1, W2, W3]
#                         # 我们打印草图的起点 W1 和终点 W3
#                         formatted_start = np.round(wp_plan[0], 2)
#                         formatted_end = np.round(wp_plan[-1], 2)
#                         print(f"  Thought #{i+1:3d}: START -> {formatted_start} | END -> {formatted_end}")

#                     # 2. 打印第二步的100个精炼预测点 (轨迹模型的蓝图)
#                     print(f"\n--- Stage 2: Refined Candidates (100 trajectories) ---")
#                     for i, trajectory in enumerate(final_candidates):
#                         # trajectory 是一条7点轨迹 [P1, ..., P7]
#                         # 我们打印轨迹的起点 P1 和终点 P7
#                         formatted_start = np.round(trajectory[0], 2)
#                         formatted_end = np.round(trajectory[-1], 2)
#                         print(f"  Refined #{i+1:3d}: START -> {formatted_start} | END -> {formatted_end}")

#                     # 3. 打印最终选择的一条轨迹的起点和终点
#                     print(f"\n--- Stage 3: Final Selected Trajectory ---")
#                     if best_waypoint is not None:
#                         formatted_start = np.round(best_waypoint[0], 2)
#                         formatted_end = np.round(best_waypoint[-1], 2)
#                         print(f"  Selected Plan: START -> {formatted_start} | END -> {formatted_end}")  

                      
#                     print(f"{'='*75}\n")

#                     # 使用最终选择的点继续流程
#                     final_refined_waypoints = [best_waypoint]

# # ======================== 新代码结束 ========================

                else:
                    # 如果没有使用 budget forcing，执行原始的、最简单的推理逻辑
                    # (这里可以放一个不带思考模块的简单模型推理，或者直接报错提示)
                    # 为了简单起见，我们假设必须启用该功能，或者你需要补充这里的 else 逻辑
                    inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions)
                    
                    # === 记录点3: 记录模型输入（标准推理） ===
                    if interceptor:
                        try:
                            input_data = interceptor.record_model_input(inputs)
                            interceptor.add_step_data(input_data)
                        except Exception as e:
                            print(f"[WARNING] 模型输入记录失败: {e}")
                    
                    final_refined_waypoints, _ = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)
                    
                    # === 记录点4: 记录模型输出（标准推理） ===
                    if interceptor:
                        try:
                            output_record = {
                                'waypoints_world': final_refined_waypoints
                            }
                            output_data = interceptor.record_model_output(output_record)
                            interceptor.add_step_data(output_data)
                        except Exception as e:
                            print(f"[WARNING] 模型输出记录失败: {e}")
                    
                    # === 新增打印 ===
                    # 确保在不使用该功能时也能看到输出
                    if final_refined_waypoints:
                        formatted_coords = np.round(final_refined_waypoints[0], 2)
                        print(f"\n{'='*20} Step [{t}]: Standard Inference {'='*20}")
                        print(f"    Predicted Coords: {formatted_coords}")
                        print(f"{'='*60}\n")
                
                # =================== 改动结束 =========================================
                
                # === 记录点5: 记录控制指令 ===
                if interceptor and final_refined_waypoints:
                    try:
                        cmd_record = {
                            'waypoints': [wp.tolist() if hasattr(wp, 'tolist') else wp for wp in final_refined_waypoints]
                        }
                        cmd_data = interceptor.record_control_command(cmd_record)
                        interceptor.add_step_data(cmd_data)
                    except Exception as e:
                        print(f"[WARNING] 控制指令记录失败: {e}")
                
                # 环境交互与状态更新
                eval_env.makeActions(final_refined_waypoints)  # 执行动作（移动 UAV）
                outputs = eval_env.get_obs()               # UAV模拟环境中获取当前的状态观测信息

                # 更新状态：观测 + done 预测 + 评估指标
                batch_state.update_from_env_output(outputs)   # 将环境返回的新观测（outputs）更新到当前这一批 episode 的状态中
                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                batch_state.update_metric()                   # 更新评估指标
                
                # === 记录点: 结束步骤 ===
                if interceptor:
                    try:
                        interceptor.end_step()
                    except Exception as e:
                        print(f"[WARNING] 步骤结束记录失败: {e}")

                # # 获取 assist 辅助提示，更新下一轮输入(环境信息、任务信息、先验知识)
                # assist_notices = batch_state.get_assist_notices()
                # inputs, rot_to_targets, processed_conversations, unbatched_inputs = model_wrapper.prepare_inputs(
                #     batch_state.episodes, batch_state.target_positions, assist_notices
                # )
            
            # === 记录点6: Episode结束 ===
            if interceptor:
                try:
                    # 获取最终的评估指标
                    final_metrics = batch_state.get_metrics() if hasattr(batch_state, 'get_metrics') else {}
                    episode_result = {
                        'success': batch_state.dones[0] if hasattr(batch_state, 'dones') and len(batch_state.dones) > 0 else False,
                        'distance_to_goal': batch_state.remain_dists[0] if hasattr(batch_state, 'remain_dists') and len(batch_state.remain_dists) > 0 else None,
                        'collision': False,  # 根据实际情况设置
                        'metrics': final_metrics
                    }
                    interceptor.end_episode(episode_result)
                except Exception as e:
                    print(f"[WARNING] Episode结束记录失败: {e}")
            
            episode_idx += 1

        # 批次循环结束后关闭进度条
        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":


    # seed = getattr(args, 'seed', 42) 
    # set_seed(seed)

    eval_save_path = args.eval_save_path   # 读取评估结果保存路径
    eval_json_path = args.eval_json_path   # 读取任务配置路径
    dataset_path = args.dataset_path       # 读取数据集路径

    if not os.path.exists(eval_save_path):    # 保存路径不存在，则自动创建
        os.makedirs(eval_save_path)

    # === 初始化数据拦截器 ===
    interceptor = None
    record_data = True  # 从参数中读取是否记录数据
    if HAS_INTERCEPTOR and record_data:
        try:
            record_dir = getattr(args, 'record_dir', './debug_data')
            interceptor = DataInterceptor(output_dir=record_dir)
            print(f"[INFO] ✓ 数据拦截器已启用，数据将保存到: {record_dir}")
            max_record_eps = getattr(args, 'max_record_eps', None)
            if max_record_eps:
                print(f"[INFO] 最多记录 {max_record_eps} 个episode")
        except Exception as e:
            print(f"[ERROR] 数据拦截器初始化失败: {e}")
            interceptor = None
    else:
        if not HAS_INTERCEPTOR:
            print("[INFO] 数据拦截器未导入，跳过数据记录")
        else:
            print("[INFO] 未启用数据记录（使用 --record_data 参数启用）")

    setup()    # 进行分布式训练或其他系统初始化设置

    assert CheckPort(), 'error port'    # 检查端口是否可用

    print("***************************************************")
    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)    # 创建UAV路径规划环境对象的实例

    if is_dist_avail_and_initialized():    # 清理分布式设置
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)    # 路径规划模型的推理逻辑
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)                 # 辅助模块

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)
    print("***************************************************")
    eval(model_wrapper=model_wrapper,      # 调用主评估函数
         assist=assist,
         eval_env=eval_env,
         eval_save_dir=eval_save_path,
         interceptor=interceptor)          # 传入拦截器

    eval_env.delete_VectorEnvUtil()
