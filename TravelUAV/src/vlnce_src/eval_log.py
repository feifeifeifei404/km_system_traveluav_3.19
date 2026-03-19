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
from utils.logger import logger
from utils.utils import *
from src.model_wrapper.travel_llm import TravelModelWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized

def set_seed(seed: int = 42):
    """
    固定所有相关的随机数种子，确保实验的可复现性。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def interpret_action_from_local_waypoint(local_waypoint):
    """
    将无人机局部坐标系下的单个航点“翻译”为人类可理解的动作描述。
    假设航点是一个 [x, y, z] 的列表或Numpy数组。
    x: 前后 (+前), y: 左右 (+右), z: 上下 (+上)
    """
    if local_waypoint is None or len(local_waypoint) < 3:
        return "Invalid Waypoint"

    x, y, z = local_waypoint[0], local_waypoint[1], local_waypoint[2]
    
    move_threshold = 0.5
    turn_threshold = 0.5
    actions = []

    if x > move_threshold: actions.append("Move Forward")
    elif x < -move_threshold: actions.append("Move Backward")

    if y > turn_threshold: actions.append("Turn Right")
    elif y < -turn_threshold: actions.append("Turn Left")

    if z > move_threshold: actions.append("Ascend (Up)")
    elif z < -move_threshold: actions.append("Descend (Down)")
    
    if not actions:
        if abs(y) > 0.1:
            side = "Right" if y > 0 else "Left"
            actions.append(f"Strafe {side}")
        else:
            actions.append("Hover / Fine-tune")

    return ", ".join(actions)

# ========= 小工具：递归找 key + 安全转型 =========
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
        if x is None: return None
        x0 = x[0]
        if hasattr(x0, "tolist"): return x0.tolist()
        if isinstance(x0, (list, tuple)): return list(x0)
        return [float(x0)]
    except Exception:
        return None

def _to_first_float(x):
    try:
        import numpy as np, torch as _torch
        if x is None: return None
        if isinstance(x, (list, tuple, np.ndarray)): return float(x[0])
        if isinstance(x, _torch.Tensor): return float(x[0].item() if x.ndim > 0 else x.item())
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

def _safe_get_episode(outputs, i):
    """从环境返回中取第 i 个 episode 的观测：
       - 若 outputs 是 list/tuple：直接 outputs[i]
       - 若是 dict：有些环境把 batch 维放在 value[i] 上，这里先返回整个 dict 让 _find_in 去找
    """
    if isinstance(outputs, (list, tuple)):
        if i < len(outputs):
            return outputs[i]
        return None
    return outputs

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
    将一个NumPy数组通常是RGB格式保存为图片文件。
    """
    try:
        # 仿真环境通常输出RGB格式的NumPy数组,
        # 而 cv2.imwrite 需要BGR格式，所以我们转换一下颜色通道。
        image_bgr = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image_bgr)
    except Exception as e:
        print(f"Error saving numpy array as image to {file_path}: {e}")


# # 循环地执行模型推理 → 环境交互 → 结果记录
# # === 【最终修改】: 在你的 eval 函数基础上进行修改 ===
# def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir):
#     model_wrapper.eval()

#     # 创建详细日志的保存目录
#     detailed_log_dir = os.path.join(eval_save_dir, "detailed_logs")
#     os.makedirs(detailed_log_dir, exist_ok=True)

#     with torch.no_grad():
#         dataset = BatchIterator(eval_env)
#         end_iter = len(dataset)
#         pbar = tqdm.tqdm(total=end_iter)

#         while True:
#             env_batchs = eval_env.next_minibatch()
#             if env_batchs is None:
#                 break
            
#             batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)
#             pbar.update(n=eval_env.batch_size)
            
#             # 初始化用于在内存中缓存本批次日志的列表
#             per_task_step_logs = [[] for _ in range(eval_env.batch_size)]

#             # 准备第一次循环的输入
#             assist_notices = batch_state.get_assist_notices()
#             inputs_for_run, preparation_logs_for_next_step = model_wrapper.prepare_inputs(
#                 batch_state.episodes, batch_state.target_positions, assist_notices
#             )

#             for t in range(int(args.maxWaypoints) + 1):
#                 logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))
                
#                 # --- [修改] 在这里捕获上一步准备的日志 ---
#                 preparation_logs_current_step = preparation_logs_for_next_step

#                 is_terminate = batch_state.check_batch_termination(t)
#                 if is_terminate:
#                     break
                
#                 # --- [修改] 修改 run 函数的调用，接收新的返回值 ---
#                 refined_waypoints, inference_logs = model_wrapper.run(
#                     inputs=inputs_for_run, 
#                     episodes=batch_state.episodes, 
#                     rot_to_targets=preparation_logs_current_step["rotation_to_target_matrices"]
#                 )
#                 eval_env.makeActions(refined_waypoints)
#                 outputs = eval_env.get_obs()
                            
#                 batch_state.update_from_env_output(outputs)
#                 batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
#                 batch_state.update_metric()
                
#                 # --- [新增] 日志记录阶段 ---
#                 for i in range(batch_state.batch_size):
#                     if batch_state.skips[i]: continue
                    
#                     step_log = {
#                         "step": t,
#                         "preparation": {
#                             key: _slice_batch(value, i) for key, value in preparation_logs_current_step.items()
#                         },
#                         "inference": {
#                             key: _slice_batch(value, i) for key, value in inference_logs.items()
#                         },
#                         "outcome": {
#                             "is_collision": batch_state.collisions[i],
#                             "is_success": batch_state.success[i],
#                             "is_oracle_success": batch_state.oracle_success[i],
#                             "distance_to_target": batch_state.distance_to_ends[i][-1],
#                             "termination_reason": batch_state.termination_reasons[i],
#                         },
#                     }
#                     per_task_step_logs[i].append(step_log)

#                 # --- [修改] 为下一次循环准备输入和日志 ---
#                 assist_notices = batch_state.get_assist_notices()
#                 inputs_for_run, preparation_logs_for_next_step = model_wrapper.prepare_inputs(
#                     batch_state.episodes, batch_state.target_positions, assist_notices
#                 )

#             # --- [新增] 批次循环结束后，统一写入日志文件 ---
#             for i in range(len(batch_state.ori_data_dirs)):
#                 task_id = batch_state.ori_data_dirs[i].split('/')[-1]
#                 log_file_path = os.path.join(detailed_log_dir, f"{task_id}_details.json")
#                 with open(log_file_path, 'w') as f:
#                     # 你已经有了 format_for_json，直接使用即可
#                     json.dump(format_for_json(per_task_step_logs[i]), f, indent=2)
#                 print(f"Saved detailed logs for task {task_id} to {log_file_path}")

#         try:    
#             pbar.close()
#         except:
#             pass

def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir):
    model_wrapper.eval()

    detailed_log_dir = os.path.join(eval_save_dir, "detailed_logs")
    os.makedirs(detailed_log_dir, exist_ok=True)

    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)
            pbar.update(n=eval_env.batch_size)
            
            per_task_step_logs = [[] for _ in range(eval_env.batch_size)]

            assist_notices = batch_state.get_assist_notices()
            inputs_for_run, preparation_logs_for_next_step = model_wrapper.prepare_inputs(
                batch_state.episodes, batch_state.target_positions, assist_notices
            )

            for t in range(int(args.maxWaypoints) + 1):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))
                
                preparation_logs_current_step = preparation_logs_for_next_step

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break
                
                refined_waypoints, inference_logs = model_wrapper.run(
                    inputs=inputs_for_run, 
                    episodes=batch_state.episodes, 
                    rot_to_targets=preparation_logs_current_step["rotation_to_target_matrices"]
                )
                
                eval_env.makeActions(refined_waypoints)
                outputs = eval_env.get_obs()
                            
                batch_state.update_from_env_output(outputs)
                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                batch_state.update_metric()
                
                # --- 【整合后的日志记录阶段】 ---
                for i in range(batch_state.batch_size):
                    if batch_state.skips[i]: continue
                    
                    task_id = batch_state.ori_data_dirs[i].split('/')[-1]
                    
                    # 1. 创建当前时间步的图片保存目录
                    step_image_dir_path = os.path.join(detailed_log_dir, task_id, f"step_{t:03d}")
                    os.makedirs(step_image_dir_path, exist_ok=True)
                    
                    # 2. 保存 MLLM 输入的5个视角图像并获取相对路径
                    mllm_input_images_tensor = _slice_batch(preparation_logs_current_step["unbatched_inputs_before_device"], i)['image']
                    mllm_input_images_np = mllm_input_images_tensor.permute(0, 2, 3, 1).cpu().numpy()
                    mean = np.array([0.48145466, 0.4578275, 0.40821073])
                    std = np.array([0.26862954, 0.26130258, 0.27577711])
                    mllm_input_images_np = np.clip((mllm_input_images_np * std + mean) * 255, 0, 255).astype(np.uint8)

                    view_names = ["front", "back", "left", "right", "down"]
                    mllm_image_paths = []
                    for view_idx, view_name in enumerate(view_names):
                        img_path = os.path.join(step_image_dir_path, f"mllm_input_view_{view_name}.png")
                        save_numpy_as_image(mllm_input_images_np[view_idx], img_path)
                        mllm_image_paths.append(os.path.relpath(img_path, detailed_log_dir))

                    # 3. 保存轨迹模型输入的前视图图像并获取相对路径
                    traj_input_img_tensor = _slice_batch(inference_logs["traj_model_input"], i)['img']
                    traj_input_img_np = traj_input_img_tensor.permute(0, 2, 3, 1).cpu().numpy()[0]
                    traj_input_img_np = np.clip((traj_input_img_np * std + mean) * 255, 0, 255).astype(np.uint8)
                    
                    traj_img_path_str = os.path.join(step_image_dir_path, "traj_model_input_front.png")
                    save_numpy_as_image(traj_input_img_np, traj_img_path_str)
                    traj_img_path_rel = os.path.relpath(traj_img_path_str, detailed_log_dir)

                    # 4. 准备写入JSON的日志字典，并将图像字段替换为路径
                    prep_logs = {key: _slice_batch(value, i) for key, value in preparation_logs_current_step.items()}
                    if 'image' in prep_logs.get("unbatched_inputs_before_device", {}):
                        prep_logs["unbatched_inputs_before_device"]["image_paths"] = mllm_image_paths
                        del prep_logs["unbatched_inputs_before_device"]["image"]

                    infer_logs = {key: _slice_batch(value, i) for key, value in inference_logs.items()}
                    if 'img' in infer_logs.get("traj_model_input", {}):
                        infer_logs["traj_model_input"]["img_path"] = traj_img_path_rel
                        del infer_logs["traj_model_input"]["img"]

                    # 5. 从日志中提取并翻译动作
                    action_description = "No Action Calculated"
                    try:
                        local_waypoints_tensor = infer_logs.get("traj_model_raw_output")
                        if local_waypoints_tensor is not None:
                            first_local_waypoint = local_waypoints_tensor[0].tolist()
                            action_description = interpret_action_from_local_waypoint(first_local_waypoint)
                    except Exception:
                        action_description = "Error Interpreting Action"
                    
                    # 6. 构建最终的单步日志
                    step_log = {
                        "step": t,
                        "action_description": action_description,
                        "preparation": prep_logs,
                        "inference": infer_logs,
                        "outcome": {
                            "is_collision": batch_state.collisions[i],
                            "is_success": batch_state.success[i],
                            "is_oracle_success": batch_state.oracle_success[i],
                            "distance_to_target": batch_state.distance_to_ends[i][-1],
                            "termination_reason": batch_state.termination_reasons[i],
                        },
                    }
                    per_task_step_logs[i].append(step_log)

                assist_notices = batch_state.get_assist_notices()
                inputs_for_run, preparation_logs_for_next_step = model_wrapper.prepare_inputs(
                    batch_state.episodes, batch_state.target_positions, assist_notices
                )

            # --- 批次结束后，统一写入JSON日志文件 ---
            for i in range(len(batch_state.ori_data_dirs)):
                task_id = batch_state.ori_data_dirs[i].split('/')[-1]
                log_file_path = os.path.join(detailed_log_dir, f"{task_id}_details.json")
                with open(log_file_path, 'w') as f:
                    json.dump(format_for_json(per_task_step_logs[i]), f, indent=2)
                print(f"Saved detailed logs (JSON) for task {task_id} to {log_file_path}")

        try:    
            pbar.close()
        except:
            pass
            


if __name__ == "__main__":

    seed = getattr(args, 'seed', 42) 
    set_seed(seed)
    
    eval_save_path = args.eval_save_path   #读取评估结果保存路径
    eval_json_path = args.eval_json_path   #读取任务配置路径
    dataset_path = args.dataset_path    #读取数据集路径
    
    if not os.path.exists(eval_save_path):    #保存路径不存在，则自动创建
        os.makedirs(eval_save_path)
    
    setup()    #进行分布式训练或其他系统初始化设置

    assert CheckPort(), 'error port'    #检查端口是否可用

    print("***************************************************")
    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)    #创建UAV路径规划环境对象的实例

    if is_dist_avail_and_initialized():    #清理分布式设置
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False
    
    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)    #路径规划模型的推理逻辑：模型结构、路径、权重等相关参数；与输入数据相关的参数
    
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)    #在模型运行过程中，是否提供 Ground Truth信息：是否始终提供提示；是否使用 ground truth 数据作为提示内容

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)
    print("***************************************************")
    eval(model_wrapper=model_wrapper,      #调用主评估函数
         assist=assist,
         eval_env=eval_env,
         eval_save_dir=eval_save_path)
    
    eval_env.delete_VectorEnvUtil()
