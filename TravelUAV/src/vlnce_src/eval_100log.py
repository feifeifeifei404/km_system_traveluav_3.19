import os
from pathlib import Path
import sys
import time
import json
import shutil
import random
import logging  # <--- 导入标准日志库

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

# --- 开始：新的日志记录设置 ---
# 1. 配置根记录器 (Root Logger)
# 定义日志文件存放的目录
log_dir = "/mnt/data/TravelUAV/scripts"
# 确保目录存在，如果不存在则创建
os.makedirs(log_dir, exist_ok=True)
# 创建一个带时间戳的完整日志文件路径
log_filename = os.path.join(log_dir, f"evaluation_run_{time.strftime('%Y%m%d-%H%M%S')}.log")


# basicConfig 会配置整个日志系统
logging.basicConfig(
    level=logging.INFO,  # 设置最低日志级别为 INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 定义日志格式
    handlers=[
        logging.FileHandler(log_filename),      # 第一个处理器：将日志写入文件
        logging.StreamHandler(sys.__stdout__)   # 第二个处理器：将日志输出到原始控制台
    ]
)

# 获取记录器实例，后续代码将使用名为 `logger` 的变量
logger = logging.getLogger(__name__)

# 2. 定义一个类，用于将 `print` 函数的输出重定向到日志记录器
class StreamToLogger:
    """
    一个类文件流对象，可以将写入操作（如 print）重定向到一个日志记录器实例。
    """
    def __init__(self, logger_instance, log_level=logging.INFO):
        self.logger = logger_instance
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        # rstrip() 移除末尾的换行符，然后 splitlines() 处理可能的多行输出
        for line in buf.rstrip().splitlines():
            # 使用日志记录器输出，它会自动处理换行
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # 这个方法对于文件类对象是必需的，但我们的日志记录器会自动处理刷新。
        pass

# --- 结束：新的日志记录设置 ---


sys.path.append(str(Path(str(os.getcwd())).resolve()))
# from utils.logger import logger  # <--- 移除了原来的 logger 导入
from utils.utils import *
from src.model_wrapper.travel_llm import TravelModelWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized
from src.vlnce_src.scoring_util import score_and_select_best_waypoint
from src.model_wrapper.utils.travel_util import transform_to_world


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
    logger.info(f"Random seed set to {seed}")


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
        logger.error(f"Error saving numpy array as image to {file_path}: {e}")


# 循环地执行模型推理 → 环境交互 → 结果记录
def eval(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir):
    model_wrapper.eval()  # 模型进入评估模式，关闭 dropout/BN 等训练特性

    with torch.no_grad():
        dataset = BatchIterator(eval_env)       # 初始化数据批次的迭代器，并设置进度条显示
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:         # 不断获取环境的下一批数据，直到没有数据可用为止
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)    # 初始化当前 batch 的状态对象
            pbar.update(n=eval_env.batch_size)

            # inputs, rot_to_targets, processed_conversations, unbatched_inputs = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions)    # 准备模型输入,追踪每个任务实例episode 的状态、位置信息、动作结果

            for t in range(int(args.maxWaypoints) + 1):    # 进入轨迹推理循环（最多 maxWaypoints 步）,每次迭代代表 UAV 执行一步
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))    # 日志记录当前步数,记录当前步数和完成的进度

                is_terminate = batch_state.check_batch_termination(t)    # 判断是否全部 episode 已结束
                if is_terminate:
                    break

                # refined_waypoints, intermediate_outputs = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)    # 模型推理，输出经过模型优化后的路径点

                if args.use_budget_forcing:
                    # 1. 获取当前状态的助理提示
                    assist_notices = batch_state.get_assist_notices()

                    # === 使用命令行传入的参数 ===
                    num_parallel_thoughts = args.num_parallel_thoughts
                    num_serial_refinements = args.num_refinement_steps

                    # --- Stage 1: 并行思考 (生成100个初始候选) ---
                    logger.info(f"Step: {t}, Stage 1: Generating {num_parallel_thoughts} initial candidates...")
                        
                    initial_inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(
                        batch_state.episodes, batch_state.target_positions, assist_notices,
                        refinement_step=0, intermediate_waypoint=None
                    )
                    token_count_initial = initial_inputs['input_ids'].shape[1]
                    batch_state.tokens_per_step[0].append({"initial_candidates": token_count_initial})

                    model_wrapper.model.train()
                    initial_candidates_relative = [] # 存放相对坐标，用于后续计算
                    initial_candidates_world = []    # 存放世界坐标，仅用于打印
                    for _ in range(num_parallel_thoughts):
                        _, intermediate_outputs = model_wrapper.run(
                            inputs=initial_inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets
                        )
                        if intermediate_outputs.get("waypoints_llm_new") is not None and len(intermediate_outputs.get("waypoints_llm_new")) > 0:
                            # 获取LLM输出的原始相对坐标
                            relative_candidate = intermediate_outputs.get("waypoints_llm_new")[0]
                            initial_candidates_relative.append(relative_candidate)
                                
                            # 转换为世界坐标并存储，仅为打印
                            world_candidate = transform_to_world([relative_candidate], batch_state.episodes)[0]
                            initial_candidates_world.append(world_candidate)
                    model_wrapper.model.eval()

                    # --- Stage 2: 串行提炼 (精炼100个候选) ---
                    final_candidates = [] # 存放100个精炼后的点
                    if initial_candidates_relative:
                        logger.info(f"Step: {t}, Stage 2: Refining {len(initial_candidates_relative)} candidates...")
                        step_refinement_tokens_all_candidates = []
                        for initial_wp in initial_candidates_relative:
                            current_wp = initial_wp
                            tokens_for_this_candidate = []
                            # 只有一个精炼步骤
                            rethink_inputs, _, _, _ = model_wrapper.prepare_inputs(
                                batch_state.episodes, batch_state.target_positions, assist_notices,
                                refinement_step=1, intermediate_waypoint=np.array(current_wp).flatten()
                            )
                                
                            # 注意：这里的 model_wrapper.run() 输出的已经是世界坐标
                            refined_wp_batch, _ = model_wrapper.run(
                                inputs=rethink_inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets
                            )
                                
                            if refined_wp_batch is not None and len(refined_wp_batch) > 0:
                                final_candidates.append(refined_wp_batch[0])
                        
                    # --- Stage 3: 最终选择 ---
                    best_waypoint = None
                    if final_candidates:
                        logger.info(f"Step: {t}, Stage 3: Scoring and selecting the best candidate...")
                        best_waypoint = score_and_select_best_waypoint(
                            candidates=final_candidates,
                            current_episode=batch_state.episodes[0],
                            target_position=batch_state.target_positions[0]
                        )
                    # 防御性代码：如果精炼失败，则退回到使用第一个初始想法
                    elif initial_candidates_relative:
                        logger.warning(f"Step: {t}, Refinement failed. Falling back to the first initial candidate.")
                        # 注意：这里的 model_wrapper.run_traj_model 输出的已经是世界坐标
                        refined_fallback_wp, _ = model_wrapper.run_traj_model(
                            batch_state.episodes, np.array([initial_candidates_relative[0]]), rot_to_targets
                        )
                        best_waypoint = refined_fallback_wp[0]
                    else:
                        logger.error(f"Step: {t}, All candidate generation failed. Terminating episode.")
                        batch_state.dones[0] = True
                        continue

                    # --- 统一打印所有结果 ---
                    print(f"\n\n{'='*25} STEP {t} DECISION SUMMARY {'='*25}")
                    # 1. 打印第一步的100个初始预测点 (LLM的草图)
                    print(f"\n--- Stage 1: Initial Parallel Thoughts (100 plans) ---")
                    for i, wp_plan in enumerate(initial_candidates_world):
                        # wp_plan 是一个3点草图 [W1, W2, W3]
                        # 我们打印草图的起点 W1 和终点 W3
                        formatted_start = np.round(wp_plan[0], 2)
                        formatted_end = np.round(wp_plan[-1], 2)
                        print(f"  Thought #{i+1:3d}: START -> {formatted_start} | END -> {formatted_end}")

                    # 2. 打印第二步的100个精炼预测点 (轨迹模型的蓝图)
                    print(f"\n--- Stage 2: Refined Candidates (100 trajectories) ---")
                    for i, trajectory in enumerate(final_candidates):
                        # trajectory 是一条7点轨迹 [P1, ..., P7]
                        # 我们打印轨迹的起点 P1 和终点 P7
                        formatted_start = np.round(trajectory[0], 2)
                        formatted_end = np.round(trajectory[-1], 2)
                        print(f"  Refined #{i+1:3d}: START -> {formatted_start} | END -> {formatted_end}")

                    # 3. 打印最终选择的一条轨迹的起点和终点
                    print(f"\n--- Stage 3: Final Selected Trajectory ---")
                    if best_waypoint is not None:
                        formatted_start = np.round(best_waypoint[0], 2)
                        formatted_end = np.round(best_waypoint[-1], 2)
                        print(f"  Selected Plan: START -> {formatted_start} | END -> {formatted_end}")  

                        
                    print(f"{'='*75}\n")

                    # 使用最终选择的点继续流程
                    final_refined_waypoints = [best_waypoint]

                else:
                    # 如果没有使用 budget forcing，执行原始的、最简单的推理逻辑
                    inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions)
                    final_refined_waypoints, _ = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)
                    # 确保在不使用该功能时也能看到输出
                    if final_refined_waypoints:
                        formatted_coords = np.round(final_refined_waypoints[0], 2)
                        print(f"\n{'='*20} Step [{t}]: Standard Inference {'='*20}")
                        print(f"    Predicted Coords: {formatted_coords}")
                        print(f"{'='*60}\n")
                
                # 环境交互与状态更新
                eval_env.makeActions(final_refined_waypoints)  # 执行动作（移动 UAV）
                outputs = eval_env.get_obs()                   # UAV模拟环境中获取当前的状态观测信息

                # 更新状态：观测 + done 预测 + 评估指标
                batch_state.update_from_env_output(outputs)   # 将环境返回的新观测（outputs）更新到当前这一批 episode 的状态中
                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)
                batch_state.update_metric()                   # 更新评估指标


        # 批次循环结束后关闭进度条
        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    # --- 在程序入口重定向 stdout 和 stderr ---
    # 这会将所有后续的 `print` 语句的输出都通过我们配置好的日志系统处理
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # seed = getattr(args, 'seed', 42) 
    # set_seed(seed)

    eval_save_path = args.eval_save_path   # 读取评估结果保存路径
    eval_json_path = args.eval_json_path   # 读取任务配置路径
    dataset_path = args.dataset_path       # 读取数据集路径

    if not os.path.exists(eval_save_path):     # 保存路径不存在，则自动创建
        os.makedirs(eval_save_path)

    setup()    # 进行分布式训练或其他系统初始化设置

    assert CheckPort(), 'error port'     # 检查端口是否可用

    logger.info("***************************************************")
    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)    # 创建UAV路径规划环境对象的实例

    if is_dist_avail_and_initialized():    # 清理分布式设置
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)    # 路径规划模型的推理逻辑
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)                 # 辅助模块

    logger.info(f"Assist setting: always_help -- {args.always_help}, use_gt -- {args.use_gt}")
    logger.info("***************************************************")
    
    eval(model_wrapper=model_wrapper,      # 调用主评估函数
         assist=assist,
         eval_env=eval_env,
         eval_save_dir=eval_save_path)

    eval_env.delete_VectorEnvUtil()

