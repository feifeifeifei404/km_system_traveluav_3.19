
import json
import random
import shutil
import time

import cv2
import numpy as np
from utils.utils import *
from utils.logger import logger
from src.common.param import args
import torch.backends.cudnn as cudnn
from src.vlnce_src.env_uav import AirVLNENV, RGB_FOLDER, DEPTH_FOLDER
from utils.env_utils_uav import env_timing_log


def setup(dagger_it=0, manual_init_distributed_mode=False):
    if not manual_init_distributed_mode:
        init_distributed_mode()

    seed = 100 + get_rank() + dagger_it
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

def CheckPort():
    pid = FromPortGetPid(int(args.DDP_MASTER_PORT))
    if pid is not None:
        print('DDP_MASTER_PORT ({}) is being used'.format(args.DDP_MASTER_PORT))
        return False

    return True

def initialize_env(dataset_path, save_path, train_json_path, activate_maps=[]):
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, eval_json_path=train_json_path, activate_maps=activate_maps)
    return train_env

def initialize_env_eval(dataset_path, save_path, eval_json_path):
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, eval_json_path=eval_json_path)
    return train_env

def save_to_dataset_dagger(episodes, path, dagger_it, teacher_after_collision_steps):
    ori_path = path
    path_parts = ori_path.strip('/').split('/')
    map_name, seq_name = path_parts[-2], path_parts[-1]
    root_path = os.path.join(args.dagger_save_path, seq_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log'] + RGB_FOLDER + DEPTH_FOLDER
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    save_logs(episodes, root_path)
    save_images(episodes, root_path)

    ori_obj = os.path.join(ori_path, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)
    with open(os.path.join(root_path, 'dagger_info.json'), 'w') as f:
        json.dump({'teacher_after_collision_steps': teacher_after_collision_steps,
                   'map_name': map_name,
                   'seq_name': seq_name}, f)
        
def save_to_dataset_eval(episodes, path, ori_traj_dir, final_metrics=None):
    root_path = os.path.join(path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log', 'complexity'] + RGB_FOLDER + DEPTH_FOLDER
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    print(root_path)
    save_logs(episodes, root_path)
    save_images(episodes, root_path)
    save_complexity(episodes, root_path)

    ori_obj = os.path.join(ori_traj_dir, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)

    result_data = {'ori_traj_dir': ori_traj_dir}
    if final_metrics is not None:
        result_data.update(final_metrics)

    with open(os.path.join(path, 'evaluation_results.json'), 'w') as f:
        json.dump(result_data, f, indent=4)

    with open(os.path.join(path, 'ori_info.json'), 'w') as f:
        json.dump({'ori_traj_dir': ori_traj_dir}, f, indent=4)

def save_logs(episodes, trajectory_dir):
    save_logs_start = time.perf_counter()
    save_dir = os.path.join(trajectory_dir, 'log')
    for idx, episode in enumerate(episodes):
        frame_start = time.perf_counter()
        info = {'frame': idx, 'sensors': episode['sensors']}
        with open(os.path.join(save_dir, str(idx).zfill(6) + '.json'), 'w') as f:
            json.dump(info, f)
        env_timing_log(f"[TIMING][save_logs] frame={idx}: {time.perf_counter() - frame_start:.3f}s")
    env_timing_log(f"[TIMING][save_logs] total frames={len(episodes)}: {time.perf_counter() - save_logs_start:.3f}s")

def save_complexity(episodes, trajectory_dir):
    """保存每步场景复杂度分数，与五视角图像并列、按同一帧 idx 对齐。

    复杂度分数在 eval 主循环中由 ComplexityManager 计算，并挂到当时所用的那一帧
    （episode['complexity']）。这里按帧 idx 落盘成 complexity/<idx>.json，使其与
    frontcamera/<idx>.png 等五视角图像一一对应。

    只有挂了复杂度的帧（通常是每个决策步带 rgb 的关键帧）才会写文件；
    中间轨迹帧没有复杂度信息，跳过。同时额外写一个 complexity_scores.json 汇总，便于分析。
    """
    save_start = time.perf_counter()
    save_dir = os.path.join(trajectory_dir, 'complexity')
    os.makedirs(save_dir, exist_ok=True)
    summary = []
    for idx, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            continue
        comp = episode.get('complexity')
        if comp is None:
            continue
        record = {'frame': idx}
        record.update(comp)
        with open(os.path.join(save_dir, str(idx).zfill(6) + '.json'), 'w') as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        summary.append(record)
    # 轨迹级汇总
    with open(os.path.join(trajectory_dir, 'complexity_scores.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    env_timing_log(f"[TIMING][save_complexity] saved {len(summary)} frames: {time.perf_counter() - save_start:.3f}s")

def save_images(episodes, trajectory_dir):
    save_images_start = time.perf_counter()
    for idx, episode in enumerate(episodes):
        frame_start = time.perf_counter()
        if 'rgb' in episode:
            for cid, camera_name in enumerate(RGB_FOLDER):
                image = episode['rgb'][cid]
                write_start = time.perf_counter()
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)
                env_timing_log(
                    f"[TIMING][save_images] rgb frame={idx} camera={camera_name}: "
                    f"{time.perf_counter() - write_start:.3f}s"
                )
        if 'depth' in episode:
            for cid, camera_name in enumerate(DEPTH_FOLDER):
                image = episode['depth'][cid]
                write_start = time.perf_counter()
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)
                env_timing_log(
                    f"[TIMING][save_images] depth frame={idx} camera={camera_name}: "
                    f"{time.perf_counter() - write_start:.3f}s"
                )
        env_timing_log(f"[TIMING][save_images] frame={idx}: {time.perf_counter() - frame_start:.3f}s")
    env_timing_log(f"[TIMING][save_images] total frames={len(episodes)}: {time.perf_counter() - save_images_start:.3f}s")

def load_object_description():
    object_desc_dict = dict()
    with open(args.object_name_json_path, 'r') as f:
        file = json.load(f)
        for item in file:
            object_desc_dict[item['object_name']] = item['object_desc']
    return object_desc_dict


def target_distance_increasing_for_10frames(lst):
    """保持原版逻辑：最近 10 帧距离目标单调不下降，视为连续远离。"""
    if len(lst) < 10:
        return False
    sublist = lst[-10:]
    for i in range(1, len(sublist)):
        if sublist[i] < sublist[i - 1]:
            return False
    return True


def target_position_stuck_for_10frames(position_lst, distance_lst, movement_threshold=0.05, distance_improve_threshold=0.05):
    """新增诊断逻辑：最近 10 帧几乎没移动，且没有明显接近目标，视为卡死。"""
    if len(position_lst) < 10 or len(distance_lst) < 10:
        return False

    recent_positions = np.array(position_lst[-10:])
    recent_distances = distance_lst[-10:]

    step_movements = np.linalg.norm(np.diff(recent_positions, axis=0), axis=1)
    total_movement = float(np.sum(step_movements))
    distance_improvement = recent_distances[0] - recent_distances[-1]

    return total_movement < movement_threshold and distance_improvement < distance_improve_threshold

class BatchIterator:
    def __init__(self, env: AirVLNENV):
        self.env = env
    
    def __len__(self):
        return len(self.env.data)
    
    def __next__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch
    
    def __iter__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch

class DaggerBatchState:
    def __init__(self, bs, env_batchs, train_env):
        self.bs = bs
        self.episodes = [[] for _ in range(bs)]
        self.train_env = train_env
        self.skips = [False] * bs
        self.dones = [False] * bs
        self.oracle_success = [False] * bs
        self.collisions = [False] * bs
        self.need_teacher = [False] * bs
        self.back_count = [dict() for _ in range(bs)]
        self.teacher_after_collision_steps = [[] for _ in range(bs)]
        self.envs_to_pause = []
        self.paths = [b['trajectory_dir'] for b in env_batchs]
        self.target_positions = [b['object_position'] for b in env_batchs]
        object_desc_dict = load_object_description()
        self.object_infos = [object_desc_dict.get(b['object']['asset_name'].replace("AA", "")) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        
    def update_from_env_output(self, outputs, check_collision_function=None):
        observations, dones, collisions, oracle_success = [list(x) for x in zip(*outputs)]
        if check_collision_function is not None:
            collisions, dones = check_collision_function(self.episodes, observations, collisions, dones)
        for i in range(self.bs):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            if oracle_success[i]:
                dones[i] = True
        self.oracle_success = oracle_success
        self.dones = dones
        self.collisions = collisions
        return
    
    
    def check_dagger_batch_termination(self, dagger_it):
        for i in range(self.bs):
            ep = self.episodes[i]
            if not self.skips[i] and ((self.dones[i] and not self.collisions[i]) or (len(self.episodes[i]) >= args.maxWaypoints * 5 // 10 and self.collisions[i])):
                ori_path = self.paths[i]
                self.skips[i] = True
                if self.collisions[i]:
                    ep = ep[:-25]
                save_to_dataset_dagger(ep, ori_path, dagger_it, self.teacher_after_collision_steps[i])
            elif len(ep) < args.maxWaypoints * 5 // 10 and self.collisions[i] and not self.skips[i]: # the dagger is not long enough, so we don't save this data
                self.skips[i] = True
        if all(self.dones):
            return True # terminate
        return False 
    
    def dagger_step_back(self):
        # if collisions without teacher action, return to last 2 frame and move with teacher action
        for i in range(self.bs):
            if self.dones[i] or i in self.envs_to_pause:
                continue
            # If no collision occurs or no teacher intervention is required, apply ModelWrapper control.
            # If a collision occurs and teacher intervention is required, the DAgger trajectory fails, and the training ends.
            # If current step is using teacher action, disable the teacher flag and apply ModelWrapper control.
            if not self.collisions[i] and self.need_teacher[i]:
                self.need_teacher[i] = False
            elif self.collisions[i] and not self.need_teacher[i]:
                if (len(self.episodes[i]) in self.back_count[i] and self.back_count[i][len(self.episodes[i])] > 3) or sum(self.back_count[i].values()) > 30:
                    continue
                else:
                    self.back_count[i][len(self.episodes[i])] = self.back_count[i].get(len(self.episodes[i]), 0) + 1
                    self.train_env.revert2frame(i)
                    self.need_teacher[i] = True
                    self.collisions[i] = False
                    # reset the done flag caused by collision
                    self.dones[i] = False
                    if len(self.episodes[i]) > 10:
                        self.episodes[i] = self.episodes[i][0:-10]
                    else:
                        self.episodes[i] = self.episodes[i][0:1]
                    assert len(self.episodes[i]) == len(self.train_env.sim_states[i].trajectory)
                    remove_index = 0
                    for teacher_after_collision_step in self.teacher_after_collision_steps[i][::-1]:
                        if teacher_after_collision_step >= len(self.episodes[i]):
                            remove_index -= 1
                    self.teacher_after_collision_steps[i] = self.teacher_after_collision_steps[i][0: (None if remove_index==0 else remove_index)]
                    self.teacher_after_collision_steps[i].append(len(self.episodes[i]))
                    
                    
class EvalBatchState:
    def __init__(self, batch_size, env_batchs, env, assist):
        self.batch_size = batch_size
        self.eval_env = env
        self.assist = assist
        self.episodes = [[] for _ in range(batch_size)]
        self.target_positions = [b['object_position'] for b in env_batchs]
        self.object_infos = [self._get_object_info(b) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        self.ori_data_dirs = [b['trajectory_dir'] for b in env_batchs]
        self.dones = [False] * batch_size

        # 保留最终输出信息，但不让它改变原版 success / done 的状态流。
        self.termination_reasons = ["进行中 (In Progress)"] * self.batch_size
        self.failure_reasons = [None] * self.batch_size
        self.failure_signals_seen = [set() for _ in range(batch_size)]
        self.eval_start_time = time.perf_counter()

        self.predict_dones = [False] * batch_size
        self.tokens_per_step = [[] for _ in range(batch_size)]  # 记录每一步思考的 token 数
        self.llm_calls_per_step = [[] for _ in range(batch_size)]  # 记录每一步 LLM 调用次数
        self.total_steps = [0] * batch_size                     # 记录总步数
        self.final_metrics = [{} for _ in range(batch_size)]    # 存储最终要保存的所有指标
        self.step_timings = [[] for _ in range(batch_size)]
        self.collisions = [False] * batch_size
        self.success = [False] * batch_size
        self.oracle_success = [False] * batch_size
        self.oracle_hit = [False] * batch_size
        self.early_end = [False] * batch_size
        self.skips = [False] * batch_size
        self.distance_to_ends = [[] for _ in range(batch_size)]
        self.position_history = [[] for _ in range(batch_size)]
        self.envs_to_pause = []
        
        self._initialize_batch_data()

    def _get_object_info(self, batch):
        object_desc_dict = self._load_object_description()
        return object_desc_dict.get(batch['object']['asset_name'].replace("AA", ""))

    def _load_object_description(self):
        with open(args.object_name_json_path, 'r') as f:
            return {item['object_name']: item['object_desc'] for item in json.load(f)}

    def _initialize_batch_data(self):
        outputs = self.eval_env.reset()
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]
        
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            current_position = observations[i][-1]['sensors']['state']['position']
            self.position_history[i].append(current_position)
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
            if self.oracle_success[i]:
                self.oracle_hit[i] = True

    def _calculate_distance(self, observation, target_position):
        return np.linalg.norm(np.array(observation['sensors']['state']['position']) - np.array(target_position))

    def _mark_failure_reason(self, i, reason):
        """只记录失败原因；不改变原版 success / done / oracle_success / early_end 的语义。"""
        self.failure_signals_seen[i].add(reason)
        if self.failure_reasons[i] is None:
            self.failure_reasons[i] = reason
            self.termination_reasons[i] = f"失败：{reason}"

    def update_from_env_output(self, outputs):
        observations, dones_from_env, collisions_from_env, self.oracle_success = [list(x) for x in zip(*outputs)]

        collisions_from_env = list(collisions_from_env)
        dones_from_env = list(dones_from_env)
        self.collisions = list(collisions_from_env)
        self.dones = list(dones_from_env)

        depth_check_start = time.perf_counter()
        self.collisions, self.dones = self.assist.check_collision_by_depth(self.episodes, observations, self.collisions, self.dones)
        logger.info(f"[TIMING] assist.check_collision_by_depth: {time.perf_counter() - depth_check_start:.3f}s")

        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue

            for j in range(len(observations[i])):
                self.episodes[i].append(observations[i][j])

            current_position = observations[i][-1]['sensors']['state']['position']
            self.position_history[i].append(current_position)
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
            if self.oracle_success[i]:
                self.oracle_hit[i] = True

            if collisions_from_env[i]:
                self._mark_failure_reason(i, "碰撞")
                self.collisions[i] = True
                self.dones[i] = True
                continue

            if self.dones[i] and self.failure_reasons[i] is None:
                if target_position_stuck_for_10frames(self.position_history[i], self.distance_to_ends[i]):
                    self._mark_failure_reason(i, "卡死")
                elif target_distance_increasing_for_10frames(self.distance_to_ends[i]):
                    self._mark_failure_reason(i, "远离")

            if not self.dones[i]:
                if target_position_stuck_for_10frames(self.position_history[i], self.distance_to_ends[i]):
                    self.dones[i] = True
                    self._mark_failure_reason(i, "卡死")
                elif target_distance_increasing_for_10frames(self.distance_to_ends[i]):
                    self.dones[i] = True
                    self._mark_failure_reason(i, "远离")

    def get_assist_notices(self):
        return self.assist.get_assist_notice(self.episodes, self.trajs, self.object_infos, self.target_positions)

    def update_metric(self):
        # 恢复原版 early_end 逻辑：模型 stop 但距离 > 20m 时，只标记 early_end，不直接失败。
        for i in range(self.batch_size):
            if self.dones[i]:
                continue
            if self.predict_dones[i] and not self.skips[i]:
                if self.distance_to_ends[i][-1] <= 20 and not self.early_end[i]:
                    self.success[i] = True
                    self.termination_reasons[i] = "成功：模型主动结束且距离目标<=20m"
                elif self.distance_to_ends[i][-1] > 20:
                    self.early_end[i] = True
                if self.oracle_success[i] and self.early_end[i]:
                    self.dones[i] = True
                    self.termination_reasons[i] = "成功：Oracle 判定成功"
                elif self.success[i]:
                    self.dones[i] = True

    def _finalize_termination_reason(self, i):
        if self.success[i]:
            self.failure_reasons[i] = None
            self.termination_reasons[i] = "成功：模型主动结束且距离目标<=20m"
        elif self.oracle_hit[i]:
            self.termination_reasons[i] = "成功：Oracle 判定成功"
        else:
            # 失败原因限定在：碰撞 / 卡死 / 远离 / 超时。
            if self.failure_reasons[i] is None:
                if self.collisions[i]:
                    self.failure_reasons[i] = "碰撞"
                else:
                    self.failure_reasons[i] = "超时"
            self.termination_reasons[i] = f"失败：{self.failure_reasons[i]}"

    def check_batch_termination(self, t):
        for i in range(self.batch_size):
            # 恢复原版：超时仍然在 batch termination 阶段作为最后兜底。
            if t == args.maxWaypoints:
                self.dones[i] = True
                if not self.success[i] and not self.oracle_hit[i] and self.failure_reasons[i] is None:
                    self._mark_failure_reason(i, "超时")

            if self.dones[i] and not self.skips[i]:
                self.envs_to_pause.append(i)
                self._finalize_termination_reason(i)

                # 任务结束，开始整合最终指标
                self.total_steps[i] = t + 1  # 步数是从 0 开始的，所以加 1
                
                # 计算总 token 数
                total_tokens = 0
                for step_tokens in self.tokens_per_step[i]:
                    total_tokens += step_tokens.get("initial_candidates", 0)
                    ref_steps = step_tokens.get("refinement_steps", [])
                    flat_steps = []
                    for item in ref_steps:
                        if isinstance(item, list):
                            flat_steps.extend(item)
                        else:
                            flat_steps.append(item)
                    total_tokens += sum(flat_steps)

                total_llm_calls = 0
                for step_calls in self.llm_calls_per_step[i]:
                    total_llm_calls += step_calls.get("initial_candidates", 0)
                    ref_calls = step_calls.get("refinement_steps", [])
                    if isinstance(ref_calls, list):
                        total_llm_calls += sum(ref_calls)
                    elif ref_calls is not None:
                        total_llm_calls += int(ref_calls)
                      
                is_success = bool(self.success[i] or self.oracle_hit[i])
                elapsed_time_seconds = sum(
                    step_timing.get("durations", {}).get("total_step_time", 0.0)
                    for step_timing in self.step_timings[i]
                )
                failure_signals_before_success = []
                if self.oracle_hit[i] and not self.success[i]:
                    failure_signals_before_success = sorted(self.failure_signals_seen[i])
                self.final_metrics[i] = {
                    "total_steps": self.total_steps[i],
                    "termination_reason": self.termination_reasons[i],
                    "failure_reason": self.failure_reasons[i],
                    "failure_signals_before_success": failure_signals_before_success,
                    "elapsed_time_seconds": elapsed_time_seconds,
                    "is_success": is_success,
                    "success_type": "success" if self.success[i] else ("oracle" if self.oracle_hit[i] else "failure"),
                    "llm_calls": total_llm_calls,
                    "llm_calls_per_step": self.llm_calls_per_step[i],
                    "total_tokens": total_tokens,
                    "tokens_per_step": self.tokens_per_step[i]
                }

                prex = ''
                if self.success[i]:
                    prex = 'success_'
                    print(i, " has succeed!")
                elif self.oracle_hit[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")
                else:
                    print(f"{i} failed, reason: {self.failure_reasons[i]}")

                new_traj_name = prex + self.ori_data_dirs[i].split('/')[-1]
                new_traj_dir = os.path.join(args.eval_save_path, new_traj_name)
                save_to_dataset_eval(self.episodes[i], new_traj_dir, self.ori_data_dirs[i], self.final_metrics[i])
                self.skips[i] = True
                print(i, " has finished!")
        return np.array(self.skips).all()
