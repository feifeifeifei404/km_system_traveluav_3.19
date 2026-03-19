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
from src.vlnce_src.closeloop_util import (
    EvalBatchState,
    BatchIterator,
    setup,
    CheckPort,
    initialize_env_eval,
    is_dist_avail_and_initialized,
)

# ==============================================================================
# 辅助函数 (HELPER FUNCTIONS)
# ==============================================================================


def _slice_batch(x, i):
    """从批量张量/数组/列表里取第 i 个样本；不是批量就原样返回"""
    if x is None:
        return None
    try:
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return x[i] if x.ndim >= 1 and x.shape[0] > i else x
        if isinstance(x, (list, tuple)):
            return x[i] if len(x) > i else x
    except (IndexError, TypeError):
        return x
    return x


def format_for_json(data):
    """递归地将数据转换为JSON兼容的格式。"""
    if isinstance(data, dict):
        return {k: format_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [format_for_json(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().float().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    else:
        return str(data)


def save_numpy_as_image(numpy_array, file_path):
    """将一个NumPy数组（通常是RGB格式）保存为图片文件。"""
    try:
        image_bgr = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image_bgr)
    except Exception as e:
        print(f"Error saving numpy array as image to {file_path}: {e}")


def interpret_action_from_local_waypoint(local_waypoint):
    """将无人机局部坐标系下的单个航点“翻译”为人类可理解的动作描述。"""
    if local_waypoint is None or len(local_waypoint) < 3:
        return "Invalid Waypoint"
    x, y, z = local_waypoint[0], local_waypoint[1], local_waypoint[2]
    move_threshold, turn_threshold = 0.5, 0.5
    actions = []
    if x > move_threshold:
        actions.append("Move Forward")
    elif x < -move_threshold:
        actions.append("Move Backward")
    if y > turn_threshold:
        actions.append("Turn Right")
    elif y < -turn_threshold:
        actions.append("Turn Left")
    if z > move_threshold:
        actions.append("Ascend (Up)")
    elif z < -move_threshold:
        actions.append("Descend (Down)")
    if not actions:
        actions.append("Hover / Fine-tune")
    return ", ".join(actions)


# ==============================================================================
# 主评估函数 (MAIN EVALUATION FUNCTION)
# ==============================================================================


def eval(
    model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir
):
    thinking_budget = getattr(args, "thinking_budget", 1)
    print(
        f"--- Running evaluation with Thinking Budget: {thinking_budget} step(s) per action ---"
    )

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

            batch_state = EvalBatchState(
                batch_size=eval_env.batch_size,
                env_batchs=env_batchs,
                env=eval_env,
                assist=assist,
            )

            pbar.update(n=eval_env.batch_size)
            per_task_step_logs = [[] for _ in range(eval_env.batch_size)]

            for t in range(int(args.maxWaypoints) + 1):
                logger.info(
                    "Step: {} \t Completed: {} / {}".format(
                        t, int(eval_env.index_data) - int(eval_env.batch_size), end_iter
                    )
                )

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break

                refined_waypoints, inference_steps_logs = model_wrapper.run(
                    episodes=batch_state.episodes,
                    target_positions=batch_state.target_positions,
                    assist_notices=batch_state.get_assist_notices(),
                    thinking_budget=thinking_budget,
                )

                eval_env.makeActions(refined_waypoints)
                outputs = eval_env.get_obs()
                batch_state.update_from_env_output(outputs)
                batch_state.predict_dones = model_wrapper.predict_done(
                    batch_state.episodes, batch_state.object_infos
                )
                batch_state.update_metric()

                for i in range(batch_state.batch_size):
                    if batch_state.skips[i]:
                        continue

                    task_id = batch_state.ori_data_dirs[i].split("/")[-1]
                    step_image_dir = os.path.join(
                        detailed_log_dir, task_id, f"step_{t:03d}"
                    )
                    os.makedirs(step_image_dir, exist_ok=True)

                    inference_log_for_sample = _slice_batch(inference_steps_logs, i)

                    if inference_log_for_sample:
                        mean, std = np.array(
                            [0.48145466, 0.4578275, 0.40821073]
                        ), np.array([0.26862954, 0.26130258, 0.27577711])
                        prep_log_first_step = inference_log_for_sample[0]["preparation"]

                        mllm_input_images_tensor = prep_log_first_step[
                            "unbatched_inputs_before_device"
                        ]["image"]
                        if mllm_input_images_tensor.dim() == 4:  # B, C, H, W
                            mllm_input_images_np = (
                                mllm_input_images_tensor.permute(0, 2, 3, 1)
                                .cpu()
                                .numpy()
                            )
                        else:  # C, H, W
                            mllm_input_images_np = (
                                mllm_input_images_tensor.permute(1, 2, 0)
                                .cpu()
                                .numpy()[None, ...]
                            )

                        mllm_input_images_np = np.clip(
                            (mllm_input_images_np * std + mean) * 255, 0, 255
                        ).astype(np.uint8)

                        view_names = ["front", "back", "left", "right", "down"]
                        for view_idx, view_name in enumerate(view_names):
                            if view_idx < mllm_input_images_np.shape[0]:
                                img_path = os.path.join(
                                    step_image_dir, f"mllm_input_view_{view_name}.png"
                                )
                                save_numpy_as_image(
                                    mllm_input_images_np[view_idx], img_path
                                )

                        final_inference_step = inference_log_for_sample[-1]
                        if (
                            "traj_model_input" in final_inference_step
                            and "img" in final_inference_step["traj_model_input"]
                        ):
                            traj_input_tensor = final_inference_step[
                                "traj_model_input"
                            ]["img"]

                            # --- 【核心修复】: 增加维度检查 ---
                            if traj_input_tensor.dim() == 4:  # Batch, C, H, W
                                traj_input_np = (
                                    traj_input_tensor.permute(0, 2, 3, 1)
                                    .cpu()
                                    .numpy()[0]
                                )
                            elif traj_input_tensor.dim() == 3:  # C, H, W
                                traj_input_np = (
                                    traj_input_tensor.permute(1, 2, 0).cpu().numpy()
                                )
                            else:
                                traj_input_np = None  # 无法处理的维度

                            if traj_input_np is not None:
                                traj_input_np = np.clip(
                                    (traj_input_np * std + mean) * 255, 0, 255
                                ).astype(np.uint8)
                                traj_img_path = os.path.join(
                                    step_image_dir, "traj_model_input_front.png"
                                )
                                save_numpy_as_image(traj_input_np, traj_img_path)

                        for ref_step_log in inference_log_for_sample:
                            if (
                                "image"
                                in ref_step_log["preparation"][
                                    "unbatched_inputs_before_device"
                                ]
                            ):
                                del ref_step_log["preparation"][
                                    "unbatched_inputs_before_device"
                                ]["image"]
                            if (
                                "traj_model_input" in ref_step_log
                                and "img" in ref_step_log["traj_model_input"]
                            ):
                                del ref_step_log["traj_model_input"]["img"]

                    action_description = "No Action"
                    if (
                        inference_log_for_sample
                        and "traj_model_raw_output" in inference_log_for_sample[-1]
                    ):
                        local_waypoints = inference_log_for_sample[-1][
                            "traj_model_raw_output"
                        ]
                        action_description = interpret_action_from_local_waypoint(
                            local_waypoints[0].tolist()
                        )

                    step_log = {
                        "step": t,
                        "action_description": action_description,
                        "inference_steps": inference_log_for_sample,
                        "outcome": {
                            "is_collision": batch_state.collisions[i],
                            "is_success": batch_state.success[i],
                            "is_oracle_success": batch_state.oracle_success[i],
                            "distance_to_target": batch_state.distance_to_ends[i][-1],
                            "termination_reason": batch_state.termination_reasons[i],
                        },
                    }
                    per_task_step_logs[i].append(step_log)

            for i in range(len(batch_state.ori_data_dirs)):
                task_id = batch_state.ori_data_dirs[i].split("/")[-1]
                log_file_path = os.path.join(
                    detailed_log_dir, f"{task_id}_details.json"
                )
                with open(log_file_path, "w") as f:
                    json.dump(format_for_json(per_task_step_logs[i]), f, indent=2)
                print(f"Saved detailed logs for task {task_id} to {log_file_path}")

        try:
            pbar.close()
        except:
            pass


# ==============================================================================
# 主程序入口 (MAIN ENTRY POINT)
# ==============================================================================

if __name__ == "__main__":
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)

    setup()
    assert CheckPort(), "error port"

    print("***************************************************")
    eval_env = initialize_env_eval(
        dataset_path=dataset_path,
        save_path=eval_save_path,
        eval_json_path=eval_json_path,
    )

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print(
        "Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt
    )
    print("***************************************************")

    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        eval_save_dir=eval_save_path,
    )

    eval_env.delete_VectorEnvUtil()
