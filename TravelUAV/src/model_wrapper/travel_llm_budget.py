import numpy as np
import torch
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor
from typing import Optional, List, Dict

# ==== Prompt / I-O Tracer ====
import os, json, time

class _PromptTracer:
    def __init__(self, save_dir=None):
        save_dir = save_dir or "./_traces"
        os.makedirs(save_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(save_dir, f"prompt_traces_{ts}.jsonl")
    def log(self, rec: dict):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print("[TRACER] write fail:", e)


class TravelModelWrapper(BaseModelWrapper):

    def __init__(self, model_args, data_args):
        self.tokenizer, self.model, self.image_processor = load_model(model_args)
        self.traj_model = load_traj_model(model_args)
        self.model.to(torch.bfloat16)
        self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args
        self.tracer = _PromptTracer(os.getenv("EVAL_SAVE_DIR"))

    def prepare_inputs(
        self,
        episodes,
        target_positions,
        assist_notices=None,
        previous_waypoint_strs: Optional[list] = None,
    ):
        inputs_list, rot_to_targets_list, all_conversations_list = [], [], []

        for i in range(len(episodes)):
            prev_wp_str = previous_waypoint_strs[i] if previous_waypoint_strs else None
            input_item, rot_to_target, processed_conversations = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None,
                previous_waypoint_str=prev_wp_str,
            )
            inputs_list.append(input_item)
            rot_to_targets_list.append(rot_to_target)
            all_conversations_list.append(processed_conversations)

        batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs_list)
        inputs_device = {
            k: v.to(self.model.device)
            for k, v in batch.items()
            if k not in ["prompts", "images", "historys"]
        }
        inputs_device.update(
            {
                "prompts": [item for item in batch["prompts"]],
                "images": [item.to(self.model.device) for item in batch["images"]],
                "historys": [
                    item.to(device=self.model.device, dtype=self.model.dtype)
                    for item in batch["historys"]
                ],
                "orientations": batch["orientations"].to(dtype=self.model.dtype),
                "return_waypoints": True,
                "use_cache": False,
            }
        )

        preparation_logs = {
            "processed_conversations": all_conversations_list,
            "unbatched_inputs_before_device": inputs_list,
            "rotation_to_target_matrices": rot_to_targets_list,
        }
        return inputs_device, preparation_logs

    def run_llm_model(self, inputs):
        waypoints_llm_raw = self.model(**inputs)
        waypoints_llm_cpu = waypoints_llm_raw.cpu().to(dtype=torch.float32).numpy()
        waypoints_llm_processed = []
        for waypoint in waypoints_llm_cpu:
            waypoint_new = (
                waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
            )
            waypoints_llm_processed.append(waypoint_new)
        return np.array(waypoints_llm_processed), waypoints_llm_raw

    def run_traj_model(self, episodes, waypoints_llm_processed, rot_to_targets):
        inputs = prepare_data_to_traj_model(
            episodes, waypoints_llm_processed, self.image_processor, rot_to_targets
        )
        waypoints_traj_raw = self.traj_model(inputs, None)
        refined_waypoints_world = transform_to_world(
            waypoints_traj_raw.cpu().to(dtype=torch.float32).numpy(), episodes
        )
        return refined_waypoints_world, inputs, waypoints_traj_raw

    # === 【核心修复】: 重写 run 方法以正确处理批处理 ===
    def run(
        self,
        episodes: List,
        target_positions: List,
        assist_notices: Optional[List],
        thinking_budget: int = 1,
    ):
        batch_size = len(episodes)
        final_refined_waypoints_world_batch = []
        batch_inference_logs = [[] for _ in range(batch_size)]  # 每个样本一个日志列表

        # MLLM的输出是整个批次的，我们需要按样本处理
        previous_waypoint_strs = [None] * batch_size

        for step in range(thinking_budget):
            current_inputs, current_prep_logs = self.prepare_inputs(
                episodes, target_positions, assist_notices, previous_waypoint_strs
            )
            llm_processed_waypoint, llm_raw_output = self.run_llm_model(current_inputs)

            # 为批次中的每个样本记录日志并准备下一次迭代
            for i in range(batch_size):
                step_log = {
                    "refinement_step": step,
                    "preparation": {k: v[i] for k, v in current_prep_logs.items()},
                    "llm_raw_output": llm_raw_output[i],
                    "llm_processed_waypoint": llm_processed_waypoint[i],
                }
                batch_inference_logs[i].append(step_log)

                # 更新下一次迭代的 "previous_waypoint_str"
                waypoint_to_pass = llm_processed_waypoint[i]
                previous_waypoint_strs[i] = (
                    f"[{waypoint_to_pass[0]:.2f}, {waypoint_to_pass[1]:.2f}, {waypoint_to_pass[2]:.2f}]"
                )

        # 轨迹模型也是批处理的
        final_llm_waypoints_batch = np.array(
            [log[-1]["llm_processed_waypoint"] for log in batch_inference_logs]
        )
        final_rot_to_targets_batch = [
            log[-1]["preparation"]["rotation_to_target_matrices"]
            for log in batch_inference_logs
        ]

        (
            refined_waypoints_world_batch,
            traj_model_inputs,
            traj_model_raw_output_batch,
        ) = self.run_traj_model(
            episodes, final_llm_waypoints_batch, final_rot_to_targets_batch
        )

        # 将最终日志追加到每个样本的日志列表中
        for i in range(batch_size):
            batch_inference_logs[i][-1]["traj_model_input"] = {
                k: v[i] for k, v in traj_model_inputs.items()
            }
            batch_inference_logs[i][-1]["traj_model_raw_output"] = (
                traj_model_raw_output_batch[i]
            )
            batch_inference_logs[i][-1]["final_refined_waypoints_world"] = (
                refined_waypoints_world_batch[i]
            )

        return refined_waypoints_world_batch, batch_inference_logs

    def eval(self):
        self.model.eval()
        self.traj_model.eval()

    def predict_done(self, episodes, object_infos):
        # ... (此函数不变) ...
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(
                episodes[i], object_infos[i]
            )
            prediction_dones.append(prediction_done)
        return prediction_dones
