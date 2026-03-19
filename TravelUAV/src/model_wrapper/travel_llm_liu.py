import numpy as np
import torch
import transformers
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor
from typing import Optional, List, Dict
import base64
import re
from io import BytesIO
import cv2
from PIL import Image

# 导入OpenAI库
from openai import OpenAI

# ==== 用于API调用的辅助函数 ====


def _numpy_to_base64(image_np_rgb: np.ndarray, format="png") -> str:
    """将一个 (H, W, 3) 的 RGB NumPy 图像数组转换为 Base64 字符串"""
    buffer = BytesIO()
    Image.fromarray(image_np_rgb).save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_waypoint_from_string(text: str) -> np.ndarray:
    """从GPT-4返回的文本中解析出航点向量"""
    try:
        # 使用正则表达式寻找类似 [num, num, num] 的模式
        match = re.search(
            r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text
        )
        if match:
            x, y, z = (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )
            # 默认为一个标准前进距离
            distance = 5.0
            return np.array([x, y, z, distance])
    except (ValueError, TypeError):
        pass
    print(f"Warning: Could not parse waypoint from GPT-4 response: '{text}'")
    return np.array([0.0, 0.0, 0.0, 0.0])  # 解析失败时默认悬停


class TravelModelWrapper(BaseModelWrapper):

    # === 【修改点 1】: __init__ 方法不再加载本地LLM，而是初始化API客户端 ===
    def __init__(self, model_args, data_args):
        print("Initializing OpenAI client for GPT-4 API...")
        try:
            self.client = OpenAI()  # 会自动从环境变量 OPENAI_API_KEY 读取密钥
        except Exception as e:
            raise ImportError(
                f"Could not initialize OpenAI client. Ensure 'openai' library is installed and OPENAI_API_KEY is set. Error: {e}"
            )

        # 轨迹模型、图像处理器和DINO监视器仍然保留
        print("Loading local Trajectory Predictor model...")
        self.traj_model = load_traj_model(model_args)
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(
            model_args.image_processor
        )
        self.traj_model.to(
            dtype=torch.bfloat16, device="cuda"
        )  # 假设轨迹模型在GPU上运行

        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

    # prepare_inputs 和 run_traj_model 的定义保持不变，因为它们仍然被run方法内部逻辑所需要
    # (省略具体代码，保持你原来的版本即可)
    def prepare_inputs(
        self,
        episodes,
        target_positions,
        assist_notices=None,
        previous_waypoint_strs: Optional[list] = None,
    ):
        # 这个函数的实现保持不变，因为它负责构建Prompt文本
        # ... (Your existing code here) ...
        inputs_list, rot_to_targets_list, all_conversations_list = [], [], []
        for i in range(len(episodes)):
            prev_wp_str = previous_waypoint_strs[i] if previous_waypoint_strs else None
            input_item, rot_to_target, processed_conversations = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=None,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None,
                previous_waypoint_str=prev_wp_str,
            )
            inputs_list.append(input_item)
            rot_to_targets_list.append(rot_to_target)
            all_conversations_list.append(processed_conversations)

        preparation_logs = {
            "processed_conversations": all_conversations_list,
            "unbatched_inputs_before_device": inputs_list,  # 注意：这里的 'image' 已经是预处理过的张量
            "rotation_to_target_matrices": rot_to_targets_list,
        }
        # 对于API调用，我们不需要返回 inputs_device
        return None, preparation_logs

    def run_traj_model(self, episodes, waypoints_llm_processed, rot_to_targets):
        # 这个函数的实现保持不变
        # ... (Your existing code here) ...
        inputs = prepare_data_to_traj_model(
            episodes, waypoints_llm_processed, self.image_processor, rot_to_targets
        )
        waypoints_traj_raw = self.traj_model(inputs, None)
        refined_waypoints_world = transform_to_world(
            waypoints_traj_raw.cpu().to(dtype=torch.float32).numpy(), episodes
        )
        return refined_waypoints_world, inputs, waypoints_traj_raw

    # === 【修改点 2】: 重写 run 方法以调用 GPT-4 API ===
    def run(
        self,
        episodes: List,
        target_positions: List,
        assist_notices: Optional[List],
        thinking_budget: int = 1,
    ):
        batch_size = len(episodes)
        batch_inference_logs = [[] for _ in range(batch_size)]
        final_llm_waypoints_batch = []

        # API不支持批处理，所以我们循环处理每个样本
        for i in range(batch_size):
            previous_waypoint_strs = [None] * batch_size

            # 思考循环
            for step in range(thinking_budget):
                # 1. 准备Prompt文本和原始图像
                _, prep_logs = self.prepare_inputs(
                    [episodes[i]],
                    [target_positions[i]],
                    [assist_notices[i] if assist_notices else None],
                    previous_waypoint_strs,
                )

                # 从日志中提取当前样本的Prompt文本
                # 注意：processed_conversations 是一个列表的列表，[sample_idx][conversation_turn]
                prompt_text = prep_logs["processed_conversations"][0][0]["value"]

                # 提取最新的5个视角的原始NumPy图像
                latest_obs_images_np = episodes[i][-1]["rgb"]

                # 2. 构建API请求
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert UAV pilot. Your task is to analyze the provided multi-view images and instructions to determine the next waypoint for navigation. Output only a single 3D vector `[x, y, z]` representing the direction in the drone's local coordinate system (+x is forward, +y is right, +z is up).",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_numpy_to_base64(latest_obs_images_np[0])}"
                                },
                            },  # Front
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_numpy_to_base64(latest_obs_images_np[1])}"
                                },
                            },  # Back
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_numpy_to_base64(latest_obs_images_np[2])}"
                                },
                            },  # Left
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_numpy_to_base64(latest_obs_images_np[3])}"
                                },
                            },  # Right
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_numpy_to_base64(latest_obs_images_np[4])}"
                                },
                            },  # Down
                        ],
                    },
                ]

                # 3. 发起API调用
                try:
                    print(f"Calling GPT-4 API for task {i}, refinement step {step}...")
                    completion = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=50,
                    )
                    gpt_response_text = completion.choices[0].message.content
                except Exception as e:
                    print(f"!! GPT-4 API call failed: {e}")
                    gpt_response_text = "[0, 0, 0]"

                # 4. 解析API响应并记录日志
                llm_processed_waypoint = _parse_waypoint_from_string(gpt_response_text)

                # 注意：prep_logs是单样本的，所以直接用
                step_log = {
                    "refinement_step": step,
                    "preparation": prep_logs,
                    "gpt_4_response": gpt_response_text,
                    "llm_processed_waypoint": llm_processed_waypoint,
                }
                batch_inference_logs[i].append(step_log)

                # 准备下一次迭代
                previous_waypoint_strs = [
                    f"[{llm_processed_waypoint[0]:.2f}, {llm_processed_waypoint[1]:.2f}, {llm_processed_waypoint[2]:.2f}]"
                ]

            final_llm_waypoints_batch.append(
                batch_inference_logs[i][-1]["llm_processed_waypoint"]
            )

        # 所有样本的思考都结束后，批处理运行轨迹模型
        final_rot_to_targets_batch = [
            log[-1]["preparation"]["rotation_to_target_matrices"][0]
            for log in batch_inference_logs
        ]

        (
            refined_waypoints_world_batch,
            traj_model_inputs,
            traj_model_raw_output_batch,
        ) = self.run_traj_model(
            episodes, np.array(final_llm_waypoints_batch), final_rot_to_targets_batch
        )

        # 追加最终日志
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
        self.traj_model.eval()

    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(
                episodes[i], object_infos[i]
            )
            prediction_dones.append(prediction_done)
        return prediction_dones
