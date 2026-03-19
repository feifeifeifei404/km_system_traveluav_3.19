import numpy as np
import torch
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor


# 把大模型（LLM）和轨迹回归模型（Trajectory Model）组合起来使用，完成路径规划任务的推理流程
class TravelModelWrapper(BaseModelWrapper):

    # 构造函数 __init__
    def __init__(self, model_args, data_args):
        self.tokenizer, self.model, self.image_processor = load_model(
            model_args
        )  # 加载语言模型部分（LLM）
        self.traj_model = load_traj_model(
            model_args
        )  # 加载轨迹模型（Trajectory Model）
        

         # === 结构探测：LLM 回归头/waypoint 相关 ===
        print("== Model modules with 'waypoint' / 'head' ==")
        for n, m in self.model.named_modules():
            lname = n.lower()
            if any(k in lname for k in ["waypoint", "head", "lm_head", "regression"]):
                if hasattr(m, "out_features"):
                    print(n, type(m).__name__, "out_features=", getattr(m, "out_features", None))
                else:
                    print(n, type(m).__name__)
        # 常见配置字段
        print("num_waypoints:", getattr(self.model, "num_waypoints", None))
        print("waypoint_dim:", getattr(self.model, "waypoint_dim", None))
        cfg = getattr(self.model, "config", None)
        print("config.num_waypoints:", getattr(cfg, "num_waypoints", None))

        # 若上面没抓到 out_features，再兜底找“最后一个 Linear”的 out_features
        try:
            last_linear = None
            for n, m in self.model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    last_linear = (n, m)
            if last_linear is not None:
                n, m = last_linear
                print(f"[fallback] last nn.Linear => {n} out_features={m.out_features}")
        except Exception as e:
            print("[fallback] scan nn.Linear failed:", repr(e))


            
        self.model.to(torch.bfloat16)  # 设置精度与设备
        self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

    # 将一个 batch的UAV任务数据（episode + 目标点 + 可选提示）转换成模型输入张量格式
    def prepare_inputs(
        self,
        episodes,
        target_positions,
        assist_notices=None,
        refinement_step=0,
        intermediate_waypoint=None,
    ):
        inputs = []
        rot_to_targets = []
        # === 新增：本 batch 的提示词/GT 收集器 ===
        _debug_prompts = []
        all_conversations = []  # <--- 修改点 1: 新增一个列表来收集对话

        # 遍历每条轨迹，调用 prepare_data_to_inputs,episode（含轨迹、图像、目标点、提示等）转换为一个 dict
        for i in range(len(episodes)):
            input_item, rot_to_target, processed_conversations = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None,
                
                refinement_step=refinement_step,
                intermediate_waypoint=intermediate_waypoint,
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)
            all_conversations.append(
                processed_conversations
            )  # <--- 修改点 2: 将对话内容追加到列表中

            # === 新增：提取“提示词”和 GT（目标点），同时记录辅助提示 ===
            prompt_text = None
            for cand in (
                "prompts",
                "prompt",
                "text",
                "input_text",
                "dialog",
                "messages",
            ):
                if cand in input_item and isinstance(
                    input_item[cand], (str, list, dict)
                ):
                    prompt_text = input_item[cand]
                    break
            if prompt_text is None:
                # 兜底：把所有字符串字段打一份快照，便于事后排查
                prompt_text = {
                    k: v for k, v in input_item.items() if isinstance(v, str)
                }

            tgt = target_positions[i]
            if hasattr(tgt, "tolist"):
                tgt = tgt.tolist()

            _debug_prompts.append(
                {
                    "episode_idx": i,
                    "prompt": prompt_text,
                    "assist_notice": (
                        assist_notices[i] if assist_notices is not None else None
                    ),
                    "gt_target": tgt,
                }
            )

        # 将 N 条数据组成 batch
        batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)
        # 送入模型前，将其送到对应 device 上
        inputs_device = {
            k: v.to(self.model.device)
            for k, v in batch.items()
            if "prompts" not in k and "images" not in k and "historys" not in k
        }
        inputs_device["prompts"] = [item for item in batch["prompts"]]
        inputs_device["images"] = [
            item.to(self.model.device) for item in batch["images"]
        ]
        inputs_device["historys"] = [
            item.to(device=self.model.device, dtype=self.model.dtype)
            for item in batch["historys"]
        ]
        inputs_device["orientations"] = inputs_device["orientations"].to(
            dtype=self.model.dtype
        )
        inputs_device["return_waypoints"] = True
        inputs_device["use_cache"] = False

        # === 新增：把本 batch 的提示词快照挂到 wrapper 上，给 eval() 外层记录用 ===
        self._last_debug_prompts = _debug_prompts

        return inputs_device, rot_to_targets, all_conversations, inputs

    # 运行 LLM 得到粗略 waypoint.给出方向
    def run_llm_model(self, inputs):
        waypoints_llm = (
            self.model(**inputs).cpu().to(dtype=torch.float32).numpy()
        )  # 调用模型获得LLM输出
        # 初始化新列表用于保存规范化后的 waypoint
        waypoints_llm_new = []
        # 遍历每个输出 waypoint 进行归一化与缩放
        for waypoint in waypoints_llm:
            waypoint_new = (
                waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
            )
            waypoints_llm_new.append(waypoint_new)
        return np.array(waypoints_llm_new)

    # 轨迹模型进一步优化,调整/平滑/合理化这些方向为更真实轨迹
    def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets):
        inputs = prepare_data_to_traj_model(
            episodes, waypoints_llm_new, self.image_processor, rot_to_targets
        )  # 构造轨迹模型的输入数据：当前批次的 UAV 任务信息，来自 LLM 的粗略 waypoint 向量，图像处理模块，表示当前位置与目标位置之间的朝向（rotation 向量）
        waypoints_traj = self.traj_model(inputs, None)  # 调用轨迹回归模型进行推理
        refined_waypoints = waypoints_traj.cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = transform_to_world(
            refined_waypoints, episodes
        )  # 转换为世界坐标系
        return refined_waypoints, inputs, waypoints_traj

    def eval(self):  # 让 self.model（语言模型） 和 self.traj_model（轨迹回归模型）都进入评估状态，从而关闭 dropout、batchnorm 等训练专用机制，使模型在推理时表现稳定
        self.model.eval()
        self.traj_model.eval()

    # 统一封装一次完整推理
    def run(self, inputs, episodes, rot_to_targets):  # inputs：输入数据；episodes：UAV任务的轨迹数据；rot_to_targets：表示当前轨迹点与目标点之间的旋转信息（用于计算朝向
        waypoints_llm_new = self.run_llm_model(inputs)# 将输入数据传入语言模型（LLM）进行推理，获得初步的路径点 waypoints_llm_new
        refined_waypoints, traj_model_inputs, waypoints_traj = self.run_traj_model(episodes, waypoints_llm_new, rot_to_targets)  # 将 waypoints_llm_new与其他任务信息（episodes 和 rot_to_targets）一起传入轨迹模型进行进一步优化，生成更精确的路径点

        # 3. 将所有需要保存的中间变量打包到一个字典中
        intermediate_outputs = {
            "waypoints_llm_new": waypoints_llm_new,
            "Img_input_for_traj_model": traj_model_inputs.get("img"),  # 使用 .get() 更安全
            "Target_input_for_traj_model": traj_model_inputs.get("target"),
            "waypoints_traj_output": waypoints_traj,
            "refined_waypoints_final": refined_waypoints,
        }

        return refined_waypoints, intermediate_outputs

    # 判断任务是否完成
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
