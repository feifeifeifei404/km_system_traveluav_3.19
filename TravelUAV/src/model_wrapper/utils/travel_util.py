import copy
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence
import transformers
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
import math


sys.path.append(str(Path(str(os.getcwd())).resolve()))
sys.path.append(str(Path(__file__).resolve().parents[3] / "Model" / "LLaMA-UAV"))
from llamavid.model.builder import load_pretrained_model
from llamavid.model.vis_traj_arch import VisionTrajectoryGenerator
from peft import PeftModel
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llamavid.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    WAYPOINT_INPUT_TOKEN,
    WAYPOINT_LABEL_TOKEN,
    DEFAULT_HISTORY_TOKEN,
    DEFAULT_WP_TOKEN,
)
from llamavid import conversation as conversation_lib


# 加载多模态 LLM 模型（如 LLaMA-UAV）,并进行必要的权重加载与 tokenizer 扩展，最终返回模型及其配套组件
def load_model(args):
    model_path = os.path.expanduser(args.model_path)  # 展开模型路径变量
    model_name = get_model_name_from_path(model_path)  # 解析模型名称
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, vision_tower=args.vision_tower
    )  # 加载预训练模型，返回对象：文本分词器，LLM 模型,图像预处理器
    smarter_tokenizer_and_embedding_resize(
        special_tokens_list=["<wp>", "<his>"], tokenizer=tokenizer, model=model
    )  # 扩展 tokenizer 和 embedding，<wp>：表示“waypoint”路径点提示，<his>：表示“history”历史轨迹提示
    model.get_special_token_id(
        {
            "<wp>": tokenizer.encode("<wp>")[1],
            "<his>": tokenizer.encode("<his>")[
                1
            ],  # 显式设置特殊 token ID（用于 prompt 构建），告诉模型这些token要重点处理的标记这些 token 会被用于构建指令格式
            ",": tokenizer.encode(",")[1],
            ";": tokenizer.encode(";")[1],
        }
    )

    # 加载 LoRA 微调权重
    lora_enable = True
    if lora_enable:
        print(f"Loading LoRA weights from {model_path}")
        model = PeftModel.from_pretrained(
            model, model_path
        )  # 把主模型替换成带 LoRA 模块的结构
        # 加载额外微调参数（非 LoRA 模块）
        non_lora_weights = torch.load(
            os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
        )
        model.load_state_dict(non_lora_weights, strict=False)
        # 加载图像投影层权重,将图像特征投影到语言空间
        mm_projector_weights = torch.load(
            os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
        )
        model.load_state_dict(mm_projector_weights, strict=False)

    return tokenizer, model, image_processor


# 加载轨迹回归模型（用于 refine waypoint）
def load_traj_model(model_args):
    vision_config = generate_vision_tower_config(
        model_args.vision_tower, model_args.image_processor
    )  # 构造视觉配置 config 文件路径
    config = transformers.AutoConfig.from_pretrained(
        vision_config, trust_remote_code=True
    )  # 加载模型配置
    traj_model = VisionTrajectoryGenerator(
        config
    )  # 以刚才的 config 为参数，实例化轨迹模型
    traj_weights = torch.load(
        os.path.join(model_args.traj_model_path, "model_5.pth"), map_location="cpu"
    )  # 加载轨迹模型权重文件
    traj_weights = {
        k: v.to(torch.bfloat16) for k, v in traj_weights.items()
    }  # 转换权重为 bfloat16
    traj_model.load_state_dict(traj_weights, strict=False)  # 加载权重到模型中
    return traj_model


# 根据vision_tower图像模型权重路径和image_processor:图像预处理器路径生成 config.json 配置文件
def generate_vision_tower_config(vision_tower, image_processor):
    default_vision_config = {
        "model_type": "clip",
        "hidden_act": "silu",
        "hidden_size": 4096,
        "image_aspect_ratio": "square",
        "image_grid_pinpoints": None,
        "image_processor": "/mnt/data/TravelUAV/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224",
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "max_token": 2048,
        "mm_hidden_size": 1408,
        "mm_projector_type": "mlp2x_gelu",
        "mm_use_im_patch_token": False,
        "mm_use_im_start_end": False,
        "mm_vision_select_feature": "patch",
        "mm_vision_select_layer": -2,
        "mm_vision_tower": "/mnt/data/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth",
        "torch_dtype": "float16",
    }
    default_vision_config["image_processor"] = image_processor
    default_vision_config["mm_vision_tower"] = vision_tower
    cf_path = os.path.join(os.path.split(vision_tower)[0], "config.json")
    with open(cf_path, "w") as f:
        json.dump(default_vision_config, f, indent=2)
    return cf_path


# 为轨迹模型构造输入数据
def prepare_data_to_traj_model(
    episodes, waypoints, image_processor, rot_to_targets=None
):
    image_list = []
    target_list = []
    for i in range(len(episodes)):
        info = episodes[i]
        rot_to_target = None
        if rot_to_targets is not None:
            if rot_to_targets[i] is not None:
                rot_to_target = rot_to_targets[i]
        target = waypoints[i][0:3]
        rot_0 = info[0]["sensors"]["imu"]["rotation"]
        rot = info[-1]["sensors"]["imu"]["rotation"]
        if rot_to_target is not None:
            target = (
                np.array(rot).T
                @ np.array(rot_0)
                @ np.array(rot_to_target)
                @ np.array(target)
            )
        else:
            target = np.array(rot).T @ np.array(rot_0) @ np.array(target)
        image_list.append(info[-1]["rgb"][0])
        target_list.append(target)
    images = np.stack(image_list, axis=0)
    image = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    target = torch.tensor(np.array(target_list))

    return {"img": image, "target": target}


# 将轨迹模型输出的waypoint从相对坐标变换为世界坐标系
def transform_to_world(waypoints, episodes):
    waypoints_world = []
    for i in range(len(waypoints)):
        waypoint = waypoints[i]
        ep = episodes[i]
        pos = ep[-1]["sensors"]["state"]["position"]
        rot = ep[-1]["sensors"]["imu"]["rotation"]
        waypoint_world = np.array(rot) @ np.array(waypoint).T + np.asarray(pos).reshape(
            3, 1
        )
        waypoint_world = waypoint_world.T
        waypoints_world.append(waypoint_world)

    return waypoints_world


# 扩展模型词表，添加自定义 token
def smarter_tokenizer_and_embedding_resize(
    special_tokens_list: List,  # 一个包含我们要添加的特殊 token 的列表，例如 ['<wp>', '<his>']
    tokenizer: transformers.PreTrainedTokenizer,  # 模型当前使用的 tokenizer
    model: transformers.PreTrainedModel,  # 模型
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(
        special_tokens_list, special_tokens=True
    )  # 把 special_tokens_list 中的 token 添加到 tokenizer 的词表里，告诉 tokenizer 这是“特殊 token”，和普通词不同
    model.resize_token_embeddings(
        len(tokenizer)
    )  # 新 token 后，扩展模型的输入输出 embedding 层

    if num_new_tokens > 0:  # 确实添加了新 token（非空）
        # 获取模型的词向量矩阵
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        # 计算原始词表中所有旧 token 的平均 embedding
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        # 用平均 embedding 初始化新 token 的向量
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# 将四元数转欧拉角
def to_eularian_angles(q):
    x, y, z, w = q
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1
    if t2 < -1.0:
        t2 = -1.0
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = math.atan2(t3, t4)
    return (pitch, roll, yaw)


# 欧拉角转旋转矩阵（轨迹坐标变换）
def euler_to_rotation_matrix(e):
    rotation = R.from_euler("xyz", e, degrees=False)
    return rotation.as_matrix()


# 将当前状态投影到目标参考坐标系下（位置 & 方向变换）
def project_this_state2target_state_axis(this_state, target_state):
    start_pos = target_state["position"]
    start_eular = to_eularian_angles(target_state["orientation"])  # (pitch, roll, yaw)
    this_pos = this_state["position"]
    this_eular = to_eularian_angles(this_state["orientation"])
    delta_pos = np.asarray(this_pos) - np.asarray(start_pos)
    delta_eular = np.asarray(this_eular) - np.asarray(start_eular)
    rot = euler_to_rotation_matrix(start_eular)
    delta_pos = rot.T @ delta_pos
    return {"position": delta_pos.tolist(), "orientation": delta_eular.tolist()}


# 配置参数类
@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)  # 文本提示模板
    refine_prompt: Optional[bool] = field(default=False)  # 是否构建细化版 prompt
    mm_use_im_start_end: bool = field(default=False)


# 管理模型加载相关的参数
@dataclass
class CommonArguments:
    model_path: Optional[str] = field(default="facebook/opt-350m")
    model_base: Optional[str] = field(default=None)


# 根据配置动态地决定是否为 <image> 添加 <im_start> 和 <im_end> 边界，以让多模态模型更明确知道图像位置
def preprocess_multimodal(
    sources: Sequence[str],  # 结构化的自然语言输入，包含 image token
    data_args: DataArguments,  # 数据配置对象（如是否加图像边界标记）
    stage=None,  # 当前飞行阶段
    delta=None,  # 前一步的位移向量描述（字符串）
    cur_pos=None,  # 当前 UAV 坐标描述（字符串）
) -> Dict:
    """
    process image token's representation
    """
    for source in sources:  # 遍历多个对话输入，每个 source 是一个对话（通常是句子序列）
        for sentence in source:
            if (
                DEFAULT_IMAGE_TOKEN in sentence["value"]
            ):  # 是否包含图像 token <image>，只有包含 <image> 的句子才会被处理
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )  # 清理 <image> 占位符
                sentence["prompt"] = copy.deepcopy(
                    sentence["value"]
                )  # 把用户原始输入内容（纯语言指令）保存到 prompt 字段中备用

                # 构造完整 Prompt 格式
                sentence["value"] = (
                    "\n\nStage:"
                    + stage
                    + "\n\nPrevious displacement:"
                    + delta
                    + "\n\nCurrent position:"
                    + cur_pos
                    + "\n\nCurrent image:"
                    + DEFAULT_IMAGE_TOKEN
                    + "\n\nInstruction:"
                    + sentence["value"]
                )
                sentence["value"] = sentence["value"].strip()

                # 如果模型使用了特定的版本（如含 "mmtag" 标记的多模态模型），需要在 <image> 外层包裹 <Image></Image> 结构，引导模型识别图像起止边界
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )

            replace_token = DEFAULT_IMAGE_TOKEN  # 把替换内容设为 <image> 本身
            if (
                data_args.mm_use_im_start_end
            ):  # 用户是否开启了“使用图像边界 token”的配置
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )  # 用 <im_start><image><im_end> 替换 <image>
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )  # 将句子中所有 <image> 替换成 replace_token

    return sources


# 从局部坐标系变换到全局（世界）坐标系
def rotation_matrix_from_vector(x, y):
    v_x = np.array([x, y, 0])
    v_x = v_x / np.linalg.norm(v_x)
    v_y = np.array([-v_x[1], v_x[0], 0])
    v_y = v_y / np.linalg.norm(v_y)
    v_z = np.array([0, 0, 1])
    rotation_matrix = np.column_stack((v_x, v_y, v_z))
    return rotation_matrix


# 将一个局部坐标 point 应用旋转矩阵变换，映射到全局坐标系
def transform_point(point, rotation_matrix):
    return np.dot(point, rotation_matrix)


# 为 LLM 构建 multimodal 输入（图像+指令+历史+朝向）,prompt
# 传入：一个 UAV 的完整轨迹历史记录（含图像+传感器+指令等）；文本 tokenizer；图像处理器；数据参数配置；当前导航目标的坐标（世界坐标）；阶段标签
def prepare_data_to_inputs(
    episodes,
    tokenizer,
    image_processor,
    data_args,
    target_point,
    assist_notice=None,
    refinement_step=0,
    intermediate_waypoint=None,
):

    ori_sources = None
    input_prompt = data_args.input_prompt  # 原始的输入 prompt 模板
    refine_prompt = data_args.refine_prompt  # 后续 refine 模型使用的 prompt 模板
    sources = episodes
    ori_sources = copy.deepcopy(sources)  # 复制一份 episodes（即 UAV 的历史轨迹）
    # 倒序遍历 sources（从最新帧往回看），找到包含 'rgb' 图像的帧，该帧的图像序列放入 images
    processor = image_processor
    images = []
    for src in sources[::-1]:
        if "rgb" in src:
            images.extend(src["rgb"])
            break
    # 将图像列表转为 shape=(T, H, W, C) 的 numpy 数组
    images = np.stack(images, axis=0)
    # 用视觉处理器将图像预处理为 pixel tensor
    image = processor.preprocess(images, return_tensors="pt")["pixel_values"]

    # 构造对话 prompt（带 <image> token）
    #结合实际情况，尤其注意前方建筑物，不要总是向下飞，可以优先考虑往上飞。
    new_hint = "Based on actual conditions, pay particular attention to structures ahead. Avoid constantly flying downward; prioritize ascending flight paths instead.\n"
    conversation_for_human = (
        "<image>\n" + new_hint + sources[-1]["instruction"]
    )  # 从最后一帧中取出人类输入的指令，前面加上 <image> 表明是多模态输入


    if refinement_step > 0 and intermediate_waypoint is not None:
        # 如果是精炼步骤，添加自我修正的提示
        waypoint_str = ','.join([f"{x:.2f}" for x in intermediate_waypoint])
        reconsider_prompt = (
            f"\nWait, let's reconsider. An initial plan was to move towards [{waypoint_str}]. "
            "Based on the instruction and current view, let's re-evaluate and provide a better plan."
        )
        # 将修正提示加在原始指令之后
        conversation_for_human += reconsider_prompt


    # 构建标准的多轮对话格式，符合 Chat 模型的输入格式
    conversation = [
        {"from": "human", "value": conversation_for_human},
        {"from": "gpt", "value": ""},
    ]
    # 判断 UAV 所处阶段
    if assist_notice is not None:
        stage = (
            assist_notice  # 如果外部提供了阶段提示（如 assist_notice="cruise"），就用它
        )
    else:
        stage = (
            "cruise" if len(sources) > 20 else "take off"
        )  # 否则根据历史长度简单判断是否为起飞或巡航阶段
    # 最早一帧中提取初始位姿（rotation & position）
    rot = np.array(ori_sources[0]["sensors"]["imu"]["rotation"])
    pos = np.array(ori_sources[0]["sensors"]["state"]["position"])
    # 计算每一帧的相对位移
    deltas = []
    for source in ori_sources:
        if "rgb" not in source.keys():
            continue
        deltas.append(
            (np.array(source["sensors"]["state"]["position"]) - pos)
        )  # 提取其位置，并减去初始位置 pos，即计算相对位移向量,得到历史路径（世界坐标系下）的 delta 序列
    history_waypoint = np.array(
        [(rot.T @ delta) for delta in deltas]
    )  # delta 从世界坐标系 → UAV 自身坐标系
    rotation_to_target = None

    target_point = np.array(
        rot.T @ (target_point - pos)
    )  # 目标点先变成相对坐标，再转为 UAV 本地坐标
    # 构造一个旋转矩阵，把 x-y 轴对齐到目标方向，再次对历史轨迹做变换，让其朝向统一为面向目标点
    x, y = target_point[0], target_point[1]
    rotation_to_target = rotation_matrix_from_vector(x, y)
    history_waypoint = transform_point(history_waypoint, rotation_to_target)

    # 计算当前位置与方向 delta
    if (
        len(history_waypoint) >= 2
    ):  # 取最新两帧间的方向向量,如果历史不足两帧，默认一个负 z 方向（下降）
        delta = history_waypoint[-1] - history_waypoint[-2]
    else:
        delta = np.array([0, 0, -4.5])
    # 单位化 delta 并格式化成逗号分隔的字符串
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    delta = ",".join([str(round(x, 1)) for x in delta])
    # 当前相对位置，也转为字符串
    cur_pos = history_waypoint[-1]
    cur_pos = ",".join([str(round(x, 1)) for x in cur_pos])
    # print('stage:', stage,'delta:', delta, 'cur_pos:', cur_pos)
    # 构造结构化 prompt,传入刚才构建的对话内容，替换 <image> token，加上 Stage / Displacement / Position 等文本说明，便于模型学习，最终返回标准化的 prompt
    sources = preprocess_multimodal(
        copy.deepcopy([conversation]),
        data_args,
        stage=stage,
        delta=delta,
        cur_pos=cur_pos,
    )
    # 对处理好的 prompt 调用 tokenizer,将文本转为 token id
    data_dict = preprocess(
        sources,
        tokenizer,
        has_image=True,
        prompt=input_prompt,
        refine_prompt=refine_prompt,
    )
    if "prompt" in data_dict:
        prompt = data_dict["prompt"]
    else:
        prompt = None
    # 打包最终输入
    data_dict = dict(
        input_ids=data_dict["input_ids"][0],  # 提取 token 序列和训练标签
        labels=data_dict["labels"][0],
    )
    data_dict["image"] = image  # 添加图像和轨迹输入
    data_dict["history_waypoint"] = torch.tensor(history_waypoint).view(-1)
    # 从最初帧到当前帧，计算朝向的相对变化（返回一个 3D 方向向量）
    ori_0 = ori_sources[0]["sensors"]["state"]
    ori = ori_sources[-1]["sensors"]["state"]
    target_relative_orientation = project_this_state2target_state_axis(ori, ori_0)[
        "orientation"
    ]
    data_dict["orientation"] = torch.tensor(target_relative_orientation).view(-1)

    # 若 tokenize 过程中提供了文本 prompt，则附加存储
    if prompt is not None:
        data_dict["prompt"] = prompt

    # 将处理后的 sources (即对话内容) 作为第三个元素返回
    processed_conversations = sources

    return (
        data_dict,
        rotation_to_target,
        processed_conversations,
    )  # 多模态 LLM 模型的全部输入;将当前位置朝向旋转对齐到目标点方向的旋转矩阵


# 将多个 input instance 转为 batch 张量
def inputs_to_batch(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels = tuple(
        [instance[key] for instance in instances] for key in ("input_ids", "labels")
    )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    input_ids = input_ids[:, : tokenizer.model_max_length]
    labels = labels[:, : tokenizer.model_max_length]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

    if "image" in instances[0]:
        images = [instance["image"] for instance in instances]
        if (
            all(x is not None and x.shape == images[0].shape for x in images)
            and len(images) > 1
            and images[0].shape[-1] < 100
        ):
            batch["images"] = torch.stack(images)
        else:
            batch["images"] = images

    if "prompt" in instances[0]:
        batch["prompts"] = [
            instance["prompt"] for instance in instances
        ]  # 收集所有样本的 prompt 字符串，用于调试或生成记录

    if "history_waypoint" in instances[0]:
        batch["historys"] = [instance["history_waypoint"] for instance in instances]

    if "orientation" in instances[0]:
        batch["orientations"] = torch.stack(
            [instance["orientation"] for instance in instances]
        )

    return batch


# 将对话内容（多轮）拼接成模型输入格式
def preprocess(
    sources: Sequence[str],  # 一个包含多条对话（每条对话是多轮问答）的列表
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,  # 是否是图文对话
    prompt: str = None,  # 预设提示词
    refine_prompt: bool = False,  # 是否精细化 prompt
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith(
        "imgsp_uav"
    ):  # 如果当前的对话模板版本是 imgsp_uav 开头的，直接交由特殊处理函数 preprocess_imgsp_uav() 去处理。
        return preprocess_imgsp_uav(
            sources, tokenizer, has_image=has_image, refine_prompt=refine_prompt
        )
    # add end signal and concatenate together
    # 构造对话内容（拼接多轮问答）
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    # 对拼接好的对话文本进行分词
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    # 构造 labels（标签）：把 GPT 回答部分留下，Human 部分设置为 IGNORE_INDEX,模型不会对 Human 的部分进行 loss 计算
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(
        input_ids=input_ids, labels=targets
    )  # 模型的输入 token ,模型要预测的目标 token（非human部分）


# 将对话中的句子添加 speaker 和换行符标记，形成标准格式输入。
# header: 前置系统提示；source: 一段多轮对话数据；get_conversation: 是否拼接为最终输出字符串
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    # 定义每一轮对话的起始标记和结尾换行符
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    # 初始化最终输出字符串，以系统提示语作为开头
    conversation = header
    # 遍历每一句话,统一化角色名称："human" → Human；"gpt" → GPT；其他 → 'unknown'
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        # 构造每一轮的格式化内容
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        # 拼接到输出字符串中
        if get_conversation:
            conversation += sentence["value"]
    # 拼接到输出字符串中
    conversation += BEGIN_SIGNAL
    return conversation  # 最终返回的是拼接好的 prompt


# 对prompt字符串列表批量进行分词（tokenization）处理，并记录每个句子的 token 长度。
def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# 构建监督信号标签 label,将人类输入（human 部分）用 IGNORE_INDEX 屏蔽，只让模型学习回答（GPT 部分）
def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


# 处理 imgsp_uav 风格多模态对话任务,prompt
# sources：多轮对话的列表；tokenizer：用于编码文本；has_image：是否输入图像；img_token：图像 token 占位符；refine_prompt：是否对 prompt 进行简化处理（只提取目标物体描述）。
def preprocess_imgsp_uav(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    img_token: str = "<image>",
    refine_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()  # 初始化部分
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates,构建多轮会话和提示词
    conversations = []  # 存储格式化后的完整对话 prompt
    guided_prompt = []  # 存储 refined 的提示词（用于训练轨迹生成器时作为指导）
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human,如果首句不是人类说的，就跳过第一轮（GPT不能主动发言）
            source = source[1:]

        conv.messages = []  # 重置对话；
        img_in_text = False  # 初始化标记是否包含图像 token
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"  # 检查人机发言是否交替，避免出错

            # add guided prompt
            if role == conv.roles[0]:  # 如果是人类发言，提取原始提示内容
                guided_sent = (
                    sentence["prompt"]
                    .replace(DEFAULT_IMAGE_TOKEN, "")
                    .replace("\n", "")
                )
                if (
                    refine_prompt
                ):  # 如果启用 refine_prompt，则只保留“目标描述”，组成一个更清晰的任务指令
                    # only keep the useful part of the prompt
                    object_description = (
                        guided_sent.split("degrees from you.")[-1]
                        .replace("Please control the drone and find the target.", "")
                        .strip()
                    )
                    guided_sent = (
                        "Please pay attention to the obstacles in images and approach the object described below: "
                        + object_description
                    )

                guided_prompt.append(guided_sent)
            # check if image token in text,如果某句话含有 <image>，说明是多模态输入
            if img_token in sentence["value"]:
                img_in_text = True
            # add image token to all sentence if multimoal input,如果 human 的输入未显示图像 token，则加在开头
            if (
                role == conv.roles[0]
                and img_in_text
                and img_token not in sentence["value"]
            ):
                # randomly add image token to the beginning or end of the sentence
                img_conv = img_token + "\n" + sentence["value"]

                conv.append_message(role, img_conv)
            else:
                conv.append_message(role, sentence["value"])
        conversations.append(
            conv.get_prompt()
        )  # 将当前完整对话（多轮拼接好）加入总列表

    # Tokenize conversations,根据是否是图像任务选择处理方式，将 <image> 替换成特殊 token，再传入 tokenizer
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # add wp embedding, input_ids[-1] is </s>, 在倒数第二位插入 waypoint token，最后一位是 </s>，保持不变
    input_ids_pad_wp = torch.zeros(
        input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long
    )
    input_ids_pad_wp[:, :-2] = input_ids[:, :-1]
    input_ids_pad_wp[:, -2] = WAYPOINT_INPUT_TOKEN
    input_ids_pad_wp[:, -1] = input_ids[:, -1]

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets,遍历每条对话，屏蔽 human 内容
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        # 防止标注错位
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
    # 在标签中也插入 <WP_LABEL>Waypoint 标签 token，对应输入中的 <WP_INPUT>
    targets_pad_wp = torch.zeros(
        targets.shape[0], targets.shape[1] + 1, dtype=torch.long
    )
    targets_pad_wp[:, :-2] = targets[:, :-1]
    targets_pad_wp[:, -2] = WAYPOINT_LABEL_TOKEN
    targets_pad_wp[:, -1] = targets[:, -1]

    # input_ids: 输入序列 + waypoint token；labels: 监督目标序列（已屏蔽 human）+ waypoint label；prompt: 提取出的 refined prompt，用于 trajectory 模型训练。
    return dict(
        input_ids=input_ids_pad_wp,
        labels=targets_pad_wp,
        prompt=guided_prompt,
    )
