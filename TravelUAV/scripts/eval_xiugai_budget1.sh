#!/bin/bash

# ==============================================================================
# 1. 配置区 (Configuration Section)
# ==============================================================================

# --- 请修改这里 ---
# 设置你想使用的思考预算
THINKING_BUDGET=2

# 设置一个你的本地高速存储路径 (例如 /home/your_username/cache)
LOCAL_STORAGE_PATH="/home/wuyou/local_models_cache"

# --- 无需修改 ---
root_dir=.
REMOTE_MODEL_ZOO_PATH="$root_dir/Model/LLaMA-UAV/model_zoo"
REMOTE_WORK_DIRS_PATH="$root_dir/Model/LLaMA-UAV/work_dirs"
BASE_MODEL_NAME="vicuna-7b-v1.5"
LORA_MODEL_NAME="llama-vid-7b-pretrain-224-uav-full-data-lora32"
TRAJ_MODEL_NAME="traj_predictor_bs_128_drop_0.1_lr_5e-4"

# ==============================================================================
# 2. 准备阶段 (Preparation - Copy models to local storage)
# ==============================================================================
echo "--- Preparing models on local storage: $LOCAL_STORAGE_PATH ---"
mkdir -p "$LOCAL_STORAGE_PATH"

LOCAL_BASE_MODEL_PATH="$LOCAL_STORAGE_PATH/$BASE_MODEL_NAME"
LOCAL_LORA_MODEL_PATH="$LOCAL_STORAGE_PATH/$LORA_MODEL_NAME"
LOCAL_TRAJ_MODEL_PATH="$LOCAL_STORAGE_PATH/$TRAJ_MODEL_NAME"

# 检查并复制所有模型文件 (如果本地不存在)
if [ ! -d "$LOCAL_BASE_MODEL_PATH" ]; then
    echo "Copying BASE model to local storage..."
    cp -r "$REMOTE_MODEL_ZOO_PATH/$BASE_MODEL_NAME" "$LOCAL_STORAGE_PATH/"
fi
if [ ! -d "$LOCAL_LORA_MODEL_PATH" ]; then
    echo "Copying LORA model to local storage..."
    cp -r "$REMOTE_WORK_DIRS_PATH/$LORA_MODEL_NAME" "$LOCAL_STORAGE_PATH/"
fi
if [ ! -d "$LOCAL_TRAJ_MODEL_PATH" ]; then
    echo "Copying Trajectory Predictor model to local storage..."
    cp -r "$REMOTE_WORK_DIRS_PATH/$TRAJ_MODEL_NAME" "$LOCAL_STORAGE_PATH/"
fi
echo "--- All models are ready on local storage. ---"

# ==============================================================================
# 3. 执行阶段 (Execution - Run a single evaluation)
# ==============================================================================

# 创建一个带时间戳和预算信息的、本次运行专属的保存路径
timestamp=$(date +"%Y%m%d_%H%M%S")
save_path="/mnt/mydisk/result/eval_closeloop/run_${timestamp}_budget${THINKING_BUDGET}"

echo "======================================================"
echo "  STARTING EVALUATION with thinking_budget = $THINKING_BUDGET"
echo "  Results will be saved to: $save_path"
echo "======================================================"

python -u $root_dir/src/vlnce_src/eval.py \
    --run_type eval \
    --name "TravelLLM_Budget_${THINKING_BUDGET}" \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --gpu_id 0 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /mnt/data/TravelUAV/Dataset/newdata/ \
    --eval_save_path "$save_path" \
    \
    --model_path "$LOCAL_LORA_MODEL_PATH" \
    --model_base "$LOCAL_BASE_MODEL_PATH" \
    --traj_model_path "$LOCAL_TRAJ_MODEL_PATH" \
    \
    --vision_tower /mnt/data/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor /mnt/data/TravelUAV/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224 \
    --eval_json_path $root_dir/Dataset/dataset_split/data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path $root_dir/Dataset/dataset_split/data/meta/map_spawnarea_info.json \
    --object_name_json_path $root_dir/Dataset/dataset_split/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
    \
    --thinking_budget "$THINKING_BUDGET"

echo "======================================================"
echo "Evaluation run has been completed."
echo "======================================================"