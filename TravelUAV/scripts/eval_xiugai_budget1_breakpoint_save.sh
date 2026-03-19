#!/bin/bash

# ==============================================================================
# 1. 配置区 (Configuration Section)
#    你只需要修改这个区域的变量
# ==============================================================================

# 设置你想使用的思考预算 (e.g., 1, 2, 3)
THINKING_BUDGET=2

# 设置一个你的本地高速存储路径 (例如 /home/your_username/cache)。
# 确保这个路径有足够的空间 (> 30GB)。
LOCAL_STORAGE_PATH="/home/wuyou/local_models_cache"

# 设置一个固定的临时工作目录。
# 这是实现“断点续跑”的关键。脚本会在这里进行读写。
FIXED_WORKING_PATH="/mnt/mydisk/result/eval_closeloop/CURRENT_RUN_BUDGET_${THINKING_BUDGET}"

# 设置最终结果的归档根目录
FINAL_ARCHIVE_PATH="/mnt/mydisk/result/eval_closeloop"

# ==============================================================================
# 2. 准备阶段 (Preparation - Copy models to local storage if not present)
# ==============================================================================
echo "--- Preparing models on local storage: $LOCAL_STORAGE_PATH ---"
mkdir -p "$LOCAL_STORAGE_PATH"

# 定义模型路径和名称
root_dir=.
REMOTE_MODEL_ZOO_PATH="$root_dir/Model/LLaMA-UAV/model_zoo"
REMOTE_WORK_DIRS_PATH="$root_dir/Model/LLaMA-UAV/work_dirs"
BASE_MODEL_NAME="vicuna-7b-v1.5"
LORA_MODEL_NAME="llama-vid-7b-pretrain-224-uav-full-data-lora32"
TRAJ_MODEL_NAME="traj_predictor_bs_128_drop_0.1_lr_5e-4"

LOCAL_BASE_MODEL_PATH="$LOCAL_STORAGE_PATH/$BASE_MODEL_NAME"
LOCAL_LORA_MODEL_PATH="$LOCAL_STORAGE_PATH/$LORA_MODEL_NAME"
LOCAL_TRAJ_MODEL_PATH="$LOCAL_STORAGE_PATH/$TRAJ_MODEL_NAME"

# 检查并复制所有模型文件
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
# 3. 执行阶段 (Execution - Run the evaluation)
# ==============================================================================
echo "======================================================"
echo "  STARTING EVALUATION with thinking_budget = $THINKING_BUDGET"
echo "  Working directory (for resume): $FIXED_WORKING_PATH"
echo "======================================================"

# 确保临时工作目录存在
mkdir -p "$FIXED_WORKING_PATH"

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
    \
    # 【关键】: eval_save_path 指向固定的临时路径，以实现断点续跑
    --eval_save_path "$FIXED_WORKING_PATH" \
    \
    # 所有模型路径都指向高速本地缓存
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

# ==============================================================================
# 4. 清理与归档阶段 (Cleanup & Archive)
# ==============================================================================

# 检查python脚本是否成功执行（$? -eq 0 表示上一个命令成功）
if [ $? -eq 0 ]; then
    # 创建一个带时间戳和预算信息的最终文件夹名
    timestamp=$(date +"%Y%m%d_%H%M%S")
    final_save_path="${FINAL_ARCHIVE_PATH}/run_${timestamp}_budget${THINKING_BUDGET}"

    echo "======================================================"
    echo "  Evaluation finished successfully."
    echo "  Moving results from $FIXED_WORKING_PATH to final destination:"
    echo "  $final_save_path"
    echo "======================================================"

    # 将临时文件夹重命名为最终的、带时间戳的文件夹
    mv "$FIXED_WORKING_PATH" "$final_save_path"
else
    echo "======================================================"
    echo "  ERROR: Evaluation script failed."
    echo "  Leaving temporary results in $FIXED_WORKING_PATH for inspection."
    echo "  You can re-run this script to resume."
    echo "======================================================"
fi```