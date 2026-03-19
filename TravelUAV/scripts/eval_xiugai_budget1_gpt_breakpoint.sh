#!/bin/bash

# ==============================================================================
# 1. 配置区 (Configuration Section)
# ==============================================================================
THINKING_BUDGET=2
LOCAL_STORAGE_PATH="/home/wuyou/local_models_cache"
FIXED_WORKING_PATH="/mnt/mydisk/result/eval_closeloop/GPT4_CURRENT_RUN_BUDGET_${THINKING_BUDGET}"
FINAL_ARCHIVE_PATH="/mnt/mydisk/result/eval_closeloop"

# ==============================================================================
# 2. 准备阶段 (Preparation - Cache the trajectory model locally)
# ==============================================================================
echo "--- Preparing models on local storage: $LOCAL_STORAGE_PATH ---"
mkdir -p "$LOCAL_STORAGE_PATH"

root_dir=.
REMOTE_WORK_DIRS_PATH="$root_dir/Model/LLaMA-UAV/work_dirs"

# --- 【核心修复】: 修正这里的文件夹名称 ---
# 请 double check 你的 Model/LLaMA-UAV/work_dirs/ 目录下，文件夹名是否完全匹配
TRAJ_MODEL_NAME="traj_predictor_bs_128_drop_0.1_lr_5e-4"
# ----------------------------------------------

LOCAL_TRAJ_MODEL_PATH="$LOCAL_STORAGE_PATH/$TRAJ_MODEL_NAME"

if [ ! -d "$LOCAL_TRAJ_MODEL_PATH" ]; then
    echo "Copying Trajectory Predictor model from '$REMOTE_WORK_DIRS_PATH/$TRAJ_MODEL_NAME'..."
    # 增加一个检查，如果源文件不存在就报错退出
    if [ ! -d "$REMOTE_WORK_DIRS_PATH/$TRAJ_MODEL_NAME" ]; then
        echo "ERROR: Source trajectory model directory not found at '$REMOTE_WORK_DIRS_PATH/$TRAJ_MODEL_NAME'"
        exit 1
    fi
    cp -r "$REMOTE_WORK_DIRS_PATH/$TRAJ_MODEL_NAME" "$LOCAL_STORAGE_PATH/"
fi
echo "--- Local models are ready. ---"

# ==============================================================================
# 3. 执行阶段 (Execution - Run the evaluation)
# ==============================================================================
echo "======================================================"
echo "  STARTING GPT-4 EVALUATION with thinking_budget = $THINKING_BUDGET"
echo "  Working directory (for resume): $FIXED_WORKING_PATH"
echo "======================================================"

mkdir -p "$FIXED_WORKING_PATH"
export PYTHONHTTPSVERIFY=0
python -u $root_dir/src/vlnce_src/eval.py \
    --run_type eval \
    --name "GPT4_Budget_${THINKING_BUDGET}" \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --gpu_id 0 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /mnt/data/TravelUAV/Dataset/newdata/ \
    --eval_save_path "$FIXED_WORKING_PATH" \
    --traj_model_path "$LOCAL_TRAJ_MODEL_PATH" \
    --image_processor /mnt/data/TravelUAV/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224 \
    --vision_tower /mnt/data/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
    --eval_json_path $root_dir/Dataset/dataset_split/data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path $root_dir/Dataset/dataset_split/data/meta/map_spawnarea_info.json \
    --object_name_json_path $root_dir/Dataset/dataset_split/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
    --thinking_budget "$THINKING_BUDGET"

# ==============================================================================
# 4. 清理与归档阶段 (Cleanup & Archive)
# ==============================================================================
if [ $? -eq 0 ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    final_save_path="${FINAL_ARCHIVE_PATH}/run_gpt4_${timestamp}_budget${THINKING_BUDGET}"
    echo "======================================================"
    echo "  Evaluation finished successfully."
    echo "  Moving results from $FIXED_WORKING_PATH to final destination:"
    echo "  $final_save_path"
    echo "======================================================"
    mv "$FIXED_WORKING_PATH" "$final_save_path"
else
    echo "======================================================"
    echo "  ERROR: Evaluation script failed."
    echo "  Leaving temporary results in $FIXED_WORKING_PATH for inspection."
    echo "  You can re-run this script to resume."
    echo "======================================================"
fi