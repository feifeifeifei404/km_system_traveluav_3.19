#!/bin/bash
# change the dataset_path to your own path

# --- 1. 配置区: 在这里设置所有固定参数 ---
root_dir=. # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV
base_save_path="/mnt/mydisk/result/eval_closeloop/eval_test" # 设置一个基础保存路径

# --- 2. 实验循环区: 遍历你想测试的思考预算 ---
for budget in 1 2 3; do

    echo "======================================================"
    echo "  STARTING EVALUATION RUN with thinking_budget = $budget"
    echo "======================================================"

    # 动态创建本次运行的保存路径
    current_save_path="${base_save_path}_budget${budget}"

    python -u $root_dir/src/vlnce_src/eval.py \
        --run_type eval \
        --name "TravelLLM_Budget_${budget}" \
        --simulator_tool_port 25000 \
        --DDP_MASTER_PORT 80005 \
        --batchSize 1 \
        --gpu_id 0 \
        --always_help True \
        --use_gt True \
        --maxWaypoints 200 \
        --dataset_path /mnt/data/TravelUAV/Dataset/newdata/ \
        --eval_save_path "$current_save_path" \
        --model_path $model_dir/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32 \
        --model_base $model_dir/model_zoo/vicuna-7b-v1.5 \
        --vision_tower /mnt/data/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
        --image_processor /mnt/data/TravelUAV/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224 \
        --traj_model_path $model_dir/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4 \
        --eval_json_path $root_dir/Dataset/dataset_split/data/uav_dataset/seen_valset.json \
        --map_spawn_area_json_path $root_dir/Dataset/dataset_split/data/meta/map_spawnarea_info.json \
        --object_name_json_path $root_dir/Dataset/dataset_split/data/meta/object_description.json \
        --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
        --thinking_budget "$budget" # <--- 使用循环变量

done

echo "======================================================"
echo "All evaluation runs have been completed."
echo "======================================================"