#!/bin/bash
# change the dataset_path to your own path

# 加载ROS2环境（用于话题通信）
source /opt/ros/humble/setup.bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "$SCRIPT_DIR/.." && pwd)"  # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV


python -u $root_dir/src/vlnce_src/eval.py \
    --run_type eval \
    --name TravelLLM \
    --simulator_tool_port 41451 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --gpu_id 0 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /mnt/data/TravelUAV/Dataset/newdata/ \
    --eval_save_path /mnt/mydisk/result/eval_closeloop/eval_test_debug \
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
    --use_budget_forcing \
    --num_parallel_thoughts 3 \
    --num_refinement_steps 1 \
    --record_dir /mnt/data/TravelUAV/readapi/debug_data
