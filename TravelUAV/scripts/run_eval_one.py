import os
import sys
import subprocess

# ---- 基本路径 ----
root_dir = "/mnt/data/TravelUAV"          # 项目根目录（用绝对路径更稳）
model_dir = os.path.join(root_dir, "Model", "LLaMA-UAV")

# ---- 环境变量 ----
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
env["EVAL_SAVE_DIR"] = os.path.join(root_dir, "result", "eval_closeloop", "eval_test")

# ---- 启动命令（与原 bash 等价）----
cmd = [
    sys.executable,  # 当前解释器
    os.path.join(root_dir, "src", "vlnce_src", "eval.py"),
    "--run_type", "eval",
    "--name", "TravelLLM",
    "--simulator_tool_port", "25000",
    "--DDP_MASTER_PORT", "80005",
    "--batchSize", "1",
    "--gpu_id", "0",
    "--always_help", "True",
    "--use_gt", "True",
    "--maxWaypoints", "200",
    "--dataset_path", "/mnt/data/TravelUAV/Dataset/newdata/",
    "--eval_save_path", os.path.join(root_dir, "result", "eval_closeloop", "eval_test_one"),
    "--model_path", os.path.join(model_dir, "work_dirs", "llama-vid-7b-pretrain-224-uav-full-data-lora32"),
    "--model_base", os.path.join(model_dir, "model_zoo", "vicuna-7b-v1.5"),
    "--vision_tower", "/mnt/data/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth",
    "--image_processor", "/mnt/data/TravelUAV/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224",
    "--traj_model_path", os.path.join(model_dir, "work_dirs", "traj_predictor_bs_128_drop_0.1_lr_5e-4"),
    "--eval_json_path", os.path.join(root_dir, "Dataset", "dataset_split", "data", "uav_dataset", "one_case.json"),  # 只跑1条
    "--map_spawn_area_json_path", os.path.join(root_dir, "Dataset", "dataset_split", "data", "meta", "map_spawnarea_info.json"),
    "--object_name_json_path", os.path.join(root_dir, "Dataset", "dataset_split", "data", "meta", "object_description.json"),
    "--groundingdino_config", os.path.join(root_dir, "src", "model_wrapper", "utils", "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"),
    "--groundingdino_model_path", os.path.join(root_dir, "src", "model_wrapper", "utils", "GroundingDINO", "groundingdino_swint_ogc.pth"),
    "--use_budget_forcing",
    "--num_refinement_steps","1" 
]

print("[INFO] Python:", sys.executable)
print("[INFO] CWD:", root_dir)
print("[INFO] CMD:\n ", " ".join(cmd))

subprocess.run(cmd, env=env, cwd=root_dir, check=True)
