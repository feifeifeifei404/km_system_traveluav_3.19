import os
import sys
import subprocess

# 跑 seen_valset 中全部 Town01 任务
# 任务列表来源：seen_valset_town01_all.json

root_dir = "/mnt/data/TravelUAV"
model_dir = os.path.join(root_dir, "Model", "LLaMA-UAV")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
env.setdefault("EVAL_SAVE_DIR", os.path.join(root_dir, "result", "eval_closeloop", "eval_town01_bridge_test"))

cmd = [
    sys.executable,
    os.path.join(root_dir, "src", "vlnce_src", "eval.py"),
    "--run_type", "eval",
    "--name", "TravelLLM_Town01_Bridge_Test",
    "--simulator_tool_port", "25000",
    "--DDP_MASTER_PORT", "80005",
    "--batchSize", "1",
    "--gpu_id", "0",
    "--always_help", "True",
    "--use_gt", "True",
    "--maxWaypoints", "200",
    "--dataset_path", os.path.join(root_dir, "Dataset", "newdata"),
    "--eval_save_path", os.path.join(root_dir, "result", "eval_closeloop", "eval_town01_bridge_test"),
    "--model_path", os.path.join(model_dir, "work_dirs", "llama-vid-7b-pretrain-224-uav-full-data-lora32"),
    "--model_base", os.path.join(model_dir, "model_zoo", "vicuna-7b-v1.5"),
    "--vision_tower", os.path.join(root_dir, "Model", "LLaMA-UAV", "model_zoo", "LAVIS", "eva_vit_g.pth"),
    "--image_processor", os.path.join(root_dir, "Model", "LLaMA-UAV", "llamavid", "processor", "clip-patch14-224"),
    "--traj_model_path", os.path.join(model_dir, "work_dirs", "traj_predictor_bs_128_drop_0.1_lr_5e-4"),
    "--eval_json_path", os.path.join(root_dir, "Dataset", "dataset_split", "data", "uav_dataset", "seen_valset_town01_all.json"),
    "--map_spawn_area_json_path", os.path.join(root_dir, "Dataset", "dataset_split", "data", "meta", "map_spawnarea_info.json"),
    "--object_name_json_path", os.path.join(root_dir, "Dataset", "dataset_split", "data", "meta", "object_description.json"),
    "--groundingdino_config", os.path.join(root_dir, "src", "model_wrapper", "utils", "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"),
    "--groundingdino_model_path", os.path.join(root_dir, "src", "model_wrapper", "utils", "GroundingDINO", "groundingdino_swint_ogc.pth"),

]

print("[INFO] Running all Town01 tasks from seen_valset")
print("[INFO] Eval JSON:", os.path.join(root_dir, "Dataset", "dataset_split", "data", "uav_dataset", "seen_valset_town01_all.json"))
print("[INFO] Python:", sys.executable)
print("[INFO] CWD:", root_dir)
print("[INFO] CMD:\n ", " ".join(cmd))

subprocess.run(cmd, env=env, cwd=root_dir, check=True)
