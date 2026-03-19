#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import re
import tqdm
import argparse
import logging

# ==== 固定导出配置（无需命令行） ====
SPL_THRESHOLD = 0.9
SPL_CSV_OUT = "/mnt/mydisk/111.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

def sort_key(filename):
    """Sort function for filenames based on the numeric part."""
    m = re.search(r'\d+', filename)
    return int(m.group()) if m else 0

def load_json(file_path):
    """
    Load a JSON file in a tolerant way.
    Returns parsed object or None if file is missing/empty/bad JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            s = f.read().strip()
        if not s:
            logging.warning(f"[metric] Empty JSON skipped: {file_path}")
            return None
        return json.loads(s)
    except FileNotFoundError:
        logging.warning(f"[metric] JSON not found: {file_path}")
        return None
    except Exception as e:
        logging.warning(f"[metric] Bad JSON skipped: {file_path} ({e})")
        return None

def _load_last_point(log_dir):
    """
    Read the last log file and return last position np.array([x,y,z]).
    Returns None if log_dir missing/empty/bad.
    """
    if not os.path.isdir(log_dir):
        logging.warning(f"[metric] Log dir missing: {log_dir}")
        return None
    logs = sorted(os.listdir(log_dir), key=sort_key)
    if not logs:
        logging.warning(f"[metric] No logs in: {log_dir}")
        return None
    last_log = os.path.join(log_dir, logs[-1])
    last_log_data = load_json(last_log)
    if last_log_data is None:
        return None
    try:
        return np.array(last_log_data["sensors"]["state"]["position"], dtype=float)
    except Exception as e:
        logging.warning(f"[metric] Malformed last log {last_log}: {e}")
        return None

def _load_oracle_traj(ori_info_path):
    """
    Given path to evaluation_results.json (or ori_info.json if你改回去了),
    load ori_traj_dir/merged_data.json and return the raw detailed trajectory list.
    Returns list or None.
    """
    ori_info = load_json(ori_info_path)
    if ori_info is None:
        return None
    ori_traj_dir = ori_info.get('ori_traj_dir')
    if not ori_traj_dir:
        logging.warning(f"[metric] ori_traj_dir missing in {ori_info_path}")
        return None
    merged_path = os.path.join(ori_traj_dir, 'merged_data.json')
    merged = load_json(merged_path)
    if merged is None:
        return None
    traj = merged.get('trajectory_raw_detailed')
    if not isinstance(traj, list) or len(traj) == 0:
        logging.warning(f"[metric] Empty/Bad trajectory in {merged_path}")
        return None
    return traj

def calculate_ne(path, dirs, success_dirs):
    """Calculate the NE (Normalized Error) between predicted and oracle trajectories."""
    ne_list = []
    skipped = 0

    for traj_dir in tqdm.tqdm(dirs, desc='Calculating NE'):
        # last predicted point
        log_dir = os.path.join(path, traj_dir, 'log')
        last_point = _load_last_point(log_dir)
        if last_point is None:
            skipped += 1
            continue

        # oracle last point
        ori_info_path = os.path.join(path, traj_dir, 'evaluation_results.json')
        ori_data = _load_oracle_traj(ori_info_path)
        if ori_data is None:
            skipped += 1
            continue
        try:
            ori_last_point = np.array(ori_data[-1]['position'], dtype=float)
        except Exception as e:
            logging.warning(f"[metric] Bad oracle last position in {ori_info_path}: {e}")
            skipped += 1
            continue

        ne = float(np.linalg.norm(ori_last_point - last_point))
        ne_list.append(ne)

    if ne_list:
        avg_ne = float(np.mean(np.array(ne_list, dtype=float)))
        logging.info(f"Average Normalized Error (NE): {avg_ne:.2f}")
    else:
        logging.info("Average Normalized Error (NE): N/A (no valid samples)")
    if skipped:
        logging.warning(f"[metric] NE skipped samples: {skipped}")

def _path_length_from_traj(traj):
    """
    Compute polyline length from list of dicts with 'position'.
    Returns float length; robust to short lists.
    """
    if not traj or len(traj) < 2:
        return 0.0
    pts = []
    for t in traj:
        try:
            pts.append(np.array(t['position'], dtype=float))
        except Exception:
            return 0.0
    length = 0.0
    for i in range(len(pts) - 1):
        length += float(np.linalg.norm(pts[i+1] - pts[i]))
    return max(length, 0.0)
def calculate_spl(path, dirs, success_dirs, threshold=SPL_THRESHOLD, csv_path=SPL_CSV_OUT):
    """
    Calculate the SPL (Success Path Length) per task and overall average.
    Returns list of tuples: (traj_dir, spl, pred_length, oracle_length).
    If threshold and csv_path are given, only rows with SPL > threshold are saved to CSV
    with columns [traj_dir, spl].
    """
    rows = []   # (traj_dir, spl, pred_length, oracle_length)
    skipped = 0

    for traj_dir in tqdm.tqdm(dirs, desc='Calculating SPL'):
        if traj_dir not in success_dirs:
            rows.append((traj_dir, 0.0, 0.0, 0.0))
            continue

        # predicted path length
        log_dir = os.path.join(path, traj_dir, 'log')
        if not os.path.isdir(log_dir):
            logging.warning(f"[metric] Log dir missing: {log_dir}")
            rows.append((traj_dir, 0.0, 0.0, 0.0)); skipped += 1; continue
        logs = sorted(os.listdir(log_dir), key=sort_key)
        if not logs:
            logging.warning(f"[metric] No logs in: {log_dir}")
            rows.append((traj_dir, 0.0, 0.0, 0.0)); skipped += 1; continue

        pred_length = 0.0
        pre_point = None
        valid = True
        for log in logs:
            log_data = load_json(os.path.join(log_dir, log))
            if log_data is None:
                valid = False; break
            try:
                point = np.array(log_data["sensors"]['state']['position'], dtype=float)
            except Exception:
                valid = False; break
            if pre_point is not None:
                pred_length += float(np.linalg.norm(pre_point - point))
            pre_point = point
        if not valid:
            rows.append((traj_dir, 0.0, 0.0, 0.0)); skipped += 1; continue

        # oracle path length
        ori_info_path = os.path.join(path, traj_dir, 'evaluation_results.json')
        ori_data = _load_oracle_traj(ori_info_path)
        if ori_data is None:
            rows.append((traj_dir, 0.0, pred_length, 0.0)); skipped += 1; continue

        path_length = _path_length_from_traj(ori_data)
        path_length = max(path_length - 20.0, 0.0)  # 你的修正

        denom = max(path_length, pred_length, 1e-8)
        spl = float(max(path_length / denom, 0.0))

        rows.append((traj_dir, spl, pred_length, path_length))

    # 汇总日志
    if rows:
        avg_spl = float(np.mean([r[1] for r in rows]) * 100.0)
        logging.info(f"Average Success Path Length (SPL): {avg_spl:.2f}%")
    else:
        logging.info("Average Success Path Length (SPL): N/A (no valid samples)")
    if skipped:
        logging.warning(f"[metric] SPL skipped samples: {skipped}")

    # CSV 导出：若指定阈值且给了 csv_path，则只写入 (traj_dir, spl) 且仅含 > 阈值 的行
    if csv_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            import csv
            if threshold is not None:
                to_write = [(d, s) for (d, s, _, _) in rows if s > threshold]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(["traj_dir", "spl"])
                    for d, s in to_write:
                        w.writerow([d, f"{s:.6f}"])
                logging.info(f"SPL> {threshold} rows saved to: {csv_path} (count={len(to_write)})")
            else:
                # 没有阈值时，写全量明细
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["traj_dir", "spl", "pred_length", "oracle_length"])
                    for r in rows:
                        w.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])
                logging.info(f"Per-task SPL CSV saved to: {csv_path}")
        except Exception as e:
            logging.warning(f"[metric] Failed to save SPL CSV to {csv_path}: {e}")

    return rows


def split_data(path, path_type):
    """Split the dataset into different categories based on the path type."""
    if not os.path.isdir(path):
        logging.warning(f"[metric] Analysis path not found: {path}")
        return []

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if path_type == 'full':
        return [traj_dir for traj_dir in dirs if 'record' not in traj_dir and 'dino' not in traj_dir]

    return_dirs = []
    for traj_dir in tqdm.tqdm(dirs, desc='Splitting data'):
        # 某些无效目录直接跳过
        if 'record' in traj_dir or 'dino' in traj_dir:
            continue

        ori_info_path = os.path.join(path, traj_dir, 'evaluation_results.json')
        ori_data = _load_oracle_traj(ori_info_path)
        if ori_data is None:
            continue

        path_length = _path_length_from_traj(ori_data)

        if path_type == 'easy' and path_length <= 250:
            return_dirs.append(traj_dir)
        elif path_type == 'hard' and path_length > 250:
            return_dirs.append(traj_dir)
        elif 'unseen' in path_type:
            # 依据你的定义：未见场景包含 ModularPark（可按需扩展）
            unseen_scenes = ['ModularPark', 'Carla_Town03']  # 如不需要 Town03，可删
            # 根据 ori_traj_dir 来判定
            ori_info = load_json(ori_info_path)
            if ori_info is None:
                continue
            ori_traj_dir = ori_info.get('ori_traj_dir', '')
            is_unseen_scene = any(s in ori_traj_dir for s in unseen_scenes)

            if path_type == 'unseen scene' and is_unseen_scene:
                return_dirs.append(traj_dir)
            elif path_type == 'unseen object' and (not is_unseen_scene):
                return_dirs.append(traj_dir)

    return return_dirs

def analyze_results(root_dir, analysis_list, path_type_list, spl_threshold=None, spl_csv=None):
    """Main function to analyze the results for different analysis types and path types."""
    for analysis_item in analysis_list:
        analysis_path = os.path.join(root_dir, analysis_item)
        if not os.path.exists(analysis_path):
            logging.warning(f"[metric] Analysis item path missing: {analysis_path}")
            continue
        logging.info(f"\nStarting analysis for type: {analysis_item}")

        for path_type in path_type_list:
            logging.info(f'\nAnalyzing for path type: {path_type}')
            analysis_dirs = split_data(analysis_path, path_type)

            total = len(analysis_dirs)
            success = 0
            oracle = 0
            success_dirs = []

            for traj_dir in analysis_dirs:
                # 约定通过目录名包含 success/oracle
                if 'success' in traj_dir:
                    success += 1
                    oracle += 1
                    success_dirs.append(traj_dir)
                elif 'oracle' in traj_dir:
                    oracle += 1

            sr = success / (total + 1e-8) * 100.0
            osr = oracle / (total + 1e-8) * 100.0
            logging.info(f"Success Rate (SR): {sr:.2f}%")
            logging.info(f"Oracle Success Rate (OSR): {osr:.2f}%")

            calculate_ne(analysis_path, analysis_dirs, success_dirs)
            # 传入阈值与 CSV 路径
            csv_out = None
            if spl_csv:
                # 如果用户传了目录，就自动拼个文件名；如果传了具体文件名，直接用
                if os.path.isdir(spl_csv):
                    csv_out = os.path.join(spl_csv, f"spl_{analysis_item}_{path_type}.csv")
                else:
                    # 若是单个文件名且多种 path_type，可能会覆盖；这里仍按用户意愿写入
                    csv_out = spl_csv
            calculate_spl(analysis_path, analysis_dirs, success_dirs,
                          threshold=spl_threshold, csv_path=csv_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the evaluation results for trajectory prediction.")
    parser.add_argument('--root_dir', type=str, required=True, help="The root directory of the dataset.")
    parser.add_argument('--analysis_list', type=str, nargs='+', required=True, help="List of analysis items to process.")
    parser.add_argument('--path_type_list', type=str, nargs='+', required=True, help="List of path types to analyze.")
    parser.add_argument('--spl_threshold', type=float, default=None,
                        help="Print tasks whose SPL is greater than this threshold (e.g., 0.9).")
    parser.add_argument('--spl_csv', type=str, default=None,
                        help="Path to save per-task SPL CSV. Can be a directory or a file path.")
    args = parser.parse_args()
    analyze_results(args.root_dir, args.analysis_list, args.path_type_list,
                    spl_threshold=args.spl_threshold, spl_csv=args.spl_csv)
