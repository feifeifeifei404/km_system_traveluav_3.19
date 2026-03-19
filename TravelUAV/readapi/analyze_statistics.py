"""
统计分析工具 - 分析多个episode的统计信息

使用方法:
    python readapi/analyze_statistics.py --data_dir ./debug_data

功能:
1. 统计成功率、碰撞率等指标
2. 分析模型推理时间
3. 分析轨迹质量
4. 生成统计报告
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict


def load_all_episodes(data_dir):
    """加载所有episode数据"""
    data_dir = Path(data_dir)
    episode_files = sorted(data_dir.glob('episode_*.json'))
    
    episodes = []
    for file_path in episode_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            episodes.append(json.load(f))
    
    return episodes


def compute_basic_statistics(episodes):
    """计算基础统计信息"""
    total = len(episodes)
    if total == 0:
        return {}
    
    success_count = sum(1 for ep in episodes if ep.get('success', False))
    collision_count = sum(1 for ep in episodes if ep.get('collision', False))
    
    total_steps = [ep.get('total_steps', 0) for ep in episodes]
    distances = [ep.get('distance_to_goal') for ep in episodes if ep.get('distance_to_goal') is not None]
    
    stats = {
        'total_episodes': total,
        'success_count': success_count,
        'success_rate': success_count / total,
        'collision_count': collision_count,
        'collision_rate': collision_count / total,
        'avg_steps': np.mean(total_steps),
        'std_steps': np.std(total_steps),
        'min_steps': np.min(total_steps),
        'max_steps': np.max(total_steps),
    }
    
    if distances:
        stats['avg_distance_to_goal'] = np.mean(distances)
        stats['std_distance_to_goal'] = np.std(distances)
    
    return stats


def compute_trajectory_statistics(episodes):
    """计算轨迹统计信息"""
    trajectory_stats = {
        'total_distance': [],
        'straight_distance': [],
        'path_efficiency': [],
        'num_waypoints': []
    }
    
    for episode in episodes:
        positions = []
        for step_data in episode.get('steps', []):
            if step_data.get('type') == 'observation':
                sensors = step_data.get('sensors', {})
                position = sensors.get('position')
                if position:
                    positions.append(position)
        
        if len(positions) > 1:
            positions = np.array(positions)
            
            # 总移动距离
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            total_dist = np.sum(distances)
            trajectory_stats['total_distance'].append(total_dist)
            
            # 直线距离
            straight_dist = np.linalg.norm(positions[-1] - positions[0])
            trajectory_stats['straight_distance'].append(straight_dist)
            
            # 路径效率
            if total_dist > 0:
                efficiency = straight_dist / total_dist
                trajectory_stats['path_efficiency'].append(efficiency)
        
        # 航点数量
        waypoint_count = sum(1 for step in episode.get('steps', []) 
                             if step.get('type') == 'model_output')
        trajectory_stats['num_waypoints'].append(waypoint_count)
    
    # 计算统计量
    summary = {}
    for key, values in trajectory_stats.items():
        if values:
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
    
    return summary


def compute_model_statistics(episodes):
    """计算模型相关统计"""
    model_stats = {
        'llm_refined_diff': [],
        'predictions_per_episode': []
    }
    
    for episode in episodes:
        llm_predictions = []
        refined_predictions = []
        
        for step_data in episode.get('steps', []):
            if step_data.get('type') == 'model_output':
                if 'waypoints_llm' in step_data:
                    llm_predictions.append(step_data['waypoints_llm'])
                if 'waypoints_refined' in step_data:
                    refined_predictions.append(step_data['waypoints_refined'])
        
        model_stats['predictions_per_episode'].append(len(llm_predictions))
        
        if llm_predictions and refined_predictions:
            llm_array = np.array(llm_predictions)
            refined_array = np.array(refined_predictions)
            
            if len(llm_array) == len(refined_array):
                diffs = np.linalg.norm(llm_array - refined_array, axis=1)
                model_stats['llm_refined_diff'].extend(diffs.tolist())
    
    # 计算统计量
    summary = {}
    for key, values in model_stats.items():
        if values:
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
    
    return summary


def print_statistics_report(basic_stats, trajectory_stats, model_stats):
    """打印统计报告"""
    print("\n" + "=" * 70)
    print(" " * 25 + "统计分析报告")
    print("=" * 70)
    
    # 基础统计
    print("\n【基础统计】")
    print("-" * 70)
    print(f"  总Episode数: {basic_stats.get('total_episodes', 0)}")
    print(f"  成功数: {basic_stats.get('success_count', 0)}")
    print(f"  成功率: {basic_stats.get('success_rate', 0):.2%}")
    print(f"  碰撞数: {basic_stats.get('collision_count', 0)}")
    print(f"  碰撞率: {basic_stats.get('collision_rate', 0):.2%}")
    print(f"\n  平均步数: {basic_stats.get('avg_steps', 0):.2f} ± {basic_stats.get('std_steps', 0):.2f}")
    print(f"  步数范围: [{basic_stats.get('min_steps', 0)}, {basic_stats.get('max_steps', 0)}]")
    
    if 'avg_distance_to_goal' in basic_stats:
        print(f"\n  平均到目标距离: {basic_stats['avg_distance_to_goal']:.2f} ± "
              f"{basic_stats.get('std_distance_to_goal', 0):.2f} m")
    
    # 轨迹统计
    print("\n【轨迹统计】")
    print("-" * 70)
    if 'total_distance_mean' in trajectory_stats:
        print(f"  总移动距离: {trajectory_stats['total_distance_mean']:.2f} ± "
              f"{trajectory_stats.get('total_distance_std', 0):.2f} m")
        print(f"  直线距离: {trajectory_stats.get('straight_distance_mean', 0):.2f} ± "
              f"{trajectory_stats.get('straight_distance_std', 0):.2f} m")
        print(f"  路径效率: {trajectory_stats.get('path_efficiency_mean', 0):.2%} ± "
              f"{trajectory_stats.get('path_efficiency_std', 0):.2%}")
    
    if 'num_waypoints_mean' in trajectory_stats:
        print(f"\n  平均航点数: {trajectory_stats['num_waypoints_mean']:.2f} ± "
              f"{trajectory_stats.get('num_waypoints_std', 0):.2f}")
        print(f"  航点数范围: [{trajectory_stats.get('num_waypoints_min', 0):.0f}, "
              f"{trajectory_stats.get('num_waypoints_max', 0):.0f}]")
    
    # 模型统计
    print("\n【模型统计】")
    print("-" * 70)
    if 'predictions_per_episode_mean' in model_stats:
        print(f"  每episode预测次数: {model_stats['predictions_per_episode_mean']:.2f} ± "
              f"{model_stats.get('predictions_per_episode_std', 0):.2f}")
    
    if 'llm_refined_diff_mean' in model_stats:
        print(f"\n  LLM vs 精化模型差异:")
        print(f"    平均: {model_stats['llm_refined_diff_mean']:.4f} m")
        print(f"    标准差: {model_stats.get('llm_refined_diff_std', 0):.4f} m")
        print(f"    范围: [{model_stats.get('llm_refined_diff_min', 0):.4f}, "
              f"{model_stats.get('llm_refined_diff_max', 0):.4f}] m")
    
    print("\n" + "=" * 70)


def save_statistics(basic_stats, trajectory_stats, model_stats, output_file):
    """保存统计结果到JSON文件"""
    all_stats = {
        'basic': basic_stats,
        'trajectory': trajectory_stats,
        'model': model_stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计结果已保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='统计分析TravelUAV交互数据')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出统计结果的JSON文件（可选）')
    args = parser.parse_args()
    
    print("=" * 70)
    print(" " * 20 + "TravelUAV 统计分析工具")
    print("=" * 70)
    print(f"\n加载数据目录: {args.data_dir}")
    
    # 加载所有episode
    episodes = load_all_episodes(args.data_dir)
    print(f"找到 {len(episodes)} 个episode")
    
    if len(episodes) == 0:
        print("\n错误: 未找到任何episode数据")
        return
    
    # 计算统计信息
    print("\n计算统计信息...")
    basic_stats = compute_basic_statistics(episodes)
    trajectory_stats = compute_trajectory_statistics(episodes)
    model_stats = compute_model_statistics(episodes)
    
    # 打印报告
    print_statistics_report(basic_stats, trajectory_stats, model_stats)
    
    # 保存结果
    if args.output:
        save_statistics(basic_stats, trajectory_stats, model_stats, args.output)


if __name__ == '__main__':
    main()

