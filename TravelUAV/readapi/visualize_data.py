"""
数据可视化工具 - 可视化拦截到的交互数据

使用方法:
    python readapi/visualize_data.py --data_file ./debug_data/episode_0000.json

功能:
1. 显示轨迹的3D路径
2. 显示模型预测的航点
3. 显示传感器数据的时间序列
4. 生成可视化报告
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，无法生成图表")
    print("安装: pip install matplotlib")


def load_episode_data(data_file):
    """加载episode数据"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_trajectory(episode_data):
    """提取轨迹数据"""
    positions = []
    timestamps = []
    
    for step_data in episode_data.get('steps', []):
        if step_data.get('type') == 'observation':
            sensors = step_data.get('sensors', {})
            position = sensors.get('position')
            if position:
                positions.append(position)
                timestamps.append(step_data.get('timestamp', ''))
    
    return np.array(positions), timestamps


def extract_waypoints(episode_data):
    """提取航点数据"""
    waypoints_llm = []
    waypoints_refined = []
    waypoints_world = []
    
    for step_data in episode_data.get('steps', []):
        if step_data.get('type') == 'model_output':
            if 'waypoints_llm' in step_data:
                waypoints_llm.append(step_data['waypoints_llm'])
            if 'waypoints_refined' in step_data:
                waypoints_refined.append(step_data['waypoints_refined'])
            if 'waypoints_world' in step_data:
                waypoints_world.append(step_data['waypoints_world'])
    
    return (
        np.array(waypoints_llm) if waypoints_llm else None,
        np.array(waypoints_refined) if waypoints_refined else None,
        np.array(waypoints_world) if waypoints_world else None
    )


def visualize_trajectory_3d(positions, waypoints_world=None, save_path=None):
    """可视化3D轨迹"""
    if not HAS_MATPLOTLIB:
        print("跳过3D轨迹可视化（matplotlib未安装）")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制实际轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='实际轨迹', marker='o', markersize=3)
    
    # 绘制起点和终点
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c='g', marker='o', s=200, label='起点')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c='r', marker='*', s=200, label='终点')
    
    # 绘制预测航点
    if waypoints_world is not None and len(waypoints_world) > 0:
        ax.scatter(waypoints_world[:, 0], waypoints_world[:, 1], waypoints_world[:, 2],
                   c='orange', marker='^', s=100, alpha=0.6, label='预测航点')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('无人机3D轨迹')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D轨迹图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_position_vs_time(positions, timestamps, save_path=None):
    """可视化位置随时间的变化"""
    if not HAS_MATPLOTLIB:
        print("跳过位置时间序列可视化（matplotlib未安装）")
        return
    
    steps = np.arange(len(positions))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(steps, positions[:, 0], 'b-', marker='o', markersize=3)
    axes[0].set_ylabel('X (m)')
    axes[0].set_title('位置随时间变化')
    axes[0].grid(True)
    
    axes[1].plot(steps, positions[:, 1], 'g-', marker='o', markersize=3)
    axes[1].set_ylabel('Y (m)')
    axes[1].grid(True)
    
    axes[2].plot(steps, positions[:, 2], 'r-', marker='o', markersize=3)
    axes[2].set_ylabel('Z (m)')
    axes[2].set_xlabel('步数')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"位置时间序列图已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary(episode_data, positions):
    """打印摘要信息"""
    print("\n" + "=" * 60)
    print("Episode 数据摘要")
    print("=" * 60)
    
    print(f"\nEpisode ID: {episode_data.get('episode_id')}")
    print(f"总步数: {episode_data.get('total_steps')}")
    print(f"成功: {episode_data.get('success')}")
    print(f"到目标距离: {episode_data.get('distance_to_goal', 'N/A')}")
    print(f"碰撞: {episode_data.get('collision', False)}")
    
    if len(positions) > 0:
        print(f"\n起点: {positions[0]}")
        print(f"终点: {positions[-1]}")
        
        # 计算总距离
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        print(f"总移动距离: {total_distance:.2f} m")
        
        # 计算直线距离
        straight_distance = np.linalg.norm(positions[-1] - positions[0])
        print(f"起终点直线距离: {straight_distance:.2f} m")
        
        if straight_distance > 0:
            efficiency = straight_distance / total_distance
            print(f"路径效率: {efficiency:.2%}")
    
    print("\n" + "=" * 60)


def analyze_model_predictions(episode_data):
    """分析模型预测"""
    print("\n" + "=" * 60)
    print("模型预测分析")
    print("=" * 60)
    
    llm_predictions = []
    refined_predictions = []
    
    for step_data in episode_data.get('steps', []):
        if step_data.get('type') == 'model_output':
            if 'waypoints_llm' in step_data:
                llm_predictions.append(step_data['waypoints_llm'])
            if 'waypoints_refined' in step_data:
                refined_predictions.append(step_data['waypoints_refined'])
    
    print(f"\n总预测步数: {len(llm_predictions)}")
    
    if llm_predictions and refined_predictions:
        llm_array = np.array(llm_predictions)
        refined_array = np.array(refined_predictions)
        
        # 计算LLM和精化模型的差异
        if len(llm_array) == len(refined_array):
            diffs = np.linalg.norm(llm_array - refined_array, axis=1)
            print(f"\nLLM vs 精化模型差异:")
            print(f"  平均差异: {np.mean(diffs):.4f} m")
            print(f"  最大差异: {np.max(diffs):.4f} m")
            print(f"  最小差异: {np.min(diffs):.4f} m")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='可视化TravelUAV交互数据')
    parser.add_argument('--data_file', type=str, required=True,
                        help='episode数据文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出图片的目录（默认为数据文件所在目录）')
    parser.add_argument('--show', action='store_true',
                        help='显示图表而不是保存')
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.data_file}")
    episode_data = load_episode_data(args.data_file)
    
    # 提取轨迹和航点
    positions, timestamps = extract_trajectory(episode_data)
    waypoints_llm, waypoints_refined, waypoints_world = extract_waypoints(episode_data)
    
    # 打印摘要
    print_summary(episode_data, positions)
    
    # 分析模型预测
    analyze_model_predictions(episode_data)
    
    # 确定输出目录
    if args.output_dir is None:
        output_dir = Path(args.data_file).parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    episode_id = episode_data.get('episode_id', 0)
    
    # 生成可视化
    if len(positions) > 0:
        if not args.show:
            trajectory_3d_path = output_dir / f'trajectory_3d_ep{episode_id:04d}.png'
            position_time_path = output_dir / f'position_time_ep{episode_id:04d}.png'
            
            visualize_trajectory_3d(positions, waypoints_world, save_path=trajectory_3d_path)
            visualize_position_vs_time(positions, timestamps, save_path=position_time_path)
        else:
            visualize_trajectory_3d(positions, waypoints_world)
            visualize_position_vs_time(positions, timestamps)
    else:
        print("\n警告: 没有找到位置数据")


if __name__ == '__main__':
    main()

