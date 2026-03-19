"""
快速入门示例 - 展示如何使用数据拦截和分析工具

这个脚本演示:
1. 如何创建模拟的交互数据
2. 如何记录数据
3. 如何可视化和分析数据

使用方法:
    python readapi/quick_start.py
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime


def create_sample_observation(step, base_position):
    """创建模拟的观测数据"""
    # 模拟无人机在移动
    position = [
        base_position[0] + step * 1.5 + np.random.randn() * 0.1,
        base_position[1] + step * 0.8 + np.random.randn() * 0.1,
        base_position[2] + step * 0.2 + np.random.randn() * 0.05
    ]
    
    observation = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'type': 'observation',
        'sensors': {
            'position': position,
            'orientation': [
                0.02 + np.random.randn() * 0.01,
                0.01 + np.random.randn() * 0.01,
                0.15 + np.random.randn() * 0.02,
                0.98
            ],
            'linear_velocity': [1.2, 0.5, 0.1],
            'collision': {'has_collided': False, 'object_name': ''}
        },
        'rgb_info': {
            'num_images': 5,
            'shape': [256, 256, 3],
            'dtype': 'uint8'
        }
    }
    
    return observation


def create_sample_model_input(step):
    """创建模拟的模型输入"""
    model_input = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'type': 'model_input',
        'input_ids': {
            'shape': [1, 512],
            'dtype': 'torch.int64',
            'device': 'cuda:0'
        },
        'prompts': [
            f"Navigate to the target. Current step: {step}"
        ],
        'images': {
            'type': 'tensor_list',
            'length': 1,
            'shapes': [[5, 3, 224, 224]],
            'dtypes': ['torch.bfloat16']
        }
    }
    
    return model_input


def create_sample_model_output(step, current_position):
    """创建模拟的模型输出"""
    # 模拟LLM输出
    direction = np.array([0.8, 0.3, 0.1])
    direction = direction / np.linalg.norm(direction)
    distance = 8.0 + np.random.randn() * 0.5
    
    waypoint_llm = (direction * distance).tolist()
    
    # 模拟精化后的输出（略有不同）
    waypoint_refined = (direction * distance + np.random.randn(3) * 0.1).tolist()
    
    # 转换到世界坐标
    waypoint_world = [
        current_position[0] + waypoint_refined[0],
        current_position[1] + waypoint_refined[1],
        current_position[2] + waypoint_refined[2]
    ]
    
    model_output = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'type': 'model_output',
        'waypoints_llm': waypoint_llm,
        'waypoints_refined': waypoint_refined,
        'waypoints_world': waypoint_world
    }
    
    return model_output


def create_sample_episode(episode_id, num_steps=10):
    """创建一个完整的模拟episode"""
    base_position = [100.0, -80.0, -40.0]
    
    episode_data = {
        'episode_id': episode_id,
        'total_steps': num_steps,
        'success': True,
        'distance_to_goal': 5.2,
        'collision': False,
        'end_time': datetime.now().isoformat(),
        'steps': []
    }
    
    current_position = base_position.copy()
    
    for step in range(num_steps):
        # 添加观测
        obs = create_sample_observation(step, current_position)
        episode_data['steps'].append(obs)
        
        # 添加模型输入
        model_input = create_sample_model_input(step)
        episode_data['steps'].append(model_input)
        
        # 添加模型输出
        model_output = create_sample_model_output(step, current_position)
        episode_data['steps'].append(model_output)
        
        # 更新位置
        current_position = obs['sensors']['position']
    
    return episode_data


def main():
    print("=" * 70)
    print(" " * 20 + "TravelUAV 快速入门示例")
    print("=" * 70)
    print()
    
    # 创建输出目录
    output_dir = Path('./readapi/sample_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"1. 创建模拟数据...")
    print(f"   输出目录: {output_dir}")
    
    # 创建3个示例episode
    num_episodes = 3
    for i in range(num_episodes):
        episode_data = create_sample_episode(i, num_steps=10)
        
        output_file = output_dir / f'episode_{i:04d}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        print(f"   创建Episode {i}: {output_file}")
    
    print(f"\n2. 数据已创建完成!")
    print(f"   共创建 {num_episodes} 个episode")
    
    print(f"\n3. 接下来你可以:")
    print(f"   - 可视化数据:")
    print(f"     python readapi/visualize_data.py --data_file {output_dir}/episode_0000.json")
    print(f"   - 统计分析:")
    print(f"     python readapi/analyze_statistics.py --data_dir {output_dir}")
    
    print(f"\n4. 数据格式说明:")
    print(f"   - observation: 模拟器返回的观测数据（位置、图像等）")
    print(f"   - model_input: 发送给大模型的输入数据")
    print(f"   - model_output: 大模型的预测输出（航点）")
    
    print(f"\n5. 查看样例数据:")
    print(f"   cat {output_dir}/episode_0000.json | head -50")
    
    print("\n" + "=" * 70)
    print("提示: 这是模拟数据，真实数据需要运行eval.py并使用data_interceptor.py记录")
    print("=" * 70)


if __name__ == '__main__':
    main()

