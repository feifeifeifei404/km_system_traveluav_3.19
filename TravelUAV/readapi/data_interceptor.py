"""
数据拦截器 - 在评估过程中拦截并记录大模型和模拟器之间的交互数据

使用方法:
    python readapi/data_interceptor.py --output_dir ./debug_data --max_episodes 5

这个脚本会:
1. 修改eval.py的关键函数，添加数据记录钩子
2. 保存每一步的观测数据、模型输入、模型输出、控制指令
3. 生成JSON格式的数据文件供后续分析
4. 保存RGB图像、深度图、instruction、prompts等完整数据
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import argparse


class DataInterceptor:
    """数据拦截器类
    
    新增功能:
    - save_images: 是否保存RGB图像和深度图到文件
    - save_prompts: 是否保存prompts文本
    - 保存instruction到episode元数据中
    """
    
    def __init__(self, output_dir='./debug_data', save_images=True, save_prompts=True):
        """
        初始化数据拦截器
        
        Args:
            output_dir: 输出目录
            save_images: 是否保存RGB图像和深度图（默认True）
            save_prompts: 是否保存prompts文本（默认True）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_images = save_images
        self.save_prompts = save_prompts
        
        self.current_episode = 0
        self.current_step = 0
        self.episode_data = []
        self.episode_meta = {}  # 保存episode的元信息（包括instruction）
        
        print(f"[DataInterceptor] 初始化完成")
        print(f"  输出目录: {self.output_dir}")
        print(f"  保存图像: {self.save_images}")
        print(f"  保存Prompts: {self.save_prompts}")
    
    def start_episode(self, episode_info):
        """开始新的episode
        
        Args:
            episode_info: 包含episode信息的字典，可能包含:
                - map_name: 地图名称
                - seq_name: 序列名称
                - instruction: 任务指令文本（重要！）
                - target_position: 目标位置
                - start_position: 起始位置
        """
        self.current_step = 0
        self.episode_data = []
        
        # 保存完整的episode元信息
        self.episode_meta = {
            'episode_id': self.current_episode,
            'map_name': episode_info.get('map_name', 'unknown'),
            'seq_name': episode_info.get('seq_name', 'unknown'),
            'instruction': episode_info.get('instruction', ''),  # 保存instruction
            'target_position': self._convert_to_list(episode_info.get('target_position')),
            'start_position': self._convert_to_list(episode_info.get('start_position')),
            'start_time': datetime.now().isoformat()
        }
        
        # 如果保存图像，创建episode的图像目录
        if self.save_images:
            self.episode_image_dir = self.output_dir / f'episode_{self.current_episode:04d}' / 'images'
            self.episode_image_dir.mkdir(parents=True, exist_ok=True)
        
        instruction_preview = self.episode_meta['instruction'][:80] if self.episode_meta['instruction'] else '(无)'
        print(f"\n[DataInterceptor] === Episode {self.current_episode} 开始 ===")
        print(f"  地图: {self.episode_meta['map_name']}")
        print(f"  序列: {self.episode_meta['seq_name']}")
        print(f"  指令: {instruction_preview}...")
        
        return self.episode_meta
    
    def record_observation(self, observation):
        """记录观测数据
        
        新增功能:
        - 保存RGB图像到PNG文件
        - 保存深度图到PNG文件
        - 记录instruction（如果observation中包含）
        """
        step_data = {
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'type': 'observation'
        }
        
        # 记录传感器数据
        if 'sensors' in observation:
            sensors = observation['sensors']
            step_data['sensors'] = {
                'position': self._convert_to_list(sensors.get('state', {}).get('position')),
                'orientation': self._convert_to_list(sensors.get('state', {}).get('orientation')),
                'linear_velocity': self._convert_to_list(sensors.get('state', {}).get('linear_velocity')),
                'angular_velocity': self._convert_to_list(sensors.get('state', {}).get('angular_velocity')),
                'collision': sensors.get('state', {}).get('collision', {})
            }
            
            if 'imu' in sensors:
                step_data['sensors']['rotation'] = self._convert_to_list(sensors['imu'].get('rotation'))
                step_data['sensors']['linear_acceleration'] = self._convert_to_list(sensors['imu'].get('linear_acceleration'))
                step_data['sensors']['angular_velocity_imu'] = self._convert_to_list(sensors['imu'].get('angular_velocity'))
        
        # 记录并保存RGB图像
        if 'rgb' in observation:
            rgb_images = observation['rgb']
            step_data['rgb_info'] = {
                'num_images': len(rgb_images) if rgb_images else 0,
                'shape': list(rgb_images[0].shape) if rgb_images and len(rgb_images) > 0 else None,
                'dtype': str(rgb_images[0].dtype) if rgb_images and len(rgb_images) > 0 else None
            }
            
            # 新增：保存RGB图像到文件
            if self.save_images and rgb_images:
                rgb_files = self._save_rgb_images(rgb_images)
                step_data['rgb_files'] = rgb_files
        
        # 新增：记录并保存深度图
        if 'depth' in observation:
            depth_images = observation['depth']
            step_data['depth_info'] = {
                'num_images': len(depth_images) if depth_images else 0,
                'shape': list(depth_images[0].shape) if depth_images and len(depth_images) > 0 else None,
                'dtype': str(depth_images[0].dtype) if depth_images and len(depth_images) > 0 else None
            }
            
            # 保存深度图到文件
            if self.save_images and depth_images:
                depth_files = self._save_depth_images(depth_images)
                step_data['depth_files'] = depth_files
        
        # 新增：记录instruction（如果在observation中）
        if 'instruction' in observation:
            step_data['instruction'] = observation['instruction']
        
        # 新增：记录目标位置和距离（如果在observation中）
        if 'object_position' in observation:
            step_data['object_position'] = self._convert_to_list(observation['object_position'])
        if 'distance_to_goal' in observation:
            step_data['distance_to_goal'] = observation['distance_to_goal']
        
        print(f"  [Step {self.current_step}] 记录观测数据: 位置={step_data.get('sensors', {}).get('position')}")
        
        return step_data
    
    def _save_rgb_images(self, rgb_images):
        """保存RGB图像到PNG文件
        
        Args:
            rgb_images: RGB图像列表，每个元素是[H, W, 3]的numpy数组
            
        Returns:
            保存的文件路径列表
        """
        try:
            from PIL import Image
        except ImportError:
            print("[警告] PIL未安装，跳过图像保存。请运行: pip install Pillow")
            return []
        
        camera_names = ['front', 'left', 'right', 'rear', 'down']
        file_paths = []
        
        for i, img in enumerate(rgb_images):
            if img is None:
                continue
            
            camera_name = camera_names[i] if i < len(camera_names) else f'camera_{i}'
            filename = f'step_{self.current_step:04d}_{camera_name}_rgb.png'
            filepath = self.episode_image_dir / filename
            
            try:
                # 确保图像是uint8类型
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    Image.fromarray(img).save(filepath)
                    file_paths.append(str(filepath))
            except Exception as e:
                print(f"[警告] 保存RGB图像失败: {e}")
        
        return file_paths
    
    def _save_depth_images(self, depth_images):
        """保存深度图到PNG文件
        
        Args:
            depth_images: 深度图列表，每个元素是[H, W]的numpy数组
            
        Returns:
            保存的文件路径列表
        """
        try:
            from PIL import Image
        except ImportError:
            print("[警告] PIL未安装，跳过深度图保存。请运行: pip install Pillow")
            return []
        
        camera_names = ['front', 'left', 'right', 'rear', 'down']
        file_paths = []
        
        for i, img in enumerate(depth_images):
            if img is None:
                continue
            
            camera_name = camera_names[i] if i < len(camera_names) else f'camera_{i}'
            filename = f'step_{self.current_step:04d}_{camera_name}_depth.png'
            filepath = self.episode_image_dir / filename
            
            try:
                if isinstance(img, np.ndarray):
                    # 深度图可能是float类型，需要归一化到0-255
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        # 归一化深度值到0-255
                        img_normalized = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                    else:
                        img_normalized = img.astype(np.uint8)
                    
                    Image.fromarray(img_normalized).save(filepath)
                    file_paths.append(str(filepath))
                    
                    # 同时保存原始深度值为numpy文件（可选，用于精确分析）
                    npy_filepath = filepath.with_suffix('.npy')
                    np.save(npy_filepath, img)
            except Exception as e:
                print(f"[警告] 保存深度图失败: {e}")
        
        return file_paths
    
    def record_model_input(self, model_input):
        """记录模型输入数据
        
        新增功能:
        - 保存prompts文本（完整的prompt内容）
        - 保存conversations（如果有）
        """
        step_data = {
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'type': 'model_input'
        }
        
        # 记录张量的形状和统计信息
        for key, value in model_input.items():
            if isinstance(value, torch.Tensor):
                step_data[key] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device),
                    'mean': float(value.float().mean().item()),
                    'std': float(value.float().std().item())
                }
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    step_data[key] = {
                        'type': 'tensor_list',
                        'length': len(value),
                        'shapes': [list(t.shape) for t in value],
                        'dtypes': [str(t.dtype) for t in value]
                    }
                elif len(value) > 0 and isinstance(value[0], str):
                    # 新增：保存文本列表（如prompts）
                    if self.save_prompts:
                        step_data[key] = value  # 保存完整文本
                    else:
                        step_data[key] = {
                            'type': 'string_list',
                            'length': len(value),
                            'preview': [s[:100] + '...' if len(s) > 100 else s for s in value]
                        }
            elif isinstance(value, str):
                # 新增：保存单个字符串（如prompt）
                if self.save_prompts:
                    step_data[key] = value
                else:
                    step_data[key] = value[:200] + '...' if len(value) > 200 else value
            elif isinstance(value, (int, float, bool)):
                step_data[key] = value
        
        # 新增：专门处理prompts字段
        if 'prompts' in model_input and self.save_prompts:
            step_data['prompts'] = model_input['prompts']
        
        # 新增：处理conversations字段（如果存在）
        if 'conversations' in model_input and self.save_prompts:
            step_data['conversations'] = model_input['conversations']
        
        print(f"  [Step {self.current_step}] 记录模型输入")
        
        return step_data
    
    def record_model_output(self, model_output):
        """记录模型输出数据"""
        step_data = {
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'type': 'model_output'
        }
        
        # 记录航点输出
        if 'waypoints_llm_new' in model_output:
            waypoints_llm = model_output['waypoints_llm_new']
            step_data['waypoints_llm'] = self._convert_to_list(waypoints_llm)
        
        if 'refined_waypoints' in model_output:
            waypoints_refined = model_output['refined_waypoints']
            step_data['waypoints_refined'] = self._convert_to_list(waypoints_refined)
        
        if 'waypoints_world' in model_output:
            waypoints_world = model_output['waypoints_world']
            step_data['waypoints_world'] = self._convert_to_list(waypoints_world)
        
        # 新增：记录中间输出（如果有）
        if 'intermediate_outputs' in model_output:
            intermediate = model_output['intermediate_outputs']
            step_data['intermediate_outputs'] = {}
            for k, v in intermediate.items():
                if isinstance(v, torch.Tensor):
                    step_data['intermediate_outputs'][k] = {
                        'shape': list(v.shape),
                        'dtype': str(v.dtype)
                    }
                else:
                    step_data['intermediate_outputs'][k] = self._convert_to_list(v)
        
        print(f"  [Step {self.current_step}] 记录模型输出: LLM航点={step_data.get('waypoints_llm')}")
        
        return step_data
    
    def record_control_command(self, command):
        """记录控制指令"""
        step_data = {
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'type': 'control_command'
        }
        
        if 'waypoints' in command:
            step_data['waypoints'] = self._convert_to_list(command['waypoints'])
        
        # 新增：记录控制参数
        if 'velocity' in command:
            step_data['velocity'] = command['velocity']
        if 'drivetrain' in command:
            step_data['drivetrain'] = str(command['drivetrain'])
        if 'yaw_mode' in command:
            step_data['yaw_mode'] = self._convert_to_list(command['yaw_mode'])
        if 'lookahead' in command:
            step_data['lookahead'] = command['lookahead']
        if 'adaptive_lookahead' in command:
            step_data['adaptive_lookahead'] = command['adaptive_lookahead']
        
        if 'parameters' in command:
            step_data['parameters'] = command['parameters']
        
        print(f"  [Step {self.current_step}] 记录控制指令")
        
        return step_data
    
    def end_step(self):
        """结束当前步骤"""
        self.current_step += 1
    
    def end_episode(self, episode_result):
        """结束episode并保存数据
        
        保存的数据包括:
        - episode_meta: 包含instruction等元信息
        - episode_result: 包含success、distance_to_goal等结果
        - steps: 所有步骤的数据
        """
        episode_file = self.output_dir / f'episode_{self.current_episode:04d}.json'
        
        # 合并元信息和结果
        episode_summary = {
            **self.episode_meta,  # 包含instruction、map_name等
            'total_steps': self.current_step,
            'success': episode_result.get('success', False),
            'distance_to_goal': episode_result.get('distance_to_goal'),
            'collision': episode_result.get('collision', False),
            'end_time': datetime.now().isoformat(),
            'steps': self.episode_data
        }
        
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(episode_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n[DataInterceptor] === Episode {self.current_episode} 结束 ===")
        print(f"  总步数: {self.current_step}")
        print(f"  成功: {episode_result.get('success', False)}")
        print(f"  Instruction: {self.episode_meta.get('instruction', '(无)')[:50]}...")
        print(f"  数据保存至: {episode_file}")
        if self.save_images:
            print(f"  图像保存至: {self.episode_image_dir}")
        
        self.current_episode += 1
        self.episode_data = []
        self.episode_meta = {}
    
    def _convert_to_list(self, data):
        """将numpy/torch数据转换为列表"""
        if data is None:
            return None
        
        if isinstance(data, (list, tuple)):
            return [self._convert_to_list(item) for item in data]
        
        if isinstance(data, np.ndarray):
            return data.tolist()
        
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
        
        if isinstance(data, (int, float, bool, str)):
            return data
        
        # 处理dict
        if isinstance(data, dict):
            return {k: self._convert_to_list(v) for k, v in data.items()}
        
        return str(data)
    
    def add_step_data(self, data):
        """添加步骤数据"""
        self.episode_data.append(data)


def create_wrapper_functions(interceptor):
    """创建包装函数来拦截数据"""
    
    def wrap_observation(original_func):
        """包装观测获取函数"""
        def wrapper(*args, **kwargs):
            obs = original_func(*args, **kwargs)
            step_data = interceptor.record_observation(obs)
            interceptor.add_step_data(step_data)
            return obs
        return wrapper
    
    def wrap_model_forward(original_func):
        """包装模型前向传播"""
        def wrapper(model_wrapper, *args, **kwargs):
            # 记录输入
            if len(args) > 0:
                model_input = args[0]
                step_data = interceptor.record_model_input(model_input)
                interceptor.add_step_data(step_data)
            
            # 执行模型
            output = original_func(model_wrapper, *args, **kwargs)
            
            # 记录输出
            if isinstance(output, dict):
                step_data = interceptor.record_model_output(output)
                interceptor.add_step_data(step_data)
            
            return output
        return wrapper
    
    def wrap_control(original_func):
        """包装控制指令发送"""
        def wrapper(*args, **kwargs):
            # 记录控制指令
            if len(args) > 0:
                command = args[0]
                step_data = interceptor.record_control_command({'waypoints': command})
                interceptor.add_step_data(step_data)
            
            result = original_func(*args, **kwargs)
            interceptor.end_step()
            return result
        return wrapper
    
    return wrap_observation, wrap_model_forward, wrap_control


def main():
    parser = argparse.ArgumentParser(description='拦截并记录TravelUAV的交互数据')
    parser.add_argument('--output_dir', type=str, default='./debug_data',
                        help='输出目录')
    parser.add_argument('--max_episodes', type=int, default=5,
                        help='最大记录episode数量')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='是否保存RGB图像和深度图（默认True）')
    parser.add_argument('--no_save_images', action='store_false', dest='save_images',
                        help='不保存图像')
    parser.add_argument('--save_prompts', action='store_true', default=True,
                        help='是否保存prompts文本（默认True）')
    parser.add_argument('--no_save_prompts', action='store_false', dest='save_prompts',
                        help='不保存prompts')
    args = parser.parse_args()
    
    print("=" * 60)
    print("TravelUAV 数据拦截器")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print(f"最大episode数: {args.max_episodes}")
    print(f"保存图像: {args.save_images}")
    print(f"保存Prompts: {args.save_prompts}")
    print()
    
    # 创建拦截器
    interceptor = DataInterceptor(
        output_dir=args.output_dir,
        save_images=args.save_images,
        save_prompts=args.save_prompts
    )
    
    print("提示: 此脚本需要与eval.py配合使用")
    print("请在eval.py中导入此模块并调用相应的拦截函数")
    print()
    print("示例代码:")
    print("  from readapi.data_interceptor import DataInterceptor")
    print("  interceptor = DataInterceptor(")
    print("      output_dir='./readapi/debug_data',")
    print("      save_images=True,")
    print("      save_prompts=True")
    print("  )")
    print()
    print("新增保存的数据:")
    print("  - instruction: 任务指令文本")
    print("  - prompts: 完整的prompt文本")
    print("  - RGB图像: PNG文件 (5个视角)")
    print("  - 深度图: PNG + NPY文件 (5个视角)")
    print()


if __name__ == '__main__':
    main()
