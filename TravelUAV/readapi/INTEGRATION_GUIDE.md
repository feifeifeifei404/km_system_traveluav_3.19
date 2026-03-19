# 数据拦截器集成指南

## 概述

本指南将教你如何在真实的评估过程（`eval.py`）中集成数据拦截器，记录大模型和模拟器之间的实际交互数据。

## 方法一: 最小侵入式集成（推荐）

这种方法只需要在 `eval.py` 中添加少量代码，不影响原有逻辑。

### 步骤1: 在eval.py开头导入拦截器

在 `src/vlnce_src/eval.py` 的开头添加：

```python
import sys
from pathlib import Path

# 添加readapi到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'readapi'))

# 导入拦截器
try:
    from data_interceptor import DataInterceptor
    ENABLE_DATA_RECORDING = True
    print("[INFO] 数据拦截器已启用")
except ImportError:
    ENABLE_DATA_RECORDING = False
    print("[WARNING] 数据拦截器未找到，跳过数据记录")
```

### 步骤2: 在main函数中初始化拦截器

在 `main()` 函数或评估循环开始前添加：

```python
def main():
    # ... 原有的参数解析和初始化代码 ...
    
    # 初始化数据拦截器
    if ENABLE_DATA_RECORDING:
        interceptor = DataInterceptor(output_dir='./debug_data')
    else:
        interceptor = None
    
    # ... 继续原有代码 ...
```

### 步骤3: 在Episode循环中添加记录点

找到评估的主循环（通常在处理每个episode的地方），添加记录调用：

```python
# Episode开始
for episode_idx, episode_data in enumerate(eval_dataset):
    
    # === 记录点1: Episode开始 ===
    if interceptor:
        episode_meta = interceptor.start_episode({
            'map_name': episode_data.get('map_name'),
            'seq_name': episode_data.get('seq_name'),
            'instruction': episode_data.get('instruction')
        })
    
    # 初始化环境
    env.reset(episode_data)
    
    # Step循环
    done = False
    step = 0
    while not done:
        
        # === 记录点2: 获取观测 ===
        observation = env.get_observation()
        if interceptor:
            obs_data = interceptor.record_observation(observation)
            interceptor.add_step_data(obs_data)
        
        # === 记录点3: 准备模型输入 ===
        model_inputs, rot_to_targets, conversations, raw_inputs = model_wrapper.prepare_inputs(
            episodes=[observation],
            target_positions=[target_position],
            assist_notices=None
        )
        if interceptor:
            input_data = interceptor.record_model_input(model_inputs)
            interceptor.add_step_data(input_data)
        
        # === 记录点4: 模型推理 ===
        model_output = model_wrapper.run(model_inputs, [observation], rot_to_targets)
        if interceptor:
            output_data = interceptor.record_model_output({
                'waypoints_llm_new': model_output.get('waypoints_llm_new'),
                'refined_waypoints': model_output.get('refined_waypoints'),
                'waypoints_world': model_output.get('waypoints_world', [])
            })
            interceptor.add_step_data(output_data)
        
        # === 记录点5: 执行动作 ===
        waypoint = model_output['waypoints_world'][0]
        if interceptor:
            cmd_data = interceptor.record_control_command({
                'waypoints': [waypoint]
            })
            interceptor.add_step_data(cmd_data)
        
        # 执行动作
        observation, reward, done, info = env.step(waypoint)
        
        # 结束步骤
        if interceptor:
            interceptor.end_step()
        
        step += 1
    
    # === 记录点6: Episode结束 ===
    if interceptor:
        interceptor.end_episode({
            'success': info.get('success', False),
            'distance_to_goal': info.get('distance_to_goal'),
            'collision': info.get('collision', False)
        })
```

### 步骤4: 运行评估并记录数据

```bash
cd /mnt/data/TravelUAV

# 运行评估（会自动记录数据到./debug_data目录）
python src/vlnce_src/eval.py \
    --model_path /path/to/model \
    --dataset_path /path/to/dataset \
    --eval_json_path /path/to/eval.json

# 记录的数据会保存在 ./debug_data/ 目录下
```

## 方法二: 使用命令行参数控制

如果你想更灵活地控制是否记录数据，可以添加命令行参数。

### 修改参数定义

在 `eval.py` 的参数解析部分添加：

```python
parser.add_argument(
    '--record_data',
    action='store_true',
    help='是否记录交互数据用于调试'
)
parser.add_argument(
    '--record_output_dir',
    type=str,
    default='./debug_data',
    help='数据记录输出目录'
)
parser.add_argument(
    '--max_record_episodes',
    type=int,
    default=None,
    help='最多记录多少个episode（None表示全部记录）'
)
```

### 条件初始化拦截器

```python
def main():
    args = parse_args()
    
    # 条件初始化拦截器
    interceptor = None
    if args.record_data:
        try:
            from readapi.data_interceptor import DataInterceptor
            interceptor = DataInterceptor(output_dir=args.record_output_dir)
            print(f"[INFO] 数据拦截器已启用，输出目录: {args.record_output_dir}")
            if args.max_record_episodes:
                print(f"[INFO] 最多记录 {args.max_record_episodes} 个episode")
        except ImportError:
            print("[WARNING] 无法导入data_interceptor，跳过数据记录")
    
    # ... 继续评估代码 ...
    
    for episode_idx, episode_data in enumerate(eval_dataset):
        # 检查是否超过记录限制
        if args.max_record_episodes and episode_idx >= args.max_record_episodes:
            interceptor = None  # 停止记录
        
        # ... 继续记录代码（如方法一） ...
```

### 运行示例

```bash
# 记录前5个episode的数据
python src/vlnce_src/eval.py \
    --record_data \
    --record_output_dir ./my_debug_data \
    --max_record_episodes 5

# 不记录数据，正常评估
python src/vlnce_src/eval.py
```

## 方法三: 创建包装脚本（最简单）

如果你不想修改 `eval.py`，可以创建一个包装脚本。

### 创建 eval_with_recording.py

```python
#!/usr/bin/env python
"""
带数据记录的评估包装脚本
使用方法: python eval_with_recording.py --output_dir ./debug_data --max_episodes 5
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'readapi'))

from data_interceptor import DataInterceptor

# 这里需要导入并修改原始的eval模块
# 由于eval.py可能复杂，建议使用方法一或方法二


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./debug_data')
    parser.add_argument('--max_episodes', type=int, default=5)
    args = parser.parse_args()
    
    # 初始化拦截器
    interceptor = DataInterceptor(output_dir=args.output_dir)
    
    print("=" * 60)
    print("评估 + 数据记录")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print(f"最大episode数: {args.max_episodes}")
    print()
    
    # TODO: 在这里调用原始的eval流程，并添加拦截器调用
    # 这需要重构eval.py使其可以作为模块导入
    
    print("提示: 此方法需要重构eval.py，推荐使用方法一或方法二")


if __name__ == '__main__':
    main()
```

## 完整示例：修改后的eval.py片段

下面是一个完整的示例，展示如何在实际的eval.py中集成拦截器：

```python
# src/vlnce_src/eval.py 修改示例

import os
import sys
from pathlib import Path
import argparse

# === 添加：导入拦截器 ===
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'readapi'))
try:
    from data_interceptor import DataInterceptor
    HAS_INTERCEPTOR = True
except:
    HAS_INTERCEPTOR = False

# ... 原有的其他导入 ...

def parse_args():
    parser = argparse.ArgumentParser()
    # ... 原有的参数 ...
    
    # === 添加：记录相关参数 ===
    parser.add_argument('--record_data', action='store_true',
                        help='记录交互数据')
    parser.add_argument('--record_dir', type=str, default='./debug_data',
                        help='数据记录目录')
    parser.add_argument('--max_record_eps', type=int, default=None,
                        help='最多记录episode数')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # === 添加：初始化拦截器 ===
    interceptor = None
    if args.record_data and HAS_INTERCEPTOR:
        interceptor = DataInterceptor(output_dir=args.record_dir)
        print(f"✓ 数据拦截器已启用: {args.record_dir}")
    
    # 加载模型
    model_wrapper = TravelModelWrapper(model_args, data_args)
    model_wrapper.eval()
    
    # 初始化环境
    env = AirVLNENV(...)
    
    # 评估循环
    for ep_idx in range(num_episodes):
        
        # 检查记录限制
        if args.max_record_eps and ep_idx >= args.max_record_eps:
            interceptor = None
        
        # 获取episode数据
        episode_batch = env.next_minibatch()
        if episode_batch is None:
            break
        
        # === 记录：Episode开始 ===
        if interceptor:
            interceptor.start_episode({
                'map_name': episode_batch[0]['map_name'],
                'seq_name': episode_batch[0]['seq_name'],
                'instruction': episode_batch[0]['instruction']
            })
        
        # 初始化场景
        env.changeToNewTrajectorys()
        
        # 导航循环
        for step in range(max_steps):
            
            # 获取观测
            observations = env.get_observations()
            
            # === 记录：观测数据 ===
            if interceptor:
                obs_data = interceptor.record_observation(observations[0])
                interceptor.add_step_data(obs_data)
            
            # 准备模型输入
            model_inputs, rot_to_targets, _, _ = model_wrapper.prepare_inputs(
                episodes=observations,
                target_positions=target_positions
            )
            
            # === 记录：模型输入 ===
            if interceptor:
                input_data = interceptor.record_model_input(model_inputs)
                interceptor.add_step_data(input_data)
            
            # 模型推理
            outputs = model_wrapper.run(model_inputs, observations, rot_to_targets)
            waypoints = outputs['refined_waypoints']
            
            # === 记录：模型输出 ===
            if interceptor:
                output_data = interceptor.record_model_output({
                    'waypoints_llm_new': outputs.get('waypoints_llm_new'),
                    'refined_waypoints': waypoints,
                    'waypoints_world': waypoints
                })
                interceptor.add_step_data(output_data)
            
            # 执行动作
            success = env.move_to_waypoints(waypoints)
            
            # === 记录：控制指令 ===
            if interceptor:
                cmd_data = interceptor.record_control_command({
                    'waypoints': waypoints[0].tolist() if hasattr(waypoints[0], 'tolist') else waypoints[0]
                })
                interceptor.add_step_data(cmd_data)
                interceptor.end_step()
            
            # 检查终止条件
            if check_done(...):
                break
        
        # === 记录：Episode结束 ===
        if interceptor:
            interceptor.end_episode({
                'success': is_success,
                'distance_to_goal': final_distance,
                'collision': has_collision
            })
        
        print(f"Episode {ep_idx} 完成")


if __name__ == '__main__':
    main()
```

## 验证集成是否成功

### 运行测试

```bash
# 测试记录1个episode
python src/vlnce_src/eval.py \
    --record_data \
    --record_dir ./test_debug \
    --max_record_eps 1

# 检查是否生成了数据文件
ls -lh ./test_debug/

# 应该看到类似这样的文件：
# episode_0000.json
```

### 查看记录的数据

```bash
# 查看JSON文件的基本信息
cat ./test_debug/episode_0000.json | jq '.episode_id, .total_steps, .success'

# 查看第一步的观测数据
cat ./test_debug/episode_0000.json | jq '.steps[0]'

# 查看模型输出
cat ./test_debug/episode_0000.json | jq '.steps[] | select(.type=="model_output")'
```

### 可视化验证

```bash
# 生成轨迹可视化
python readapi/visualize_data.py \
    --data_file ./test_debug/episode_0000.json

# 统计分析
python readapi/analyze_statistics.py \
    --data_dir ./test_debug
```

## 常见问题排查

### 问题1: ImportError: No module named 'data_interceptor'

**解决方法**:
```python
# 确保正确添加了路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'readapi'))
```

### 问题2: 记录的数据不完整

**检查清单**:
- ✓ 确认在所有关键点都调用了 `interceptor.record_*` 方法
- ✓ 确认调用了 `interceptor.add_step_data()`
- ✓ 确认调用了 `interceptor.end_step()` 和 `interceptor.end_episode()`

### 问题3: JSON文件太大

**优化方法**:
```python
# 在DataInterceptor的record方法中只保存摘要信息
# 已经默认实现，不保存完整张量，只保存shape和统计量
```

### 问题4: 影响评估性能

**说明**: 数据记录的开销很小（<5ms/步），如果担心性能：
```python
# 只记录前N个episode
if ep_idx < 10:  # 只记录前10个
    if interceptor:
        interceptor.record_observation(obs)
```

## 下一步

记录数据后，你可以：

1. **可视化轨迹**
   ```bash
   python readapi/visualize_data.py --data_file ./debug_data/episode_0000.json
   ```

2. **统计分析**
   ```bash
   python readapi/analyze_statistics.py --data_dir ./debug_data
   ```

3. **深入分析**
   - 查看模型在哪些情况下预测不准
   - 对比LLM和轨迹模型的输出差异
   - 分析失败案例的原因
   - 优化模型或参数

## 总结

推荐使用 **方法一（最小侵入式）** 或 **方法二（命令行参数控制）**，它们：
- ✓ 代码改动最小
- ✓ 不影响原有逻辑
- ✓ 可以灵活开关
- ✓ 易于维护

现在就可以开始集成了！

