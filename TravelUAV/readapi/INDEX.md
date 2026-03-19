# TravelUAV readapi 工具集索引

## 📚 文档

| 文件 | 描述 | 推荐阅读顺序 |
|------|------|--------------|
| [README.md](README.md) | 系统架构和数据流概述 | ⭐ 首先阅读 |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | **如何使用数据记录功能** | ⭐⭐⭐ **必读** |
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | 集成指南（技术细节） | ⭐⭐ 进阶 |
| [data_flow_diagram.md](data_flow_diagram.md) | 详细的数据流图解 | ⭐⭐ 深入理解 |
| [INDEX.md](INDEX.md) | 本文件，工具集索引 | - |

## 🔧 工具脚本

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| [quick_start.py](quick_start.py) | 生成模拟数据并演示工具使用 | ⭐ 快速入门 |
| [data_interceptor.py](data_interceptor.py) | 拦截真实评估数据 | 调试和分析真实运行 |
| [visualize_data.py](visualize_data.py) | 可视化轨迹和数据 | 查看episode轨迹 |
| [analyze_statistics.py](analyze_statistics.py) | 统计分析多个episode | 性能评估和对比 |

## 📊 数据样例

| 文件 | 内容 | 用途 |
|------|------|------|
| [sample_observation.json](sample_observation.json) | 模拟器观测数据格式 | 了解观测数据结构 |
| [sample_model_input.json](sample_model_input.json) | 模型输入数据格式 | 了解模型输入结构 |
| [sample_model_output.json](sample_model_output.json) | 模型输出数据格式 | 了解模型输出结构 |
| [sample_control_command.json](sample_control_command.json) | 控制指令格式 | 了解AirSim控制接口 |

## 🚀 快速开始

### 1. 生成和查看模拟数据

```bash
# 生成模拟数据
python readapi/quick_start.py

# 可视化第一个episode
python readapi/visualize_data.py --data_file readapi/sample_data/episode_0000.json

# 统计分析所有episode
python readapi/analyze_statistics.py --data_dir readapi/sample_data
```

### 2. 查看数据格式样例

```bash
# 查看观测数据格式
cat readapi/sample_observation.json | jq '.'

# 查看模型输入格式
cat readapi/sample_model_input.json | jq '.'

# 查看模型输出格式
cat readapi/sample_model_output.json | jq '.'

# 查看控制指令格式
cat readapi/sample_control_command.json | jq '.'
```

### 3. 拦截真实评估数据（高级）

```bash
# 在eval.py中添加数据拦截代码（参考data_interceptor.py中的说明）

# 运行评估并记录数据
python src/vlnce_src/eval.py --record_data --output_dir ./debug_data

# 分析记录的数据
python readapi/visualize_data.py --data_file ./debug_data/episode_0000.json
python readapi/analyze_statistics.py --data_dir ./debug_data
```

## 📖 学习路径

### 初学者

1. ✅ 阅读 [README.md](README.md) 了解系统架构
2. ✅ 运行 `quick_start.py` 生成样例数据
3. ✅ 查看 `sample_*.json` 了解数据格式
4. ✅ 运行 `visualize_data.py` 查看轨迹可视化

### 进阶用户

1. ✅ 深入阅读 [data_flow_diagram.md](data_flow_diagram.md)
2. ✅ 在真实评估中使用 `data_interceptor.py`
3. ✅ 使用 `analyze_statistics.py` 进行性能分析
4. ✅ 修改和扩展工具脚本以满足自己的需求

### 开发者

1. ✅ 研究源代码中的关键模块：
   - `airsim_plugin/AirVLNSimulatorClientTool.py` - 模拟器客户端
   - `src/model_wrapper/travel_llm.py` - 模型封装
   - `src/model_wrapper/utils/travel_util.py` - 数据处理
   - `src/vlnce_src/eval.py` - 评估流程
2. ✅ 扩展数据拦截器以记录更多信息
3. ✅ 创建自定义可视化和分析工具

## 🎯 常见用例

### 用例1: 理解模型如何预测航点

```bash
# 1. 生成样例数据
python readapi/quick_start.py

# 2. 查看模型输出格式
cat readapi/sample_model_output.json | jq '.llm_output_raw'
cat readapi/sample_model_output.json | jq '.waypoints_world'

# 3. 可视化预测的航点
python readapi/visualize_data.py \
    --data_file readapi/sample_data/episode_0000.json \
    --show
```

### 用例2: 分析轨迹质量

```bash
# 运行评估并记录数据
python src/vlnce_src/eval.py --record_data

# 统计分析
python readapi/analyze_statistics.py \
    --data_dir ./debug_data \
    --output ./debug_data/statistics.json

# 查看结果
cat ./debug_data/statistics.json | jq '.trajectory'
```

### 用例3: 调试模型输入

```bash
# 1. 在data_interceptor.py中设置断点
# 2. 运行评估
# 3. 检查记录的input_ids、images、historys等张量的shape和统计信息

# 查看第一步的模型输入
cat ./debug_data/episode_0000.json | jq '.steps[] | select(.type=="model_input") | .images'
```

### 用例4: 对比不同模型/参数的性能

```bash
# 运行实验A
python src/vlnce_src/eval.py --model_path ./model_A --record_data --output_dir ./results_A

# 运行实验B
python src/vlnce_src/eval.py --model_path ./model_B --record_data --output_dir ./results_B

# 对比统计
python readapi/analyze_statistics.py --data_dir ./results_A --output ./stats_A.json
python readapi/analyze_statistics.py --data_dir ./results_B --output ./stats_B.json

# 手动对比stats_A.json和stats_B.json
diff <(jq '.basic' ./stats_A.json) <(jq '.basic' ./stats_B.json)
```

## 💡 关键概念速查

### 坐标系

- **世界坐标系**: AirSim的全局坐标系 (NED: North-East-Down)
- **相对坐标系**: 以当前无人机为原点的局部坐标系
- **转换**: `world = R @ relative + current_position`

### 数据流方向

```
模拟器 → 观测数据 → 模型输入预处理 → LLM推理 → 轨迹精化 → 控制指令 → 模拟器
```

### 关键数据大小

- **观测数据**: ~5MB (主要是图像)
- **模型输入**: ~100MB (张量+历史)
- **模型输出**: ~100 bytes (航点坐标)
- **控制指令**: ~1KB (JSON)

### 时间开销

- **观测获取**: ~50ms
- **数据预处理**: ~30ms
- **LLM推理**: ~200ms
- **轨迹精化**: ~50ms
- **执行动作**: ~500ms
- **总计**: ~830ms/步

## 📝 代码结构参考

```
TravelUAV/
├── readapi/                          # 本工具集
│   ├── README.md                    # 系统架构说明
│   ├── INDEX.md                     # 本索引文件
│   ├── data_flow_diagram.md         # 数据流图解
│   ├── quick_start.py               # 快速入门脚本
│   ├── data_interceptor.py          # 数据拦截器
│   ├── visualize_data.py            # 可视化工具
│   ├── analyze_statistics.py        # 统计分析工具
│   ├── sample_*.json                # 数据格式样例
│   └── sample_data/                 # 生成的样例数据
├── airsim_plugin/                    # AirSim交互模块
│   ├── AirVLNSimulatorClientTool.py # 模拟器客户端
│   └── AirVLNSimulatorServerTool.py # 模拟器服务端
├── src/
│   ├── model_wrapper/               # 模型封装
│   │   ├── travel_llm.py           # 主要模型接口
│   │   └── utils/
│   │       └── travel_util.py      # 数据处理工具
│   └── vlnce_src/                   # 评估和训练
│       ├── eval.py                  # 评估主程序
│       ├── env_uav.py              # 环境管理
│       └── ...
└── Model/LLaMA-UAV/                 # 模型权重和配置
```

## ❓ 常见问题

### Q: 如何修改数据拦截器以记录更多信息？

A: 编辑 `data_interceptor.py`，在 `DataInterceptor` 类中添加新的 `record_*` 方法。

### Q: 可视化工具需要什么依赖？

A: `matplotlib` 用于绘图。安装: `pip install matplotlib`

### Q: 如何在eval.py中集成数据拦截器？

A: 参考 `data_interceptor.py` 中的注释和示例代码，在关键位置调用拦截方法。

### Q: 生成的JSON文件很大怎么办？

A: 数据拦截器默认只保存元信息（shape、统计量）而不保存完整张量。如需完整数据，可以使用pickle格式。

## 🔗 相关资源

- [AirSim API文档](https://microsoft.github.io/AirSim/api_docs/html/)
- [LLaMA-VID论文](https://arxiv.org/abs/2311.17043)
- [TravelUAV论文](https://arxiv.org/abs/2410.07087)
- [项目主README](../README.md)

## 📧 反馈

如有问题或建议，请在项目中提issue或联系开发者。

