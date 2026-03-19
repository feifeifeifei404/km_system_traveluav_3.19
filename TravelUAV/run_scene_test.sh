#!/bin/bash
#
# 场景连接测试脚本
#
# 功能:
# 1. 激活正确的 conda 环境
# 2. 运行场景连接测试
# 3. 诊断字节串问题
#

set -e

echo "========================================"
echo "  TravelUAV 场景连接测试"
echo "========================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活 conda 环境
echo "激活 conda 环境: llama"
eval "$(conda shell.bash hook)"
conda activate llama

echo "Python 版本: $(python --version)"
echo "Python 路径: $(which python)"
echo ""

# 检查场景服务器是否运行
echo "检查场景服务器状态..."
if lsof -i :25000 >/dev/null 2>&1; then
    echo "✓ 场景服务器正在运行 (端口 25000)"
else
    echo "✗ 场景服务器未运行 (端口 25000)"
    echo ""
    echo "请先启动场景服务器:"
    echo "  cd $SCRIPT_DIR/airsim_plugin"
    echo "  python AirVLNSimulatorServerTool.py --port 25000 --root_path /mnt/data/TravelUAV/envs"
    exit 1
fi

echo ""
echo "========================================"
echo "  运行测试..."
echo "========================================"
echo ""

# 运行测试
cd "$SCRIPT_DIR"
python test_scene_connection.py

echo ""
echo "测试完成！"
