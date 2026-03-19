#!/bin/bash

# 测试脚本运行器（自动退出conda环境）

# 退出conda环境（如果在conda环境中）
conda deactivate 2>/dev/null || true

# 清除conda相关环境变量
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

echo "✓ 已退出conda环境，使用系统Python"
echo ""

# 运行测试
if [ $# -eq 0 ]; then
    # 默认运行连接测试
    echo "运行AirSim连接测试..."
    /usr/bin/python3 test_airsim_connection.py
else
    # 运行指定的脚本
    /usr/bin/python3 "$@"
fi


