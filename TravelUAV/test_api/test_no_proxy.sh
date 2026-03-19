#!/bin/bash
# 清除所有代理设置并测试API

echo "=========================================="
echo "取消代理设置并测试 OpenAI API"
echo "=========================================="

# 取消所有代理环境变量
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY

echo ""
echo "✓ 已取消所有代理设置"
echo ""

# 激活conda环境
source ~/anaconda3/bin/activate llama

# 运行测试
cd /mnt/data/TravelUAV/test_api
python test_openai_only.py

