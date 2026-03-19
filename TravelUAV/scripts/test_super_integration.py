#!/usr/bin/env python3
"""
测试SUPER集成的独立脚本
在TravelUAV的conda环境(llama)中运行
"""
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/vlnce_src'))

from super_socket_client import get_super_client
import time

def test_super_integration():
    """测试Socket通信"""
    print("="*70)
    print("测试SUPER Socket通信")
    print("="*70)
    
    # 获取客户端
    client = get_super_client()
    
    # 测试1: 发送目标点
    print("\n[测试1] 发送目标点...")
    success = client.send_goal(5.0, 5.0, -2.0)
    
    if not success:
        print("✗ 发送失败！请确认:")
        print("  1. run_bidirectional_bridge.sh 是否运行?")
        print("  2. SUPER是否启动?")
        return
    
    print("✓ 目标点已发送")
    
    # 测试2: 查询状态
    print("\n[测试2] 查询SUPER状态...")
    for i in range(10):
        status_info = client.get_status()
        if status_info:
            print(f"  状态: {status_info['status']}")
            if status_info['current_pos']:
                print(f"  当前位置: {status_info['current_pos']}")
        else:
            print("  ✗ 无法获取状态")
        
        time.sleep(2)
    
    # 测试3: 等待到达
    print("\n[测试3] 等待到达...")
    success, final_status = client.wait_for_arrival(timeout=30.0)
    
    if success:
        print(f"✓ 测试成功！最终状态: {final_status}")
    else:
        print(f"✗ 测试失败！最终状态: {final_status}")
    
    print("\n"+"="*70)

if __name__ == '__main__':
    test_super_integration()
