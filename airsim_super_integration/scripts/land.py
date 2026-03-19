#!/usr/bin/env python3
"""
AirSim无人机降落脚本

功能：
1. 安全降落无人机
2. 加锁
3. 释放API控制

使用：
    python3 land.py
"""

import airsim
import sys

def main():
    print("=" * 70)
    print("         AirSim无人机降落")
    print("=" * 70)
    print()
    
    # 连接到AirSim
    print("[1/3] 连接到AirSim...")
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✓ 连接成功")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return 1
    
    # 降落
    print("\n[2/3] 降落中...")
    print("请稍候...")
    client.landAsync().join()
    print("✓ 降落完成")
    
    # 加锁和释放控制
    print("\n[3/3] 加锁和释放控制...")
    client.armDisarm(False)
    print("✓ 无人机已加锁")
    
    client.enableApiControl(False)
    print("✓ API控制已释放")
    
    print("\n" + "=" * 70)
    print("         降落完成！")
    print("=" * 70)
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


