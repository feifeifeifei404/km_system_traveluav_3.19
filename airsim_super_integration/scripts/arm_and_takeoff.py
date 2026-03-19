#!/usr/bin/env python3
"""
AirSim无人机解锁和起飞脚本

功能：
1. 连接到AirSim
2. 解锁无人机
3. 起飞到指定高度

使用：
    python3 arm_and_takeoff.py [高度]
    默认高度：2米
"""

import airsim
import sys
import time

def main():
    # 获取目标高度
    target_height = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    
    print("=" * 70)
    print("         AirSim无人机解锁和起飞")
    print("=" * 70)
    print()
    
    # 连接到AirSim
    print("[1/4] 连接到AirSim...")
    try:
        client = airsim.MultirotorClient(port=25001)
        client.confirmConnection()
        print("✓ 连接成功")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("\n请确保AirSim环境正在运行！")
        return 1
    
    # 检查当前状态
    print("\n[2/4] 检查无人机状态...")
    state = client.getMultirotorState()
    print(f"  当前位置: ({state.kinematics_estimated.position.x_val:.2f}, "
          f"{state.kinematics_estimated.position.y_val:.2f}, "
          f"{state.kinematics_estimated.position.z_val:.2f})")
    print(f"  是否着陆: {state.landed_state}")
    
    # 解锁无人机
    print("\n[3/4] 解锁无人机...")
    client.enableApiControl(True)
    print("✓ API控制已启用")
    
    client.armDisarm(True)
    print("✓ 无人机已解锁")
    
    # 起飞
    print(f"\n[4/4] 起飞到 {target_height} 米...")
    print("请稍候...")
    
    # 起飞（AirSim使用NED坐标系，Z向下为正）
    client.takeoffAsync().join()
    print("✓ 起飞完成")
    
    # 移动到指定高度
    print(f"调整到目标高度 {target_height} 米...")
    client.moveToZAsync(-target_height, 2).join()  # NED: 负Z = 向上
    
    time.sleep(1)
    
    # 检查最终状态
    final_state = client.getMultirotorState()
    final_z = -final_state.kinematics_estimated.position.z_val  # 转换为向上为正
    print(f"✓ 当前高度: {final_z:.2f} 米")
    
    print("\n" + "=" * 70)
    print("         无人机已准备就绪！")
    print("=" * 70)
    print()
    print("现在可以发送目标点：")
    print("  ./send_goal.sh 10 5 2")
    print()
    print("如需降落：")
    print("  python3 land.py")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


