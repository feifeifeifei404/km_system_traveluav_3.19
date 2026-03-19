#!/usr/bin/env python3
"""
重置无人机到正常飞行高度
"""
import airsim
import time

print("连接 AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("✓ 已连接")

print("\n重置无人机...")
client.reset()
time.sleep(1)

print("启用 API 控制...")
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(1)

print("起飞到 2 米高度...")
client.takeoffAsync().join()
time.sleep(2)

# 移动到安全位置（x=0, y=0, z=-2 in NED = 2米高）
print("移动到安全位置 (0, 0, -2) NED...")
client.moveToPositionAsync(0, 0, -2, 1).join()
time.sleep(1)

# 查看最终位置
state = client.getMultirotorState()
pos = state.kinematics_estimated.position
print(f"\n✓ 无人机已就位！")
print(f"AirSim 位置: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")
print(f"ROS 位置（预计）: x={pos.y_val:.2f}, y={pos.x_val:.2f}, z={-pos.z_val:.2f}")
print(f"\n无人机现在在地面以上约 2 米，可以开始规划了！")
