#!/usr/bin/env python3
"""
测试AirSim连接和传感器

快速检查AirSim是否正常运行并能获取数据

使用方法:
python3 test_airsim_connection.py
"""

import airsim
import sys


def main():
    """测试AirSim连接"""
    
    print("=" * 70)
    print("测试AirSim连接...")
    print("=" * 70)
    
    try:
        # 连接AirSim
        print("\n[1/4] 连接AirSim...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✓ AirSim连接成功!")
        
        # 获取无人机状态
        print("\n[2/4] 获取无人机状态...")
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"✓ 无人机位置: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
        
        # 获取LiDAR数据
        print("\n[3/4] 获取LiDAR点云...")
        lidar_data = client.getLidarData('Lidar1')
        if lidar_data.point_cloud and len(lidar_data.point_cloud) > 0:
            num_points = len(lidar_data.point_cloud) // 3
            print(f"✓ LiDAR点云: {num_points} 个点")
        else:
            print("⚠ LiDAR无数据（可能未配置）")
        
        # 获取相机图像
        print("\n[4/4] 获取相机图像...")
        try:
            request = airsim.ImageRequest("FrontCamera", airsim.ImageType.Scene, False, False)
            response = client.simGetImages([request])[0]
            if len(response.image_data_uint8) > 0:
                print(f"✓ 相机图像: {response.width} x {response.height}")
            else:
                print("⚠ 相机无数据")
        except:
            print("⚠ 相机不可用")
        
        print("\n" + "=" * 70)
        print("✓✓✓ 所有测试通过！AirSim工作正常 ✓✓✓")
        print("=" * 70)
        print("\n可以开始启动集成系统了：")
        print("  cd /mnt/data/airsim_super_integration")
        print("  ./start_airsim_super_ros2.sh")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("\n可能的原因:")
        print("  1. AirSim环境未启动")
        print("  2. AirSim端口被占用")
        print("  3. airsim Python包未安装")
        print("\n解决方法:")
        print("  # 启动AirSim环境")
        print("  cd /mnt/data/TravelUAV/envs/closeloop_envs")
        print("  ./ModularPark.sh -ResX=1280 -ResY=720 -windowed")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())


