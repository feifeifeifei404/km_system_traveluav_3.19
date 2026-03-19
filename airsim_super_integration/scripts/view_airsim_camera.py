#!/usr/bin/env python3
"""
实时查看 AirSim 无人机视角
从 AirSim 获取相机图像并实时显示
"""
import sys
sys.path.insert(0, '/mnt/data/TravelUAV')

import airsim
import cv2
import numpy as np
import time

def main():
    # 连接 AirSim
    print("连接 AirSim...")
    client = airsim.MultirotorClient(ip='127.0.0.1', port=25001)
    client.confirmConnection()
    print("✅ 连接成功")

    print("按 Q 退出，按 1/2/3 切换相机")
    cameras = ['FrontCamera', 'LeftCamera', 'RightCamera']
    cam_idx = 0

    while True:
        try:
            # 获取相机图像
            cam_name = cameras[cam_idx]
            responses = client.simGetImages([
                airsim.ImageRequest(cam_name, airsim.ImageType.Scene, False, False)
            ], vehicle_name='Drone_1')

            if responses and len(responses) > 0:
                resp = responses[0]
                img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
                img = img1d.reshape(resp.height, resp.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 获取无人机位置
                state = client.getMultirotorState(vehicle_name='Drone_1')
                pos = state.kinematics_estimated.position
                # NED -> ENU 显示
                x_enu = pos.y_val
                y_enu = pos.x_val
                z_enu = -pos.z_val

                # 在图像上叠加信息
                cv2.putText(img, f"ENU: ({x_enu:.1f}, {y_enu:.1f}, {z_enu:.1f})m",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Camera: {cam_name}",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(img, "1/2/3: 切换相机  Q: 退出",
                           (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('AirSim 无人机视角', img)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                cam_idx = 0
            elif key == ord('2'):
                cam_idx = 1
            elif key == ord('3'):
                cam_idx = 2

        except Exception as e:
            print(f"错误: {e}")
            time.sleep(0.5)

    cv2.destroyAllWindows()
    print("已退出")

if __name__ == '__main__':
    main()
