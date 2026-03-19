import airsim
import cv2  # 如果没有安装，请运行: pip install opencv-python
import numpy as np
import time
import os

# -------------------------------------------------------------------
# 配置参数 (你可以修改这里的值来进行不同的测试)
# -------------------------------------------------------------------
# 图像分辨率
IMG_WIDTH = 640
IMG_HEIGHT = 480

# 是否请求浮点数格式的图像 (这是导致性能问题的关键参数)
# True: 数据量大，可能慢 | False: 数据量小，应该很快
PIXELS_AS_FLOAT = False 

# AirSim服务器的IP地址 (如果UE4运行在另一台电脑上，请修改)
AIRSIM_IP = "127.0.0.1"

# 连接超时时间 (秒)，如果请求超过这个时间，就会报错
# 如果你觉得服务器慢，可以适当增加这个值
TIMEOUT = 10 

# 保存图片的文件名
FILENAME = "test_output.png"
# -------------------------------------------------------------------


def test_image_retrieval():
    """
    连接到AirSim并测试图像获取功能、速度和正确性
    """
    # 1. 创建并连接客户端
    client = airsim.MultirotorClient(ip=AIRSIM_IP, timeout_value=TIMEOUT)
    print(f"正在尝试连接到 AirSim 服务器 (IP: {AIRSIM_IP}, 超时: {TIMEOUT}s)...")
    
    try:
        client.confirmConnection()
        print("连接成功！")
    except Exception as e:
        print(f"连接失败: {e}")
        print("请确保 Unreal Engine / AirSim 模拟器正在运行。")
        return

    print("\n" + "="*50)
    print("测试配置:")
    print(f"  - 分辨率: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"  - 浮点像素 (pixels_as_float): {PIXELS_AS_FLOAT}")
    print("="*50 + "\n")

    try:
        # 2. 准备图像请求
        # airsim.ImageType.Scene 表示获取正常的场景图像
        request = airsim.ImageRequest(
            "0",  # "0" 是默认的前置摄像头
            airsim.ImageType.Scene,
            pixels_as_float=PIXELS_AS_FLOAT
        )

        # 预先设置一下相机分辨率，确保获取的是我们想要的大小
        client.simSetCameraResolution("0", IMG_WIDTH, IMG_HEIGHT)
        print("已设置相机分辨率。")

        # 3. 发送请求并计时
        print("正在发送图像请求...")
        start_time = time.time()
        
        responses = client.simGetImages([request])
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"成功收到响应！耗时: {duration:.4f} 秒")

        # 4. 处理并保存图像
        response = responses[0]
        
        if PIXELS_AS_FLOAT:
            # 浮点图是单通道HDR图像，需要转换为的可视的8位图
            img_float = np.array(response.image_data_float, dtype=np.float32)
            img_float = img_float.reshape(response.height, response.width)
            # 将像素值从 [min, max] 范围归一化到 [0, 255]
            img_to_save = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:
            # 8位图是BGRA格式
            img_bgra = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_bgra = img_bgra.reshape(response.height, response.width, 4)
            # 去掉Alpha通道，保留BGR
            img_to_save = img_bgra[:, :, :3]

        cv2.imwrite(FILENAME, img_to_save)
        print(f"图像已成功保存到: {os.path.abspath(FILENAME)}")

    except Exception as e:
        print("\n" + "!"*50)
        print(f"测试失败！在请求图像时发生错误。")
        print(f"错误详情: {e}")
        print("!"*50)


if __name__ == "__main__":
    test_image_retrieval()