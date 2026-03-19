#!/usr/bin/env python3
"""
综合API测试脚本
测试TravelUAV项目中使用的所有主要API接口

支持的API:
1. OpenAI GPT-4 API
2. AirSim 模拟器 API  
3. msgpackrpc 服务器连接

使用方法:
    python test_api_comprehensive.py --test all
    python test_api_comprehensive.py --test openai
    python test_api_comprehensive.py --test airsim
    python test_api_comprehensive.py --test msgpack
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path


# ============================================================================
# 1. OpenAI GPT API 测试
# ============================================================================

def test_openai_api():
    """测试OpenAI GPT-4 API连接"""
    print("\n" + "="*70)
    print(" 测试 1: OpenAI GPT-4 API")
    print("="*70)
    
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 错误: 未安装openai库")
        print("   请运行: pip install openai")
        return False
    
    # 检查API密钥
    api_key = os.environ.get('OPENAI_API_KEY', 'sk-7zZMomkPoQGrnxe1Xqar5O17AfaqShEMALVqucYCIuzr6dr5')
    if not api_key:
        print("❌ 错误: 未设置环境变量 OPENAI_API_KEY")
        print("\n请设置API密钥:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return False
    
    print(f"✓ API Key已设置 (前5位: {api_key[:5]}...)")
    
    # 创建客户端 - 使用第三方API服务 (aizex.top)
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://aizex.top/v1"
        )
        print("✓ OpenAI客户端创建成功 (使用第三方服务: aizex.top)")
    except Exception as e:
        print(f"❌ 创建客户端失败: {e}")
        return False
    
    # 测试基础文本生成
    try:
        print("\n[测试 1.1] 基础文本生成...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一个无人机导航助手。"},
                {"role": "user", "content": "简单介绍一下无人机视觉导航的基本原理。"}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        print(f"✓ GPT-4响应成功")
        print(f"  回复内容: {content[:100]}...")
        print(f"  Token使用: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return False
    
    # 测试带图像的Vision API
    try:
        print("\n[测试 1.2] GPT-4 Vision API...")
        import base64
        import cv2
        
        # 创建一个简单的测试图像（模拟无人机视角）
        test_img = np.zeros((256, 256, 3), dtype=np.uint8)
        # 绘制一些简单的几何形状
        cv2.rectangle(test_img, (50, 50), (200, 200), (0, 255, 0), 3)
        cv2.circle(test_img, (128, 128), 40, (0, 0, 255), -1)
        cv2.putText(test_img, 'UAV VIEW', (60, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 编码为base64
        _, img_encoded = cv2.imencode('.jpg', test_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "描述这张图像中的内容。"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "low"
                    }}
                ]
            }],
            max_tokens=150
        )
        
        content = response.choices[0].message.content
        print(f"✓ Vision API响应成功")
        print(f"  回复内容: {content[:100]}...")
        print(f"  Token使用: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"⚠️  Vision API测试失败: {e}")
        print("   (可能需要更高权限的API key)")
    
    print("\n✅ OpenAI API测试通过！")
    return True


# ============================================================================
# 2. AirSim API 测试
# ============================================================================

def test_airsim_api():
    """测试AirSim模拟器API连接"""
    print("\n" + "="*70)
    print(" 测试 2: AirSim 模拟器 API")
    print("="*70)
    
    try:
        import airsim
        import cv2
    except ImportError as e:
        print(f"❌ 错误: 缺少必要的库 - {e}")
        print("   请运行: pip install airsim opencv-python")
        return False
    
    # 连接参数
    airsim_ip = os.environ.get('AIRSIM_IP', '127.0.0.1')
    timeout = 10
    
    print(f"正在连接到AirSim服务器...")
    print(f"  IP地址: {airsim_ip}")
    print(f"  超时时间: {timeout}秒")
    
    # 创建客户端
    try:
        client = airsim.MultirotorClient(ip=airsim_ip, timeout_value=timeout)
        client.confirmConnection()
        print("✓ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n提示:")
        print("  1. 确保Unreal Engine/AirSim模拟器正在运行")
        print("  2. 检查IP地址是否正确")
        print("  3. 检查防火墙设置")
        return False
    
    # 测试基本状态获取
    try:
        print("\n[测试 2.1] 获取无人机状态...")
        state = client.getMultirotorState()
        
        position = state.kinematics_estimated.position
        print(f"✓ 成功获取状态")
        print(f"  位置: x={position.x_val:.2f}, y={position.y_val:.2f}, z={position.z_val:.2f}")
        print(f"  时间戳: {state.timestamp}")
        
    except Exception as e:
        print(f"❌ 获取状态失败: {e}")
        return False
    
    # 测试碰撞检测
    try:
        print("\n[测试 2.2] 检测碰撞信息...")
        collision_info = client.simGetCollisionInfo()
        print(f"✓ 成功获取碰撞信息")
        print(f"  是否碰撞: {collision_info.has_collided}")
        if collision_info.has_collided:
            print(f"  碰撞对象: {collision_info.object_name}")
            
    except Exception as e:
        print(f"⚠️  碰撞检测失败: {e}")
    
    # 测试图像获取
    try:
        print("\n[测试 2.3] 获取相机图像...")
        
        # 设置相机分辨率
        img_width, img_height = 256, 256
        client.simSetCameraResolution("0", img_width, img_height)
        
        # 请求RGB图像
        start_time = time.time()
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False)
        ])
        duration = time.time() - start_time
        
        response = responses[0]
        print(f"✓ 成功获取图像")
        print(f"  分辨率: {response.width}x{response.height}")
        print(f"  耗时: {duration:.4f}秒")
        
        # 保存图像
        img_bgra = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_bgra = img_bgra.reshape(response.height, response.width, 4)
        img_bgr = img_bgra[:, :, :3]
        
        output_file = "test_airsim_image.png"
        cv2.imwrite(output_file, img_bgr)
        print(f"  图像已保存: {output_file}")
        
    except Exception as e:
        print(f"❌ 图像获取失败: {e}")
        return False
    
    # 测试GPS数据
    try:
        print("\n[测试 2.4] 获取GPS数据...")
        gps_data = client.getGpsData()
        print(f"✓ 成功获取GPS数据")
        print(f"  纬度: {gps_data.gnss.geo_point.latitude}")
        print(f"  经度: {gps_data.gnss.geo_point.longitude}")
        print(f"  高度: {gps_data.gnss.geo_point.altitude}")
        
    except Exception as e:
        print(f"⚠️  GPS数据获取失败: {e}")
    
    print("\n✅ AirSim API测试通过！")
    return True


# ============================================================================
# 3. msgpackrpc 测试
# ============================================================================

def test_msgpack_rpc():
    """测试msgpackrpc服务器连接"""
    print("\n" + "="*70)
    print(" 测试 3: msgpackrpc 服务器")
    print("="*70)
    
    try:
        import msgpackrpc
    except ImportError:
        print("❌ 错误: 未安装msgpackrpc库")
        print("   请运行: pip install msgpack-rpc-python")
        return False
    
    # 连接参数
    server_ip = os.environ.get('MSGPACK_SERVER_IP', '127.0.0.1')
    server_port = int(os.environ.get('MSGPACK_SERVER_PORT', '18300'))
    
    print(f"正在连接到msgpackrpc服务器...")
    print(f"  地址: {server_ip}:{server_port}")
    
    try:
        client = msgpackrpc.Client(
            msgpackrpc.Address(server_ip, server_port),
            timeout=10
        )
        print("✓ 客户端创建成功")
        
    except Exception as e:
        print(f"❌ 创建客户端失败: {e}")
        print("\n提示:")
        print("  1. 确保msgpackrpc服务器正在运行")
        print(f"  2. 检查地址 {server_ip}:{server_port} 是否正确")
        return False
    
    # 测试简单的RPC调用（ping/echo）
    try:
        print("\n[测试 3.1] 测试基础RPC调用...")
        # 尝试调用一个简单的测试方法
        result = client.call('ping')
        print(f"✓ ping调用成功: {result}")
        
    except Exception as e:
        print(f"⚠️  基础RPC调用失败: {e}")
        print("   (服务器可能未实现ping方法)")
    
    print("\n✅ msgpackrpc测试完成！")
    return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TravelUAV API综合测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试所有API
  python test_api_comprehensive.py --test all
  
  # 只测试OpenAI
  python test_api_comprehensive.py --test openai
  
  # 只测试AirSim
  python test_api_comprehensive.py --test airsim
  
  # 只测试msgpackrpc
  python test_api_comprehensive.py --test msgpack

环境变量:
  OPENAI_API_KEY       OpenAI API密钥
  AIRSIM_IP           AirSim服务器IP (默认: 127.0.0.1)
  MSGPACK_SERVER_IP   msgpackrpc服务器IP (默认: 127.0.0.1)
  MSGPACK_SERVER_PORT msgpackrpc服务器端口 (默认: 18300)
        """
    )
    
    parser.add_argument(
        '--test',
        type=str,
        default='all',
        choices=['all', 'openai', 'airsim', 'msgpack'],
        help='选择要测试的API类型'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*20 + "TravelUAV API 综合测试")
    print("="*70)
    print(f"\n测试类型: {args.test}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 执行测试
    if args.test in ['all', 'openai']:
        results['openai'] = test_openai_api()
    
    if args.test in ['all', 'airsim']:
        results['airsim'] = test_airsim_api()
    
    if args.test in ['all', 'msgpack']:
        results['msgpack'] = test_msgpack_rpc()
    
    # 总结
    print("\n" + "="*70)
    print(" "*25 + "测试总结")
    print("="*70)
    
    for api_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {api_name.upper():15s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！API连接正常。")
    else:
        print("⚠️  部分测试失败，请检查相应的配置。")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

