#!/usr/bin/env python3
"""
简单API测试脚本 - 快速验证API是否可用

使用方法:
    python test_api_simple.py
"""

import os
import sys


def test_openai():
    """快速测试OpenAI API"""
    print("\n🔍 测试OpenAI API...")
    
    try:
        from openai import OpenAI
        
        api_key = os.environ.get('OPENAI_API_KEY', 'sk-7zZMomkPoQGrnxe1Xqar5O17AfaqShEMALVqucYCIuzr6dr5')
        if not api_key:
            print("❌ 请先设置环境变量: export OPENAI_API_KEY='your-key'")
            return False
        
        # 使用第三方API服务 (aizex.top)
        client = OpenAI(
            api_key=api_key,
            base_url="https://aizex.top/v1"
        )
        
        # 简单的测试请求
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "说'你好，TravelUAV!'"}],
            max_tokens=50
        )
        
        print(f"✅ OpenAI API正常")
        print(f"   响应: {response.choices[0].message.content}")
        print(f"   Token使用: {response.usage.total_tokens}")
        return True
        
    except ImportError:
        print("❌ 请安装openai库: pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAI API错误: {e}")
        return False


def test_airsim():
    """快速测试AirSim API"""
    print("\n🔍 测试AirSim API...")
    
    try:
        import airsim
        
        client = airsim.MultirotorClient(timeout_value=5)
        client.confirmConnection()
        
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        print(f"✅ AirSim API正常")
        print(f"   无人机位置: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
        return True
        
    except ImportError:
        print("❌ 请安装airsim库: pip install airsim")
        return False
    except Exception as e:
        print(f"❌ AirSim API错误: {e}")
        print(f"   提示: 确保Unreal Engine/AirSim模拟器正在运行")
        return False


def main():
    print("="*60)
    print("       TravelUAV - 快速API测试")
    print("="*60)
    
    results = {
        'OpenAI': test_openai(),
        'AirSim': test_airsim()
    }
    
    print("\n" + "="*60)
    print("测试结果:")
    for name, success in results.items():
        status = "✅ 正常" if success else "❌ 失败"
        print(f"  {name:10s}: {status}")
    
    print("="*60)
    
    if all(results.values()):
        print("🎉 所有API测试通过！")
        return 0
    else:
        print("⚠️  部分API测试失败，请检查配置")
        return 1


if __name__ == '__main__':
    sys.exit(main())

