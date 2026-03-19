#!/usr/bin/env python3
"""
OpenAI API测试 - 自动禁用代理版本

使用方法:
    python test_openai_no_proxy.py
"""

import os
import sys


def test_openai():
    """测试OpenAI API - 禁用代理"""
    
    # 临时禁用代理（仅在这个脚本运行期间）
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
    old_proxy_settings = {}
    
    for var in proxy_vars:
        if var in os.environ:
            old_proxy_settings[var] = os.environ[var]
            del os.environ[var]
    
    print("="*70)
    print(" "*20 + "OpenAI API 测试")
    print(" "*18 + "(已禁用代理)")
    print("="*70)
    
    try:
        from openai import OpenAI
        
        api_key = 'sk-7zZMomkPoQGrnxe1Xqar5O17AfaqShEMALVqucYCIuzr6dr5'
        base_url = 'https://aizex.top/v1'
        
        print(f"\n配置信息:")
        print(f"  API密钥: {api_key[:20]}...")
        print(f"  服务地址: {base_url}")
        print(f"  模型: gpt-4")
        print(f"  代理: 已禁用 ✓")
        
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=15.0)
        
        print(f"\n正在发送测试请求...")
        
        # 测试1: 简单对话
        print("\n[测试 1] 简单对话")
        response = client.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'user', 'content': '你好，请用一句话介绍你自己'}
            ],
            max_tokens=100
        )
        
        print(f"  ✅ 成功")
        print(f"  回复: {response.choices[0].message.content}")
        print(f"  Token: {response.usage.total_tokens}")
        
        # 测试2: 无人机相关问题
        print("\n[测试 2] 无人机导航问题")
        response = client.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': '你是一个无人机导航专家。'},
                {'role': 'user', 'content': '如何让无人机避开障碍物？请简短回答。'}
            ],
            max_tokens=150
        )
        
        print(f"  ✅ 成功")
        print(f"  回复: {response.choices[0].message.content[:100]}...")
        print(f"  Token: {response.usage.total_tokens}")
        
        print("\n" + "="*70)
        print("✅ 所有测试通过！OpenAI API 工作正常")
        print("="*70)
        
        return True
        
    except ImportError as e:
        print(f"❌ 缺少必要的库: {e}")
        print("   请运行: pip install openai")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    finally:
        # 恢复原来的代理设置
        for var, value in old_proxy_settings.items():
            os.environ[var] = value


if __name__ == '__main__':
    success = test_openai()
    sys.exit(0 if success else 1)

