#!/usr/bin/env python3
"""
第三方API服务测试脚本 - 适用于 aizex.top 等中转服务

使用方法:
    python test_third_party_api.py
"""

import os
import sys
from openai import OpenAI


def test_third_party_openai():
    """测试第三方OpenAI API服务"""
    print("="*70)
    print(" "*15 + "第三方OpenAI API测试 (aizex.top)")
    print("="*70)
    
    # API配置
    api_key = os.environ.get('OPENAI_API_KEY', 'sk-7zZMomkPoQGrnxe1Xqar5O17AfaqShEMALVqucYCIuzr6dr5')
    
    # 第三方服务的常见base_url格式
    possible_base_urls = [
        "https://api.aizex.top/v1",
        "https://aizex.top/v1",
        "https://api.aizex.top",
        "https://one.aizex.top/v1",
    ]
    
    print(f"\n✅ API密钥: {api_key[:20]}...")
    print(f"\n🔍 正在尝试不同的API地址...\n")
    
    for base_url in possible_base_urls:
        print(f"尝试连接: {base_url}")
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=10.0
            )
            
            # 发送测试请求
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "user", "content": "你好，请回复：API连接成功！"}
                ],
                max_tokens=50
            )
            
            content = response.choices[0].message.content
            
            print(f"  ✅ 成功！这是正确的API地址")
            print(f"  📝 GPT回复: {content}")
            print(f"  💰 Token使用: {response.usage.total_tokens}")
            print(f"\n" + "="*70)
            print(f"✅ API测试通过！")
            print(f"正确的配置:")
            print(f"  base_url = '{base_url}'")
            print(f"  api_key = '{api_key[:20]}...'")
            print("="*70)
            
            return True, base_url
            
        except Exception as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower():
                print(f"  ❌ 超时")
            elif '404' in error_msg or 'not found' in error_msg.lower():
                print(f"  ❌ 地址不存在")
            elif '401' in error_msg or 'authentication' in error_msg.lower():
                print(f"  ❌ 认证失败（密钥可能无效）")
            elif 'connection' in error_msg.lower():
                print(f"  ❌ 连接失败")
            else:
                print(f"  ❌ 错误: {error_msg[:60]}...")
        print()
    
    print("="*70)
    print("❌ 所有常见地址都无法连接")
    print("\n💡 建议:")
    print("  1. 访问 https://aizex.top 查看API文档")
    print("  2. 找到正确的 API Base URL")
    print("  3. 检查API密钥是否有效且有余额")
    print("  4. 确认网络连接正常")
    print("="*70)
    
    return False, None


def update_test_scripts(base_url):
    """更新测试脚本以使用正确的base_url"""
    print("\n" + "="*70)
    print("📝 更新测试脚本配置")
    print("="*70)
    
    # 读取并更新 test_api_simple.py
    try:
        simple_file = '/mnt/data/TravelUAV/test_api_simple.py'
        with open(simple_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修改 OpenAI 客户端创建部分
        old_line = 'client = OpenAI(api_key=api_key)'
        new_line = f'client = OpenAI(api_key=api_key, base_url="{base_url}")'
        
        if old_line in content and base_url not in content:
            content = content.replace(old_line, new_line)
            with open(simple_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已更新: test_api_simple.py")
        else:
            print(f"⚠️  test_api_simple.py 可能已包含base_url配置")
    except Exception as e:
        print(f"⚠️  更新test_api_simple.py失败: {e}")
    
    print("\n现在你可以运行:")
    print("  python test_api_simple.py")
    print("来测试更新后的配置！")
    print("="*70)


def main():
    success, base_url = test_third_party_openai()
    
    if success and base_url:
        # 询问是否更新测试脚本
        print("\n是否要更新其他测试脚本以使用这个配置？")
        update_test_scripts(base_url)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

