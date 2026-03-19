#!/usr/bin/env python3
"""
API问题诊断工具 - 帮助找出API连接失败的具体原因

使用方法:
    python diagnose_api.py
"""

import os
import sys
import socket
import subprocess


def check_env_variable(var_name):
    """检查环境变量是否设置"""
    value = os.environ.get(var_name)
    if value:
        # 对于API密钥，只显示前几位
        if 'KEY' in var_name or 'TOKEN' in var_name:
            display_value = f"{value[:8]}..." if len(value) > 8 else "***"
        else:
            display_value = value
        print(f"  ✅ {var_name} = {display_value}")
        return True
    else:
        print(f"  ❌ {var_name} 未设置")
        return False


def check_network_connectivity(host, port=443, timeout=5):
    """检查网络连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"  ✅ 可以连接到 {host}:{port}")
            return True
        else:
            print(f"  ❌ 无法连接到 {host}:{port} (错误码: {result})")
            return False
    except socket.gaierror:
        print(f"  ❌ 无法解析域名 {host}")
        return False
    except Exception as e:
        print(f"  ❌ 连接测试失败: {e}")
        return False


def check_openai_issues():
    """诊断OpenAI API问题"""
    print("\n" + "="*70)
    print("🔍 诊断 OpenAI API")
    print("="*70)
    
    issues = []
    suggestions = []
    
    # 1. 检查API密钥
    print("\n[1] 检查API密钥...")
    if not check_env_variable('OPENAI_API_KEY'):
        issues.append("未设置OPENAI_API_KEY环境变量")
        suggestions.append("运行: export OPENAI_API_KEY='your-api-key-here'")
    
    # 2. 检查网络连接
    print("\n[2] 检查网络连接...")
    
    # 测试能否访问OpenAI服务器
    openai_reachable = check_network_connectivity('api.openai.com', 443, timeout=5)
    if not openai_reachable:
        issues.append("无法连接到api.openai.com")
        suggestions.append("检查网络连接或配置代理")
        
        # 检查是否在中国大陆（需要代理）
        print("\n  💡 如果你在中国大陆，可能需要配置代理:")
        print("     export https_proxy=http://127.0.0.1:7890")
        print("     export http_proxy=http://127.0.0.1:7890")
    
    # 3. 检查是否安装了openai库
    print("\n[3] 检查Python包...")
    try:
        import openai
        print(f"  ✅ openai 库已安装 (版本: {openai.__version__})")
    except ImportError:
        issues.append("未安装openai库")
        suggestions.append("运行: pip install openai")
    
    # 4. 测试实际API调用（如果有密钥）
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key and openai_reachable:
        print("\n[4] 测试实际API调用...")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # 尝试一个最小的请求
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 使用更便宜的模型测试
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            print(f"  ✅ API调用成功！")
            print(f"     响应: {response.choices[0].message.content}")
            
        except Exception as e:
            error_msg = str(e)
            issues.append(f"API调用失败: {error_msg}")
            
            if "authentication" in error_msg.lower() or "401" in error_msg:
                suggestions.append("API密钥无效，请检查密钥是否正确")
            elif "quota" in error_msg.lower() or "429" in error_msg:
                suggestions.append("API配额已用完，请检查账户余额")
            elif "connection" in error_msg.lower():
                suggestions.append("网络连接问题，尝试配置代理")
            else:
                suggestions.append(f"其他错误: {error_msg}")
    
    return issues, suggestions


def check_airsim_issues():
    """诊断AirSim API问题"""
    print("\n" + "="*70)
    print("🔍 诊断 AirSim API")
    print("="*70)
    
    issues = []
    suggestions = []
    
    # 1. 检查是否安装了airsim库
    print("\n[1] 检查Python包...")
    try:
        import airsim
        print(f"  ✅ airsim 库已安装")
    except ImportError:
        issues.append("未安装airsim库")
        suggestions.append("运行: pip install airsim")
        return issues, suggestions
    
    # 2. 检查AirSim服务器连接
    print("\n[2] 检查AirSim服务器...")
    airsim_ip = os.environ.get('AIRSIM_IP', '127.0.0.1')
    airsim_port = 41451  # AirSim默认端口
    
    print(f"  目标地址: {airsim_ip}:{airsim_port}")
    
    if check_network_connectivity(airsim_ip, airsim_port, timeout=3):
        print("  ✅ AirSim服务器正在运行")
    else:
        issues.append("无法连接到AirSim服务器")
        suggestions.append("启动Unreal Engine并运行AirSim模拟器")
        suggestions.append("确认模拟器设置中启用了API服务器")
        
        # 检查是否有进程在运行
        print("\n  💡 检查UE4进程...")
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'UE4'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"  ⚠️  发现UE4进程，但无法连接API")
                suggestions.append("检查AirSim配置文件settings.json")
            else:
                print(f"  ❌ 未发现UE4/Unreal进程在运行")
        except Exception as e:
            print(f"  ⚠️  进程检查失败: {e}")
    
    # 3. 检查配置文件
    print("\n[3] 检查AirSim配置文件...")
    settings_paths = [
        os.path.expanduser("~/Documents/AirSim/settings.json"),
        os.path.expanduser("~/.airsim/settings.json"),
    ]
    
    found_settings = False
    for path in settings_paths:
        if os.path.exists(path):
            print(f"  ✅ 找到配置文件: {path}")
            found_settings = True
            
            # 读取并检查配置
            try:
                import json
                with open(path, 'r') as f:
                    settings = json.load(f)
                    if 'ApiServerPort' in settings:
                        print(f"     API端口: {settings['ApiServerPort']}")
            except:
                pass
            break
    
    if not found_settings:
        print(f"  ⚠️  未找到AirSim配置文件")
        suggestions.append("创建配置文件: ~/Documents/AirSim/settings.json")
    
    return issues, suggestions


def main():
    print("="*70)
    print(" "*20 + "API 问题诊断工具")
    print("="*70)
    
    all_issues = []
    all_suggestions = []
    
    # 诊断OpenAI
    openai_issues, openai_suggestions = check_openai_issues()
    all_issues.extend([f"[OpenAI] {i}" for i in openai_issues])
    all_suggestions.extend([f"[OpenAI] {s}" for s in openai_suggestions])
    
    # 诊断AirSim
    airsim_issues, airsim_suggestions = check_airsim_issues()
    all_issues.extend([f"[AirSim] {i}" for i in airsim_issues])
    all_suggestions.extend([f"[AirSim] {s}" for s in airsim_suggestions])
    
    # 总结
    print("\n" + "="*70)
    print(" "*25 + "诊断总结")
    print("="*70)
    
    if all_issues:
        print("\n❌ 发现的问题:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    
    if all_suggestions:
        print("\n💡 建议的解决方案:")
        for i, suggestion in enumerate(all_suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    if not all_issues:
        print("\n✅ 未发现明显问题，API应该可以正常工作")
    
    print("\n" + "="*70)
    
    # 快速修复指南
    if all_issues:
        print("\n🚀 快速修复步骤:\n")
        
        if any('OPENAI_API_KEY' in i for i in all_issues):
            print("1️⃣  设置OpenAI API密钥:")
            print("   export OPENAI_API_KEY='sk-your-key-here'")
            print()
        
        if any('openai.com' in i for i in all_issues):
            print("2️⃣  配置网络代理（如果在中国大陆）:")
            print("   export https_proxy=http://127.0.0.1:7890")
            print("   export http_proxy=http://127.0.0.1:7890")
            print()
        
        if any('AirSim' in i for i in all_issues):
            print("3️⃣  启动AirSim模拟器:")
            print("   - 启动Unreal Engine")
            print("   - 加载AirSim项目")
            print("   - 点击Play开始模拟")
            print()


if __name__ == '__main__':
    main()

