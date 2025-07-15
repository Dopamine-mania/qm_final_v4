#!/usr/bin/env python3
"""
🌙 音乐疗愈AI系统 4.0版本 - 快速启动脚本
一键初始化并启动系统
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    # 检查Python版本
    if sys.version_info < (3.7, 0):
        print("❌ 需要Python 3.7或更高版本")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    
    # 检查必要的Python包
    required_packages = ['gradio', 'numpy', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: 未安装")
    
    if missing_packages:
        print(f"\\n⚠️ 缺少必要包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 检查ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ ffmpeg: 已安装")
        else:
            print("❌ ffmpeg: 不可用")
            return False
    except Exception:
        print("❌ ffmpeg: 未安装")
        print("请安装ffmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: 下载ffmpeg并添加到PATH")
        return False
    
    return True

def check_video_files():
    """检查视频文件"""
    print("\\n🎬 检查视频文件...")
    
    video_dir = Path("materials/video")
    
    if not video_dir.exists():
        print(f"❌ 视频目录不存在: {video_dir}")
        return False
    
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        print(f"❌ 未找到视频文件: {video_dir}")
        print("请确保32.mp4和56.mp4位于materials/video/目录下")
        return False
    
    print(f"✅ 找到视频文件: {len(video_files)} 个")
    for video in video_files:
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  - {video.name}: {size_mb:.1f} MB")
    
    return True

def run_tests():
    """运行MVP测试"""
    print("\\n🧪 运行MVP测试...")
    
    try:
        result = subprocess.run([sys.executable, 'test_mvp.py'], 
                              capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ MVP测试通过")
            return True
        else:
            print("❌ MVP测试失败")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ MVP测试超时")
        return False
    except Exception as e:
        print(f"❌ MVP测试异常: {e}")
        return False

def start_application():
    """启动应用程序"""
    print("\\n🚀 启动应用程序...")
    
    try:
        # 启动Gradio应用
        subprocess.run([sys.executable, 'gradio_retrieval_4.0.py'])
    except KeyboardInterrupt:
        print("\\n👋 应用程序已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("🌙 音乐疗愈AI系统 4.0版本 - 快速启动")
    print("🔍 从生成到检索的智能疗愈系统")
    print("=" * 60)
    
    # 1. 检查依赖项
    if not check_dependencies():
        print("\\n❌ 依赖项检查失败，请解决问题后重试")
        return False
    
    # 2. 检查视频文件
    if not check_video_files():
        print("\\n❌ 视频文件检查失败，请解决问题后重试")
        return False
    
    # 3. 询问是否运行测试
    print("\\n" + "=" * 40)
    run_test = input("是否运行MVP测试？(y/n): ").lower().strip()
    
    if run_test in ['y', 'yes', '是']:
        if not run_tests():
            print("\\n⚠️ 测试未完全通过，但您仍可以尝试启动应用")
            continue_anyway = input("是否继续启动应用？(y/n): ").lower().strip()
            if continue_anyway not in ['y', 'yes', '是']:
                return False
    
    # 4. 启动应用
    print("\\n" + "=" * 40)
    print("🎯 即将启动Web界面...")
    print("📱 启动后将自动打开浏览器")
    print("🌐 或手动访问显示的本地地址")
    print("⚡ 首次使用请点击'初始化系统'按钮")
    print("=" * 40)
    
    input("按回车键继续...")
    start_application()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\\n💡 启动失败，请检查上述错误信息")
            print("🔧 您可以手动运行以下命令进行调试:")
            print("   python test_mvp.py")
            print("   python gradio_retrieval_4.0.py")
    except KeyboardInterrupt:
        print("\\n👋 已取消启动")
    except Exception as e:
        print(f"\\n❌ 启动过程中发生错误: {e}")
        print("🔧 请检查错误信息并重试")