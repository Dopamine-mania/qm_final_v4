#!/usr/bin/env python3
"""
单个视频音乐特征提取测试
"""
import os
import sys
import subprocess
import shutil
import time

def test_single_extraction():
    """测试单个视频的特征提取"""
    # 路径配置
    video_path = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_1min/32_1min_01.mp4"
    
    # 创建临时目录
    temp_dir = "temp_test"
    audio_dir = os.path.join(temp_dir, "audio")
    features_dir = os.path.join(temp_dir, "features")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    try:
        print("🔍 测试单个视频特征提取...")
        print(f"视频文件: {video_path}")
        
        # 1. 提取音频
        audio_path = os.path.join(audio_dir, "32_1min_01.wav")
        print("1. 提取音频...")
        
        cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', '-y', audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 音频提取失败: {result.stderr}")
            return False
        
        if not os.path.exists(audio_path):
            print("❌ 音频文件未生成")
            return False
        
        print(f"✅ 音频提取成功: {os.path.getsize(audio_path)} bytes")
        
        # 2. 测试CLAMP3加载
        print("2. 测试CLAMP3模型加载...")
        test_code = '''
import sys
sys.path.append("code")
from utils import *
print("✅ CLAMP3模块加载成功")
'''
        
        result = subprocess.run(['python', '-c', test_code], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ CLAMP3加载失败: {result.stderr}")
            return False
        
        print(result.stdout)
        
        # 3. 运行特征提取
        print("3. 运行特征提取...")
        cmd = ['python', 'clamp3_embd.py', audio_dir, features_dir, '--get_global']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 特征提取失败: {result.stderr}")
            return False
        
        # 检查输出
        feature_file = os.path.join(features_dir, "32_1min_01.npy")
        if os.path.exists(feature_file):
            print(f"✅ 特征文件生成成功: {os.path.getsize(feature_file)} bytes")
            
            # 测试加载特征
            import numpy as np
            features = np.load(feature_file)
            print(f"✅ 特征维度: {features.shape}")
            return True
        else:
            print("❌ 特征文件未生成")
            return False
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("🧹 临时文件已清理")

if __name__ == "__main__":
    success = test_single_extraction()
    if success:
        print("\n✅ 单个视频特征提取测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 单个视频特征提取测试失败！")
        sys.exit(1)