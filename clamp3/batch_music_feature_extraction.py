#!/usr/bin/env python3
"""
批量音乐特征提取脚本
从所有视频片段中提取CLAMP3音乐特征，保持文件名一致性
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
import glob

def get_video_duration(video_path):
    """获取视频时长"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'csv=p=0', video_path
        ], capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0

def extract_audio_from_video(video_path, audio_path):
    """从视频中提取音频"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', 
            '-y', audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"音频提取失败 {video_path}: {e}")
        return False

def extract_features_for_segment(input_dir, output_dir, segment_type):
    """为特定时长的视频片段提取特征"""
    print(f"\n{'='*60}")
    print(f"处理 {segment_type} 视频片段...")
    print(f"{'='*60}")
    
    # 创建输出目录
    features_dir = os.path.join(output_dir, f"features_{segment_type}")
    audio_temp_dir = os.path.join(output_dir, f"audio_temp_{segment_type}")
    
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(audio_temp_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_files = glob.glob(os.path.join(input_dir, f"segments_{segment_type}", "*.mp4"))
    video_files.sort()
    
    if not video_files:
        print(f"⚠️  {segment_type} 目录中没有找到视频文件")
        return False
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 统计信息
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        
        # 定义路径
        audio_path = os.path.join(audio_temp_dir, f"{base_name}.wav")
        feature_path = os.path.join(features_dir, f"{base_name}.npy")
        
        # 检查是否已经处理过
        if os.path.exists(feature_path):
            print(f"[{i:2d}/{len(video_files)}] ✓ 已存在: {base_name}")
            processed_count += 1
            continue
        
        print(f"[{i:2d}/{len(video_files)}] 🔄 处理中: {base_name}")
        
        try:
            # 1. 提取音频
            if not extract_audio_from_video(video_path, audio_path):
                print(f"    ❌ 音频提取失败")
                failed_count += 1
                continue
            
            # 2. 检查音频文件大小
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print(f"    ❌ 音频文件为空")
                failed_count += 1
                continue
            
            # 3. 使用CLAMP3提取特征
            # 只处理当前音频文件
            temp_single_dir = os.path.join(audio_temp_dir, f"single_{base_name}")
            temp_features_dir = os.path.join(audio_temp_dir, f"features_{base_name}")
            
            # 确保临时目录不存在
            if os.path.exists(temp_single_dir):
                shutil.rmtree(temp_single_dir)
            if os.path.exists(temp_features_dir):
                shutil.rmtree(temp_features_dir)
                
            os.makedirs(temp_single_dir, exist_ok=True)
            temp_audio_path = os.path.join(temp_single_dir, f"{base_name}.wav")
            shutil.copy2(audio_path, temp_audio_path)
            
            # 提取特征
            result = subprocess.run([
                'python', 'clamp3_embd.py', 
                temp_single_dir, temp_features_dir, '--get_global'
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            # 复制特征文件到最终目录
            temp_feature_file = os.path.join(temp_features_dir, f"{base_name}.npy")
            if os.path.exists(temp_feature_file):
                shutil.copy2(temp_feature_file, feature_path)
            
            # 清理临时文件
            shutil.rmtree(temp_single_dir)
            if os.path.exists(temp_features_dir):
                shutil.rmtree(temp_features_dir)
            os.remove(audio_path)
            
            if result.returncode == 0 and os.path.exists(feature_path):
                print(f"    ✅ 特征提取成功")
                processed_count += 1
            else:
                print(f"    ❌ 特征提取失败: {result.stderr}")
                failed_count += 1
                
        except Exception as e:
            print(f"    ❌ 处理异常: {e}")
            failed_count += 1
            
            # 清理可能的临时文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    # 清理临时目录
    if os.path.exists(audio_temp_dir):
        shutil.rmtree(audio_temp_dir)
    
    # 统计结果
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{segment_type} 处理完成:")
    print(f"  ✅ 成功: {processed_count}")
    print(f"  ❌ 失败: {failed_count}")
    print(f"  ⏱️  用时: {duration:.1f}秒")
    
    return failed_count == 0

def main():
    """主函数"""
    # 路径配置
    base_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4"
    materials_dir = os.path.join(base_dir, "materials", "retrieve_libraries")
    output_dir = os.path.join(base_dir, "materials", "music_features")
    
    # 检查输入目录
    if not os.path.exists(materials_dir):
        print(f"❌ 素材目录不存在: {materials_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义要处理的时长版本
    segments = ["1min", "3min", "5min", "10min", "20min", "30min"]
    
    print("开始批量音乐特征提取...")
    print(f"输入目录: {materials_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理版本: {', '.join(segments)}")
    
    # 检查依赖
    dependencies = ['ffmpeg', 'ffprobe']
    for dep in dependencies:
        if shutil.which(dep) is None:
            print(f"❌ 缺少依赖: {dep}")
            print("请安装 ffmpeg: brew install ffmpeg")
            sys.exit(1)
    
    # 总体统计
    total_start = time.time()
    success_segments = 0
    
    # 逐个处理每个时长版本
    for segment in segments:
        success = extract_features_for_segment(materials_dir, output_dir, segment)
        if success:
            success_segments += 1
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    print(f"\n{'='*60}")
    print("批量处理完成!")
    print(f"成功处理版本: {success_segments}/{len(segments)}")
    print(f"总用时: {total_duration:.1f}秒")
    print(f"输出目录: {output_dir}")
    
    # 验证结果
    print(f"\n{'='*60}")
    print("验证提取结果...")
    for segment in segments:
        features_dir = os.path.join(output_dir, f"features_{segment}")
        if os.path.exists(features_dir):
            feature_files = glob.glob(os.path.join(features_dir, "*.npy"))
            print(f"  {segment}: {len(feature_files)} 个特征文件")
        else:
            print(f"  {segment}: 目录不存在")

if __name__ == "__main__":
    main()