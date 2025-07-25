#!/usr/bin/env python3
"""
1分钟视频音乐特征提取
"""
import os
import sys
import subprocess
import shutil
import time
import glob

def extract_1min_features():
    """提取1分钟视频的音乐特征"""
    # 路径配置
    materials_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries"
    segments_dir = os.path.join(materials_dir, "segments_1min")
    output_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/music_features"
    features_dir = os.path.join(output_dir, "features_1min")
    
    # 创建输出目录
    os.makedirs(features_dir, exist_ok=True)
    
    # 获取所有1分钟视频
    video_files = glob.glob(os.path.join(segments_dir, "*.mp4"))
    video_files.sort()
    
    print(f"找到 {len(video_files)} 个1分钟视频文件")
    
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        
        # 检查是否已处理
        feature_path = os.path.join(features_dir, f"{base_name}.npy")
        if os.path.exists(feature_path):
            print(f"[{i:2d}/{len(video_files)}] ✓ 已存在: {base_name}")
            processed_count += 1
            continue
        
        print(f"[{i:2d}/{len(video_files)}] 🔄 处理中: {base_name}")
        
        try:
            # 创建临时目录
            temp_dir = f"temp_{base_name}"
            audio_dir = os.path.join(temp_dir, "audio")
            temp_features_dir = os.path.join(temp_dir, "features")
            
            os.makedirs(audio_dir, exist_ok=True)
            
            # 提取音频
            audio_path = os.path.join(audio_dir, f"{base_name}.wav")
            cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', '-y', audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"    ❌ 音频提取失败")
                failed_count += 1
                continue
            
            # 提取特征
            result = subprocess.run([
                'python', 'clamp3_embd.py', 
                audio_dir, temp_features_dir, '--get_global'
            ], capture_output=True, text=True)
            
            # 复制特征文件
            temp_feature_file = os.path.join(temp_features_dir, f"{base_name}.npy")
            if os.path.exists(temp_feature_file):
                shutil.copy2(temp_feature_file, feature_path)
                print(f"    ✅ 特征提取成功")
                processed_count += 1
            else:
                print(f"    ❌ 特征文件未生成")
                failed_count += 1
            
        except Exception as e:
            print(f"    ❌ 处理异常: {e}")
            failed_count += 1
        
        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n1分钟视频特征提取完成:")
    print(f"  ✅ 成功: {processed_count}")
    print(f"  ❌ 失败: {failed_count}")
    print(f"  ⏱️  用时: {duration:.1f}秒")
    print(f"  📁 输出目录: {features_dir}")
    
    return processed_count, failed_count

if __name__ == "__main__":
    processed, failed = extract_1min_features()
    print(f"\n{'='*60}")
    if failed == 0:
        print("🎉 所有1分钟视频特征提取成功!")
    else:
        print(f"⚠️  {failed} 个视频处理失败")
    print(f"{'='*60}")