#!/usr/bin/env python3
"""
批量提取所有剩余时长的视频音乐特征
"""
import os
import sys
import subprocess
import shutil
import time
import glob

def extract_features_for_duration(duration):
    """提取指定时长视频的音乐特征"""
    # 路径配置
    materials_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries"
    segments_dir = os.path.join(materials_dir, f"segments_{duration}")
    output_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/music_features"
    features_dir = os.path.join(output_dir, f"features_{duration}")
    
    # 创建输出目录
    os.makedirs(features_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_files = glob.glob(os.path.join(segments_dir, "*.mp4"))
    video_files.sort()
    
    print(f"\n{'='*60}")
    print(f"处理 {duration} 视频片段...")
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"{'='*60}")
    
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
    duration_time = end_time - start_time
    
    print(f"\n{duration} 视频特征提取完成:")
    print(f"  ✅ 成功: {processed_count}")
    print(f"  ❌ 失败: {failed_count}")
    print(f"  ⏱️  用时: {duration_time:.1f}秒")
    print(f"  📁 输出目录: {features_dir}")
    
    return processed_count, failed_count

def main():
    """主函数"""
    durations = ["3min", "5min", "10min", "20min", "30min"]
    
    print("开始批量提取所有剩余时长的音乐特征...")
    print(f"处理时长: {', '.join(durations)}")
    
    total_processed = 0
    total_failed = 0
    total_start_time = time.time()
    
    for duration in durations:
        processed, failed = extract_features_for_duration(duration)
        total_processed += processed
        total_failed += failed
        
        # 短暂休息，避免过热
        time.sleep(2)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("所有剩余时长特征提取完成!")
    print(f"总计处理:")
    print(f"  ✅ 成功: {total_processed}")
    print(f"  ❌ 失败: {total_failed}")
    print(f"  ⏱️  总用时: {total_duration:.1f}秒")
    print(f"  📁 输出目录: /Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/music_features/")
    
    if total_failed == 0:
        print("🎉 所有视频特征提取成功!")
    else:
        print(f"⚠️  {total_failed} 个视频处理失败")

if __name__ == "__main__":
    main()