#!/usr/bin/env python3
"""
带详细进度条的音乐特征提取脚本
"""
import os
import sys
import subprocess
import shutil
import time
import glob
from tqdm import tqdm
import threading

class ProgressTracker:
    def __init__(self):
        self.current_step = ""
        self.start_time = None
        self.is_running = False
        
    def start_step(self, step_name):
        self.current_step = step_name
        self.start_time = time.time()
        self.is_running = True
        
    def end_step(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"    完成时间: {elapsed:.1f}秒")
        self.is_running = False
        
    def show_progress(self):
        """显示实时进度"""
        while self.is_running:
            if self.start_time:
                elapsed = time.time() - self.start_time
                print(f"\r    {self.current_step} - 已用时: {elapsed:.1f}秒", end='', flush=True)
            time.sleep(1)

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
    print(f"处理 {duration} 视频片段 - 共 {len(video_files)} 个文件")
    print(f"{'='*60}")
    
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    # 创建进度条
    progress_bar = tqdm(video_files, desc=f"处理{duration}视频", unit="个")
    
    for i, video_path in enumerate(progress_bar):
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        
        # 更新进度条描述
        progress_bar.set_description(f"处理{duration}视频 [{base_name}]")
        
        # 检查是否已处理
        feature_path = os.path.join(features_dir, f"{base_name}.npy")
        if os.path.exists(feature_path):
            progress_bar.write(f"✓ 已存在: {base_name}")
            processed_count += 1
            continue
        
        progress_bar.write(f"🔄 处理中: {base_name}")
        
        # 创建进度追踪器
        tracker = ProgressTracker()
        
        try:
            # 创建临时目录
            temp_dir = f"temp_{base_name}"
            audio_dir = os.path.join(temp_dir, "audio")
            temp_features_dir = os.path.join(temp_dir, "features")
            
            os.makedirs(audio_dir, exist_ok=True)
            
            # 1. 提取音频
            tracker.start_step("提取音频")
            progress_thread = threading.Thread(target=tracker.show_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            audio_path = os.path.join(audio_dir, f"{base_name}.wav")
            cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', '-y', audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            tracker.end_step()
            
            if result.returncode != 0:
                progress_bar.write(f"    ❌ 音频提取失败")
                failed_count += 1
                continue
            
            # 检查音频文件大小
            audio_size = os.path.getsize(audio_path) / 1024 / 1024  # MB
            progress_bar.write(f"    ✅ 音频提取成功 ({audio_size:.1f}MB)")
            
            # 2. 提取特征
            tracker.start_step("提取CLAMP3特征")
            progress_thread = threading.Thread(target=tracker.show_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            result = subprocess.run([
                'python', 'clamp3_embd.py', 
                audio_dir, temp_features_dir, '--get_global'
            ], capture_output=True, text=True)
            
            tracker.end_step()
            
            # 复制特征文件
            temp_feature_file = os.path.join(temp_features_dir, f"{base_name}.npy")
            if os.path.exists(temp_feature_file):
                shutil.copy2(temp_feature_file, feature_path)
                progress_bar.write(f"    ✅ 特征提取成功")
                processed_count += 1
            else:
                progress_bar.write(f"    ❌ 特征文件未生成")
                if result.stderr:
                    progress_bar.write(f"    错误信息: {result.stderr[:200]}...")
                failed_count += 1
            
        except Exception as e:
            tracker.end_step()
            progress_bar.write(f"    ❌ 处理异常: {e}")
            failed_count += 1
        
        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    progress_bar.close()
    
    end_time = time.time()
    duration_time = end_time - start_time
    avg_time = duration_time / len(video_files) if video_files else 0
    
    print(f"\n{duration} 视频特征提取完成:")
    print(f"  ✅ 成功: {processed_count}")
    print(f"  ❌ 失败: {failed_count}")
    print(f"  ⏱️  总用时: {duration_time:.1f}秒")
    print(f"  ⚡ 平均每个: {avg_time:.1f}秒")
    print(f"  📁 输出目录: {features_dir}")
    
    return processed_count, failed_count

def main():
    """主函数"""
    # 先检查哪些版本还没完成
    output_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/music_features"
    
    all_durations = ["5min", "10min", "20min", "30min"]
    pending_durations = []
    
    for duration in all_durations:
        features_dir = os.path.join(output_dir, f"features_{duration}")
        if os.path.exists(features_dir):
            feature_files = glob.glob(os.path.join(features_dir, "*.npy"))
            if len(feature_files) < 20:
                pending_durations.append(duration)
                print(f"📂 {duration}: {len(feature_files)}/20 个文件已完成")
            else:
                print(f"✅ {duration}: 已完成所有20个文件")
        else:
            pending_durations.append(duration)
            print(f"📂 {duration}: 0/20 个文件已完成")
    
    if not pending_durations:
        print("🎉 所有时长版本都已完成!")
        return
    
    print(f"\n开始处理剩余时长: {', '.join(pending_durations)}")
    
    total_processed = 0
    total_failed = 0
    total_start_time = time.time()
    
    for duration in pending_durations:
        processed, failed = extract_features_for_duration(duration)
        total_processed += processed
        total_failed += failed
        
        # 短暂休息，避免过热
        print("😴 休息2秒...")
        time.sleep(2)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("所有剩余时长特征提取完成!")
    print(f"总计处理:")
    print(f"  ✅ 成功: {total_processed}")
    print(f"  ❌ 失败: {total_failed}")
    print(f"  ⏱️  总用时: {total_duration:.1f}秒")
    print(f"  📁 输出目录: {output_dir}")
    
    if total_failed == 0:
        print("🎉 所有视频特征提取成功!")
    else:
        print(f"⚠️  {total_failed} 个视频处理失败")

if __name__ == "__main__":
    main()