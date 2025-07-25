#!/usr/bin/env python3
"""
优化版音乐特征提取 - 专门处理长音频
"""
import os
import sys
import subprocess
import shutil
import time
import glob
import gc
import psutil
from tqdm import tqdm

def get_memory_usage():
    """获取内存使用情况"""
    return psutil.virtual_memory().percent

def extract_features_optimized(duration):
    """优化的特征提取"""
    # 路径配置
    materials_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries"
    segments_dir = os.path.join(materials_dir, f"segments_{duration}")
    output_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/music_features"
    features_dir = os.path.join(output_dir, f"features_{duration}")
    
    os.makedirs(features_dir, exist_ok=True)
    
    # 获取视频文件并按大小排序（小文件优先）
    video_files = []
    for f in glob.glob(os.path.join(segments_dir, "*.mp4")):
        size = os.path.getsize(f)
        video_files.append((f, size))
    
    # 按文件大小排序
    video_files.sort(key=lambda x: x[1])
    video_files = [f[0] for f in video_files]
    
    print(f"\n{'='*60}")
    print(f"优化处理 {duration} 视频 - 共 {len(video_files)} 个文件")
    print(f"预估处理时间: {len(video_files) * 15:.0f}秒 (约{len(video_files) * 15/60:.1f}分钟)")
    print(f"{'='*60}")
    
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    # 分批处理，每批3个文件
    batch_size = 3
    for batch_start in range(0, len(video_files), batch_size):
        batch_files = video_files[batch_start:batch_start+batch_size]
        
        print(f"\n🔄 处理批次 {batch_start//batch_size + 1}/{(len(video_files)-1)//batch_size + 1}")
        print(f"内存使用: {get_memory_usage():.1f}%")
        
        for video_path in batch_files:
            video_name = os.path.basename(video_path)
            base_name = os.path.splitext(video_name)[0]
            
            # 检查是否已处理
            feature_path = os.path.join(features_dir, f"{base_name}.npy")
            if os.path.exists(feature_path):
                print(f"  ✓ 已存在: {base_name}")
                processed_count += 1
                continue
            
            file_size = os.path.getsize(video_path) / 1024 / 1024
            print(f"  🔄 处理: {base_name} ({file_size:.1f}MB)")
            
            step_start = time.time()
            
            try:
                # 创建临时目录
                temp_dir = f"temp_{base_name}"
                audio_dir = os.path.join(temp_dir, "audio")
                temp_features_dir = os.path.join(temp_dir, "features")
                
                # 清理可能存在的临时目录
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
                os.makedirs(audio_dir, exist_ok=True)
                
                # 1. 提取音频 (优化参数)
                audio_path = os.path.join(audio_dir, f"{base_name}.wav")
                cmd = [
                    'ffmpeg', '-i', video_path, 
                    '-q:a', '2',  # 稍微降低音质以减少文件大小
                    '-ar', '44100',  # 固定采样率
                    '-ac', '2',  # 立体声
                    '-map', 'a', '-y', audio_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"    ❌ 音频提取失败: {base_name}")
                    failed_count += 1
                    continue
                
                audio_size = os.path.getsize(audio_path) / 1024 / 1024
                print(f"    ✅ 音频提取完成 ({audio_size:.1f}MB)")
                
                # 2. 特征提取 (设置更长的超时)
                print(f"    🔄 开始特征提取...")
                
                # 增加超时时间和内存限制
                env = os.environ.copy()
                env['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 降低GPU内存使用
                
                result = subprocess.run([
                    'python', 'clamp3_embd.py', 
                    audio_dir, temp_features_dir, '--get_global'
                ], capture_output=True, text=True, timeout=60, env=env)  # 60秒超时
                
                temp_feature_file = os.path.join(temp_features_dir, f"{base_name}.npy")
                if os.path.exists(temp_feature_file):
                    shutil.copy2(temp_feature_file, feature_path)
                    
                    step_time = time.time() - step_start
                    print(f"    ✅ 特征提取成功 ({step_time:.1f}秒)")
                    processed_count += 1
                else:
                    print(f"    ❌ 特征文件未生成")
                    # 输出更详细的错误信息
                    if result.stderr:
                        error_lines = result.stderr.split('\n')
                        for line in error_lines[-5:]:  # 只显示最后5行错误
                            if line.strip():
                                print(f"    错误: {line.strip()}")
                    failed_count += 1
                
            except subprocess.TimeoutExpired:
                print(f"    ❌ 处理超时 (>60秒)")
                failed_count += 1
                
            except Exception as e:
                print(f"    ❌ 处理异常: {e}")
                failed_count += 1
            
            finally:
                # 强制清理临时目录
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                
                # 强制垃圾回收
                gc.collect()
        
        # 批次间休息
        if batch_start + batch_size < len(video_files):
            print(f"  💤 批次休息3秒...")
            time.sleep(3)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(video_files) if video_files else 0
    
    print(f"\n{duration} 视频特征提取完成:")
    print(f"  ✅ 成功: {processed_count}")
    print(f"  ❌ 失败: {failed_count}")
    print(f"  ⏱️  总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"  ⚡ 平均每个: {avg_time:.1f}秒")
    print(f"  📁 输出目录: {features_dir}")
    
    return processed_count, failed_count

def main():
    """主函数"""
    print("🚀 优化版音乐特征提取启动")
    print(f"系统内存: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"可用内存: {psutil.virtual_memory().available / 1024**3:.1f}GB")
    
    # 检查剩余任务
    durations = ["5min", "10min", "20min", "30min"]
    
    for duration in durations:
        features_dir = f"/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/music_features/features_{duration}"
        if os.path.exists(features_dir):
            count = len(glob.glob(os.path.join(features_dir, "*.npy")))
            print(f"📊 {duration}: {count}/20 已完成")
        else:
            print(f"📊 {duration}: 0/20 已完成")
    
    # 开始处理
    total_start = time.time()
    for duration in durations:
        processed, failed = extract_features_optimized(duration)
        
        if failed > 0:
            print(f"⚠️  {duration} 有 {failed} 个文件失败，是否继续？")
            # 这里可以添加用户确认逻辑
    
    total_time = time.time() - total_start
    print(f"\n🎉 所有任务完成！总用时: {total_time/60:.1f}分钟")

if __name__ == "__main__":
    main()