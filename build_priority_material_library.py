#!/usr/bin/env python3
"""
优先级素材库构建脚本
优先构建关键时长的素材库，为MVP测试提供足够的素材
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.video_processor import VideoProcessor

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_priority_material_library():
    """构建优先级素材库 - 专注于5分钟和10分钟片段"""
    print("🎯 构建优先级疗愈视频素材库")
    print("专注于5分钟和10分钟片段，确保MVP有足够素材")
    print("=" * 60)
    
    start_time = time.time()
    
    # 初始化处理器
    processor = VideoProcessor()
    
    # 检查ffmpeg
    if not processor.check_ffmpeg_availability():
        print("❌ ffmpeg不可用，无法进行视频处理")
        return False
    
    # 扫描源视频
    print("📹 扫描源视频文件...")
    videos = processor.scan_source_videos()
    
    if not videos:
        print("❌ 未找到源视频文件")
        return False
    
    # 显示视频信息和优先级片段计算
    print("📊 视频信息及优先级片段:")
    priority_durations = [5, 10]  # 专注于5分钟和10分钟
    total_priority_segments = 0
    
    for video in videos:
        duration_hours = video['duration'] / 3600
        print(f"   📹 {video['file_name']}")
        print(f"      时长: {duration_hours:.1f}小时")
        
        for duration_min in priority_durations:
            duration_sec = duration_min * 60
            num_segments = int(video['duration'] // duration_sec)
            total_priority_segments += num_segments
            print(f"      {duration_min}分钟: {num_segments}个片段")
    
    print(f"\n🎯 优先级片段总计: {total_priority_segments}个")
    print(f"   预估存储需求: 约 {total_priority_segments * 120 / 1024:.1f} GB")
    print(f"   预估处理时间: 约 {total_priority_segments * 2 / 60:.0f} 分钟")
    
    # 检查当前5分钟和10分钟片段
    segments_dir = Path("materials/segments")
    current_5min = len(list((segments_dir / "5min").glob("*.mp4"))) if (segments_dir / "5min").exists() else 0
    current_10min = len(list((segments_dir / "10min").glob("*.mp4"))) if (segments_dir / "10min").exists() else 0
    
    print(f"\n📈 当前状态:")
    print(f"   5分钟片段: {current_5min}个")
    print(f"   10分钟片段: {current_10min}个")
    
    # 计算需要生成的片段
    target_5min = sum(int(video['duration'] // 300) for video in videos)  # 5分钟 = 300秒
    target_10min = sum(int(video['duration'] // 600) for video in videos)  # 10分钟 = 600秒
    
    needed_5min = max(0, target_5min - current_5min)
    needed_10min = max(0, target_10min - current_10min)
    
    print(f"\n🔄 需要生成:")
    print(f"   5分钟片段: {needed_5min}个")
    print(f"   10分钟片段: {needed_10min}个")
    
    if needed_5min == 0 and needed_10min == 0:
        print("✅ 优先级素材库已完整！")
        return True
    
    # 开始构建
    print(f"\n🔪 开始构建优先级素材库...")
    
    try:
        # 创建自定义视频处理器，只处理5分钟和10分钟
        custom_processor = VideoProcessor(durations=[5, 10])
        
        # 切分关键时长片段
        segments = custom_processor.segment_videos(
            extract_intro_only=False, 
            force_resegment=False
        )
        
        # 统计结果
        final_5min = len(segments.get('5min', []))
        final_10min = len(segments.get('10min', []))
        total_generated = final_5min + final_10min
        
        print(f"\n✅ 优先级素材库构建完成！")
        print(f"   5分钟片段: {final_5min}个")
        print(f"   10分钟片段: {final_10min}个")
        print(f"   总计: {total_generated}个")
        
        # 显示存储使用情况
        summary = custom_processor.get_processing_summary()
        priority_size = 0
        if (segments_dir / "5min").exists():
            priority_size += sum(f.stat().st_size for f in (segments_dir / "5min").glob("*.mp4"))
        if (segments_dir / "10min").exists():
            priority_size += sum(f.stat().st_size for f in (segments_dir / "10min").glob("*.mp4"))
        
        priority_size_mb = priority_size / (1024**2)
        print(f"\n💾 优先级素材存储: {priority_size_mb:.1f} MB ({priority_size_mb/1024:.1f} GB)")
        
        # 显示时间统计
        end_time = time.time()
        elapsed = end_time - start_time
        elapsed_min = elapsed / 60
        print(f"\n⏱️  处理时间: {elapsed_min:.1f} 分钟")
        
        print(f"\n🎉 优先级素材库构建成功！")
        print(f"现在有足够的5分钟和10分钟片段用于MVP测试")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 优先级素材库构建失败: {e}")
        logger.error(f"优先级素材库构建失败: {e}")
        return False

def extend_to_full_library():
    """扩展到完整素材库 - 添加其他时长"""
    print("\n🔄 扩展到完整素材库...")
    print("添加1分钟、3分钟、20分钟、30分钟片段")
    
    try:
        # 处理其他时长
        processor = VideoProcessor(durations=[1, 3, 20, 30])
        
        segments = processor.segment_videos(
            extract_intro_only=False, 
            force_resegment=False
        )
        
        total_new = sum(len(seg_list) for seg_list in segments.values())
        print(f"✅ 添加了 {total_new} 个其他时长片段")
        
        return True
        
    except Exception as e:
        print(f"❌ 扩展到完整素材库失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始优先级素材库构建流程")
    
    # 第一步：构建优先级素材库
    success = build_priority_material_library()
    
    if not success:
        print("❌ 优先级素材库构建失败")
        sys.exit(1)
    
    # 询问是否继续构建完整库
    print("\n" + "="*60)
    extend = input("是否继续构建完整素材库（包含所有时长）？[y/N]: ")
    
    if extend.lower() == 'y':
        success = extend_to_full_library()
        if success:
            print("🎉 完整素材库构建成功！")
        else:
            print("⚠️  完整素材库构建部分失败，但优先级素材可用")
    else:
        print("✅ 优先级素材库已可用于MVP测试")
    
    # 显示最终状态
    print("\n📊 最终素材库状态:")
    segments_dir = Path("materials/segments")
    if segments_dir.exists():
        total_segments = 0
        total_size = 0
        
        for duration_dir in segments_dir.iterdir():
            if duration_dir.is_dir():
                video_files = list(duration_dir.glob("*.mp4"))
                count = len(video_files)
                total_segments += count
                
                size_mb = sum(f.stat().st_size for f in video_files) / (1024**2)
                total_size += size_mb
                
                print(f"   {duration_dir.name}: {count:3d} 个片段 ({size_mb:6.1f} MB)")
        
        print(f"\n总计: {total_segments} 个片段")
        print(f"存储: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    
    print("✅ 素材库构建流程完成！")

if __name__ == "__main__":
    main()