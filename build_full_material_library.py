#!/usr/bin/env python3
"""
完整素材库构建脚本
将7小时视频切分成完整的素材库，包含所有时长的所有片段
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('material_library_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def build_full_material_library():
    """构建完整的素材库"""
    print("🏗️  开始构建完整的疗愈视频素材库")
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
    
    # 显示视频信息
    print("📊 视频信息:")
    total_duration = 0
    for video in videos:
        duration_hours = video['duration'] / 3600
        total_duration += video['duration']
        print(f"   📹 {video['file_name']}")
        print(f"      时长: {duration_hours:.1f}小时 ({video['duration']:.0f}秒)")
        print(f"      大小: {video['file_size'] / (1024**3):.1f} GB")
    
    total_hours = total_duration / 3600
    print(f"\n总时长: {total_hours:.1f}小时")
    
    # 计算理论片段数量
    print("\n📈 理论片段数量预估:")
    durations = [1, 3, 5, 10, 20, 30]
    total_theoretical = 0
    
    for duration_min in durations:
        duration_sec = duration_min * 60
        count_per_video = []
        for video in videos:
            num_segments = int(video['duration'] // duration_sec)
            count_per_video.append(num_segments)
            total_theoretical += num_segments
        
        total_for_duration = sum(count_per_video)
        print(f"   {duration_min:2d}分钟: {total_for_duration:3d}个 "
              f"({' + '.join(map(str, count_per_video))})")
    
    print(f"\n总计理论片段: {total_theoretical}个")
    
    # 检查当前进度
    print("\n🔍 检查当前切分进度...")
    try:
        processor.load_segment_index()
        current_segments = processor.segment_index
        current_total = sum(len(segments) for segments in current_segments.values())
        print(f"当前已有片段: {current_total}个")
        
        if current_total > 0:
            for duration, segments in current_segments.items():
                print(f"   {duration}: {len(segments)}个")
    except:
        current_total = 0
        print("当前无片段")
    
    # 确认是否继续
    remaining = total_theoretical - current_total
    if remaining > 0:
        print(f"\n⚠️  需要切分 {remaining} 个片段")
        print("💾 预估存储空间需求: 约 10-20GB")
        print("⏱️  预估处理时间: 约 30-60分钟")
        
        confirm = input("\n是否继续构建完整素材库? [y/N]: ")
        if confirm.lower() != 'y':
            print("❌ 用户取消操作")
            return False
    else:
        print("✅ 素材库已完整，无需重新构建")
        return True
    
    # 开始切分
    print(f"\n🔪 开始切分所有视频片段...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 切分所有片段（非intro模式）
        segments = processor.segment_videos(
            extract_intro_only=False, 
            force_resegment=False
        )
        
        # 统计结果
        final_total = sum(len(seg_list) for seg_list in segments.values())
        
        print(f"\n✅ 素材库构建完成！")
        print(f"总计生成: {final_total} 个片段")
        
        for duration, seg_list in segments.items():
            print(f"   {duration}: {len(seg_list)} 个片段")
        
        # 显示存储使用情况
        summary = processor.get_processing_summary()
        print(f"\n💾 存储使用: {summary['total_disk_usage_mb']:.1f} MB")
        print(f"   约 {summary['total_disk_usage_mb']/1024:.1f} GB")
        
        # 显示时间统计
        end_time = time.time()
        elapsed = end_time - start_time
        elapsed_min = elapsed / 60
        print(f"\n⏱️  总处理时间: {elapsed_min:.1f} 分钟")
        
        if final_total > 0:
            avg_time_per_segment = elapsed / final_total
            print(f"   平均每片段: {avg_time_per_segment:.1f} 秒")
        
        print(f"\n🎉 素材库构建成功！现在有 {final_total} 个疗愈视频片段可用于检索")
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断操作")
        current_total = len([f for f in Path("materials/segments").rglob("*.mp4")])
        print(f"当前已生成 {current_total} 个片段")
        return False
        
    except Exception as e:
        print(f"\n❌ 素材库构建失败: {e}")
        logger.error(f"素材库构建失败: {e}")
        return False

def show_current_status():
    """显示当前素材库状态"""
    print("📊 当前素材库状态")
    print("=" * 40)
    
    segments_dir = Path("materials/segments")
    if not segments_dir.exists():
        print("❌ 素材库目录不存在")
        return
    
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

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        show_current_status()
        return
    
    try:
        success = build_full_material_library()
        if success:
            print("\n📈 构建后的素材库状态:")
            show_current_status()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()