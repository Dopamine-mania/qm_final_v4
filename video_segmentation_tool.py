#!/usr/bin/env python3
"""
视频切分工具 - 为音乐疗愈AI系统生成多时长版本素材
从两个长视频中随机切分出不同时长的片段，用于模型检索

功能：
- 支持1分钟、3分钟、5分钟、10分钟、20分钟、30分钟切分
- 每个时长版本从两个源视频各随机选择10个片段
- 总计生成120条素材（6个版本 × 20条素材）
"""

import os
import random
import subprocess
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoSegmentationTool:
    """视频切分工具类"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.video_path = self.base_path / "materials" / "video"
        self.segments_base = self.base_path / "materials"
        
        # 定义切分配置
        self.durations = {
            '1min': 60,
            '3min': 180,
            '5min': 300,
            '10min': 600,
            '20min': 1200,
            '30min': 1800
        }
        
        # 视频文件信息
        self.video_files = {
            '32.mp4': None,  # 将存储视频时长
            '56.mp4': None
        }
        
        # 随机种子设置
        random.seed(42)
    
    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"获取视频时长失败 {video_path}: {e}")
            return 0
    
    def initialize_video_info(self):
        """初始化视频信息"""
        logger.info("正在获取视频文件信息...")
        
        for video_file in self.video_files.keys():
            video_path = self.video_path / video_file
            if not video_path.exists():
                logger.error(f"视频文件不存在: {video_path}")
                continue
            
            duration = self.get_video_duration(video_path)
            self.video_files[video_file] = duration
            logger.info(f"{video_file}: {duration:.2f}秒 ({duration/3600:.2f}小时)")
    
    def generate_random_segments(self, video_duration: float, segment_duration: int, count: int) -> List[float]:
        """为指定视频生成随机切分时间点"""
        if video_duration <= segment_duration:
            logger.warning(f"视频时长 {video_duration}s 小于目标片段时长 {segment_duration}s")
            return [0]
        
        max_start_time = video_duration - segment_duration
        segments = []
        
        for _ in range(count):
            start_time = random.uniform(0, max_start_time)
            segments.append(start_time)
        
        return sorted(segments)
    
    def cut_video_segment(self, input_video: Path, output_video: Path, 
                         start_time: float, duration: int) -> bool:
        """切分视频片段"""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_video),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # 使用流复制，避免重编码
                '-avoid_negative_ts', 'make_zero',
                str(output_video),
                '-y'  # 覆盖输出文件
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"切分成功: {output_video.name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"切分失败 {output_video.name}: {e}")
            return False
    
    def process_duration_segments(self, duration_name: str, segment_duration: int):
        """处理特定时长的所有切分"""
        output_dir = self.segments_base / f"segments_{duration_name}"
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"开始处理 {duration_name} 切分...")
        
        segment_count = 0
        
        for video_file, video_duration in self.video_files.items():
            if video_duration is None or video_duration == 0:
                logger.warning(f"跳过无效视频: {video_file}")
                continue
            
            # 为每个视频生成10个随机片段
            segments = self.generate_random_segments(video_duration, segment_duration, 10)
            
            input_video = self.video_path / video_file
            video_name = video_file.split('.')[0]
            
            for i, start_time in enumerate(segments):
                output_filename = f"{video_name}_{duration_name}_{i+1:02d}.mp4"
                output_path = output_dir / output_filename
                
                success = self.cut_video_segment(
                    input_video, output_path, start_time, segment_duration
                )
                
                if success:
                    segment_count += 1
                    logger.info(f"  → {output_filename} (起始: {start_time:.2f}s)")
        
        logger.info(f"{duration_name} 切分完成: {segment_count} 个片段")
        return segment_count
    
    def generate_all_segments(self):
        """生成所有时长版本的切分"""
        logger.info("开始视频切分任务...")
        
        # 初始化视频信息
        self.initialize_video_info()
        
        # 验证视频文件
        valid_videos = [f for f, d in self.video_files.items() if d and d > 0]
        if len(valid_videos) < 2:
            logger.error("需要至少2个有效视频文件")
            return False
        
        total_segments = 0
        
        # 处理每个时长版本
        for duration_name, segment_duration in self.durations.items():
            try:
                count = self.process_duration_segments(duration_name, segment_duration)
                total_segments += count
            except Exception as e:
                logger.error(f"处理 {duration_name} 时出错: {e}")
                continue
        
        logger.info(f"视频切分任务完成！总计生成 {total_segments} 个片段")
        return True
    
    def validate_output(self) -> Dict[str, int]:
        """验证输出结果"""
        logger.info("正在验证切分结果...")
        
        results = {}
        
        for duration_name in self.durations.keys():
            output_dir = self.segments_base / f"segments_{duration_name}"
            if output_dir.exists():
                files = list(output_dir.glob("*.mp4"))
                results[duration_name] = len(files)
                logger.info(f"  {duration_name}: {len(files)} 个文件")
            else:
                results[duration_name] = 0
                logger.warning(f"  {duration_name}: 目录不存在")
        
        total = sum(results.values())
        logger.info(f"验证完成: 总计 {total} 个片段")
        
        return results

def main():
    """主函数"""
    base_path = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4"
    
    # 创建工具实例
    tool = VideoSegmentationTool(base_path)
    
    # 执行切分
    success = tool.generate_all_segments()
    
    if success:
        # 验证结果
        results = tool.validate_output()
        
        # 输出摘要
        print("\n" + "="*50)
        print("视频切分完成摘要")
        print("="*50)
        
        for duration_name, count in results.items():
            print(f"{duration_name:8s}: {count:3d} 个片段")
        
        total = sum(results.values())
        print(f"{'总计':8s}: {total:3d} 个片段")
        
        if total == 120:
            print("✅ 成功生成全部120个片段！")
        else:
            print(f"⚠️  预期120个片段，实际生成{total}个")
    
    else:
        print("❌ 视频切分失败")

if __name__ == "__main__":
    main()