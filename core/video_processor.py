#!/usr/bin/env python3
"""
视频处理模块 - 4.0版本核心组件
负责将长视频切分成不同时长的片段，并提取音频特征
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    视频处理器
    负责切分、索引和管理疗愈视频素材库
    """
    
    def __init__(self, 
                 materials_dir: str = "materials",
                 durations: List[int] = [1, 3, 5, 10, 20, 30]):
        """
        初始化视频处理器
        
        Args:
            materials_dir: 素材目录路径
            durations: 要切分的时长列表（分钟）
        """
        self.materials_dir = Path(materials_dir)
        self.video_dir = self.materials_dir / "video"
        self.segments_dir = self.materials_dir / "segments"
        self.features_dir = self.materials_dir / "features"
        self.durations = durations
        
        # 确保目录存在
        self._ensure_directories()
        
        # 视频索引
        self.video_index = []
        self.segment_index = {}
        
    def _ensure_directories(self):
        """确保所有必要的目录都存在"""
        self.materials_dir.mkdir(exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)
        self.segments_dir.mkdir(exist_ok=True)
        self.features_dir.mkdir(exist_ok=True)
        
        # 为每个时长创建子目录
        for duration in self.durations:
            (self.segments_dir / f"{duration}min").mkdir(exist_ok=True)
    
    def scan_source_videos(self) -> List[Dict[str, Any]]:
        """
        扫描原始视频文件
        
        Returns:
            List[Dict]: 视频文件信息列表
        """
        video_files = []
        
        # 支持的视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        for file_path in self.video_dir.iterdir():
            if file_path.suffix.lower() in video_extensions:
                # 获取视频信息
                video_info = self._get_video_info(file_path)
                if video_info:
                    video_files.append(video_info)
                    logger.info(f"发现视频: {file_path.name} - 时长: {video_info['duration']:.1f}秒")
        
        self.video_index = video_files
        return video_files
    
    def _get_video_info(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """
        获取视频基本信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            Dict: 视频信息，包含时长、分辨率等
        """
        try:
            # 使用ffprobe获取视频信息
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"ffprobe失败: {result.stderr}")
                return None
            
            data = json.loads(result.stdout)
            
            # 提取视频流信息
            video_stream = None
            audio_stream = None
            
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video' and not video_stream:
                    video_stream = stream
                elif stream.get('codec_type') == 'audio' and not audio_stream:
                    audio_stream = stream
            
            if not video_stream:
                logger.error(f"视频文件无视频流: {video_path}")
                return None
            
            duration = float(data['format'].get('duration', 0))
            
            return {
                'file_path': str(video_path),
                'file_name': video_path.name,
                'file_size': video_path.stat().st_size,
                'duration': duration,
                'duration_formatted': self._format_duration(duration),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'has_audio': audio_stream is not None,
                'audio_sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else None,
                'video_codec': video_stream.get('codec_name'),
                'audio_codec': audio_stream.get('codec_name') if audio_stream else None,
                'created_at': datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"ffprobe超时: {video_path}")
            return None
        except Exception as e:
            logger.error(f"获取视频信息失败: {video_path}, 错误: {e}")
            return None
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时长显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def segment_videos(self, 
                      force_resegment: bool = False,
                      extract_intro_only: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        切分视频为不同时长的片段
        
        Args:
            force_resegment: 是否强制重新切分
            extract_intro_only: 是否只提取前25%用于特征提取
            
        Returns:
            Dict: 按时长分组的片段信息
        """
        if not self.video_index:
            self.scan_source_videos()
        
        all_segments = {}
        
        for duration_min in self.durations:
            duration_sec = duration_min * 60
            segments_for_duration = []
            
            logger.info(f"开始切分 {duration_min} 分钟片段...")
            
            for video_info in self.video_index:
                video_path = Path(video_info['file_path'])
                video_duration = video_info['duration']
                
                # 计算可以切分多少个片段
                num_segments = int(video_duration // duration_sec)
                
                if num_segments == 0:
                    logger.warning(f"视频 {video_path.name} 时长不足 {duration_min} 分钟，跳过")
                    continue
                
                # 为每个片段生成输出路径和信息
                for i in range(num_segments):
                    start_time = i * duration_sec
                    
                    # 如果只提取intro，只处理第一个片段
                    if extract_intro_only and i > 0:
                        break
                    
                    segment_info = self._create_segment(
                        video_path, 
                        start_time, 
                        duration_sec, 
                        duration_min, 
                        i,
                        force_resegment
                    )
                    
                    if segment_info:
                        segments_for_duration.append(segment_info)
            
            all_segments[f"{duration_min}min"] = segments_for_duration
            logger.info(f"完成 {duration_min} 分钟片段切分，共 {len(segments_for_duration)} 个")
        
        # 保存片段索引
        self.segment_index = all_segments
        self._save_segment_index()
        
        return all_segments
    
    def _create_segment(self, 
                       video_path: Path, 
                       start_time: float, 
                       duration: float, 
                       duration_min: int, 
                       segment_index: int,
                       force_resegment: bool) -> Optional[Dict[str, Any]]:
        """
        创建单个视频片段
        
        Args:
            video_path: 原始视频路径
            start_time: 开始时间（秒）
            duration: 片段时长（秒）
            duration_min: 片段时长（分钟）
            segment_index: 片段索引
            force_resegment: 是否强制重新切分
            
        Returns:
            Dict: 片段信息
        """
        # 生成输出文件名
        base_name = video_path.stem
        output_name = f"{base_name}_seg{segment_index:03d}_{duration_min}min.mp4"
        output_path = self.segments_dir / f"{duration_min}min" / output_name
        
        # 检查文件是否已存在
        if output_path.exists() and not force_resegment:
            logger.info(f"片段已存在，跳过: {output_name}")
            return self._get_existing_segment_info(output_path, video_path, start_time, duration, segment_index)
        
        try:
            # 使用ffmpeg切分视频
            cmd = [
                'ffmpeg', '-y',  # 覆盖输出文件
                '-i', str(video_path),
                '-ss', str(start_time),  # 开始时间
                '-t', str(duration),     # 持续时间
                '-c', 'copy',            # 直接复制流，避免重新编码
                '-avoid_negative_ts', 'make_zero',
                str(output_path)
            ]
            
            logger.info(f"切分视频片段: {output_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg切分失败: {result.stderr}")
                return None
            
            # 验证输出文件
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"输出文件无效: {output_path}")
                return None
            
            logger.info(f"✅ 成功创建片段: {output_name}")
            
            return {
                'segment_path': str(output_path),
                'segment_name': output_name,
                'source_video': str(video_path),
                'start_time': start_time,
                'duration': duration,
                'duration_min': duration_min,
                'segment_index': segment_index,
                'file_size': output_path.stat().st_size,
                'created_at': datetime.now().isoformat(),
                'is_intro_segment': segment_index == 0,  # 标记是否为intro片段
                'intro_ratio': 0.25 if segment_index == 0 else 0  # ISO原则匹配阶段
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg切分超时: {output_name}")
            return None
        except Exception as e:
            logger.error(f"创建片段失败: {output_name}, 错误: {e}")
            return None
    
    def _get_existing_segment_info(self, 
                                  output_path: Path, 
                                  video_path: Path, 
                                  start_time: float, 
                                  duration: float, 
                                  segment_index: int) -> Dict[str, Any]:
        """获取已存在片段的信息"""
        duration_min = int(duration // 60)
        
        return {
            'segment_path': str(output_path),
            'segment_name': output_path.name,
            'source_video': str(video_path),
            'start_time': start_time,
            'duration': duration,
            'duration_min': duration_min,
            'segment_index': segment_index,
            'file_size': output_path.stat().st_size,
            'created_at': datetime.fromtimestamp(output_path.stat().st_mtime).isoformat(),
            'is_intro_segment': segment_index == 0,
            'intro_ratio': 0.25 if segment_index == 0 else 0
        }
    
    def _save_segment_index(self):
        """保存片段索引到文件"""
        index_file = self.features_dir / "segment_index.json"
        
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_index': self.video_index,
                    'segment_index': self.segment_index,
                    'updated_at': datetime.now().isoformat(),
                    'durations': self.durations
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 片段索引已保存: {index_file}")
            
        except Exception as e:
            logger.error(f"保存片段索引失败: {e}")
    
    def load_segment_index(self) -> bool:
        """从文件加载片段索引"""
        index_file = self.features_dir / "segment_index.json"
        
        if not index_file.exists():
            logger.info("片段索引文件不存在")
            return False
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.video_index = data.get('video_index', [])
            self.segment_index = data.get('segment_index', {})
            
            logger.info(f"✅ 加载片段索引成功，共 {len(self.video_index)} 个原始视频")
            
            # 统计片段数量
            total_segments = sum(len(segments) for segments in self.segment_index.values())
            logger.info(f"   总计 {total_segments} 个片段")
            
            return True
            
        except Exception as e:
            logger.error(f"加载片段索引失败: {e}")
            return False
    
    def get_intro_segments(self, duration_min: int = 5) -> List[Dict[str, Any]]:
        """
        获取所有intro片段（用于特征提取）
        
        Args:
            duration_min: 指定时长的片段
            
        Returns:
            List[Dict]: intro片段列表
        """
        duration_key = f"{duration_min}min"
        
        if duration_key not in self.segment_index:
            logger.warning(f"没有找到 {duration_min} 分钟的片段")
            return []
        
        intro_segments = [
            segment for segment in self.segment_index[duration_key]
            if segment.get('is_intro_segment', False)
        ]
        
        logger.info(f"找到 {len(intro_segments)} 个 {duration_min} 分钟的intro片段")
        return intro_segments
    
    def get_segments_by_duration(self, duration_min: int) -> List[Dict[str, Any]]:
        """获取指定时长的所有片段"""
        duration_key = f"{duration_min}min"
        return self.segment_index.get(duration_key, [])
    
    def check_ffmpeg_availability(self) -> bool:
        """检查ffmpeg是否可用"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ ffmpeg可用")
                return True
            else:
                logger.error("❌ ffmpeg不可用")
                return False
        except Exception as e:
            logger.error(f"❌ ffmpeg检查失败: {e}")
            return False
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理总结"""
        total_segments = sum(len(segments) for segments in self.segment_index.values())
        
        summary = {
            'source_videos': len(self.video_index),
            'total_segments': total_segments,
            'durations': self.durations,
            'segments_by_duration': {
                duration: len(self.segment_index.get(f"{duration}min", []))
                for duration in self.durations
            },
            'intro_segments': sum(
                1 for segments in self.segment_index.values()
                for segment in segments
                if segment.get('is_intro_segment', False)
            ),
            'total_disk_usage_mb': self._calculate_disk_usage()
        }
        
        return summary
    
    def _calculate_disk_usage(self) -> float:
        """计算磁盘使用量（MB）"""
        total_size = 0
        
        for segments in self.segment_index.values():
            for segment in segments:
                segment_path = Path(segment['segment_path'])
                if segment_path.exists():
                    total_size += segment_path.stat().st_size
        
        return round(total_size / (1024 * 1024), 2)

if __name__ == "__main__":
    # 测试视频处理器
    processor = VideoProcessor()
    
    # 检查ffmpeg
    if not processor.check_ffmpeg_availability():
        print("❌ 需要安装ffmpeg才能运行视频处理")
        exit(1)
    
    # 扫描视频
    print("🔍 扫描原始视频...")
    videos = processor.scan_source_videos()
    
    if not videos:
        print("❌ 未找到视频文件")
        exit(1)
    
    print(f"✅ 找到 {len(videos)} 个视频文件:")
    for video in videos:
        print(f"  - {video['file_name']}: {video['duration_formatted']}")
    
    # 切分视频
    print("\n🔪 开始切分视频...")
    segments = processor.segment_videos(extract_intro_only=True)
    
    # 显示结果
    print("\n📊 处理完成，结果统计:")
    summary = processor.get_processing_summary()
    
    for key, value in summary.items():
        print(f"  {key}: {value}")