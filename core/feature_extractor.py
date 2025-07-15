#!/usr/bin/env python3
"""
特征提取模块 - 4.0版本核心组件
负责从视频片段中提取音乐特征，为检索提供基础
"""

import os
import json
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import hashlib

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """
    音频特征提取器
    从视频中提取音频并分析其音乐特征
    """
    
    def __init__(self, features_dir: str = "materials/features"):
        """
        初始化特征提取器
        
        Args:
            features_dir: 特征存储目录
        """
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # 特征缓存
        self.features_cache = {}
        self.load_features_cache()
    
    def extract_video_features(self, video_path: str, 
                             extract_ratio: float = 0.25) -> Optional[Dict[str, Any]]:
        """
        从视频中提取音乐特征
        
        Args:
            video_path: 视频文件路径
            extract_ratio: 提取比例（默认前25%，对应ISO匹配阶段）
            
        Returns:
            Dict: 音乐特征字典
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return None
        
        # 生成特征ID
        feature_id = self._generate_feature_id(video_path, extract_ratio)
        
        # 检查缓存
        if feature_id in self.features_cache:
            logger.info(f"使用缓存特征: {video_path.name}")
            return self.features_cache[feature_id]
        
        try:
            logger.info(f"提取特征: {video_path.name} (前{extract_ratio*100:.0f}%)")
            
            # 1. 提取音频
            audio_data = self._extract_audio_segment(video_path, extract_ratio)
            if audio_data is None:
                return None
            
            # 2. 分析音频特征
            features = self._analyze_audio_features(audio_data)
            if features is None:
                return None
            
            # 3. 添加元数据
            features.update({
                'video_path': str(video_path),
                'video_name': video_path.name,
                'extract_ratio': extract_ratio,
                'feature_id': feature_id,
                'extracted_at': datetime.now().isoformat(),
                'file_size': video_path.stat().st_size,
                'extractor_version': '4.0.0'
            })
            
            # 4. 保存到缓存
            self.features_cache[feature_id] = features
            self._save_features_cache()
            
            logger.info(f"✅ 特征提取完成: {video_path.name}")
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {video_path.name}, 错误: {e}")
            return None
    
    def _generate_feature_id(self, video_path: Path, extract_ratio: float) -> str:
        """生成特征的唯一ID"""
        # 基于文件路径、大小、修改时间和提取比例生成哈希
        stat = video_path.stat()
        content = f"{video_path.name}_{stat.st_size}_{stat.st_mtime}_{extract_ratio}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_audio_segment(self, video_path: Path, extract_ratio: float) -> Optional[np.ndarray]:
        """
        从视频中提取音频片段
        
        Args:
            video_path: 视频文件路径
            extract_ratio: 提取比例
            
        Returns:
            np.ndarray: 音频数据
        """
        try:
            # 首先获取视频时长
            duration = self._get_video_duration(video_path)
            if duration is None:
                return None
            
            # 计算提取时长
            extract_duration = duration * extract_ratio
            
            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                # 使用ffmpeg提取音频
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),
                    '-t', str(extract_duration),  # 只提取前N秒
                    '-vn',  # 不要视频
                    '-acodec', 'pcm_s16le',  # 16位PCM
                    '-ar', '22050',  # 22.05kHz采样率
                    '-ac', '1',  # 单声道
                    temp_audio_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    logger.error(f"ffmpeg音频提取失败: {result.stderr}")
                    return None
                
                # 读取音频数据
                audio_data = self._load_audio_file(temp_audio_path)
                return audio_data
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            return None
    
    def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """获取视频时长"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            return float(data['format'].get('duration', 0))
            
        except Exception:
            return None
    
    def _load_audio_file(self, audio_path: str) -> Optional[np.ndarray]:
        """加载音频文件为numpy数组"""
        try:
            # 尝试使用librosa（如果可用）
            try:
                import librosa
                audio_data, sr = librosa.load(audio_path, sr=22050, mono=True)
                return audio_data
            except ImportError:
                pass
            
            # 如果没有librosa，使用scipy
            try:
                from scipy.io import wavfile
                sr, audio_data = wavfile.read(audio_path)
                
                # 转换为float32并归一化
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                
                return audio_data
            except ImportError:
                pass
            
            # 最后尝试使用ffmpeg读取
            return self._load_audio_with_ffmpeg(audio_path)
            
        except Exception as e:
            logger.error(f"加载音频文件失败: {e}")
            return None
    
    def _load_audio_with_ffmpeg(self, audio_path: str) -> Optional[np.ndarray]:
        """使用ffmpeg读取音频数据"""
        try:
            cmd = [
                'ffmpeg', '-i', audio_path, '-f', 'f32le', '-acodec', 'pcm_f32le', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode != 0:
                return None
            
            # 将字节数据转换为float32数组
            audio_data = np.frombuffer(result.stdout, dtype=np.float32)
            return audio_data
            
        except Exception:
            return None
    
    def _analyze_audio_features(self, audio_data: np.ndarray, sr: int = 22050) -> Optional[Dict[str, Any]]:
        """
        分析音频特征
        
        Args:
            audio_data: 音频数据
            sr: 采样率
            
        Returns:
            Dict: 音频特征
        """
        try:
            features = {}
            
            # 1. 基础统计特征
            features.update(self._extract_basic_features(audio_data))
            
            # 2. 频域特征
            features.update(self._extract_frequency_features(audio_data, sr))
            
            # 3. 时域特征
            features.update(self._extract_temporal_features(audio_data, sr))
            
            # 4. 感知特征（简化版）
            features.update(self._extract_perceptual_features(audio_data, sr))
            
            return features
            
        except Exception as e:
            logger.error(f"音频特征分析失败: {e}")
            return None
    
    def _extract_basic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """提取基础统计特征"""
        return {
            'duration': len(audio_data) / 22050,  # 时长（秒）
            'rms_energy': float(np.sqrt(np.mean(audio_data**2))),  # RMS能量
            'zero_crossing_rate': float(np.mean(np.abs(np.diff(np.sign(audio_data))) / 2)),  # 过零率
            'amplitude_mean': float(np.mean(np.abs(audio_data))),  # 平均振幅
            'amplitude_std': float(np.std(audio_data)),  # 振幅标准差
            'amplitude_max': float(np.max(np.abs(audio_data))),  # 最大振幅
            'dynamic_range': float(np.max(audio_data) - np.min(audio_data))  # 动态范围
        }
    
    def _extract_frequency_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """提取频域特征"""
        # 计算FFT
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # 频率轴
        freqs = np.fft.fftfreq(len(audio_data), 1/sr)[:len(fft)//2]
        
        # 归一化
        if np.sum(magnitude) > 0:
            magnitude = magnitude / np.sum(magnitude)
        
        # 特征提取
        features = {}
        
        # 质心频率
        if np.sum(magnitude) > 0:
            features['spectral_centroid'] = float(np.sum(freqs * magnitude) / np.sum(magnitude))
        else:
            features['spectral_centroid'] = 0.0
        
        # 频谱带宽
        if features['spectral_centroid'] > 0:
            features['spectral_bandwidth'] = float(
                np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * magnitude) / np.sum(magnitude))
            )
        else:
            features['spectral_bandwidth'] = 0.0
        
        # 频谱滚降点
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            features['spectral_rolloff'] = float(freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0)
        else:
            features['spectral_rolloff'] = 0.0
        
        # 频谱平坦度
        if np.mean(magnitude) > 0:
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            features['spectral_flatness'] = float(geometric_mean / arithmetic_mean)
        else:
            features['spectral_flatness'] = 0.0
        
        return features
    
    def _extract_temporal_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """提取时域特征"""
        features = {}
        
        # 分帧分析
        frame_length = int(0.025 * sr)  # 25ms帧
        hop_length = int(0.01 * sr)     # 10ms跳跃
        
        frames = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frames.append(audio_data[i:i + frame_length])
        
        if not frames:
            return {
                'tempo_estimate': 0.0,
                'rhythm_regularity': 0.0,
                'onset_density': 0.0
            }
        
        # 简化的节拍估计
        frame_energies = [np.sum(frame**2) for frame in frames]
        energy_diffs = np.abs(np.diff(frame_energies))
        
        # 寻找能量峰值（简单的onset检测）
        threshold = np.mean(energy_diffs) + np.std(energy_diffs)
        onsets = np.where(energy_diffs > threshold)[0]
        
        # 估计tempo
        if len(onsets) > 1:
            onset_intervals = np.diff(onsets) * hop_length / sr
            if len(onset_intervals) > 0:
                avg_interval = np.median(onset_intervals)
                tempo_estimate = 60.0 / avg_interval if avg_interval > 0 else 0
                features['tempo_estimate'] = float(min(tempo_estimate, 200))  # 限制在合理范围
            else:
                features['tempo_estimate'] = 0.0
            
            # 节奏规律性
            if len(onset_intervals) > 0:
                features['rhythm_regularity'] = float(1.0 / (1.0 + np.std(onset_intervals)))
            else:
                features['rhythm_regularity'] = 0.0
        else:
            features['tempo_estimate'] = 0.0
            features['rhythm_regularity'] = 0.0
        
        # Onset密度
        features['onset_density'] = float(len(onsets) / (len(audio_data) / sr))
        
        return features
    
    def _extract_perceptual_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """提取感知特征（简化版）"""
        features = {}
        
        # 响度估计（基于RMS）
        rms = np.sqrt(np.mean(audio_data**2))
        features['loudness_estimate'] = float(20 * np.log10(rms + 1e-10))  # dB
        
        # 亮度估计（高频能量比例）
        fft = np.abs(np.fft.fft(audio_data))
        total_energy = np.sum(fft**2)
        
        if total_energy > 0:
            # 高频（>1kHz）能量比例
            high_freq_start = int(1000 * len(fft) / sr)
            high_freq_energy = np.sum(fft[high_freq_start:]**2)
            features['brightness'] = float(high_freq_energy / total_energy)
        else:
            features['brightness'] = 0.0
        
        # 温暖度估计（低频能量比例）
        if total_energy > 0:
            # 低频（<500Hz）能量比例
            low_freq_end = int(500 * len(fft) / sr)
            low_freq_energy = np.sum(fft[:low_freq_end]**2)
            features['warmth'] = float(low_freq_energy / total_energy)
        else:
            features['warmth'] = 0.0
        
        # 粗糙度估计（基于频谱不规则性）
        if len(fft) > 2:
            spectral_diff = np.diff(fft[:len(fft)//2])
            features['roughness'] = float(np.std(spectral_diff))
        else:
            features['roughness'] = 0.0
        
        return features
    
    def extract_batch_features(self, video_list: List[Dict[str, Any]], 
                             extract_ratio: float = 0.25) -> Dict[str, Dict[str, Any]]:
        """
        批量提取特征
        
        Args:
            video_list: 视频信息列表
            extract_ratio: 提取比例
            
        Returns:
            Dict: 视频路径到特征的映射
        """
        features_db = {}
        
        logger.info(f"开始批量特征提取，共 {len(video_list)} 个视频")
        
        for i, video_info in enumerate(video_list, 1):
            video_path = video_info.get('segment_path') or video_info.get('file_path')
            
            if not video_path:
                logger.warning(f"视频信息缺少路径: {video_info}")
                continue
            
            logger.info(f"处理 {i}/{len(video_list)}: {Path(video_path).name}")
            
            features = self.extract_video_features(video_path, extract_ratio)
            
            if features:
                features_db[video_path] = features
                # 添加视频信息
                features.update({
                    'source_info': video_info
                })
            else:
                logger.warning(f"特征提取失败: {video_path}")
        
        logger.info(f"✅ 批量特征提取完成，成功 {len(features_db)}/{len(video_list)}")
        
        # 保存特征数据库
        self._save_features_database(features_db)
        
        return features_db
    
    def _save_features_database(self, features_db: Dict[str, Dict[str, Any]]):
        """保存特征数据库"""
        db_file = self.features_dir / "features_database.json"
        
        try:
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'features_database': features_db,
                    'created_at': datetime.now().isoformat(),
                    'total_videos': len(features_db),
                    'extractor_version': '4.0.0'
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 特征数据库已保存: {db_file}")
            
        except Exception as e:
            logger.error(f"保存特征数据库失败: {e}")
    
    def load_features_database(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """加载特征数据库"""
        db_file = self.features_dir / "features_database.json"
        
        if not db_file.exists():
            logger.info("特征数据库文件不存在")
            return None
        
        try:
            with open(db_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features_db = data.get('features_database', {})
            logger.info(f"✅ 加载特征数据库成功，共 {len(features_db)} 个视频特征")
            
            return features_db
            
        except Exception as e:
            logger.error(f"加载特征数据库失败: {e}")
            return None
    
    def _save_features_cache(self):
        """保存特征缓存"""
        cache_file = self.features_dir / "features_cache.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.features_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存特征缓存失败: {e}")
    
    def load_features_cache(self):
        """加载特征缓存"""
        cache_file = self.features_dir / "features_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.features_cache = json.load(f)
                logger.info(f"加载特征缓存: {len(self.features_cache)} 项")
            except Exception as e:
                logger.error(f"加载特征缓存失败: {e}")
                self.features_cache = {}
        else:
            self.features_cache = {}

if __name__ == "__main__":
    # 测试特征提取器
    extractor = AudioFeatureExtractor()
    
    # 测试单个视频
    test_video = "materials/video/32.mp4"
    
    if os.path.exists(test_video):
        print(f"🎵 测试特征提取: {test_video}")
        features = extractor.extract_video_features(test_video)
        
        if features:
            print("✅ 特征提取成功!")
            print(f"   时长: {features.get('duration', 0):.1f}秒")
            print(f"   RMS能量: {features.get('rms_energy', 0):.4f}")
            print(f"   频谱质心: {features.get('spectral_centroid', 0):.1f}Hz")
            print(f"   估计节拍: {features.get('tempo_estimate', 0):.1f}BPM")
            print(f"   亮度: {features.get('brightness', 0):.3f}")
            print(f"   温暖度: {features.get('warmth', 0):.3f}")
        else:
            print("❌ 特征提取失败")
    else:
        print(f"❌ 测试视频不存在: {test_video}")