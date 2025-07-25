#!/usr/bin/env python3
"""
特征提取模块 - 4.0版本核心组件
负责从视频片段中提取音乐特征，为检索提供基础
使用CLAMP3音乐理解大模型进行高级音乐特征提取
"""

import os
import json
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import hashlib

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLAMP3FeatureExtractor:
    """
    CLAMP3音乐特征提取器
    使用CLAMP3模型从视频中提取高级音乐特征
    """
    
    def __init__(self, clamp3_dir: str = "CLAMP3", features_dir: str = "materials/features"):
        """
        初始化CLAMP3特征提取器
        
        Args:
            clamp3_dir: CLAMP3项目目录
            features_dir: 特征存储目录
        """
        self.clamp3_dir = Path(clamp3_dir)
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证CLAMP3目录
        if not self.clamp3_dir.exists():
            raise FileNotFoundError(f"CLAMP3目录不存在: {self.clamp3_dir}")
        
        # 验证关键文件
        required_files = [
            "clamp3_embd.py",
            "utils.py",
            "code/extract_clamp3.py"
        ]
        
        for file_path in required_files:
            full_path = self.clamp3_dir / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"CLAMP3关键文件不存在: {full_path}")
        
        # 特征缓存
        self.features_cache = {}
        self.load_features_cache()
        
        # 临时目录 - 使用绝对路径
        self.temp_dir = Path("temp_clamp3_extraction").resolve()
        
        logger.info(f"✅ CLAMP3特征提取器初始化完成")
        logger.info(f"   CLAMP3目录: {self.clamp3_dir}")
        logger.info(f"   特征目录: {self.features_dir}")
    
    def extract_video_features(self, video_path: str, 
                             extract_ratio: float = 0.25) -> Optional[Dict[str, Any]]:
        """
        使用CLAMP3从视频中提取音乐特征
        
        Args:
            video_path: 视频文件路径
            extract_ratio: 提取比例（默认前25%，对应ISO匹配阶段）
            
        Returns:
            Dict: 包含CLAMP3特征向量的字典
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
            logger.info(f"使用CLAMP3提取特征: {video_path.name} (前{extract_ratio*100:.0f}%)")
            
            # 1. 提取音频片段
            audio_path = self._extract_audio_segment(video_path, extract_ratio)
            if audio_path is None:
                return None
            
            # 2. 使用CLAMP3提取特征
            clamp3_features = self._extract_clamp3_features(audio_path)
            if clamp3_features is None:
                logger.warning("CLAMP3特征提取失败，降级到传统音频特征")
                # 降级到传统音频特征
                return self._extract_fallback_features(audio_path, video_path, extract_ratio, feature_id)
            
            # 3. 构建特征字典
            features = {
                'clamp3_features': clamp3_features,
                'feature_vector': clamp3_features,  # 主要特征向量
                'video_path': str(video_path),
                'video_name': video_path.name,
                'extract_ratio': extract_ratio,
                'feature_id': feature_id,
                'extracted_at': datetime.now().isoformat(),
                'file_size': video_path.stat().st_size,
                'extractor_version': '4.0.0-clamp3',
                'model_type': 'clamp3-saas'
            }
            
            # 4. 保存到缓存
            self.features_cache[feature_id] = features
            self._save_features_cache()
            
            logger.info(f"✅ CLAMP3特征提取完成: {video_path.name}")
            logger.info(f"   特征维度: {clamp3_features.shape if isinstance(clamp3_features, np.ndarray) else 'unknown'}")
            
            return features
            
        except Exception as e:
            logger.error(f"CLAMP3特征提取失败: {video_path.name}, 错误: {e}")
            logger.warning("降级到传统音频特征")
            
            # 尝试降级特征提取
            try:
                audio_path = self._extract_audio_segment(video_path, extract_ratio)
                if audio_path:
                    return self._extract_fallback_features(audio_path, video_path, extract_ratio, feature_id)
            except Exception as fallback_error:
                logger.error(f"降级特征提取也失败: {fallback_error}")
            
            return None
        finally:
            # 清理临时文件
            self._cleanup_temp_files()
    
    def _extract_fallback_features(self, audio_path: str, video_path: Path, 
                                 extract_ratio: float, feature_id: str) -> Optional[Dict[str, Any]]:
        """
        降级特征提取方法 - 使用librosa提取传统音频特征
        """
        try:
            import librosa
            
            logger.info("使用Librosa提取传统音频特征")
            
            # 加载音频
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 提取多种特征
            features = {}
            
            # 1. MFCC特征 (13维)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 2. 色谱特征 (12维)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # 3. 谱质心
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = np.mean(centroid)
            
            # 4. 谱带宽
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth'] = np.mean(bandwidth)
            
            # 5. 过零率
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zero_crossing_rate'] = np.mean(zcr)
            
            # 6. RMS能量
            rms = librosa.feature.rms(y=y)
            features['rms_energy'] = np.mean(rms)
            
            # 7. 谱回滚点
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff'] = np.mean(rolloff)
            
            # 组合所有特征为一个向量
            feature_vector = np.concatenate([
                features['mfcc_mean'],
                features['mfcc_std'], 
                features['chroma_mean'],
                features['chroma_std'],
                [features['spectral_centroid']],
                [features['spectral_bandwidth']],
                [features['zero_crossing_rate']],
                [features['rms_energy']],
                [features['spectral_rolloff']]
            ])
            
            # 构建结果字典
            result = {
                'librosa_features': features,
                'feature_vector': feature_vector,  # 主要特征向量
                'video_path': str(video_path),
                'video_name': video_path.name,
                'extract_ratio': extract_ratio,
                'feature_id': feature_id,
                'extracted_at': datetime.now().isoformat(),
                'file_size': video_path.stat().st_size,
                'extractor_version': '4.0.0-fallback',
                'model_type': 'librosa-traditional'
            }
            
            # 保存到缓存
            self.features_cache[feature_id] = result
            self._save_features_cache()
            
            logger.info(f"✅ 降级特征提取完成: {video_path.name}")
            logger.info(f"   特征维度: {feature_vector.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"降级特征提取失败: {e}")
            return None
    
    def _generate_feature_id(self, video_path: Path, extract_ratio: float) -> str:
        """生成特征的唯一ID"""
        stat = video_path.stat()
        content = f"clamp3_{video_path.name}_{stat.st_size}_{stat.st_mtime}_{extract_ratio}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_audio_segment(self, video_path: Path, extract_ratio: float) -> Optional[str]:
        """
        从视频中提取音频片段
        
        Args:
            video_path: 视频文件路径
            extract_ratio: 提取比例
            
        Returns:
            str: 临时音频文件路径
        """
        try:
            # 首先获取视频时长
            duration = self._get_video_duration(video_path)
            if duration is None:
                return None
            
            # 计算提取时长
            extract_duration = duration * extract_ratio
            
            # 创建临时目录
            self.temp_dir.mkdir(exist_ok=True)
            
            # 生成临时音频文件路径
            temp_audio_path = self.temp_dir / f"{video_path.stem}_segment.wav"
            
            # 使用ffmpeg提取音频
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-t', str(extract_duration),  # 只提取前N秒
                '-vn',  # 不要视频
                '-acodec', 'pcm_s16le',  # 16位PCM
                '-ar', '22050',  # 22.05kHz采样率
                '-ac', '1',  # 单声道
                str(temp_audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg音频提取失败: {result.stderr}")
                return None
            
            if not temp_audio_path.exists():
                logger.error(f"音频文件提取失败: {temp_audio_path}")
                return None
            
            logger.info(f"✅ 音频提取完成: {temp_audio_path.name} ({extract_duration:.1f}s)")
            return str(temp_audio_path)
            
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            return None
    
    def _extract_clamp3_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        使用CLAMP3提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            np.ndarray: CLAMP3特征向量
        """
        try:
            # 创建临时输入目录（CLAMP3需要目录作为输入）
            temp_input_dir = self.temp_dir / "input"
            temp_input_dir.mkdir(exist_ok=True)
            
            # 复制音频文件到临时输入目录
            audio_path = Path(audio_path)
            temp_audio_in_dir = temp_input_dir / audio_path.name
            shutil.copy(audio_path, temp_audio_in_dir)
            
            # 创建唯一的特征输出目录（CLAMP3会创建）
            import time
            timestamp = str(int(time.time() * 1000))
            temp_output_dir = self.temp_dir / f"output_{timestamp}"
            # 确保目录不存在（CLAMP3要求不存在的目录）
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)
            # 不要预先创建目录，让CLAMP3创建
            
            # 保存当前工作目录
            original_cwd = os.getcwd()
            
            try:
                # 切换到CLAMP3目录
                os.chdir(self.clamp3_dir)
                
                # 使用绝对路径
                cmd = [
                    'python', 'clamp3_embd.py', 
                    str(temp_input_dir), 
                    str(temp_output_dir),
                    '--get_global'
                ]
                
                logger.info(f"执行CLAMP3特征提取: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                # 记录CLAMP3的输出信息  
                if result.stdout:
                    logger.info(f"CLAMP3输出: {result.stdout}")
                if result.stderr:
                    logger.warning(f"CLAMP3错误输出: {result.stderr}")
                
                if result.returncode != 0:
                    logger.error(f"CLAMP3执行失败 (返回码: {result.returncode})")
                    
                    # 检查是否是依赖问题
                    if "ModuleNotFoundError" in result.stderr:
                        logger.warning("CLAMP3依赖缺失，请安装requirements.txt中的依赖")
                        logger.warning("或者运行：pip install -r CLAMP3/requirements.txt")
                    
                    return None
                
                # 查找输出的特征文件
                feature_files = list(temp_output_dir.glob("*.npy"))
                
                if not feature_files:
                    logger.error("CLAMP3未生成特征文件")
                    return None
                
                # 加载特征文件
                feature_file = feature_files[0]
                features = np.load(feature_file)
                
                logger.info(f"✅ CLAMP3特征加载成功: {features.shape}")
                
                return features
                
            finally:
                # 恢复工作目录
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"CLAMP3特征提取失败: {e}")
            
            # 如果是依赖问题，给出提示
            if "ModuleNotFoundError" in str(e):
                logger.warning("CLAMP3依赖缺失，请安装requirements.txt中的依赖")
                logger.warning("或者运行：pip install -r CLAMP3/requirements.txt")
            
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
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug(f"清理临时目录: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    
    def extract_batch_features(self, video_list: List[Dict[str, Any]], 
                             extract_ratio: float = 0.25) -> Dict[str, Dict[str, Any]]:
        """
        批量提取CLAMP3特征
        
        Args:
            video_list: 视频信息列表
            extract_ratio: 提取比例
            
        Returns:
            Dict: 视频路径到特征的映射
        """
        features_db = {}
        
        logger.info(f"开始批量CLAMP3特征提取，共 {len(video_list)} 个视频")
        
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
                logger.warning(f"CLAMP3特征提取失败: {video_path}")
        
        logger.info(f"✅ 批量CLAMP3特征提取完成，成功 {len(features_db)}/{len(video_list)}")
        
        # 保存特征数据库
        self._save_features_database(features_db)
        
        return features_db
    
    def _save_features_database(self, features_db: Dict[str, Dict[str, Any]]):
        """保存特征数据库"""
        db_file = self.features_dir / "clamp3_features_database.json"
        
        try:
            # 将numpy数组转换为列表以便JSON序列化
            serializable_db = {}
            for video_path, features in features_db.items():
                serializable_features = dict(features)
                
                # 转换numpy数组
                if 'clamp3_features' in serializable_features:
                    clamp3_feat = serializable_features['clamp3_features']
                    if isinstance(clamp3_feat, np.ndarray):
                        serializable_features['clamp3_features'] = clamp3_feat.tolist()
                        serializable_features['clamp3_features_shape'] = clamp3_feat.shape
                
                if 'feature_vector' in serializable_features:
                    feat_vec = serializable_features['feature_vector']
                    if isinstance(feat_vec, np.ndarray):
                        serializable_features['feature_vector'] = feat_vec.tolist()
                        serializable_features['feature_vector_shape'] = feat_vec.shape
                
                serializable_db[video_path] = serializable_features
            
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'features_database': serializable_db,
                    'created_at': datetime.now().isoformat(),
                    'total_videos': len(features_db),
                    'extractor_version': '4.0.0-clamp3',
                    'model_type': 'clamp3-saas'
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ CLAMP3特征数据库已保存: {db_file}")
            
        except Exception as e:
            logger.error(f"保存CLAMP3特征数据库失败: {e}")
    
    def load_features_database(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """加载特征数据库"""
        db_file = self.features_dir / "clamp3_features_database.json"
        
        if not db_file.exists():
            logger.info("CLAMP3特征数据库文件不存在")
            return None
        
        try:
            with open(db_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features_db = data.get('features_database', {})
            
            # 将列表转换回numpy数组
            for video_path, features in features_db.items():
                if 'clamp3_features' in features and isinstance(features['clamp3_features'], list):
                    features['clamp3_features'] = np.array(features['clamp3_features'])
                if 'feature_vector' in features and isinstance(features['feature_vector'], list):
                    features['feature_vector'] = np.array(features['feature_vector'])
            
            logger.info(f"✅ 加载CLAMP3特征数据库成功，共 {len(features_db)} 个视频特征")
            
            return features_db
            
        except Exception as e:
            logger.error(f"加载CLAMP3特征数据库失败: {e}")
            return None
    
    def _save_features_cache(self):
        """保存特征缓存"""
        cache_file = self.features_dir / "clamp3_features_cache.json"
        
        try:
            # 将numpy数组转换为列表以便JSON序列化
            serializable_cache = {}
            for feature_id, features in self.features_cache.items():
                serializable_features = dict(features)
                
                # 转换numpy数组
                if 'clamp3_features' in serializable_features:
                    clamp3_feat = serializable_features['clamp3_features']
                    if isinstance(clamp3_feat, np.ndarray):
                        serializable_features['clamp3_features'] = clamp3_feat.tolist()
                        serializable_features['clamp3_features_shape'] = clamp3_feat.shape
                
                if 'feature_vector' in serializable_features:
                    feat_vec = serializable_features['feature_vector']
                    if isinstance(feat_vec, np.ndarray):
                        serializable_features['feature_vector'] = feat_vec.tolist()
                        serializable_features['feature_vector_shape'] = feat_vec.shape
                
                # 处理降级特征中的嵌套numpy数组和float32
                if 'librosa_features' in serializable_features:
                    librosa_feat = serializable_features['librosa_features']
                    if isinstance(librosa_feat, dict):
                        for key, value in librosa_feat.items():
                            if isinstance(value, np.ndarray):
                                librosa_feat[key] = value.tolist()
                            elif isinstance(value, (np.float32, np.float64)):
                                librosa_feat[key] = float(value)
                
                serializable_cache[feature_id] = serializable_features
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_cache, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存CLAMP3特征缓存失败: {e}")
    
    def load_features_cache(self):
        """加载特征缓存"""
        cache_file = self.features_dir / "clamp3_features_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # 将列表转换回numpy数组
                for feature_id, features in cached_data.items():
                    if 'clamp3_features' in features and isinstance(features['clamp3_features'], list):
                        features['clamp3_features'] = np.array(features['clamp3_features'])
                    if 'feature_vector' in features and isinstance(features['feature_vector'], list):
                        features['feature_vector'] = np.array(features['feature_vector'])
                
                self.features_cache = cached_data
                logger.info(f"加载CLAMP3特征缓存: {len(self.features_cache)} 项")
                
            except Exception as e:
                logger.error(f"加载CLAMP3特征缓存失败: {e}")
                self.features_cache = {}
        else:
            self.features_cache = {}

class AudioFeatureExtractor:
    """
    音频特征提取器（兼容性保持）
    从视频中提取音频并分析其音乐特征
    现在使用CLAMP3作为主要特征提取引擎
    """
    
    def __init__(self, features_dir: str = "materials/features"):
        """
        初始化特征提取器
        
        Args:
            features_dir: 特征存储目录
        """
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化CLAMP3特征提取器
        try:
            self.clamp3_extractor = CLAMP3FeatureExtractor(features_dir=features_dir)
            self.use_clamp3 = True
            logger.info("✅ AudioFeatureExtractor使用CLAMP3作为后端")
        except Exception as e:
            logger.warning(f"CLAMP3初始化失败，使用传统特征提取: {e}")
            self.use_clamp3 = False
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
        # 如果可用，使用CLAMP3提取特征
        if self.use_clamp3:
            return self.clamp3_extractor.extract_video_features(video_path, extract_ratio)
        
        # 否则使用传统特征提取方法
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
            logger.info(f"提取传统特征: {video_path.name} (前{extract_ratio*100:.0f}%)")
            
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
                'extractor_version': '4.0.0-fallback'
            })
            
            # 4. 保存到缓存
            self.features_cache[feature_id] = features
            self._save_features_cache()
            
            logger.info(f"✅ 传统特征提取完成: {video_path.name}")
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
        # 如果可用，使用CLAMP3批量提取
        if self.use_clamp3:
            return self.clamp3_extractor.extract_batch_features(video_list, extract_ratio)
        
        # 否则使用传统批量提取
        features_db = {}
        
        logger.info(f"开始批量传统特征提取，共 {len(video_list)} 个视频")
        
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
        
        logger.info(f"✅ 批量传统特征提取完成，成功 {len(features_db)}/{len(video_list)}")
        
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
        # 如果可用，使用CLAMP3加载数据库
        if self.use_clamp3:
            return self.clamp3_extractor.load_features_database()
        
        # 否则使用传统方法
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