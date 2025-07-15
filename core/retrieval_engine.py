#!/usr/bin/env python3
"""
检索引擎模块 - 4.0版本核心组件
基于情绪和音乐特征进行视频片段检索匹配
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import random

from .emotion_mapper import emotion_recognizer, music_feature_mapper

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoRetrievalEngine:
    """
    视频检索引擎
    基于情绪特征和音乐特征进行相似度计算和检索
    """
    
    def __init__(self, features_dir: str = "materials/features"):
        """
        初始化检索引擎
        
        Args:
            features_dir: 特征数据目录
        """
        self.features_dir = Path(features_dir)
        self.features_database = {}
        self.emotion_database = {}
        
        # 加载数据
        self.load_databases()
        
        # 特征权重配置
        self.feature_weights = {
            # 基础特征权重
            'tempo': 0.25,           # 节拍重要性高
            'energy': 0.20,          # 能量重要性高  
            'brightness': 0.15,      # 亮度
            'warmth': 0.15,          # 温暖度
            'rhythm': 0.10,          # 节奏规律性
            'spectral': 0.10,        # 频谱特征
            'dynamics': 0.05         # 动态范围
        }
        
        # 情绪特征映射权重
        self.emotion_weights = {
            'matching_stage': 1.0,   # ISO匹配阶段权重最高
            'mood_similarity': 0.8,  # 情绪相似度
            'energy_level': 0.6      # 能量水平匹配
        }
    
    def load_databases(self):
        """加载特征数据库和情绪数据库"""
        # 加载音频特征数据库
        features_db_file = self.features_dir / "features_database.json"
        if features_db_file.exists():
            try:
                with open(features_db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.features_database = data.get('features_database', {})
                logger.info(f"✅ 加载特征数据库: {len(self.features_database)} 个视频")
            except Exception as e:
                logger.error(f"加载特征数据库失败: {e}")
        
        # 构建情绪数据库
        self._build_emotion_database()
    
    def _build_emotion_database(self):
        """构建情绪特征数据库"""
        self.emotion_database = {}
        
        # 获取所有支持的情绪
        supported_emotions = music_feature_mapper.get_supported_emotions()
        
        for emotion in supported_emotions:
            # 获取匹配阶段特征（对应前25%视频内容）
            matching_features = music_feature_mapper.extract_matching_stage_features(emotion)
            
            # 转换为数值特征向量
            feature_vector = self._emotion_features_to_vector(matching_features)
            
            self.emotion_database[emotion] = {
                'features': matching_features,
                'feature_vector': feature_vector,
                'normalized_vector': self._normalize_vector(feature_vector)
            }
        
        logger.info(f"✅ 构建情绪数据库: {len(self.emotion_database)} 种情绪")
    
    def _emotion_features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        将情绪特征转换为数值向量
        
        Args:
            features: 情绪特征字典
            
        Returns:
            np.ndarray: 特征向量
        """
        vector = []
        
        # 节拍特征
        tempo_val = self._extract_tempo_value(features.get('tempo', ''))
        vector.append(tempo_val)
        
        # 调性特征
        key_val = self._extract_key_value(features.get('key', ''))
        vector.append(key_val)
        
        # 动态特征
        dynamics_val = self._extract_dynamics_value(features.get('dynamics', ''))
        vector.append(dynamics_val)
        
        # 情绪强度（基于mood描述）
        mood_intensity = self._extract_mood_intensity(features.get('mood', ''))
        vector.append(mood_intensity)
        
        # 乐器复杂度
        instrument_complexity = self._extract_instrument_complexity(features.get('instrumental', ''))
        vector.append(instrument_complexity)
        
        # 织体复杂度
        texture_complexity = self._extract_texture_complexity(features.get('texture', ''))
        vector.append(texture_complexity)
        
        return np.array(vector, dtype=np.float32)
    
    def _extract_tempo_value(self, tempo_desc: str) -> float:
        """从tempo描述中提取数值"""
        tempo_map = {
            'slow': 0.2, 'deeply': 0.1, 'profound': 0.1,
            'moderate': 0.5, 'gradually': 0.4, 'natural': 0.5,
            'fast': 0.8, 'urgent': 0.9, 'agitated': 0.85,
            'tired': 0.25, 'sluggish': 0.2, 'gentle': 0.3,
            'smooth': 0.4, 'flowing': 0.45, 'weightless': 0.15
        }
        
        for key, value in tempo_map.items():
            if key in tempo_desc.lower():
                return value
        
        return 0.5  # 默认中等
    
    def _extract_key_value(self, key_desc: str) -> float:
        """从调性描述中提取数值"""
        # 简化映射：minor=负面情绪(0.2-0.4), major=正面情绪(0.6-0.8), neutral=中性(0.5)
        if 'minor' in key_desc.lower():
            if 'anxious' in key_desc or 'tense' in key_desc:
                return 0.2
            elif 'weary' in key_desc:
                return 0.25
            else:
                return 0.3
        elif 'major' in key_desc.lower():
            if 'calm' in key_desc:
                return 0.7
            elif 'warm' in key_desc:
                return 0.75
            else:
                return 0.6
        else:
            return 0.5  # neutral
    
    def _extract_dynamics_value(self, dynamics_desc: str) -> float:
        """从动态描述中提取数值"""
        dynamics_map = {
            'whisper': 0.1, 'gentle': 0.2, 'soft': 0.25,
            'restless': 0.7, 'sharp': 0.8, 'heavy': 0.6,
            'compressed': 0.75, 'embracing': 0.3, 'peaceful': 0.2,
            'expanding': 0.5, 'free': 0.4
        }
        
        for key, value in dynamics_map.items():
            if key in dynamics_desc.lower():
                return value
        
        return 0.4  # 默认
    
    def _extract_mood_intensity(self, mood_desc: str) -> float:
        """从情绪描述中提取强度"""
        intensity_keywords = {
            'deep': 0.8, 'profound': 0.9, 'transcendent': 1.0,
            'anxiety': 0.7, 'stress': 0.8, 'overload': 0.9,
            'exhausted': 0.6, 'fatigue': 0.5, 'tired': 0.4,
            'irritated': 0.7, 'angry': 0.8, 'calm': 0.3,
            'peace': 0.2, 'tranquil': 0.25, 'serene': 0.2
        }
        
        max_intensity = 0.5
        for keyword, intensity in intensity_keywords.items():
            if keyword in mood_desc.lower():
                max_intensity = max(max_intensity, intensity)
        
        return max_intensity
    
    def _extract_instrument_complexity(self, instrument_desc: str) -> float:
        """从乐器描述中提取复杂度"""
        if not instrument_desc:
            return 0.5
        
        # 统计乐器数量和类型
        instruments = instrument_desc.lower().split(',')
        complexity = len(instruments) * 0.2  # 基础复杂度
        
        # 特定乐器的复杂度加权
        complex_instruments = ['percussion', 'brass', 'strings']
        simple_instruments = ['piano', 'ambient', 'pads']
        
        for inst in instruments:
            inst = inst.strip()
            if any(ci in inst for ci in complex_instruments):
                complexity += 0.3
            elif any(si in inst for si in simple_instruments):
                complexity += 0.1
        
        return min(complexity, 1.0)
    
    def _extract_texture_complexity(self, texture_desc: str) -> float:
        """从织体描述中提取复杂度"""
        texture_map = {
            'minimal': 0.1, 'simple': 0.2, 'spacious': 0.2,
            'complex': 0.8, 'layered': 0.7, 'polyphonic': 0.9,
            'angular': 0.6, 'flowing': 0.4, 'smooth': 0.3,
            'balanced': 0.5, 'compressed': 0.7, 'intense': 0.8
        }
        
        for key, value in texture_map.items():
            if key in texture_desc.lower():
                return value
        
        return 0.5
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """归一化向量"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _audio_features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        将音频特征转换为向量
        
        Args:
            features: 音频特征字典
            
        Returns:
            np.ndarray: 特征向量
        """
        vector = []
        
        # 标准化特征提取
        vector.append(min(features.get('tempo_estimate', 0) / 200.0, 1.0))  # 节拍(0-200 BPM)
        vector.append(min(features.get('rms_energy', 0) * 10, 1.0))         # 能量
        vector.append(min(features.get('brightness', 0), 1.0))              # 亮度
        vector.append(min(features.get('warmth', 0), 1.0))                  # 温暖度
        vector.append(min(features.get('rhythm_regularity', 0), 1.0))       # 节奏规律性
        vector.append(min(features.get('spectral_centroid', 0) / 8000.0, 1.0))  # 频谱质心
        vector.append(min(features.get('dynamic_range', 0) / 2.0, 1.0))     # 动态范围
        
        return np.array(vector, dtype=np.float32)
    
    def calculate_similarity(self, 
                           emotion: str, 
                           video_features: Dict[str, Any]) -> float:
        """
        计算情绪与视频特征的相似度
        
        Args:
            emotion: 目标情绪
            video_features: 视频的音频特征
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if emotion not in self.emotion_database:
            logger.warning(f"未找到情绪 '{emotion}' 的特征数据")
            return 0.0
        
        try:
            # 检查是否有CLAMP3特征
            if 'clamp3_features' in video_features or 'feature_vector' in video_features:
                return self._calculate_clamp3_similarity(emotion, video_features)
            else:
                return self._calculate_traditional_similarity(emotion, video_features)
                
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def _calculate_clamp3_similarity(self, emotion: str, video_features: Dict[str, Any]) -> float:
        """
        使用CLAMP3特征计算相似度
        
        Args:
            emotion: 目标情绪
            video_features: 包含CLAMP3特征的视频特征
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 获取CLAMP3特征向量
        clamp3_vector = video_features.get('clamp3_features') or video_features.get('feature_vector')
        
        if clamp3_vector is None:
            logger.warning("CLAMP3特征向量为空")
            return 0.0
        
        # 确保是numpy数组
        if not isinstance(clamp3_vector, np.ndarray):
            clamp3_vector = np.array(clamp3_vector)
        
        # 如果是2D数组，取第一个维度
        if clamp3_vector.ndim > 1:
            clamp3_vector = clamp3_vector.flatten()
        
        # 获取情绪特征向量
        emotion_vector = self.emotion_database[emotion]['normalized_vector']
        
        # 对于CLAMP3特征，我们需要先将其映射到情绪空间
        # 这里使用简化的方法：基于CLAMP3特征的统计特性
        
        # 计算CLAMP3特征的统计特性
        clamp3_stats = self._extract_clamp3_statistics(clamp3_vector)
        
        # 将统计特性映射到情绪特征空间
        mapped_audio_vector = self._map_clamp3_to_emotion_space(clamp3_stats)
        mapped_audio_vector_norm = self._normalize_vector(mapped_audio_vector)
        
        # 确保向量长度一致
        min_len = min(len(emotion_vector), len(mapped_audio_vector_norm))
        emotion_vec = emotion_vector[:min_len]
        audio_vec = mapped_audio_vector_norm[:min_len]
        
        # 计算余弦相似度
        dot_product = np.dot(emotion_vec, audio_vec)
        cosine_similarity = max(0.0, dot_product)  # 确保非负
        
        # 计算欧几里得距离相似度
        euclidean_distance = np.linalg.norm(emotion_vec - audio_vec)
        euclidean_similarity = 1.0 / (1.0 + euclidean_distance)
        
        # 加权组合（CLAMP3特征更依赖余弦相似度）
        final_similarity = 0.8 * cosine_similarity + 0.2 * euclidean_similarity
        
        return float(final_similarity)
    
    def _extract_clamp3_statistics(self, clamp3_vector: np.ndarray) -> Dict[str, float]:
        """
        从CLAMP3特征向量中提取统计特性
        
        Args:
            clamp3_vector: CLAMP3特征向量
            
        Returns:
            Dict: 统计特性
        """
        stats = {}
        
        # 基础统计量
        stats['mean'] = float(np.mean(clamp3_vector))
        stats['std'] = float(np.std(clamp3_vector))
        stats['max'] = float(np.max(clamp3_vector))
        stats['min'] = float(np.min(clamp3_vector))
        stats['median'] = float(np.median(clamp3_vector))
        
        # 分位数特征
        stats['q25'] = float(np.percentile(clamp3_vector, 25))
        stats['q75'] = float(np.percentile(clamp3_vector, 75))
        stats['iqr'] = stats['q75'] - stats['q25']
        
        # 形状特征
        stats['skewness'] = float(self._calculate_skewness(clamp3_vector))
        stats['kurtosis'] = float(self._calculate_kurtosis(clamp3_vector))
        
        # 能量特征
        stats['energy'] = float(np.sum(clamp3_vector ** 2))
        stats['rms'] = float(np.sqrt(np.mean(clamp3_vector ** 2)))
        
        # 频域特征（将CLAMP3特征视为时间序列）
        fft_features = np.fft.fft(clamp3_vector)
        magnitude = np.abs(fft_features)
        stats['spectral_centroid'] = float(np.sum(np.arange(len(magnitude)) * magnitude) / np.sum(magnitude))
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _map_clamp3_to_emotion_space(self, clamp3_stats: Dict[str, float]) -> np.ndarray:
        """
        将CLAMP3统计特性映射到情绪特征空间
        
        Args:
            clamp3_stats: CLAMP3统计特性
            
        Returns:
            np.ndarray: 映射后的情绪空间特征向量
        """
        # 创建映射向量（与情绪特征向量相同长度）
        mapped_vector = []
        
        # 1. 节拍特征（基于能量和变化）
        tempo_proxy = min(abs(clamp3_stats['energy']) / 10.0, 1.0)
        mapped_vector.append(tempo_proxy)
        
        # 2. 调性特征（基于频谱质心）
        key_proxy = min(abs(clamp3_stats['spectral_centroid']) / 100.0, 1.0)
        mapped_vector.append(key_proxy)
        
        # 3. 动态特征（基于标准差）
        dynamics_proxy = min(clamp3_stats['std'] * 2.0, 1.0)
        mapped_vector.append(dynamics_proxy)
        
        # 4. 情绪强度（基于RMS）
        intensity_proxy = min(clamp3_stats['rms'] * 5.0, 1.0)
        mapped_vector.append(intensity_proxy)
        
        # 5. 复杂度（基于偏度）
        complexity_proxy = min(abs(clamp3_stats['skewness']) / 2.0, 1.0)
        mapped_vector.append(complexity_proxy)
        
        # 6. 织体特征（基于峰度）
        texture_proxy = min(abs(clamp3_stats['kurtosis']) / 5.0, 1.0)
        mapped_vector.append(texture_proxy)
        
        return np.array(mapped_vector, dtype=np.float32)
    
    def _calculate_traditional_similarity(self, emotion: str, video_features: Dict[str, Any]) -> float:
        """
        使用传统特征计算相似度
        
        Args:
            emotion: 目标情绪
            video_features: 传统音频特征
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 获取情绪特征向量
        emotion_vector = self.emotion_database[emotion]['normalized_vector']
        
        # 转换音频特征为向量
        audio_vector = self._audio_features_to_vector(video_features)
        audio_vector_norm = self._normalize_vector(audio_vector)
        
        # 确保向量长度一致
        min_len = min(len(emotion_vector), len(audio_vector_norm))
        emotion_vec = emotion_vector[:min_len]
        audio_vec = audio_vector_norm[:min_len]
        
        # 计算余弦相似度
        dot_product = np.dot(emotion_vec, audio_vec)
        cosine_similarity = max(0.0, dot_product)  # 确保非负
        
        # 计算欧几里得距离相似度
        euclidean_distance = np.linalg.norm(emotion_vec - audio_vec)
        euclidean_similarity = 1.0 / (1.0 + euclidean_distance)
        
        # 加权组合
        final_similarity = 0.7 * cosine_similarity + 0.3 * euclidean_similarity
        
        return float(final_similarity)
    
    def retrieve_videos(self, 
                       emotion: str, 
                       top_k: int = 5,
                       min_similarity: float = 0.1) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        检索最匹配的视频
        
        Args:
            emotion: 目标情绪
            top_k: 返回top-k结果
            min_similarity: 最小相似度阈值
            
        Returns:
            List[Tuple]: (视频路径, 相似度分数, 视频信息) 的列表
        """
        if not self.features_database:
            logger.error("特征数据库为空，请先提取视频特征")
            return []
        
        logger.info(f"检索情绪 '{emotion}' 的匹配视频 (top-{top_k})")
        
        similarities = []
        
        for video_path, video_features in self.features_database.items():
            # 计算相似度
            similarity = self.calculate_similarity(emotion, video_features)
            
            if similarity >= min_similarity:
                similarities.append((video_path, similarity, video_features))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top-k结果
        top_results = similarities[:top_k]
        
        logger.info(f"找到 {len(similarities)} 个匹配视频，返回top-{len(top_results)}")
        
        for i, (path, score, _) in enumerate(top_results, 1):
            logger.info(f"  {i}. {Path(path).name}: {score:.3f}")
        
        return top_results
    
    def get_random_from_top_k(self, 
                             emotion: str, 
                             top_k: int = 5) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        """
        从top-k结果中随机选择一个
        
        Args:
            emotion: 目标情绪
            top_k: top-k范围
            
        Returns:
            Tuple: (视频路径, 相似度分数, 视频信息) 或 None
        """
        top_results = self.retrieve_videos(emotion, top_k)
        
        if not top_results:
            logger.warning(f"没有找到匹配情绪 '{emotion}' 的视频")
            return None
        
        # 随机选择一个
        selected = random.choice(top_results)
        
        logger.info(f"从top-{len(top_results)}中随机选择: {Path(selected[0]).name} (相似度: {selected[1]:.3f})")
        
        return selected
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            'total_videos': len(self.features_database),
            'supported_emotions': len(self.emotion_database),
            'emotion_list': list(self.emotion_database.keys()),
            'feature_weights': self.feature_weights,
            'emotion_weights': self.emotion_weights,
            'last_updated': datetime.now().isoformat()
        }
    
    def update_feature_weights(self, new_weights: Dict[str, float]):
        """更新特征权重"""
        self.feature_weights.update(new_weights)
        logger.info(f"更新特征权重: {new_weights}")
    
    def rebuild_emotion_database(self):
        """重建情绪数据库"""
        logger.info("重建情绪数据库...")
        self._build_emotion_database()

class TherapyVideoSelector:
    """
    疗愈视频选择器
    基于用户输入和情绪状态选择最适合的疗愈视频
    """
    
    def __init__(self, retrieval_engine: VideoRetrievalEngine):
        """
        初始化视频选择器
        
        Args:
            retrieval_engine: 检索引擎实例
        """
        self.retrieval_engine = retrieval_engine
        self.selection_history = []
    
    def select_therapy_video(self, 
                           user_input: str,
                           duration_preference: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        选择疗愈视频
        
        Args:
            user_input: 用户情绪输入
            duration_preference: 时长偏好（分钟）
            
        Returns:
            Dict: 选择的视频信息
        """
        # 1. 情绪识别
        emotion, confidence = emotion_recognizer.detect_emotion(user_input)
        
        logger.info(f"情绪识别: {emotion} (置信度: {confidence:.2%})")
        
        # 2. 检索匹配视频
        selected_result = self.retrieval_engine.get_random_from_top_k(emotion, top_k=5)
        
        if not selected_result:
            return None
        
        video_path, similarity, video_features = selected_result
        
        # 3. 构建返回信息
        video_info = {
            'video_path': video_path,
            'video_name': Path(video_path).name,
            'detected_emotion': emotion,
            'emotion_confidence': confidence,
            'similarity_score': similarity,
            'selected_at': datetime.now().isoformat(),
            'user_input': user_input,
            'video_features': video_features,
            'therapy_stage': 'ISO匹配阶段',
            'therapy_duration': video_features.get('duration', 0)
        }
        
        # 4. 记录选择历史
        self.selection_history.append(video_info)
        
        # 5. 获取ISO三阶段特征
        iso_features = music_feature_mapper.get_music_features(emotion)
        video_info['iso_features'] = iso_features
        
        logger.info(f"✅ 选择疗愈视频: {video_info['video_name']}")
        logger.info(f"   情绪: {emotion} | 相似度: {similarity:.3f} | 时长: {video_info['therapy_duration']:.1f}秒")
        
        return video_info
    
    def get_selection_history(self) -> List[Dict[str, Any]]:
        """获取选择历史"""
        return self.selection_history.copy()
    
    def clear_history(self):
        """清空选择历史"""
        self.selection_history = []
        logger.info("选择历史已清空")

if __name__ == "__main__":
    # 测试检索引擎
    engine = VideoRetrievalEngine()
    
    print("🔍 测试检索引擎:")
    print(f"特征数据库: {len(engine.features_database)} 个视频")
    print(f"情绪数据库: {len(engine.emotion_database)} 种情绪")
    
    # 测试检索
    test_emotion = "焦虑"
    results = engine.retrieve_videos(test_emotion, top_k=3)
    
    print(f"\n🎯 检索测试 - 情绪: {test_emotion}")
    for i, (path, score, _) in enumerate(results, 1):
        print(f"  {i}. {Path(path).name}: {score:.3f}")
    
    # 测试疗愈视频选择器
    print(f"\n🌙 测试疗愈视频选择器:")
    selector = TherapyVideoSelector(engine)
    
    test_inputs = [
        "我感到很焦虑，心跳加速",
        "太累了，身体和精神都很疲惫",
        "比较平静，但希望更深层的放松"
    ]
    
    for user_input in test_inputs:
        result = selector.select_therapy_video(user_input)
        if result:
            print(f"输入: {user_input}")
            print(f"选择: {result['video_name']} | {result['detected_emotion']} | {result['similarity_score']:.3f}")
            print("-" * 50)