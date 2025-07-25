#!/usr/bin/env python3
"""
简化版语义检索 - 暂时使用模拟的文本特征映射
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSemanticSearcher:
    """简化版语义检索器"""
    
    def __init__(self):
        """初始化简化版语义检索器"""
        # 创建文本特征映射表 (简化版，基于关键词)
        self.text_feature_mapping = {
            # 节拍类特征 (影响特征向量的某些维度)
            "tempo": 0.8, "bpm": 0.8, "节奏": 0.7, "节拍": 0.7,
            "90": 0.6, "100": 0.65, "110": 0.7, "120": 0.75,
            "缓慢": 0.3, "慢": 0.3, "快": 0.8, "明快": 0.85,
            
            # 调式类特征
            "大调": 0.7, "小调": 0.3, "调式": 0.5,
            
            # 情绪类特征
            "轻松": 0.8, "愉悦": 0.85, "快乐": 0.9, "开心": 0.9,
            "放松": 0.6, "平静": 0.5, "安静": 0.4, "宁静": 0.45,
            "冥想": 0.3, "深度": 0.2, "思考": 0.25,
            "专注": 0.7, "工作": 0.75, "学习": 0.7,
            "焦虑": 0.1, "紧张": 0.15, "压力": 0.1,
            
            # 音乐风格特征
            "活泼": 0.85, "明亮": 0.8, "温暖": 0.7, "柔和": 0.4,
            "治愈": 0.5, "疗愈": 0.45, "舒缓": 0.4,
            
            # 和声特征
            "和声": 0.6, "协和": 0.7, "不协和": 0.3,
            "简单": 0.6, "复杂": 0.4, "丰富": 0.8
        }
        
        logger.info("✅ 简化版语义检索器初始化完成")
    
    def text_to_feature_vector(self, text: str, dim: int = 768) -> np.ndarray:
        """
        将文本转换为特征向量 (简化版)
        
        Args:
            text: 输入文本
            dim: 特征向量维度
            
        Returns:
            特征向量
        """
        # 基础随机向量
        base_vector = np.random.normal(0, 0.1, dim)
        
        # 根据文本内容调整向量
        text_lower = text.lower()
        
        # 计算文本特征强度
        feature_strength = 0.0
        matched_features = []
        
        for keyword, strength in self.text_feature_mapping.items():
            if keyword in text_lower:
                feature_strength += strength
                matched_features.append(keyword)
        
        # 标准化特征强度
        if feature_strength > 0:
            feature_strength = min(feature_strength / len(matched_features), 1.0)
        
        # 调整向量的某些维度来反映文本特征
        # 这是一个简化的映射方法
        if matched_features:
            # 根据匹配的特征调整向量的前100个维度
            for i, keyword in enumerate(matched_features[:100]):
                strength = self.text_feature_mapping[keyword]
                base_vector[i] = strength * 2 - 1  # 映射到 [-1, 1]
        
        # 归一化向量
        norm = np.linalg.norm(base_vector)
        if norm > 0:
            base_vector = base_vector / norm
        
        logger.info(f"🔄 文本特征映射: '{text}' -> 匹配关键词: {matched_features}")
        logger.info(f"   特征强度: {feature_strength:.3f}")
        
        return base_vector
    
    def compute_text_audio_similarity(self, text: str, audio_features: np.ndarray) -> float:
        """
        计算文本和音频特征的相似度
        
        Args:
            text: 文本描述
            audio_features: 音频特征向量
            
        Returns:
            相似度分数 (0-1)
        """
        # 获取文本特征向量
        text_features = self.text_to_feature_vector(text, audio_features.shape[0])
        
        # 计算余弦相似度
        dot_product = np.dot(text_features, audio_features)
        norm_text = np.linalg.norm(text_features)
        norm_audio = np.linalg.norm(audio_features)
        
        if norm_text == 0 or norm_audio == 0:
            return 0.0
        
        similarity = dot_product / (norm_text * norm_audio)
        
        # 将相似度从 [-1, 1] 映射到 [0, 1]
        similarity = (similarity + 1) / 2
        
        # 添加基于关键词的额外加权
        keyword_bonus = self._get_keyword_bonus(text)
        similarity = min(1.0, similarity + keyword_bonus)
        
        return float(similarity)
    
    def _get_keyword_bonus(self, text: str) -> float:
        """
        基于关键词匹配获得额外的相似度加分
        
        Args:
            text: 文本描述
            
        Returns:
            额外加分 (0-0.2)
        """
        text_lower = text.lower()
        bonus = 0.0
        
        # 特别强的情感关键词给额外加分
        strong_keywords = {
            "疗愈": 0.15, "治愈": 0.15, "放松": 0.12, "冥想": 0.12,
            "轻松": 0.1, "愉悦": 0.1, "舒缓": 0.12, "专注": 0.1
        }
        
        for keyword, score in strong_keywords.items():
            if keyword in text_lower:
                bonus += score
        
        return min(bonus, 0.2)  # 最大加分0.2

def main():
    """测试简化版语义检索"""
    print("🧪 测试简化版语义检索...")
    
    # 初始化检索器
    searcher = SimpleSemanticSearcher()
    
    # 模拟音频特征
    audio_features = np.random.normal(0, 0.5, 768)
    audio_features = audio_features / np.linalg.norm(audio_features)
    
    # 测试文本
    test_texts = [
        "tempo 90 BPM, 大调, 轻松愉悦的音乐",
        "节奏缓慢, 小调, 适合放松冥想",
        "明快活泼的音乐, 适合专注工作",
        "深沉内敛, 60 BPM, 适合深度思考"
    ]
    
    print("\n🔍 相似度计算结果:")
    for text in test_texts:
        similarity = searcher.compute_text_audio_similarity(text, audio_features)
        print(f"   '{text}' -> 相似度: {similarity:.4f}")
    
    print("\n🎉 简化版语义检索测试完成!")

if __name__ == "__main__":
    main()