#!/usr/bin/env python3
"""
GoEmotions到Cowen & Keltner (2017) 27维情绪映射处理器

实现GoEmotions数据集标签到C&K情绪分类体系的精确映射
支持多标签情绪强度聚合和归一化
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Any
from collections import defaultdict

from config import (
    COWEN_KELTNER_EMOTIONS, 
    GOEMOTIONS_LABELS, 
    GOEMOTIONS_TO_CK_MAPPING
)

logger = logging.getLogger(__name__)

class GoEmotionsMapper:
    """GoEmotions到Cowen & Keltner映射器"""
    
    def __init__(self):
        """初始化映射器"""
        self.ck_emotions = COWEN_KELTNER_EMOTIONS
        self.goemotions_labels = GOEMOTIONS_LABELS
        self.mapping = GOEMOTIONS_TO_CK_MAPPING
        
        # 创建情绪索引映射
        self.ck_to_index = {emotion: i for i, emotion in enumerate(self.ck_emotions)}
        self.goemotions_to_index = {label: i for i, label in enumerate(self.goemotions_labels)}
        
        # 创建反向映射 (C&K -> GoEmotions列表)
        self.ck_to_goemotions = defaultdict(list)
        for ge_label, ck_emotion in self.mapping.items():
            self.ck_to_goemotions[ck_emotion].append(ge_label)
        
        logger.info("✅ GoEmotions映射器初始化完成")
        logger.info(f"   支持映射: {len(self.mapping)} GoEmotions标签 → {len(self.ck_emotions)} C&K情绪")
    
    def map_goemotions_to_ck_vector(self, goemotions_scores: Union[Dict[str, float], List[float], np.ndarray]) -> np.ndarray:
        """
        将GoEmotions多标签分数映射为27维C&K情绪向量
        
        Args:
            goemotions_scores: GoEmotions分数，支持多种格式:
                - Dict[str, float]: {"joy": 0.8, "anger": 0.3, ...}
                - List[float]: 按GOEMOTIONS_LABELS顺序的27维列表
                - np.ndarray: 27维numpy数组
                
        Returns:
            np.ndarray: 27维C&K情绪向量 [0, 1]
        """
        try:
            # 统一转换为字典格式
            if isinstance(goemotions_scores, dict):
                ge_dict = goemotions_scores
            elif isinstance(goemotions_scores, (list, np.ndarray)):
                if len(goemotions_scores) != len(self.goemotions_labels):
                    raise ValueError(f"GoEmotions向量长度错误: 期望{len(self.goemotions_labels)}，实际{len(goemotions_scores)}")
                ge_dict = {label: float(score) for label, score in zip(self.goemotions_labels, goemotions_scores)}
            else:
                raise ValueError(f"不支持的输入格式: {type(goemotions_scores)}")
            
            # 初始化27维C&K向量
            ck_vector = np.zeros(27, dtype=np.float32)
            
            # 映射GoEmotions到C&K情绪
            for ge_label, ge_score in ge_dict.items():
                if ge_label in self.mapping and ge_score > 0:
                    ck_emotion = self.mapping[ge_label]
                    ck_index = self.ck_to_index[ck_emotion]
                    
                    # 累加强度到对应的C&K情绪
                    ck_vector[ck_index] += ge_score
            
            # 归一化到[0, 1]范围
            ck_vector = np.clip(ck_vector, 0, 1)
            
            return ck_vector
            
        except Exception as e:
            logger.error(f"❌ GoEmotions映射失败: {e}")
            # 返回零向量作为fallback
            return np.zeros(27, dtype=np.float32)
    
    def map_ck_vector_to_dict(self, ck_vector: np.ndarray) -> Dict[str, float]:
        """
        将27维C&K向量转换为情绪字典
        
        Args:
            ck_vector: 27维C&K情绪向量
            
        Returns:
            Dict[str, float]: 情绪字典 {"情绪名": 强度值}
        """
        if len(ck_vector) != 27:
            raise ValueError(f"C&K向量维度错误: 期望27维，实际{len(ck_vector)}维")
        
        return {emotion: float(ck_vector[i]) for i, emotion in enumerate(self.ck_emotions)}
    
    def process_goemotions_dataset(self, data_path: str, output_path: str) -> None:
        """
        处理GoEmotions数据集，转换为C&K格式
        
        Args:
            data_path: GoEmotions数据集路径 (CSV格式)
            output_path: 处理后输出路径
        """
        try:
            logger.info(f"📂 开始处理GoEmotions数据集: {data_path}")
            
            # 读取原始数据
            df = pd.read_csv(data_path)
            logger.info(f"   原始数据: {len(df)} 条样本")
            
            # 检查必要列
            required_cols = ['text'] + self.goemotions_labels
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必要列: {missing_cols}")
            
            # 处理每一行
            processed_data = []
            
            for idx, row in df.iterrows():
                text = row['text']
                
                # 提取GoEmotions分数
                ge_scores = {label: row[label] for label in self.goemotions_labels if label in row}
                
                # 映射到C&K向量
                ck_vector = self.map_goemotions_to_ck_vector(ge_scores)
                
                # 构建输出行
                output_row = {'text': text}
                
                # 添加C&K情绪列
                for i, emotion in enumerate(self.ck_emotions):
                    output_row[emotion] = ck_vector[i]
                
                # 添加元数据
                output_row['original_labels'] = ','.join([label for label, score in ge_scores.items() if score > 0])
                output_row['max_ck_emotion'] = self.ck_emotions[np.argmax(ck_vector)]
                output_row['emotion_intensity'] = float(np.max(ck_vector))
                
                processed_data.append(output_row)
                
                if (idx + 1) % 1000 == 0:
                    logger.info(f"   处理进度: {idx + 1}/{len(df)}")
            
            # 保存处理结果
            processed_df = pd.DataFrame(processed_data)
            processed_df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"✅ 数据集处理完成: {output_path}")
            logger.info(f"   输出数据: {len(processed_df)} 条样本")
            logger.info(f"   情绪分布: {processed_df['max_ck_emotion'].value_counts().head()}")
            
        except Exception as e:
            logger.error(f"❌ 数据集处理失败: {e}")
            raise
    
    def analyze_mapping_coverage(self) -> Dict[str, Any]:
        """
        分析映射覆盖情况
        
        Returns:
            映射分析结果
        """
        # 统计每个C&K情绪对应的GoEmotions标签数量
        coverage_stats = {}
        
        for ck_emotion in self.ck_emotions:
            mapped_ge_labels = self.ck_to_goemotions[ck_emotion]
            coverage_stats[ck_emotion] = {
                'mapped_count': len(mapped_ge_labels),
                'mapped_labels': mapped_ge_labels
            }
        
        # 未映射的C&K情绪
        unmapped_ck = [emotion for emotion in self.ck_emotions if len(self.ck_to_goemotions[emotion]) == 0]
        
        # 未使用的GoEmotions标签
        unmapped_ge = [label for label in self.goemotions_labels if label not in self.mapping]
        
        return {
            'total_ck_emotions': len(self.ck_emotions),
            'total_ge_labels': len(self.goemotions_labels),
            'mapped_ge_labels': len(self.mapping),
            'coverage_by_ck': coverage_stats,
            'unmapped_ck_emotions': unmapped_ck,
            'unmapped_ge_labels': unmapped_ge,
            'mapping_rate': len(self.mapping) / len(self.goemotions_labels)
        }
    
    def get_top_emotions_from_vector(self, ck_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        从C&K向量中获取top-k情绪
        
        Args:
            ck_vector: 27维C&K情绪向量
            top_k: 返回前k个情绪
            
        Returns:
            List[Tuple[str, float]]: [(情绪名, 强度值), ...]
        """
        if len(ck_vector) != 27:
            raise ValueError(f"C&K向量维度错误: 期望27维，实际{len(ck_vector)}维")
        
        # 获取非零情绪
        emotion_scores = [(self.ck_emotions[i], float(ck_vector[i])) 
                         for i in range(27) if ck_vector[i] > 0]
        
        # 按强度排序
        emotion_scores.sort(key=lambda x: x[1], reverse=True)
        
        return emotion_scores[:top_k]
    
    def validate_vector(self, ck_vector: np.ndarray) -> bool:
        """
        验证C&K向量格式
        
        Args:
            ck_vector: C&K情绪向量
            
        Returns:
            bool: 是否有效
        """
        try:
            if not isinstance(ck_vector, np.ndarray):
                return False
            
            if ck_vector.shape[0] != 27:
                return False
            
            if np.any(ck_vector < 0) or np.any(ck_vector > 1):
                return False
            
            return True
            
        except Exception:
            return False

def main():
    """测试映射器功能"""
    print("🔄 GoEmotions映射器测试")
    print("=" * 50)
    
    # 初始化映射器
    mapper = GoEmotionsMapper()
    
    # 测试映射分析
    analysis = mapper.analyze_mapping_coverage()
    print(f"\n📊 映射覆盖分析:")
    print(f"   GoEmotions标签: {analysis['total_ge_labels']}")
    print(f"   已映射标签: {analysis['mapped_ge_labels']}")
    print(f"   映射覆盖率: {analysis['mapping_rate']:.2%}")
    print(f"   未映射的C&K情绪: {analysis['unmapped_ck_emotions']}")
    
    # 测试向量映射
    print(f"\n🧪 测试向量映射:")
    
    # 测试1: 字典输入
    ge_scores = {"joy": 0.8, "anger": 0.3, "fear": 0.1}
    ck_vector = mapper.map_goemotions_to_ck_vector(ge_scores)
    top_emotions = mapper.get_top_emotions_from_vector(ck_vector, 3)
    
    print(f"输入GoEmotions: {ge_scores}")
    print(f"输出C&K向量: {ck_vector[:5]}... (前5维)")
    print(f"主要情绪: {top_emotions}")
    
    # 测试2: 列表输入
    ge_list = [0.1] * 27  # 27维均匀分布
    ge_list[17] = 0.9     # joy高分
    ck_vector2 = mapper.map_goemotions_to_ck_vector(ge_list)
    top_emotions2 = mapper.get_top_emotions_from_vector(ck_vector2, 3)
    
    print(f"\n输入GoEmotions列表: joy=0.9, others=0.1")
    print(f"输出C&K向量: {ck_vector2[:5]}... (前5维)")
    print(f"主要情绪: {top_emotions2}")
    
    print(f"\n✅ 映射器测试完成!")

if __name__ == "__main__":
    main()