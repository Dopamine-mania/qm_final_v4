#!/usr/bin/env python3
"""
情感计算模块推理API

提供统一的情感分析接口，与KG模块完美集成
支持单文本、批量文本的情感识别和27维向量输出
"""

import sys
import os
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Any
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from .emotion_classifier import EmotionClassifier
from .emotion_mapper import GoEmotionsMapper
from .config import COWEN_KELTNER_EMOTIONS, INFERENCE_CONFIG, MODEL_PATHS

logger = logging.getLogger(__name__)

class EmotionInferenceAPI:
    """情感分析推理API"""
    
    def __init__(self, model_path: str = None, load_finetuned: bool = True):
        """
        初始化推理API
        
        Args:
            model_path: 自定义模型路径
            load_finetuned: 是否加载微调模型
        """
        self.emotion_names = COWEN_KELTNER_EMOTIONS
        self.model_path = model_path
        
        # 初始化分类器
        self.classifier = EmotionClassifier(load_pretrained=not load_finetuned)
        
        # 尝试加载微调模型
        if load_finetuned:
            try:
                finetuned_path = model_path or MODEL_PATHS["finetuned_model"]
                if Path(finetuned_path).exists():
                    self.classifier.load_finetuned_model(str(finetuned_path))
                    logger.info("✅ 使用微调模型")
                else:
                    logger.warning("⚠️  微调模型不存在，使用预训练模型")
            except Exception as e:
                logger.warning(f"⚠️  微调模型加载失败，使用预训练模型: {e}")
        
        # 初始化映射器
        self.mapper = GoEmotionsMapper()
        
        logger.info("✅ 情感推理API初始化完成")
    
    def analyze_single_text(self, text: str, output_format: str = "vector") -> Union[np.ndarray, Dict[str, float], List[tuple]]:
        """
        分析单个文本的情感
        
        Args:
            text: 输入文本
            output_format: 输出格式 ("vector", "dict", "top_k")
            
        Returns:
            根据format返回不同格式的结果
        """
        try:
            # 基础验证
            if not text or len(text.strip()) < 1:
                logger.warning("⚠️  输入文本为空")
                if output_format == "vector":
                    return np.zeros(27, dtype=np.float32)
                elif output_format == "dict":
                    return {emotion: 0.0 for emotion in self.emotion_names}
                else:  # top_k
                    return []
            
            # 获取情感向量
            emotion_vector = self.classifier.predict_single(text)
            
            # 根据输出格式返回结果
            if output_format == "vector":
                return emotion_vector
            elif output_format == "dict":
                return self.mapper.map_ck_vector_to_dict(emotion_vector)
            elif output_format == "top_k":
                return self.mapper.get_top_emotions_from_vector(emotion_vector, 5)
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")
                
        except Exception as e:
            logger.error(f"❌ 单文本情感分析失败: {e}")
            # 返回默认结果
            if output_format == "vector":
                return np.zeros(27, dtype=np.float32)
            elif output_format == "dict":
                return {emotion: 0.0 for emotion in self.emotion_names}
            else:
                return []
    
    def analyze_batch_texts(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        批量分析文本情感
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            (N, 27) 情感向量矩阵
        """
        try:
            if not texts:
                return np.zeros((0, 27), dtype=np.float32)
            
            batch_size = batch_size or INFERENCE_CONFIG["max_batch_size"]
            return self.classifier.predict_batch(texts, batch_size)
            
        except Exception as e:
            logger.error(f"❌ 批量情感分析失败: {e}")
            return np.zeros((len(texts), 27), dtype=np.float32)
    
    def get_emotion_for_kg_module(self, text: str) -> np.ndarray:
        """
        为KG模块提供标准化的27维情感向量
        
        这是与KG模块集成的核心接口
        
        Args:
            text: 用户输入文本
            
        Returns:
            np.ndarray: 标准化的27维C&K情感向量 [0, 1]
        """
        try:
            logger.info(f"🧠 为KG模块分析情感: {text[:50]}...")
            
            # 获取情感向量
            emotion_vector = self.analyze_single_text(text, output_format="vector")
            
            # 验证向量格式
            if not self.mapper.validate_vector(emotion_vector):
                logger.error("❌ 情感向量格式验证失败")
                return np.zeros(27, dtype=np.float32)
            
            # 记录主要情感
            top_emotions = self.mapper.get_top_emotions_from_vector(emotion_vector, 3)
            logger.info(f"   主要情感: {[(name, f'{score:.3f}') for name, score in top_emotions]}")
            
            return emotion_vector
            
        except Exception as e:
            logger.error(f"❌ KG模块情感分析失败: {e}")
            return np.zeros(27, dtype=np.float32)
    
    def analyze_emotion_with_context(self, text: str) -> Dict[str, Any]:
        """
        带上下文的情感分析
        
        Args:
            text: 输入文本
            
        Returns:
            包含详细分析信息的字典
        """
        try:
            # 基础情感分析
            emotion_vector = self.analyze_single_text(text, output_format="vector")
            emotion_dict = self.mapper.map_ck_vector_to_dict(emotion_vector)
            top_emotions = self.mapper.get_top_emotions_from_vector(emotion_vector, 5)
            
            # 计算统计信息
            total_intensity = float(np.sum(emotion_vector))
            max_intensity = float(np.max(emotion_vector))
            active_emotions = len([score for score in emotion_vector if score > 0.1])
            
            # 情感分类
            positive_emotions = ["快乐", "兴奋", "娱乐", "钦佩", "崇拜", "审美欣赏", "敬畏", "入迷", "兴趣", "浪漫"]
            negative_emotions = ["愤怒", "焦虑", "悲伤", "恐惧", "内疚", "恐怖", "失望", "厌恶", "嫉妒", "蔑视"]
            neutral_emotions = ["平静", "无聊", "困惑", "尴尬", "同情", "渴望", "怀旧"]
            
            positive_score = sum(emotion_dict[e] for e in positive_emotions if e in emotion_dict)
            negative_score = sum(emotion_dict[e] for e in negative_emotions if e in emotion_dict)
            neutral_score = sum(emotion_dict[e] for e in neutral_emotions if e in emotion_dict)
            
            return {
                "input_text": text,
                "emotion_vector": emotion_vector.tolist(),
                "emotion_dict": emotion_dict,
                "top_emotions": top_emotions,
                "statistics": {
                    "total_intensity": total_intensity,
                    "max_intensity": max_intensity,
                    "active_emotions_count": active_emotions,
                    "emotion_balance": {
                        "positive": positive_score,
                        "negative": negative_score,
                        "neutral": neutral_score
                    }
                },
                "primary_emotion": top_emotions[0] if top_emotions else ("平静", 0.0),
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ 上下文情感分析失败: {e}")
            return {
                "input_text": text,
                "emotion_vector": np.zeros(27).tolist(),
                "error": str(e)
            }
    
    def test_kg_integration(self, test_texts: List[str] = None) -> Dict[str, Any]:
        """
        测试与KG模块的集成
        
        Args:
            test_texts: 测试文本列表
            
        Returns:
            测试结果
        """
        default_texts = [
            "我今天感到非常焦虑，难以入睡",
            "这首音乐让我感到平静和放松",
            "我对这个结果感到非常开心和兴奋",
            "I feel sad and disappointed about the news",
            "Cette musique me rend nostalgique"
        ]
        
        test_texts = test_texts or default_texts
        results = []
        
        logger.info("🧪 开始KG集成测试")
        
        for text in test_texts:
            try:
                # 使用KG接口
                emotion_vector = self.get_emotion_for_kg_module(text)
                
                # 分析结果
                top_emotions = self.mapper.get_top_emotions_from_vector(emotion_vector, 3)
                
                result = {
                    "text": text,
                    "emotion_vector_shape": emotion_vector.shape,
                    "vector_sum": float(np.sum(emotion_vector)),
                    "top_emotions": top_emotions,
                    "vector_valid": self.mapper.validate_vector(emotion_vector)
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"❌ 测试失败: {text[:30]}... -> {e}")
                results.append({
                    "text": text,
                    "error": str(e)
                })
        
        # 统计
        successful_tests = len([r for r in results if "error" not in r])
        
        summary = {
            "total_tests": len(test_texts),
            "successful_tests": successful_tests,
            "success_rate": successful_tests / len(test_texts),
            "test_results": results
        }
        
        logger.info(f"✅ KG集成测试完成: {successful_tests}/{len(test_texts)} 成功")
        
        return summary
    
    def get_api_status(self) -> Dict[str, Any]:
        """获取API状态信息"""
        return {
            "api_version": "1.0.0",
            "model_loaded": self.classifier is not None,
            "model_path": str(self.model_path) if self.model_path else "default",
            "supported_emotions": len(self.emotion_names),
            "emotion_names": self.emotion_names,
            "inference_config": INFERENCE_CONFIG,
            "device": getattr(self.classifier, 'device', 'unknown') if self.classifier else 'unknown'
        }

# 全局API实例 (单例模式)
_global_api_instance = None

def get_emotion_api(model_path: str = None, load_finetuned: bool = True) -> EmotionInferenceAPI:
    """
    获取全局情感API实例
    
    Args:
        model_path: 模型路径
        load_finetuned: 是否加载微调模型
        
    Returns:
        EmotionInferenceAPI实例
    """
    global _global_api_instance
    
    if _global_api_instance is None:
        _global_api_instance = EmotionInferenceAPI(model_path, load_finetuned)
    
    return _global_api_instance

def analyze_text_emotion(text: str) -> np.ndarray:
    """
    快捷函数：分析文本情感并返回27维向量
    
    Args:
        text: 输入文本
        
    Returns:
        27维情感向量
    """
    api = get_emotion_api()
    return api.get_emotion_for_kg_module(text)

def main():
    """测试推理API"""
    print("🔮 情感推理API测试")
    print("=" * 50)
    
    # 初始化API
    api = EmotionInferenceAPI(load_finetuned=False)  # 使用预训练模型测试
    
    # 显示状态
    status = api.get_api_status()
    print(f"\n📊 API状态:")
    print(f"   模型已加载: {status['model_loaded']}")
    print(f"   支持情绪数: {status['supported_emotions']}")
    print(f"   使用设备: {status['device']}")
    
    # 测试单文本分析
    print(f"\n🧪 单文本情感分析测试:")
    test_text = "我今天感到非常开心和兴奋，这首音乐让我想起了美好的回忆"
    
    # 不同输出格式
    vector_result = api.analyze_single_text(test_text, "vector")
    dict_result = api.analyze_single_text(test_text, "dict")
    topk_result = api.analyze_single_text(test_text, "top_k")
    
    print(f"文本: {test_text}")
    print(f"向量形状: {vector_result.shape}")
    print(f"向量强度: {np.sum(vector_result):.3f}")
    print(f"主要情绪: {topk_result[:3]}")
    
    # 测试上下文分析
    print(f"\n📈 上下文情感分析:")
    context_result = api.analyze_emotion_with_context(test_text)
    print(f"总强度: {context_result['statistics']['total_intensity']:.3f}")
    print(f"活跃情绪数: {context_result['statistics']['active_emotions_count']}")
    print(f"情绪平衡: {context_result['statistics']['emotion_balance']}")
    
    # 测试KG集成
    print(f"\n🌉 KG模块集成测试:")
    kg_test = api.test_kg_integration()
    print(f"测试成功率: {kg_test['success_rate']:.2%}")
    
    print(f"\n✅ 推理API测试完成!")

if __name__ == "__main__":
    import pandas as pd  # 导入pandas用于时间戳
    main()