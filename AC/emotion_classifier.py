#!/usr/bin/env python3
"""
情感分类器核心模块

基于xlm-roberta的多语言情感分类器
支持文本输入到27维C&K情绪向量的端到端转换
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Union, Tuple, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from .config import MODEL_CONFIG, MODEL_PATHS, COWEN_KELTNER_EMOTIONS, INFERENCE_CONFIG
from .emotion_mapper import GoEmotionsMapper

logger = logging.getLogger(__name__)

class EmotionClassifier(nn.Module):
    """基于xlm-roberta的情感分类器"""
    
    def __init__(self, model_name: str = None, num_labels: int = 27, load_pretrained: bool = True):
        """
        初始化情感分类器
        
        Args:
            model_name: 预训练模型名称
            num_labels: 输出标签数量 (27维C&K情绪)
            load_pretrained: 是否加载预训练权重
        """
        super().__init__()
        
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.num_labels = num_labels
        self.emotion_names = COWEN_KELTNER_EMOTIONS
        
        # 设备检测
        self.device = self._detect_device()
        logger.info(f"🔧 使用设备: {self.device}")
        
        # 初始化分词器和模型
        if load_pretrained:
            self._load_pretrained_model()
        
        # 初始化映射器
        self.mapper = GoEmotionsMapper()
        
        logger.info("✅ 情感分类器初始化完成")
    
    def _detect_device(self) -> str:
        """检测可用设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_pretrained_model(self):
        """加载预训练模型"""
        try:
            logger.info(f"📥 加载预训练模型: {self.model_name}")
            
            # 配置
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="multi_label_classification",
                cache_dir=MODEL_PATHS["pretrained_cache"]
            )
            
            # 分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=MODEL_PATHS["pretrained_cache"]
            )
            
            # 模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
                cache_dir=MODEL_PATHS["pretrained_cache"]
            )
            
            # 移到设备
            self.model.to(self.device)
            
            logger.info(f"✅ 预训练模型加载成功")
            
        except Exception as e:
            logger.error(f"❌ 预训练模型加载失败: {e}")
            raise
    
    def load_finetuned_model(self, model_path: str = None):
        """
        加载微调后的模型
        
        Args:
            model_path: 微调模型路径
        """
        try:
            model_path = model_path or MODEL_PATHS["finetuned_model"]
            logger.info(f"📥 加载微调模型: {model_path}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ 微调模型加载成功")
            
        except Exception as e:
            logger.error(f"❌ 微调模型加载失败: {e}")
            # 回退到预训练模型
            logger.info("🔄 回退到预训练模型")
            self._load_pretrained_model()
    
    def predict_single(self, text: str, return_dict: bool = False) -> Union[np.ndarray, Dict[str, float]]:
        """
        单文本情感预测
        
        Args:
            text: 输入文本
            return_dict: 是否返回字典格式
            
        Returns:
            27维情绪向量或情绪字典
        """
        try:
            if not text or len(text.strip()) < 1:
                logger.warning("⚠️  输入文本为空，返回零向量")
                zero_vector = np.zeros(27, dtype=np.float32)
                return self.mapper.map_ck_vector_to_dict(zero_vector) if return_dict else zero_vector
            
            # 文本预处理和分词
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=MODEL_CONFIG["max_length"],
                return_tensors="pt"
            )
            
            # 移到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 应用sigmoid激活 (多标签分类)
                probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # 确保输出维度正确
            if len(probabilities) != 27:
                logger.error(f"❌ 模型输出维度错误: 期望27维，实际{len(probabilities)}维")
                probabilities = np.zeros(27, dtype=np.float32)
            
            # 应用置信度阈值
            threshold = INFERENCE_CONFIG["confidence_threshold"]
            probabilities = np.where(probabilities > threshold, probabilities, 0.0)
            
            # 归一化到[0, 1]
            probabilities = np.clip(probabilities, 0, 1)
            
            if return_dict:
                return self.mapper.map_ck_vector_to_dict(probabilities)
            else:
                return probabilities.astype(np.float32)
                
        except Exception as e:
            logger.error(f"❌ 单文本预测失败: {e}")
            zero_vector = np.zeros(27, dtype=np.float32)
            return self.mapper.map_ck_vector_to_dict(zero_vector) if return_dict else zero_vector
    
    def predict_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        批量文本情感预测
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            (N, 27) 情绪向量矩阵
        """
        try:
            if not texts:
                return np.zeros((0, 27), dtype=np.float32)
            
            batch_size = batch_size or INFERENCE_CONFIG["max_batch_size"]
            results = []
            
            logger.info(f"🔄 开始批量预测: {len(texts)} 条文本")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 批量分词
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MODEL_CONFIG["max_length"],
                    return_tensors="pt"
                )
                
                # 移到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 批量推理
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # 应用sigmoid激活
                    probabilities = torch.sigmoid(logits).cpu().numpy()
                
                results.append(probabilities)
                
                if (i + batch_size) % 100 == 0:
                    logger.info(f"   批量预测进度: {min(i + batch_size, len(texts))}/{len(texts)}")
            
            # 合并结果
            all_results = np.vstack(results)
            
            # 应用阈值和归一化
            threshold = INFERENCE_CONFIG["confidence_threshold"]
            all_results = np.where(all_results > threshold, all_results, 0.0)
            all_results = np.clip(all_results, 0, 1)
            
            logger.info(f"✅ 批量预测完成: {all_results.shape}")
            return all_results.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ 批量预测失败: {e}")
            return np.zeros((len(texts), 27), dtype=np.float32)
    
    def get_top_emotions(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        获取文本的top-k情绪
        
        Args:
            text: 输入文本
            top_k: 返回前k个情绪
            
        Returns:
            [(情绪名, 强度值), ...]
        """
        emotion_vector = self.predict_single(text)
        return self.mapper.get_top_emotions_from_vector(emotion_vector, top_k)
    
    def analyze_emotion_distribution(self, texts: List[str]) -> Dict[str, any]:
        """
        分析文本集合的情绪分布
        
        Args:
            texts: 文本列表
            
        Returns:
            情绪分布统计
        """
        try:
            # 批量预测
            emotion_matrix = self.predict_batch(texts)
            
            # 统计分析
            stats = {}
            
            # 每个情绪的平均强度
            mean_intensities = np.mean(emotion_matrix, axis=0)
            stats["mean_emotions"] = {
                emotion: float(mean_intensities[i]) 
                for i, emotion in enumerate(self.emotion_names)
            }
            
            # 最常见的主导情绪
            dominant_emotions = [self.emotion_names[np.argmax(row)] for row in emotion_matrix]
            from collections import Counter
            stats["dominant_distribution"] = dict(Counter(dominant_emotions))
            
            # 情绪活跃度 (非零情绪的平均数量)
            active_emotions = np.sum(emotion_matrix > 0, axis=1)
            stats["avg_active_emotions"] = float(np.mean(active_emotions))
            
            # 总体情绪强度
            total_intensities = np.sum(emotion_matrix, axis=1)
            stats["avg_total_intensity"] = float(np.mean(total_intensities))
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ 情绪分布分析失败: {e}")
            return {}
    
    def save_model(self, output_path: str = None):
        """
        保存微调后的模型
        
        Args:
            output_path: 输出路径
        """
        try:
            output_path = output_path or MODEL_PATHS["finetuned_model"]
            logger.info(f"💾 保存模型到: {output_path}")
            
            # 保存模型和分词器
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            logger.info("✅ 模型保存成功")
            
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {e}")
            raise

def main():
    """测试情感分类器"""
    print("🧠 情感分类器测试")
    print("=" * 50)
    
    # 初始化分类器
    classifier = EmotionClassifier(load_pretrained=True)
    
    # 测试文本
    test_texts = [
        "我今天感到非常开心和兴奋！",
        "I am feeling very anxious about the exam tomorrow.",
        "这首音乐让我感到平静和放松。",
        "Je suis très en colère contre cette situation!",
        "The sunset is absolutely beautiful and awe-inspiring."
    ]
    
    print(f"\n🧪 测试单文本预测:")
    for text in test_texts:
        # 预测情绪向量
        emotion_vector = classifier.predict_single(text)
        
        # 获取top-3情绪
        top_emotions = classifier.get_top_emotions(text, 3)
        
        print(f"\n文本: {text}")
        print(f"主要情绪: {top_emotions}")
        print(f"向量强度: {np.sum(emotion_vector):.3f}")
    
    # 测试批量预测
    print(f"\n🔄 测试批量预测:")
    batch_results = classifier.predict_batch(test_texts)
    print(f"批量结果形状: {batch_results.shape}")
    
    # 分析情绪分布
    print(f"\n📊 情绪分布分析:")
    distribution = classifier.analyze_emotion_distribution(test_texts)
    print(f"平均活跃情绪数: {distribution.get('avg_active_emotions', 0):.2f}")
    print(f"主导情绪分布: {distribution.get('dominant_distribution', {})}")
    
    print(f"\n✅ 情感分类器测试完成!")

if __name__ == "__main__":
    main()