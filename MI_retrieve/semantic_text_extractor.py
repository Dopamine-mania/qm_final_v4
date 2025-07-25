#!/usr/bin/env python3
"""
CLAMP3文本特征提取器 - 实现文本到768维特征向量的转换
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional
import logging

# 添加code目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from code.utils import CLaMP3Model
from code.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticTextExtractor:
    """CLAMP3语义文本特征提取器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化文本特征提取器
        
        Args:
            model_path: CLAMP3模型权重路径
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'code', 
                                    'weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth')
        
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化CLAMP3模型和tokenizer"""
        try:
            logger.info("🔄 初始化CLAMP3模型...")
            
            # 初始化tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
            logger.info(f"✅ 加载tokenizer: {TEXT_MODEL}")
            
            # 初始化CLAMP3模型
            self.model = CLaMP3Model()
            
            # 加载预训练权重
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info(f"✅ 加载模型权重: {self.model_path}")
            else:
                logger.warning(f"⚠️  模型权重文件不存在: {self.model_path}")
                logger.info("💡 使用随机初始化的权重")
            
            # 移动模型到设备
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ CLAMP3模型初始化完成 (设备: {self.device})")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    def extract_text_features(self, text: Union[str, List[str]], 
                            max_length: int = 128) -> np.ndarray:
        """
        提取文本特征向量
        
        Args:
            text: 文本字符串或文本列表
            max_length: 最大文本长度
            
        Returns:
            形状为 (batch_size, 768) 的特征向量
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未初始化")
        
        # 处理输入
        if isinstance(text, str):
            text = [text]
        
        try:
            logger.info(f"🔄 提取文本特征，输入数量: {len(text)}")
            
            # 文本tokenization
            encoded = self.tokenizer(
                text,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # 提取特征
            with torch.no_grad():
                text_features = self.model.get_text_features(
                    input_ids, 
                    attention_mask, 
                    get_global=True
                )
            
            # 转换为numpy数组
            features = text_features.cpu().numpy()
            
            logger.info(f"✅ 文本特征提取完成，特征形状: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"❌ 文本特征提取失败: {e}")
            raise
    
    def extract_single_text_feature(self, text: str) -> np.ndarray:
        """
        提取单个文本的特征向量
        
        Args:
            text: 文本字符串
            
        Returns:
            形状为 (768,) 的特征向量
        """
        features = self.extract_text_features(text)
        return features[0]  # 返回第一个(也是唯一一个)特征向量
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数 (0-1)
        """
        features1 = self.extract_single_text_feature(text1)
        features2 = self.extract_single_text_feature(text2)
        
        # 计算余弦相似度
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )
        
        return float(similarity)
    
    def batch_extract_text_features(self, texts: List[str], 
                                  batch_size: int = 32) -> np.ndarray:
        """
        批量提取文本特征
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            形状为 (len(texts), 768) 的特征矩阵
        """
        all_features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_features = self.extract_text_features(batch_texts)
            all_features.append(batch_features)
        
        return np.vstack(all_features)

def main():
    """测试文本特征提取功能"""
    print("🧪 测试CLAMP3文本特征提取...")
    
    # 初始化提取器
    extractor = SemanticTextExtractor()
    
    # 测试文本
    test_texts = [
        "tempo 90 BPM, 大调, 轻松愉悦的音乐",
        "节奏缓慢, 小调, 适合放松冥想",
        "明快活泼, 120 BPM, 适合专注工作",
        "深沉内敛, 60 BPM, 适合深度思考"
    ]
    
    # 提取特征
    for text in test_texts:
        try:
            features = extractor.extract_single_text_feature(text)
            print(f"✅ 文本: {text}")
            print(f"   特征形状: {features.shape}")
            print(f"   特征范围: [{features.min():.4f}, {features.max():.4f}]")
            print()
        except Exception as e:
            print(f"❌ 提取失败: {e}")
    
    # 测试相似度计算
    print("🔍 测试文本相似度计算...")
    similarity = extractor.compute_text_similarity(
        "tempo 90 BPM, 大调, 轻松愉悦", 
        "节奏90, 大调, 放松音乐"
    )
    print(f"✅ 相似度: {similarity:.4f}")

if __name__ == "__main__":
    main()