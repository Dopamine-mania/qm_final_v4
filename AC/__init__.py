#!/usr/bin/env python3
"""
AC (Affective Computing) Module - 情感计算模块

基于xlm-roberta和GoEmotions数据集的多语言情感识别系统
严格遵循Cowen & Keltner (2017) 27维情绪分类标准
与KG知识图谱模块完美集成

模块结构:
- emotion_mapper.py: GoEmotions到C&K 27维情绪映射
- model_trainer.py: xlm-roberta微调训练
- emotion_classifier.py: 核心情感分类器
- inference_api.py: 推理接口
- config.py: 配置文件
"""

from .emotion_classifier import EmotionClassifier
from .inference_api import EmotionInferenceAPI
from .emotion_mapper import GoEmotionsMapper

__version__ = "1.0.0"
__author__ = "SuperClaude qm_final4"

__all__ = [
    "EmotionClassifier",
    "EmotionInferenceAPI", 
    "GoEmotionsMapper"
]