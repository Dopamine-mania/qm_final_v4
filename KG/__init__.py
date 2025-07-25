#!/usr/bin/env python3
"""
KG (Knowledge Graph) 模块 - 情绪驱动音乐治疗系统

基于GEMS模型的情绪到音乐参数映射知识图谱
支持27维情绪向量到音乐治疗参数的智能转换
"""

from .knowledge_graph import KnowledgeGraph
from .emotion_music_bridge import EmotionMusicBridge
from .parameter_mapping import ParameterMapper

__version__ = "1.0.0"
__author__ = "SuperClaude Music Therapy System"

# 导出主要类
__all__ = [
    "KnowledgeGraph",
    "EmotionMusicBridge", 
    "ParameterMapper"
]