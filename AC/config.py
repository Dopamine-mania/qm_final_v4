#!/usr/bin/env python3
"""
AC模块配置文件
"""

import os
from pathlib import Path

# 基础路径
AC_MODULE_ROOT = Path(__file__).parent
PROJECT_ROOT = AC_MODULE_ROOT.parent

# Cowen & Keltner (2017) 27维情绪标准
COWEN_KELTNER_EMOTIONS = [
    "钦佩", "崇拜", "审美欣赏", "娱乐", "愤怒", "焦虑", "敬畏", "尴尬",
    "无聊", "平静", "困惑", "蔑视", "渴望", "失望", "厌恶", "同情",
    "入迷", "嫉妒", "兴奋", "恐惧", "内疚", "恐怖", "兴趣", "快乐",
    "怀旧", "浪漫", "悲伤"
]

# GoEmotions原始标签 (27个类别)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise"
]

# GoEmotions到Cowen & Keltner映射表
GOEMOTIONS_TO_CK_MAPPING = {
    # 直接对应 (1:1映射)
    "admiration": "钦佩",
    "amusement": "娱乐", 
    "anger": "愤怒",
    "confusion": "困惑",
    "desire": "渴望",
    "disappointment": "失望",
    "disgust": "厌恶",
    "embarrassment": "尴尬",
    "excitement": "兴奋",
    "fear": "恐惧",
    "joy": "快乐",
    "sadness": "悲伤",
    
    # 语义映射 (强度加到对应C&K情绪)
    "annoyance": "愤怒",      # 烦恼→愤怒
    "approval": "钦佩",       # 赞同→钦佩  
    "caring": "同情",         # 关爱→同情
    "curiosity": "兴趣",      # 好奇→兴趣
    "disapproval": "蔑视",    # 不赞同→蔑视
    "gratitude": "钦佩",      # 感激→钦佩
    "grief": "悲伤",          # 悲痛→悲伤
    "love": "浪漫",           # 爱→浪漫
    "nervousness": "焦虑",    # 紧张→焦虑
    "optimism": "快乐",       # 乐观→快乐
    "pride": "钦佩",          # 自豪→钦佩
    "realization": "兴趣",    # 领悟→兴趣
    "relief": "平静",         # 解脱→平静
    "remorse": "内疚",        # 懊悔→内疚
    "surprise": "敬畏"        # 惊讶→敬畏
}

# 模型配置
MODEL_CONFIG = {
    "model_name": "xlm-roberta-base",
    "num_labels": 27,
    "max_length": 512,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1
}

# 数据路径
DATA_PATHS = {
    "goemotions_train": AC_MODULE_ROOT / "data" / "goemotions_train.csv",
    "goemotions_val": AC_MODULE_ROOT / "data" / "goemotions_val.csv", 
    "goemotions_test": AC_MODULE_ROOT / "data" / "goemotions_test.csv",
    "processed_train": AC_MODULE_ROOT / "data" / "processed_train.csv",
    "processed_val": AC_MODULE_ROOT / "data" / "processed_val.csv"
}

# 模型路径
MODEL_PATHS = {
    "pretrained_cache": AC_MODULE_ROOT / "models" / "pretrained",
    "finetuned_model": AC_MODULE_ROOT / "models" / "finetuned_xlm_roberta",
    "tokenizer": AC_MODULE_ROOT / "models" / "finetuned_xlm_roberta"
}

# 确保目录存在
for path in [AC_MODULE_ROOT / "data", AC_MODULE_ROOT / "models", 
             AC_MODULE_ROOT / "models" / "pretrained"]:
    path.mkdir(parents=True, exist_ok=True)

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": AC_MODULE_ROOT / "logs" / "ac_module.log"
}

# 确保日志目录存在
(AC_MODULE_ROOT / "logs").mkdir(exist_ok=True)

# 推理配置
INFERENCE_CONFIG = {
    "confidence_threshold": 0.1,    # 情绪强度阈值
    "max_batch_size": 32,           # 批处理大小
    "device": "auto",               # 设备选择 ("auto", "cpu", "cuda")
    "output_format": "vector"       # 输出格式 ("vector", "dict", "top_k")
}