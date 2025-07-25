#!/usr/bin/env python3
"""
测试训练完成的情感模型
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def test_trained_model():
    """测试训练好的模型"""
    print("🧪 测试训练完成的情感模型...")
    
    try:
        from emotion_classifier import EmotionClassifier
        from inference_api import EmotionInferenceAPI
        
        # 初始化推理API
        api = EmotionInferenceAPI()
        
        # 测试用例
        test_texts = [
            "我今天非常开心，天气很好！",
            "这首歌让我很感动，想起了童年时光。",
            "我对这件事感到很愤怒和失望。",
            "看到这个消息我很震惊，简直不敢相信。",
            "他的表现让我感到非常钦佩。"
        ]
        
        print("\n📊 情感分析测试结果:")
        print("-" * 80)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n🔍 测试 {i}: {text}")
            
            # 获取情绪向量
            emotion_vector = api.get_emotion_for_kg_module(text)
            
            # 分析结果
            max_emotion_idx = np.argmax(emotion_vector)
            max_emotion_value = emotion_vector[max_emotion_idx]
            total_intensity = np.sum(emotion_vector)
            active_emotions = np.sum(emotion_vector > 0.1)
            
            print(f"   📈 27维向量: [{emotion_vector[0]:.3f}, {emotion_vector[1]:.3f}, {emotion_vector[2]:.3f}, ...]")
            print(f"   🎯 最强情绪强度: {max_emotion_value:.3f}")
            print(f"   📊 总体情绪强度: {total_intensity:.3f}")
            print(f"   🔢 活跃情绪数量: {active_emotions}")
        
        print("\n✅ 模型测试完成!")
        print("🎉 你的AC模块现在可以将文本转换为27维情绪向量了!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_trained_model()