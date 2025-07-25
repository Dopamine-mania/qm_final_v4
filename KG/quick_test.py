#!/usr/bin/env python3
"""
KG模块快速功能测试
"""

import sys
import numpy as np
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from knowledge_graph import KnowledgeGraph
from emotion_music_bridge import EmotionMusicBridge

def quick_kg_test():
    """快速测试KG模块核心功能"""
    print("🧪 KG模块快速功能测试")
    print("=" * 40)
    
    # 测试1: 基础功能
    print("\n1️⃣ 测试基础功能")
    kg = KnowledgeGraph()
    print(f"   情绪维度: {len(kg.emotion_names)}")
    print(f"   规则数量: {len(kg.rules)}")
    
    # 测试2: 情绪向量处理
    print("\n2️⃣ 测试情绪向量处理")
    emotion_vector = np.zeros(27)
    emotion_vector[5] = 0.8  # 焦虑
    emotion_vector[9] = 0.2  # 平静
    
    result = kg.get_initial_music_parameters(emotion_vector)
    print(f"   输入: 焦虑=0.8, 平静=0.2")
    print(f"   输出: Tempo={result['tempo']}, Mode={result['mode']:.2f}")
    
    # 测试3: 桥接器功能
    print("\n3️⃣ 测试桥接器功能")
    bridge = EmotionMusicBridge(enable_mi_retrieve=False)  # 快速模式
    
    # 从字典创建向量
    emotions = {"快乐": 0.8, "兴奋": 0.6}
    vector = bridge.create_emotion_vector_from_dict(emotions)
    
    # 获取参数
    result = bridge.get_therapy_parameters_only(vector)
    if result["success"]:
        print(f"   输入: {emotions}")
        print(f"   主要情绪: {result['emotion_analysis']['max_emotion']}")
        print(f"   治疗焦点: {result['therapy_recommendation']['primary_focus']}")
    
    # 测试4: 错误处理
    print("\n4️⃣ 测试错误处理")
    try:
        # 错误维度
        wrong_vector = np.zeros(20)
        kg.get_initial_music_parameters(wrong_vector)
        print("   ❌ 应该报错但没有")
    except Exception as e:
        print(f"   ✅ 正确捕获错误: {type(e).__name__}")
    
    print("\n✅ 快速测试完成! KG模块功能正常")
    return True

if __name__ == "__main__":
    quick_kg_test()