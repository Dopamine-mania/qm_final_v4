#!/usr/bin/env python3
"""
情绪驱动音乐检索示例

展示如何使用KG模块进行完整的情绪驱动音乐治疗流程
包含多种使用场景和最佳实践
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加KG模块路径
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_graph import KnowledgeGraph
from emotion_music_bridge import EmotionMusicBridge
from parameter_mapping import ParameterMapper

def example_1_basic_usage():
    """示例1: 基础使用流程"""
    print("🌟 示例1: 基础情绪分析与音乐推荐")
    print("-" * 50)
    
    # 初始化桥接器
    bridge = EmotionMusicBridge(enable_mi_retrieve=True)
    
    # 定义用户情绪状态
    user_emotions = {
        "焦虑": 0.8,     # 高度焦虑
        "平静": 0.1,     # 低平静度  
        "恐惧": 0.3      # 轻微恐惧
    }
    
    print(f"🧠 用户情绪状态: {user_emotions}")
    
    # 创建情绪向量
    emotion_vector = bridge.create_emotion_vector_from_dict(user_emotions)
    
    # 获取音乐推荐
    result = bridge.analyze_emotion_and_recommend_music(emotion_vector, duration="3min", top_k=3)
    
    if result["success"]:
        print(f"✅ 分析成功")
        print(f"主要情绪: {result['emotion_analysis']['max_emotion']}")
        print(f"音乐参数: {result['music_parameters']}")
        print(f"治疗建议: {result['therapy_recommendation']['primary_focus']}")
        
        if result["music_search_results"] and result["music_search_results"]["success"]:
            print(f"推荐音乐: {len(result['music_search_results']['results'])} 首")
        else:
            print("音乐检索不可用")
    else:
        print(f"❌ 分析失败: {result['error']}")
    
    return result

def example_2_detailed_analysis():
    """示例2: 详细情绪分析"""
    print("\n🌟 示例2: 详细情绪分析与参数映射")
    print("-" * 50)
    
    # 初始化组件
    kg = KnowledgeGraph()
    mapper = ParameterMapper()
    
    # 复杂情绪状态
    complex_emotions = {
        "悲伤": 0.7,
        "怀旧": 0.5,
        "平静": 0.3,
        "审美欣赏": 0.4
    }
    
    print(f"🎭 复杂情绪状态: {complex_emotions}")
    
    # 创建情绪向量
    emotion_vector = np.zeros(27)
    for emotion_name, value in complex_emotions.items():
        if emotion_name in kg.emotion_names:
            index = kg.emotion_names.index(emotion_name)
            emotion_vector[index] = value
    
    # 详细分析
    emotion_analysis = kg.analyze_emotion_vector(emotion_vector)
    print(f"\n📊 详细情绪分析:")
    print(f"   前3情绪: {emotion_analysis['top_emotions'][:3]}")
    print(f"   情绪平衡: {emotion_analysis['emotion_balance']}")
    print(f"   整体强度: {emotion_analysis['overall_intensity']:.3f}")
    print(f"   情绪多样性: {emotion_analysis['emotion_diversity']}")
    
    # 获取音乐参数
    music_params = kg.get_initial_music_parameters(emotion_vector)
    print(f"\n🎵 音乐参数:")
    for key, value in music_params.items():
        print(f"   {key}: {value}")
    
    # 参数映射
    text_desc = mapper.kg_to_text_description(music_params)
    structured_params = mapper.kg_to_structured_params(music_params)
    
    print(f"\n📝 自然语言描述:")
    print(f"   {text_desc}")
    
    print(f"\n🏗️  结构化参数:")
    for key, value in structured_params.items():
        print(f"   {key}: {value}")

def example_3_therapy_scenarios():
    """示例3: 不同治疗场景"""
    print("\n🌟 示例3: 不同音乐治疗场景")
    print("-" * 50)
    
    bridge = EmotionMusicBridge(enable_mi_retrieve=True)
    
    # 定义多种治疗场景
    therapy_scenarios = [
        {
            "name": "考试焦虑缓解",
            "emotions": {"焦虑": 0.8, "恐惧": 0.4, "平静": 0.1},
            "goal": "降低焦虑，提升专注力"
        },
        {
            "name": "失恋情感支持",
            "emotions": {"悲伤": 0.9, "失望": 0.7, "愤怒": 0.3, "怀旧": 0.5},
            "goal": "情感宣泄，逐步愈合"
        },
        {
            "name": "工作压力释放",
            "emotions": {"愤怒": 0.6, "焦虑": 0.5, "疲劳": 0.8},  # 注意: 疲劳不在27维中
            "goal": "压力释放，身心放松"
        },
        {
            "name": "庆祝成功喜悦",
            "emotions": {"快乐": 0.9, "兴奋": 0.8, "钦佩": 0.6},
            "goal": "维持积极情绪，分享喜悦"
        },
        {
            "name": "深度冥想",
            "emotions": {"平静": 0.9, "审美欣赏": 0.7, "敬畏": 0.5},
            "goal": "深层放松，内在探索"
        }
    ]
    
    for i, scenario in enumerate(therapy_scenarios, 1):
        print(f"\n🎬 场景{i}: {scenario['name']}")
        print(f"   治疗目标: {scenario['goal']}")
        print(f"   情绪状态: {scenario['emotions']}")
        
        # 分析情绪并获取治疗建议
        emotion_vector = bridge.create_emotion_vector_from_dict(scenario["emotions"])
        result = bridge.get_therapy_parameters_only(emotion_vector)
        
        if result["success"]:
            therapy = result["therapy_recommendation"]
            music = result["music_parameters"]
            
            print(f"   💊 治疗建议:")
            print(f"      焦点: {therapy['primary_focus']}")
            print(f"      方法: {therapy['therapy_approach']}")
            print(f"      时长: {therapy['session_duration']}")
            
            print(f"   🎵 音乐特征:")
            print(f"      节拍: {music['tempo']} BPM")
            print(f"      调式: {music['mode']}")
            print(f"      音色: {music['timbre']}")
        else:
            print(f"   ❌ 分析失败: {result['error']}")

def example_4_batch_processing():
    """示例4: 批量处理"""
    print("\n🌟 示例4: 批量情绪分析")
    print("-" * 50)
    
    bridge = EmotionMusicBridge(enable_mi_retrieve=False)  # 仅参数模式，避免过多检索
    
    # 创建多个情绪向量
    emotion_sets = [
        {"快乐": 0.8, "兴奋": 0.6},
        {"悲伤": 0.7, "失望": 0.5},
        {"焦虑": 0.9, "恐惧": 0.4},
        {"平静": 0.8, "审美欣赏": 0.6},
        {"愤怒": 0.7, "厌恶": 0.4}
    ]
    
    emotion_vectors = []
    for emotions in emotion_sets:
        vector = bridge.create_emotion_vector_from_dict(emotions)
        emotion_vectors.append(vector)
    
    print(f"📦 批量处理 {len(emotion_vectors)} 个情绪状态...")
    
    # 批量分析
    results = bridge.batch_emotion_analysis(emotion_vectors, duration="3min")
    
    print(f"\n📊 批量分析结果:")
    for i, result in enumerate(results, 1):
        if result["success"]:
            max_emotion = result["emotion_analysis"]["max_emotion"]
            tempo = result["music_parameters"]["tempo"]
            print(f"   {i}. 主要情绪: {max_emotion[0]} ({max_emotion[1]:.2f}) -> Tempo: {tempo} BPM")
        else:
            print(f"   {i}. ❌ 分析失败")

def example_5_advanced_usage():
    """示例5: 高级使用场景"""
    print("\n🌟 示例5: 高级使用场景与自定义")
    print("-" * 50)
    
    kg = KnowledgeGraph()
    
    # 情绪向量插值 (两种情绪状态之间的过渡)
    emotion_start = np.zeros(27)
    emotion_start[5] = 0.8  # 焦虑
    
    emotion_end = np.zeros(27) 
    emotion_end[9] = 0.8    # 平静
    
    print("🔄 情绪状态过渡分析:")
    print("   起始状态: 高焦虑")
    print("   目标状态: 高平静")
    
    # 生成过渡序列
    steps = 5
    for i in range(steps + 1):
        t = i / steps
        transition_vector = (1 - t) * emotion_start + t * emotion_end
        
        music_params = kg.get_initial_music_parameters(transition_vector)
        
        print(f"   步骤{i+1}: 焦虑{(1-t)*0.8:.1f}/平静{t*0.8:.1f} -> Tempo: {music_params['tempo']:.0f} BPM")
    
    # 情绪向量分析
    print(f"\n🔍 情绪向量统计分析:")
    
    # 创建随机情绪状态样本
    np.random.seed(42)  # 可重复性
    random_vectors = []
    
    for _ in range(10):
        # 随机选择2-4个情绪，给予随机强度
        vector = np.zeros(27)
        num_emotions = np.random.randint(2, 5)
        selected_indices = np.random.choice(27, num_emotions, replace=False)
        
        for idx in selected_indices:
            vector[idx] = np.random.uniform(0.2, 1.0)
        
        random_vectors.append(vector)
    
    # 分析随机向量的模式
    tempos = []
    modes = []
    
    for vector in random_vectors:
        params = kg.get_initial_music_parameters(vector)
        tempos.append(params['tempo'])
        modes.append(params['mode'])
    
    print(f"   随机样本数: {len(random_vectors)}")
    print(f"   Tempo范围: {min(tempos):.0f} - {max(tempos):.0f} BPM")
    print(f"   平均Tempo: {np.mean(tempos):.0f} BPM")
    print(f"   Mode范围: {min(modes):.2f} - {max(modes):.2f}")
    print(f"   平均Mode: {np.mean(modes):.2f}")

def example_6_error_handling():
    """示例6: 错误处理与异常情况"""
    print("\n🌟 示例6: 错误处理与边界情况")
    print("-" * 50)
    
    bridge = EmotionMusicBridge(enable_mi_retrieve=False)
    
    # 测试各种异常输入
    test_cases = [
        {
            "name": "空情绪字典",
            "emotions": {},
            "expected": "应返回默认参数"
        },
        {
            "name": "未知情绪名称",
            "emotions": {"开心": 0.8, "生气": 0.6},  # 不在27维中
            "expected": "应忽略未知情绪"
        },
        {
            "name": "超出范围的值",
            "emotions": {"快乐": 1.5, "悲伤": -0.3},
            "expected": "应裁剪到[0,1]范围"
        },
        {
            "name": "极小值情绪",
            "emotions": {"快乐": 0.001, "悲伤": 0.002},
            "expected": "应正常处理"
        }
    ]
    
    for case in test_cases:
        print(f"\n🧪 测试: {case['name']}")
        print(f"   输入: {case['emotions']}")
        print(f"   期望: {case['expected']}")
        
        try:
            emotion_vector = bridge.create_emotion_vector_from_dict(case["emotions"])
            result = bridge.get_therapy_parameters_only(emotion_vector)
            
            if result["success"]:
                tempo = result["music_parameters"]["tempo"]
                primary_focus = result["therapy_recommendation"]["primary_focus"]
                print(f"   ✅ 处理成功: Tempo={tempo} BPM, 焦点={primary_focus}")
            else:
                print(f"   ⚠️  处理失败: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ 异常: {e}")

def main():
    """运行所有示例"""
    print("🎵 情绪驱动音乐检索完整示例")
    print("=" * 80)
    
    examples = [
        example_1_basic_usage,
        example_2_detailed_analysis,
        example_3_therapy_scenarios,
        example_4_batch_processing,
        example_5_advanced_usage,
        example_6_error_handling
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ 示例 {example.__name__} 执行失败: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 所有示例演示完成!")
    print("\n💡 使用提示:")
    print("   1. 根据具体需求选择合适的情绪分析模式")
    print("   2. 启用MI_retrieve可获得真实音乐推荐") 
    print("   3. 治疗建议应结合专业心理评估")
    print("   4. 批量处理适合研究和统计分析")
    print("   5. 注意情绪向量的27维标准格式")

if __name__ == "__main__":
    main()