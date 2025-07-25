#!/usr/bin/env python3
"""
AC模块与KG模块集成测试

验证情感计算模块与知识图谱模块的完整集成流程
测试从文本输入到音乐推荐的端到端功能
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "KG"))
sys.path.append(str(project_root / "AC"))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ac_module_basic():
    """测试AC模块基础功能"""
    print("\n🧪 测试1: AC模块基础功能")
    print("-" * 40)
    
    try:
        from AC.inference_api import EmotionInferenceAPI
        from AC.emotion_mapper import GoEmotionsMapper
        
        # 初始化
        api = EmotionInferenceAPI(load_finetuned=False)  # 使用预训练模型
        mapper = GoEmotionsMapper()
        
        # 测试文本
        test_texts = [
            "我今天感到非常焦虑，心跳加速，难以入睡",
            "这首音乐让我感到平静和放松",
            "I am feeling very happy and excited today!",
            "Cette musique me rend nostalgique et émue",
            "我对这次考试感到很有压力"
        ]
        
        print("📋 测试文本情感分析:")
        results = []
        
        for i, text in enumerate(test_texts, 1):
            # 分析情感
            emotion_vector = api.get_emotion_for_kg_module(text)
            top_emotions = mapper.get_top_emotions_from_vector(emotion_vector, 3)
            
            result = {
                "text": text,
                "vector_shape": emotion_vector.shape,
                "vector_sum": float(np.sum(emotion_vector)),
                "top_emotions": top_emotions,
                "is_valid": mapper.validate_vector(emotion_vector)
            }
            
            results.append(result)
            
            print(f"   {i}. {text[:30]}...")
            print(f"      向量形状: {result['vector_shape']}")
            print(f"      总强度: {result['vector_sum']:.3f}")
            print(f"      主要情绪: {result['top_emotions']}")
            print(f"      向量有效: {result['is_valid']}")
            print()
        
        # 批量测试
        print("🔄 批量处理测试:")
        batch_results = api.analyze_batch_texts(test_texts)
        print(f"   批量结果形状: {batch_results.shape}")
        print(f"   平均强度: {np.mean(np.sum(batch_results, axis=1)):.3f}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ AC模块测试失败: {e}")
        return False, []

def test_kg_module_compatibility():
    """测试KG模块兼容性"""
    print("\n🧪 测试2: KG模块兼容性")
    print("-" * 40)
    
    try:
        from KG.knowledge_graph import KnowledgeGraph
        
        # 初始化KG
        kg = KnowledgeGraph()
        
        # 创建测试情绪向量
        test_vectors = [
            # 高焦虑
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 焦虑=0.8
            # 高快乐
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),  # 快乐=0.9, 兴奋=0.6
            # 平静状态
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # 平静=0.8
        ]
        
        scenarios = ["高焦虑状态", "快乐兴奋状态", "平静状态"]
        
        print("📊 KG模块处理27维向量:")
        kg_results = []
        
        for i, (vector, scenario) in enumerate(zip(test_vectors, scenarios)):
            try:
                # 测试情绪分析
                emotion_analysis = kg.analyze_emotion_vector(vector)
                
                # 测试音乐参数生成
                music_params = kg.get_music_search_parameters(vector)
                
                result = {
                    "scenario": scenario,
                    "input_vector_sum": float(np.sum(vector)),
                    "max_emotion": emotion_analysis["max_emotion"],
                    "emotion_balance": emotion_analysis["emotion_balance"],
                    "music_description": music_params["text_description"][:100] + "...",
                    "structured_params": music_params["structured_params"]
                }
                
                kg_results.append(result)
                
                print(f"   {i+1}. {scenario}:")
                print(f"      最强情绪: {result['max_emotion']}")
                print(f"      情绪平衡: {result['emotion_balance']}")
                print(f"      音乐描述: {result['music_description']}")
                print(f"      结构参数: {result['structured_params']}")
                print()
                
            except Exception as e:
                print(f"❌ KG处理失败 ({scenario}): {e}")
                return False, []
        
        return True, kg_results
        
    except Exception as e:
        print(f"❌ KG模块测试失败: {e}")
        return False, []

def test_full_integration():
    """测试完整集成流程"""
    print("\n🧪 测试3: AC-KG完整集成")
    print("-" * 40)
    
    try:
        from AC.inference_api import EmotionInferenceAPI
        from KG.emotion_music_bridge import EmotionMusicBridge
        
        # 初始化组件
        api = EmotionInferenceAPI(load_finetuned=False)
        bridge = EmotionMusicBridge(enable_mi_retrieve=False)  # 不启用音乐检索，专注测试AC-KG
        
        # 测试场景
        test_scenarios = [
            {
                "text": "我今天工作压力很大，感到焦虑不安，需要一些音乐来放松",
                "expected_primary": ["焦虑", "压力", "愤怒"]
            },
            {
                "text": "I'm feeling so happy and excited about my vacation tomorrow!",
                "expected_primary": ["快乐", "兴奋", "娱乐"]
            },
            {
                "text": "这首古典音乐让我感到内心平静，仿佛回到了童年时光",
                "expected_primary": ["平静", "怀旧", "审美欣赏"]
            },
            {
                "text": "Je me sens triste et nostalgique en écoutant cette chanson",
                "expected_primary": ["悲伤", "怀旧", "失望"]
            }
        ]
        
        print("🌉 端到端集成测试:")
        integration_results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            try:
                text = scenario["text"]
                
                # Step 1: AC模块情感分析
                emotion_vector = api.get_emotion_for_kg_module(text)
                
                # Step 2: KG模块处理
                result = bridge.get_therapy_parameters_only(emotion_vector)
                
                if result["success"]:
                    analysis = result["emotion_analysis"]
                    music_params = result["music_parameters"]
                    therapy_rec = result["therapy_recommendation"]
                    
                    integration_result = {
                        "scenario": i,
                        "input_text": text,
                        "ac_vector_sum": float(np.sum(emotion_vector)),
                        "kg_max_emotion": analysis["max_emotion"],
                        "kg_emotion_balance": analysis["emotion_balance"],
                        "music_tempo": music_params["tempo"],
                        "music_mode": music_params["mode"],
                        "therapy_focus": therapy_rec["primary_focus"],
                        "integration_success": True
                    }
                    
                    print(f"   {i}. 文本: {text[:50]}...")
                    print(f"      AC输出强度: {integration_result['ac_vector_sum']:.3f}")
                    print(f"      KG识别情绪: {integration_result['kg_max_emotion']}")
                    print(f"      音乐节拍: {integration_result['music_tempo']:.1f} BPM")
                    print(f"      治疗焦点: {integration_result['therapy_focus']}")
                    print(f"      ✅ 集成成功")
                    
                else:
                    integration_result = {
                        "scenario": i,
                        "input_text": text,
                        "error": result.get("error", "未知错误"),
                        "integration_success": False
                    }
                    print(f"   {i}. ❌ 集成失败: {integration_result['error']}")
                
                integration_results.append(integration_result)
                print()
                
            except Exception as e:
                print(f"   {i}. ❌ 场景测试失败: {e}")
                integration_results.append({
                    "scenario": i,
                    "error": str(e),
                    "integration_success": False
                })
        
        # 统计成功率
        successful_integrations = len([r for r in integration_results if r.get("integration_success", False)])
        success_rate = successful_integrations / len(test_scenarios)
        
        print(f"📊 集成测试统计:")
        print(f"   总测试场景: {len(test_scenarios)}")
        print(f"   成功集成: {successful_integrations}")
        print(f"   成功率: {success_rate:.2%}")
        
        return success_rate >= 0.75, integration_results  # 75%以上算成功
        
    except Exception as e:
        print(f"❌ 完整集成测试失败: {e}")
        return False, []

def test_performance_benchmark():
    """性能基准测试"""
    print("\n🧪 测试4: 性能基准测试")
    print("-" * 40)
    
    try:
        import time
        from AC.inference_api import EmotionInferenceAPI
        
        api = EmotionInferenceAPI(load_finetuned=False)
        
        # 准备测试数据
        test_texts = [
            "我今天心情不错",
            "I feel anxious about the meeting",
            "Cette musique est magnifique",
            "今天的天气让我感到很开心",
            "The movie made me feel sad"
        ] * 20  # 100条文本
        
        print("⏱️  单文本处理性能:")
        
        # 单文本处理时间
        start_time = time.time()
        for text in test_texts[:10]:  # 测试前10条
            emotion_vector = api.get_emotion_for_kg_module(text)
        single_processing_time = time.time() - start_time
        
        avg_single_time = single_processing_time / 10
        print(f"   10条文本处理时间: {single_processing_time:.3f}秒")
        print(f"   平均单文本时间: {avg_single_time:.3f}秒")
        
        # 批量处理时间
        print("\n⚡ 批量处理性能:")
        start_time = time.time()
        batch_results = api.analyze_batch_texts(test_texts[:50])  # 测试50条
        batch_processing_time = time.time() - start_time
        
        avg_batch_time = batch_processing_time / 50
        print(f"   50条文本批量处理: {batch_processing_time:.3f}秒")
        print(f"   平均单文本时间: {avg_batch_time:.3f}秒")
        print(f"   批量处理加速比: {avg_single_time / avg_batch_time:.2f}x")
        
        # 内存使用估算
        vector_memory = batch_results.nbytes / (1024 * 1024)  # MB
        print(f"   结果内存占用: {vector_memory:.2f} MB")
        
        performance_metrics = {
            "avg_single_time": avg_single_time,
            "avg_batch_time": avg_batch_time,
            "speedup_ratio": avg_single_time / avg_batch_time,
            "memory_usage_mb": vector_memory
        }
        
        return True, performance_metrics
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False, {}

def main():
    """主测试函数"""
    print("🚀 AC模块与KG模块集成测试")
    print("=" * 60)
    
    test_results = {
        "ac_basic": {"passed": False, "data": None},
        "kg_compatibility": {"passed": False, "data": None},
        "full_integration": {"passed": False, "data": None},
        "performance": {"passed": False, "data": None}
    }
    
    # 测试1: AC模块基础功能
    try:
        passed, data = test_ac_module_basic()
        test_results["ac_basic"] = {"passed": passed, "data": data}
    except Exception as e:
        print(f"❌ AC基础测试异常: {e}")
    
    # 测试2: KG模块兼容性
    try:
        passed, data = test_kg_module_compatibility()
        test_results["kg_compatibility"] = {"passed": passed, "data": data}
    except Exception as e:
        print(f"❌ KG兼容性测试异常: {e}")
    
    # 测试3: 完整集成
    try:
        passed, data = test_full_integration()
        test_results["full_integration"] = {"passed": passed, "data": data}
    except Exception as e:
        print(f"❌ 完整集成测试异常: {e}")
    
    # 测试4: 性能基准
    try:
        passed, data = test_performance_benchmark()
        test_results["performance"] = {"passed": passed, "data": data}
    except Exception as e:
        print(f"❌ 性能测试异常: {e}")
    
    # 总结报告
    print("\n📋 集成测试总结报告")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = len([r for r in test_results.values() if r["passed"]])
    
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"📊 总体成功率: {passed_tests/total_tests:.2%}")
    print()
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result["passed"] else "❌ 失败"
        print(f"{test_name}: {status}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有集成测试通过！AC模块已准备好与KG模块协作。")
        return True
    else:
        print(f"\n⚠️  部分测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)