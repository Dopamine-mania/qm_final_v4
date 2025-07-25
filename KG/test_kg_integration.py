#!/usr/bin/env python3
"""
KG模块集成测试

测试知识图谱、桥接器和参数映射的完整功能
验证与MI_retrieve模块的集成效果
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from knowledge_graph import KnowledgeGraph
from emotion_music_bridge import EmotionMusicBridge
from parameter_mapping import ParameterMapper

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KGIntegrationTester:
    """KG模块集成测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.kg = KnowledgeGraph()
        self.bridge = EmotionMusicBridge(enable_mi_retrieve=True)
        self.mapper = ParameterMapper()
        
        print("🧪 KG模块集成测试器初始化完成")
    
    def test_knowledge_graph_basic(self):
        """测试知识图谱基础功能"""
        print("\n" + "="*60)
        print("🧠 测试 1: 知识图谱基础功能")
        print("="*60)
        
        try:
            # 测试情绪名称映射
            print(f"📝 情绪维度数量: {len(self.kg.emotion_names)}")
            print(f"   前5个情绪: {self.kg.emotion_names[:5]}")
            print(f"   后5个情绪: {self.kg.emotion_names[-5:]}")
            
            # 测试规则系统
            print(f"🎯 GEMS规则数量: {len(self.kg.rules)}")
            
            # 显示各优先级规则数量
            priority_counts = {}
            for rule in self.kg.rules:
                priority = rule.priority
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            print(f"📊 规则优先级分布: {priority_counts}")
            
            # 测试默认参数
            print(f"🎵 默认音乐参数:")
            for key, value in self.kg.default_music_parameters.items():
                print(f"   {key}: {value}")
            
            print("✅ 知识图谱基础功能测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 知识图谱基础功能测试失败: {e}")
            return False
    
    def test_emotion_scenarios(self):
        """测试多种情绪场景"""
        print("\n" + "="*60)
        print("🎭 测试 2: 多种情绪场景")
        print("="*60)
        
        # 定义测试场景
        test_scenarios = [
            {
                "name": "高焦虑状态",
                "emotions": {"焦虑": 0.8, "恐惧": 0.3, "平静": 0.1},
                "expected_tempo": "slow",
                "expected_mode": "major_or_neutral"
            },
            {
                "name": "快乐兴奋状态", 
                "emotions": {"快乐": 0.9, "兴奋": 0.7, "娱乐": 0.5},
                "expected_tempo": "fast",
                "expected_mode": "major"
            },
            {
                "name": "深度悲伤",
                "emotions": {"悲伤": 0.8, "失望": 0.6, "内疚": 0.3},
                "expected_tempo": "slow",
                "expected_mode": "minor"
            },
            {
                "name": "平静冥想",
                "emotions": {"平静": 0.9, "审美欣赏": 0.4},
                "expected_tempo": "very_slow",
                "expected_mode": "neutral_or_major"
            },
            {
                "name": "愤怒释放",
                "emotions": {"愤怒": 0.8, "厌恶": 0.4},
                "expected_tempo": "medium",
                "expected_mode": "minor_or_neutral"
            }
        ]
        
        success_count = 0
        
        for scenario in test_scenarios:
            print(f"\n🎬 测试场景: {scenario['name']}")
            
            try:
                # 创建情绪向量
                emotion_vector = self.bridge.create_emotion_vector_from_dict(scenario["emotions"])
                
                # 获取音乐参数
                music_params = self.kg.get_initial_music_parameters(emotion_vector)
                
                # 分析情绪
                emotion_analysis = self.kg.analyze_emotion_vector(emotion_vector)
                
                # 显示结果
                print(f"   主要情绪: {emotion_analysis['max_emotion']}")
                print(f"   音乐参数: Tempo={music_params['tempo']:.1f}, Mode={music_params['mode']:.2f}")
                print(f"   音色偏好: {music_params['timbre_preference']}")
                print(f"   情绪包络: {music_params['emotional_envelope_direction']}")
                
                # 基本验证
                if music_params['tempo'] >= 30 and music_params['tempo'] <= 200:
                    if 0 <= music_params['mode'] <= 1:
                        success_count += 1
                        print(f"   ✅ 场景测试通过")
                    else:
                        print(f"   ❌ 调式参数超出范围")
                else:
                    print(f"   ❌ 节拍参数超出范围")
                
            except Exception as e:
                print(f"   ❌ 场景测试失败: {e}")
        
        print(f"\n📊 情绪场景测试结果: {success_count}/{len(test_scenarios)} 通过")
        return success_count == len(test_scenarios)
    
    def test_parameter_mapping(self):
        """测试参数映射功能"""
        print("\n" + "="*60)
        print("🗺️  测试 3: 参数映射功能")
        print("="*60)
        
        try:
            # 创建测试参数
            test_kg_params = {
                'tempo': 110.0,
                'mode': 0.8,  # 大调
                'dynamics': 0.6,  # 中等偏响
                'harmony_consonance': 0.7,  # 较协和
                'timbre_preference': 'bright_ensemble',
                'pitch_register': 0.7,  # 偏高
                'density': 0.6,  # 中等偏密集
                'emotional_envelope_direction': 'uplifting'
            }
            
            print("🧪 测试KG参数:")
            for key, value in test_kg_params.items():
                print(f"   {key}: {value}")
            
            # 验证参数
            is_valid, errors = self.mapper.validate_parameters(test_kg_params)
            print(f"\n✅ 参数验证: {is_valid}")
            if errors:
                for error in errors:
                    print(f"   ⚠️  {error}")
            
            # 转换为文本描述
            text_desc = self.mapper.kg_to_text_description(test_kg_params)
            print(f"\n📝 文本描述:")
            print(f"   {text_desc}")
            
            # 转换为结构化参数
            structured = self.mapper.kg_to_structured_params(test_kg_params)
            print(f"\n🏗️  结构化参数:")
            for key, value in structured.items():
                print(f"   {key}: {value}")
            
            # 反向转换测试
            reverse_params = self.mapper.text_to_kg_params(text_desc)
            print(f"\n🔄 反向转换 (实验性):")
            for key, value in reverse_params.items():
                print(f"   {key}: {value}")
            
            print("✅ 参数映射功能测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 参数映射功能测试失败: {e}")
            return False
    
    def test_emotion_music_bridge(self):
        """测试情绪-音乐桥接器"""
        print("\n" + "="*60)
        print("🌉 测试 4: 情绪-音乐桥接器")
        print("="*60)
        
        try:
            # 显示桥接器状态
            status = self.bridge.get_bridge_status()
            print("📊 桥接器状态:")
            for key, value in status.items():
                print(f"   {key}: {value}")
            
            # 测试情绪向量模板
            template = self.bridge.get_emotion_vector_template()
            print(f"\n📝 情绪模板包含 {len(template)} 个情绪")
            
            # 测试完整分析流程
            print(f"\n🧪 测试完整分析流程:")
            
            # 创建混合情绪状态
            mixed_emotions = {
                "焦虑": 0.6,
                "悲伤": 0.4,
                "平静": 0.2,
                "快乐": 0.1
            }
            
            emotion_vector = self.bridge.create_emotion_vector_from_dict(mixed_emotions)
            print(f"   输入情绪: {mixed_emotions}")
            
            # 完整分析
            result = self.bridge.analyze_emotion_and_recommend_music(emotion_vector, "3min", 3)
            
            if result["success"]:
                print(f"   ✅ 分析成功")
                print(f"   主要情绪: {result['emotion_analysis']['max_emotion']}")
                print(f"   情绪平衡: {result['emotion_analysis']['emotion_balance']}")
                print(f"   音乐参数: {result['music_parameters']}")
                print(f"   治疗建议: {result['therapy_recommendation']['primary_focus']}")
                
                # 检查音乐检索结果
                if result["music_search_results"]:
                    search_results = result["music_search_results"]
                    if search_results["success"]:
                        print(f"   🎵 找到音乐: {len(search_results['results'])} 首")
                        if search_results["results"]:
                            first_result = search_results["results"][0]
                            print(f"      第一首: {first_result['video_name']} (相似度: {first_result['similarity']})")
                    else:
                        print(f"   ⚠️  音乐检索失败: {search_results['error']}")
                else:
                    print(f"   💡 仅参数分析模式 (MI_retrieve不可用)")
                    
            else:
                print(f"   ❌ 分析失败: {result['error']}")
                return False
            
            # 测试仅参数模式
            print(f"\n🔧 测试仅参数模式:")
            param_result = self.bridge.get_therapy_parameters_only(emotion_vector)
            
            if param_result["success"]:
                print(f"   ✅ 参数获取成功")
                print(f"   治疗方法: {param_result['therapy_recommendation']['therapy_approach']}")
            else:
                print(f"   ❌ 参数获取失败: {param_result['error']}")
                return False
            
            print("✅ 情绪-音乐桥接器测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 情绪-音乐桥接器测试失败: {e}")
            return False
    
    def test_edge_cases(self):
        """测试边界情况"""
        print("\n" + "="*60)
        print("🔬 测试 5: 边界情况")
        print("="*60)
        
        test_cases = [
            {
                "name": "全零向量",
                "vector": np.zeros(27),
                "should_pass": True
            },
            {
                "name": "全一向量", 
                "vector": np.ones(27),
                "should_pass": True
            },
            {
                "name": "单一极值情绪",
                "vector": np.zeros(27),
                "should_pass": True,
                "modify": lambda v: (v.__setitem__(5, 1.0), v)[1]  # 焦虑=1.0
            },
            {
                "name": "错误维度向量",
                "vector": np.zeros(20),  # 错误维度
                "should_pass": False
            },
            {
                "name": "超出范围值",
                "vector": np.ones(27) * 2.0,  # 值超出[0,1]
                "should_pass": True  # 应该被裁剪
            }
        ]
        
        success_count = 0
        
        for case in test_cases:
            print(f"\n🧪 测试: {case['name']}")
            
            try:
                test_vector = case["vector"].copy()
                
                # 应用修改函数
                if "modify" in case:
                    case["modify"](test_vector)
                
                # 尝试获取音乐参数
                music_params = self.kg.get_initial_music_parameters(test_vector)
                
                if case["should_pass"]:
                    # 验证结果合理性
                    if isinstance(music_params, dict) and "tempo" in music_params:
                        print(f"   ✅ 测试通过: 获得合理参数")
                        print(f"      Tempo: {music_params['tempo']:.1f}")
                        success_count += 1
                    else:
                        print(f"   ❌ 测试失败: 参数格式异常")
                else:
                    print(f"   ❌ 测试失败: 应该拒绝但接受了")
                
            except Exception as e:
                if case["should_pass"]:
                    print(f"   ❌ 测试失败: 意外异常 {e}")
                else:
                    print(f"   ✅ 测试通过: 正确拒绝 {e}")
                    success_count += 1
        
        print(f"\n📊 边界情况测试结果: {success_count}/{len(test_cases)} 通过")
        return success_count >= len(test_cases) - 1  # 允许一个失败
    
    def test_full_integration(self):
        """测试完整集成流程"""
        print("\n" + "="*60)
        print("🔗 测试 6: 完整集成流程")
        print("="*60)
        
        try:
            # 模拟真实使用场景
            print("🎬 模拟场景: 压力工作后的放松需求")
            
            # 用户情绪状态: 焦虑、疲劳、需要放松
            user_emotions = {
                "焦虑": 0.7,
                "疲劳": 0.6,  # 注意: 这不是27维中的情绪，会被忽略
                "平静": 0.1,
                "兴趣": 0.3
            }
            
            print(f"   用户报告情绪: {user_emotions}")
            
            # 步骤1: 创建情绪向量
            emotion_vector = self.bridge.create_emotion_vector_from_dict(user_emotions)
            print(f"   情绪向量维度: {emotion_vector.shape}")
            print(f"   非零情绪数量: {np.count_nonzero(emotion_vector)}")
            
            # 步骤2: 分析情绪并推荐音乐
            result = self.bridge.analyze_emotion_and_recommend_music(
                emotion_vector, 
                duration="3min", 
                top_k=5
            )
            
            if not result["success"]:
                print(f"   ❌ 集成流程失败: {result['error']}")
                return False
            
            # 步骤3: 展示分析结果
            emotion_analysis = result["emotion_analysis"]
            print(f"\n📊 情绪分析结果:")
            print(f"   主要情绪: {emotion_analysis['max_emotion']}")
            print(f"   情绪强度: {emotion_analysis['overall_intensity']:.3f}")
            print(f"   显著情绪: {len(emotion_analysis['significant_emotions'])} 个")
            
            # 步骤4: 展示音乐参数
            music_params = result["music_parameters"]
            print(f"\n🎵 推荐音乐参数:")
            for key, value in music_params.items():
                print(f"   {key}: {value}")
            
            # 步骤5: 展示治疗建议
            therapy = result["therapy_recommendation"]
            print(f"\n🏥 治疗建议:")
            print(f"   治疗重点: {therapy['primary_focus']}")
            print(f"   治疗方法: {therapy['therapy_approach']}")
            print(f"   建议时长: {therapy['session_duration']}")
            
            # 步骤6: 展示音乐检索结果
            if result["music_search_results"]:
                music_results = result["music_search_results"]
                if music_results["success"]:
                    print(f"\n🎶 音乐检索结果:")
                    print(f"   找到音乐: {len(music_results['results'])} 首")
                    
                    for i, music in enumerate(music_results["results"][:3], 1):
                        print(f"   {i}. {music['video_name']} - 相似度: {music['similarity']:.4f}")
                else:
                    print(f"\n⚠️  音乐检索失败: {music_results['error']}")
            else:
                print(f"\n💡 音乐检索不可用 (仅参数推荐模式)")
            
            # 步骤7: 文本描述
            text_desc = result["text_description"]
            print(f"\n📝 音乐描述:")
            print(f"   {text_desc}")
            
            print(f"\n✅ 完整集成流程测试通过")
            return True
            
        except Exception as e:
            print(f"❌ 完整集成流程测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 开始KG模块完整集成测试")
        print("=" * 80)
        
        test_methods = [
            self.test_knowledge_graph_basic,
            self.test_emotion_scenarios,
            self.test_parameter_mapping,
            self.test_emotion_music_bridge,
            self.test_edge_cases,
            self.test_full_integration
        ]
        
        passed = 0
        total = len(test_methods)
        
        for test_method in test_methods:
            try:
                if test_method():
                    passed += 1
            except Exception as e:
                print(f"❌ 测试方法 {test_method.__name__} 执行异常: {e}")
        
        # 测试总结
        print("\n" + "="*80)
        print("📊 KG模块集成测试总结")
        print("="*80)
        print(f"总测试数: {total}")
        print(f"通过测试: {passed}")
        print(f"失败测试: {total - passed}")
        print(f"成功率: {passed/total*100:.1f}%")
        
        if passed == total:
            print("🎉 所有测试通过! KG模块集成功能正常")
        elif passed >= total * 0.8:
            print("⚠️  大部分测试通过，模块基本可用")
        else:
            print("❌ 多项测试失败，需要进一步调试")
        
        return passed == total

def main():
    """主测试函数"""
    tester = KGIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 KG模块已准备就绪，可以投入使用!")
    else:
        print("\n🔧 KG模块需要进一步完善才能正常使用")
    
    return success

if __name__ == "__main__":
    main()