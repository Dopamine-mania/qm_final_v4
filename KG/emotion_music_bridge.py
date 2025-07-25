#!/usr/bin/env python3
"""
情绪-音乐桥接器 (Emotion-Music Bridge)

连接KG知识图谱模块与MI_retrieve音乐检索模块
实现端到端的情绪驱动音乐治疗流程
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加MI_retrieve模块路径
sys.path.append(str(Path(__file__).parent.parent / "MI_retrieve"))

from knowledge_graph import KnowledgeGraph

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionMusicBridge:
    """
    情绪-音乐桥接器
    
    整合情绪分析、知识图谱推理和音乐检索
    提供从情绪状态到音乐播放的完整解决方案
    """
    
    def __init__(self, enable_mi_retrieve: bool = True):
        """
        初始化桥接器
        
        Args:
            enable_mi_retrieve: 是否启用MI_retrieve模块集成
        """
        self.kg = KnowledgeGraph()
        self.enable_mi_retrieve = enable_mi_retrieve
        self.mi_retrieve_api = None
        
        # 尝试加载MI_retrieve模块
        if enable_mi_retrieve:
            try:
                from music_search_api import MusicSearchAPI
                self.mi_retrieve_api = MusicSearchAPI()
                logger.info("✅ MI_retrieve音乐检索模块加载成功")
            except ImportError as e:
                logger.warning(f"⚠️  MI_retrieve模块加载失败: {e}")
                logger.info("💡 将以独立模式运行，仅提供音乐参数推荐")
                self.enable_mi_retrieve = False
        
        logger.info("🌉 情绪-音乐桥接器初始化完成")
    
    def analyze_emotion_and_recommend_music(self, emotion_vector: np.ndarray, 
                                          duration: str = "3min", 
                                          top_k: int = 5) -> Dict[str, Any]:
        """
        分析情绪并推荐音乐 (核心方法)
        
        Args:
            emotion_vector: 27维情绪向量
            duration: 音乐时长版本
            top_k: 返回结果数量
            
        Returns:
            完整的情绪分析和音乐推荐结果
        """
        try:
            logger.info("🧠 开始情绪分析和音乐推荐流程...")
            
            # 1. 情绪向量验证
            if not self._validate_emotion_vector(emotion_vector):
                raise ValueError("情绪向量格式错误")
            
            # 2. 情绪分析
            emotion_analysis = self.kg.analyze_emotion_vector(emotion_vector)
            logger.info(f"📊 情绪分析完成，主要情绪: {emotion_analysis['max_emotion']}")
            
            # 3. 获取音乐搜索参数
            search_params = self.kg.get_music_search_parameters(emotion_vector)
            logger.info(f"🎵 生成音乐搜索参数: {search_params['structured_params']}")
            
            # 4. 构建基础结果
            result = {
                "success": True,
                "emotion_analysis": emotion_analysis,
                "music_parameters": search_params["structured_params"],
                "text_description": search_params["text_description"],
                "emotion_context": search_params["emotion_context"],
                "therapy_recommendation": self._generate_therapy_recommendation(emotion_analysis),
                "music_search_results": None
            }
            
            # 5. 音乐检索 (如果启用)
            if self.enable_mi_retrieve and self.mi_retrieve_api:
                try:
                    logger.info("🔍 执行音乐检索...")
                    search_result = self.mi_retrieve_api.search_by_description(
                        description=search_params["text_description"],
                        duration=duration,
                        top_k=top_k
                    )
                    
                    if search_result["success"]:
                        result["music_search_results"] = search_result
                        logger.info(f"✅ 音乐检索成功，找到 {len(search_result['results'])} 首匹配音乐")
                    else:
                        logger.warning(f"⚠️  音乐检索失败: {search_result['error']}")
                        result["music_search_error"] = search_result["error"]
                        
                except Exception as search_error:
                    logger.error(f"❌ 音乐检索异常: {search_error}")
                    result["music_search_error"] = str(search_error)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 情绪音乐分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "emotion_analysis": None,
                "music_parameters": None,
                "music_search_results": None
            }
    
    def search_music_by_emotion(self, emotion_vector: np.ndarray,
                              duration: str = "3min", 
                              top_k: int = 5) -> Dict[str, Any]:
        """
        基于情绪搜索音乐 (简化接口)
        
        Args:
            emotion_vector: 27维情绪向量
            duration: 音乐时长版本
            top_k: 返回结果数量
            
        Returns:
            音乐搜索结果
        """
        full_result = self.analyze_emotion_and_recommend_music(emotion_vector, duration, top_k)
        
        if full_result["success"] and full_result["music_search_results"]:
            return full_result["music_search_results"]
        else:
            return {
                "success": False,
                "error": full_result.get("error", "音乐检索不可用"),
                "results": []
            }
    
    def get_therapy_parameters_only(self, emotion_vector: np.ndarray) -> Dict[str, Any]:
        """
        仅获取治疗参数，不进行音乐检索
        
        Args:
            emotion_vector: 27维情绪向量
            
        Returns:
            治疗参数和建议
        """
        try:
            if not self._validate_emotion_vector(emotion_vector):
                raise ValueError("情绪向量格式错误")
            
            # 情绪分析
            emotion_analysis = self.kg.analyze_emotion_vector(emotion_vector)
            
            # 获取音乐参数
            search_params = self.kg.get_music_search_parameters(emotion_vector)
            
            # 生成治疗建议
            therapy_recommendation = self._generate_therapy_recommendation(emotion_analysis)
            
            return {
                "success": True,
                "emotion_analysis": emotion_analysis,
                "music_parameters": search_params["structured_params"],
                "therapy_recommendation": therapy_recommendation,
                "text_description": search_params["text_description"],
                "emotion_context": search_params["emotion_context"]
            }
            
        except Exception as e:
            logger.error(f"❌ 获取治疗参数失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def batch_emotion_analysis(self, emotion_vectors: List[np.ndarray], 
                             duration: str = "3min") -> List[Dict[str, Any]]:
        """
        批量情绪分析和音乐推荐
        
        Args:
            emotion_vectors: 情绪向量列表
            duration: 音乐时长版本
            
        Returns:
            分析结果列表
        """
        results = []
        
        for i, emotion_vector in enumerate(emotion_vectors, 1):
            logger.info(f"📊 批量分析进度: {i}/{len(emotion_vectors)}")
            result = self.analyze_emotion_and_recommend_music(emotion_vector, duration)
            results.append(result)
        
        logger.info(f"✅ 批量分析完成，处理了 {len(results)} 个情绪状态")
        return results
    
    def create_emotion_vector_from_dict(self, emotion_dict: Dict[str, float]) -> np.ndarray:
        """
        从情绪字典创建27维向量
        
        Args:
            emotion_dict: 情绪字典，如 {"快乐": 0.8, "兴奋": 0.6}
            
        Returns:
            27维情绪向量
        """
        emotion_vector = np.zeros(27)
        
        for emotion_name, value in emotion_dict.items():
            if emotion_name in self.kg.emotion_names:
                index = self.kg.emotion_names.index(emotion_name)
                emotion_vector[index] = max(0, min(1, value))  # 确保在[0,1]范围内
            else:
                logger.warning(f"⚠️  未知情绪名称: {emotion_name}")
        
        return emotion_vector
    
    def get_emotion_vector_template(self) -> Dict[str, float]:
        """
        获取情绪向量模板
        
        Returns:
            包含所有27个情绪的模板字典
        """
        return {emotion_name: 0.0 for emotion_name in self.kg.emotion_names}
    
    def _validate_emotion_vector(self, emotion_vector: np.ndarray) -> bool:
        """验证情绪向量格式"""
        try:
            if not isinstance(emotion_vector, np.ndarray):
                emotion_vector = np.array(emotion_vector)
            
            if emotion_vector.shape[0] != 27:
                logger.error(f"情绪向量维度错误: 期望27维，实际{emotion_vector.shape[0]}维")
                return False
            
            if np.any(emotion_vector < 0) or np.any(emotion_vector > 1):
                logger.warning("⚠️  情绪向量值超出[0,1]范围")
                # 这里我们选择容忍并进行裁剪，而不是拒绝
            
            return True
            
        except Exception as e:
            logger.error(f"情绪向量验证失败: {e}")
            return False
    
    def _generate_therapy_recommendation(self, emotion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于情绪分析生成治疗建议
        
        Args:
            emotion_analysis: 情绪分析结果
            
        Returns:
            治疗建议字典
        """
        max_emotion_name, max_emotion_value = emotion_analysis["max_emotion"]
        emotion_balance = emotion_analysis["emotion_balance"]
        
        # 基础治疗建议
        recommendations = {
            "primary_focus": "",
            "therapy_approach": "",
            "session_duration": "15-30分钟",
            "precautions": [],
            "follow_up": ""
        }
        
        # 根据主要情绪制定建议
        if max_emotion_name in ["焦虑", "恐惧", "恐怖"]:
            recommendations["primary_focus"] = "焦虑缓解和情绪稳定"
            recommendations["therapy_approach"] = "使用缓慢、协和的音乐，逐步降低生理激活水平"
            recommendations["precautions"] = ["避免突然的音量变化", "监控呼吸和心率反应"]
            recommendations["follow_up"] = "观察15分钟后的放松效果，必要时延长治疗时间"
            
        elif max_emotion_name in ["愤怒", "厌恶", "蔑视"]:
            recommendations["primary_focus"] = "情绪疏导和内心平衡"
            recommendations["therapy_approach"] = "先匹配情绪，再逐步引导到更平静的状态"
            recommendations["precautions"] = ["允许情绪表达", "避免过度压抑"]
            recommendations["follow_up"] = "评估愤怒水平是否降低，考虑进行认知重构"
            
        elif max_emotion_name in ["悲伤", "失望", "内疚"]:
            recommendations["primary_focus"] = "情感支持和希望重建"
            recommendations["therapy_approach"] = "从共情音乐开始，逐步引入温暖、上升的音乐元素"
            recommendations["precautions"] = ["避免过度催泪的音乐", "注意自杀风险评估"]
            recommendations["follow_up"] = "关注情绪提升效果，必要时配合心理咨询"
            
        elif max_emotion_name in ["快乐", "兴奋", "娱乐"]:
            recommendations["primary_focus"] = "积极情绪维持和能量平衡"
            recommendations["therapy_approach"] = "维持积极状态，同时避免过度兴奋"
            recommendations["precautions"] = ["注意能量过度消耗", "维持情绪稳定性"]
            recommendations["follow_up"] = "确保积极状态的持续性"
            
        elif max_emotion_name in ["平静", "审美欣赏"]:
            recommendations["primary_focus"] = "深度放松和内在和谐"
            recommendations["therapy_approach"] = "维持当前平静状态，深化放松体验"
            recommendations["precautions"] = ["避免过度刺激"]
            recommendations["follow_up"] = "评估放松深度和持续时间"
            
        else:
            recommendations["primary_focus"] = "情绪平衡和自我觉察"
            recommendations["therapy_approach"] = "使用中性、平衡的音乐促进情绪整合"
            recommendations["follow_up"] = "观察情绪变化趋势"
        
        # 根据情绪强度调整
        if max_emotion_value > 0.8:
            recommendations["session_duration"] = "30-45分钟"
            recommendations["precautions"].append("高强度情绪状态，需要更多关注")
        elif max_emotion_value < 0.3:
            recommendations["session_duration"] = "10-20分钟"
        
        # 根据情绪平衡调整
        if emotion_balance["negative"] > 0.6:
            recommendations["precautions"].append("负面情绪较强，需要额外支持")
        
        return recommendations
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """获取桥接器状态信息"""
        return {
            "kg_loaded": self.kg is not None,
            "mi_retrieve_enabled": self.enable_mi_retrieve,
            "mi_retrieve_available": self.mi_retrieve_api is not None,
            "emotion_dimensions": len(self.kg.emotion_names),
            "gems_rules_count": len(self.kg.rules)
        }

def main():
    """演示桥接器使用"""
    print("🌉 情绪-音乐桥接器演示")
    print("=" * 50)
    
    # 初始化桥接器
    bridge = EmotionMusicBridge(enable_mi_retrieve=True)
    
    # 显示状态
    status = bridge.get_bridge_status()
    print(f"📊 桥接器状态:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # 获取情绪模板
    print(f"\n📝 情绪向量模板 (前10个):")
    template = bridge.get_emotion_vector_template()
    for i, (emotion, value) in enumerate(list(template.items())[:10]):
        print(f"   {i}: {emotion} = {value}")
    
    # 测试场景1: 高焦虑状态
    print(f"\n🧪 测试场景1: 高焦虑状态")
    anxiety_emotions = {"焦虑": 0.8, "恐惧": 0.3, "平静": 0.1}
    anxiety_vector = bridge.create_emotion_vector_from_dict(anxiety_emotions)
    
    result1 = bridge.analyze_emotion_and_recommend_music(anxiety_vector, "3min", 3)
    
    if result1["success"]:
        print(f"✅ 分析成功")
        print(f"主要情绪: {result1['emotion_analysis']['max_emotion']}")
        print(f"音乐参数: {result1['music_parameters']}")
        print(f"治疗建议: {result1['therapy_recommendation']['primary_focus']}")
        
        if result1["music_search_results"]:
            print(f"找到音乐: {len(result1['music_search_results']['results'])} 首")
        else:
            print("⚠️  音乐检索不可用")
    else:
        print(f"❌ 分析失败: {result1['error']}")
    
    # 测试场景2: 快乐兴奋状态
    print(f"\n🧪 测试场景2: 快乐兴奋状态")
    happy_emotions = {"快乐": 0.9, "兴奋": 0.7, "娱乐": 0.5}
    happy_vector = bridge.create_emotion_vector_from_dict(happy_emotions)
    
    # 仅获取治疗参数 (不进行音乐检索)
    result2 = bridge.get_therapy_parameters_only(happy_vector)
    
    if result2["success"]:
        print(f"✅ 参数获取成功")
        print(f"主要情绪: {result2['emotion_analysis']['max_emotion']}")
        print(f"治疗方法: {result2['therapy_recommendation']['therapy_approach']}")
    else:
        print(f"❌ 参数获取失败: {result2['error']}")
    
    print(f"\n✅ 桥接器演示完成!")

if __name__ == "__main__":
    main()