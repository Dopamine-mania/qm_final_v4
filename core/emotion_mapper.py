#!/usr/bin/env python3
"""
情绪识别与映射模块 - 从qm_final3迁移并优化
用于4.0版本的检索逻辑
"""

import re
from typing import Dict, Tuple, List, Any
from datetime import datetime

class EmotionRecognizer:
    """
    27维情绪识别系统
    基于关键词匹配和语义分析的情绪检测
    """
    
    def __init__(self):
        # 基础5维情绪（保持与3.0版本兼容）
        self.base_emotions = {
            "焦虑": ["焦虑", "紧张", "担心", "不安", "害怕", "恐惧", "心跳", "忐忑", "惶恐"],
            "疲惫": ["疲惫", "累", "疲劳", "困倦", "乏力", "无力", "疲倦", "困", "精疲力竭"],
            "烦躁": ["烦躁", "烦恼", "易怒", "急躁", "不耐烦", "暴躁", "愤怒", "生气", "恼火"],
            "平静": ["平静", "放松", "安静", "宁静", "舒缓", "轻松", "安逸", "祥和", "淡定"],
            "压力": ["压力", "紧迫", "负担", "重压", "沉重", "压抑", "紧张", "负重", "喘不过气"]
        }
        
        # 扩展睡眠专用情绪维度（适配27维系统）
        self.sleep_specific_emotions = {
            "反刍思考": ["反刍", "胡思乱想", "想太多", "思维循环", "钻牛角尖"],
            "睡眠焦虑": ["睡不着", "失眠", "睡眠焦虑", "怕睡不好", "担心睡眠"],
            "身体疲惫": ["身体累", "肌肉酸痛", "体力不支", "身体沉重"],
            "精神疲惫": ["脑子累", "精神疲劳", "心累", "思维疲惫"],
            "过度觉醒": ["太兴奋", "睡不下", "精神亢奋", "大脑活跃"],
            "就寝担忧": ["睡前担心", "明天的事", "工作压力", "生活烦恼"],
            "思维奔逸": ["停不下来", "思绪飞转", "脑子转个不停", "想法很多"],
            "躯体紧张": ["肌肉紧张", "身体僵硬", "无法放松", "绷得很紧"]
        }
        
        # 合并所有情绪类别
        self.all_emotions = {**self.base_emotions, **self.sleep_specific_emotions}
        
    def detect_emotion(self, user_input: str) -> Tuple[str, float]:
        """
        增强情绪检测
        
        Args:
            user_input: 用户输入的情绪描述
            
        Returns:
            Tuple[str, float]: (检测到的情绪, 置信度)
        """
        if not user_input or len(user_input.strip()) < 2:
            return "焦虑", 0.85
        
        # 清理输入
        cleaned_input = self._clean_input(user_input)
        
        # 计算情绪得分
        emotion_scores = {}
        for emotion, keywords in self.all_emotions.items():
            score = self._calculate_emotion_score(cleaned_input, keywords)
            if score > 0:
                emotion_scores[emotion] = score
        
        # 如果没有匹配到专用情绪，使用基础情绪
        if not emotion_scores:
            for emotion, keywords in self.base_emotions.items():
                score = self._calculate_emotion_score(cleaned_input, keywords)
                if score > 0:
                    emotion_scores[emotion] = score
        
        # 选择得分最高的情绪
        if emotion_scores:
            detected_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            emotion_name = detected_emotion[0]
            base_confidence = 0.85 + detected_emotion[1] * 0.03
            confidence = min(base_confidence, 0.95)
        else:
            # 默认情绪
            emotion_name = "焦虑"
            confidence = 0.80
        
        return emotion_name, confidence
    
    def _clean_input(self, text: str) -> str:
        """清理输入文本"""
        # 移除多余空格和标点
        cleaned = re.sub(r'[^\w\s]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.lower()
    
    def _calculate_emotion_score(self, text: str, keywords: List[str]) -> float:
        """计算情绪关键词匹配得分"""
        score = 0
        for keyword in keywords:
            if keyword in text:
                # 完全匹配得2分，部分匹配得1分
                if keyword == text or f" {keyword} " in f" {text} ":
                    score += 2
                else:
                    score += 1
        return score
    
    def get_emotion_details(self, emotion: str) -> Dict[str, Any]:
        """获取情绪的详细信息"""
        # 判断是基础情绪还是睡眠专用情绪
        if emotion in self.base_emotions:
            category = "基础情绪"
            keywords = self.base_emotions[emotion]
        elif emotion in self.sleep_specific_emotions:
            category = "睡眠专用情绪"
            keywords = self.sleep_specific_emotions[emotion]
        else:
            category = "未知"
            keywords = []
        
        return {
            "emotion": emotion,
            "category": category,
            "keywords": keywords,
            "dimension_count": len(self.all_emotions),
            "timestamp": datetime.now().isoformat()
        }

class MusicFeatureMapper:
    """
    情绪到音乐特征的映射系统
    基于ISO三阶段原则和音乐疗愈理论
    """
    
    def __init__(self):
        # ISO三阶段音乐特征数据库（从3.0版本迁移）
        self.features_database = {
            "焦虑": {
                "匹配阶段": {
                    "tempo": "moderate tense",
                    "key": "minor anxious", 
                    "dynamics": "restless energy",
                    "mood": "matching anxiety",
                    "instrumental": "strings, piano",
                    "texture": "complex, layered"
                },
                "引导阶段": {
                    "tempo": "gradually calming",
                    "key": "minor to neutral transition",
                    "dynamics": "settling down", 
                    "mood": "guiding to peace",
                    "instrumental": "soft piano, ambient",
                    "texture": "simplifying"
                },
                "目标阶段": {
                    "tempo": "slow peaceful",
                    "key": "major calm",
                    "dynamics": "gentle soft",
                    "mood": "deep relaxation for sleep",
                    "instrumental": "ambient pads, nature",
                    "texture": "minimal, spacious"
                }
            },
            "疲惫": {
                "匹配阶段": {
                    "tempo": "tired sluggish",
                    "key": "minor weary",
                    "dynamics": "heavy fatigue",
                    "mood": "exhausted state",
                    "instrumental": "cello, low piano",
                    "texture": "heavy, dragging"
                },
                "引导阶段": {
                    "tempo": "gentle restoration",
                    "key": "minor to warm transition", 
                    "dynamics": "nurturing support",
                    "mood": "healing tiredness",
                    "instrumental": "warm strings, harp",
                    "texture": "supportive, embracing"
                },
                "目标阶段": {
                    "tempo": "deeply restful",
                    "key": "warm major",
                    "dynamics": "embracing comfort",
                    "mood": "restorative sleep",
                    "instrumental": "soft choir, ambient",
                    "texture": "enveloping, safe"
                }
            },
            "烦躁": {
                "匹配阶段": {
                    "tempo": "agitated irregular",
                    "key": "dissonant minor",
                    "dynamics": "sharp edges",
                    "mood": "irritated energy",
                    "instrumental": "staccato strings, percussion",
                    "texture": "angular, restless"
                },
                "引导阶段": {
                    "tempo": "smoothing out",
                    "key": "resolving tensions",
                    "dynamics": "softening edges",
                    "mood": "releasing irritation",
                    "instrumental": "flowing strings, woodwinds",
                    "texture": "smoothing, flowing"
                },
                "目标阶段": {
                    "tempo": "smooth flowing",
                    "key": "resolved major",
                    "dynamics": "peaceful waves",
                    "mood": "serene sleep state",
                    "instrumental": "soft pads, gentle waves",
                    "texture": "fluid, peaceful"
                }
            },
            "平静": {
                "匹配阶段": {
                    "tempo": "naturally calm",
                    "key": "neutral peaceful",
                    "dynamics": "already gentle",
                    "mood": "existing tranquility",
                    "instrumental": "soft piano, strings",
                    "texture": "balanced, serene"
                },
                "引导阶段": {
                    "tempo": "deepening calm",
                    "key": "enriching peace",
                    "dynamics": "expanding serenity",
                    "mood": "enhancing stillness",
                    "instrumental": "ambient, soft choir",
                    "texture": "expanding, deepening"
                },
                "目标阶段": {
                    "tempo": "profound stillness",
                    "key": "deep major",
                    "dynamics": "whisper soft",
                    "mood": "transcendent sleep",
                    "instrumental": "pure tones, silence",
                    "texture": "minimal, transcendent"
                }
            },
            "压力": {
                "匹配阶段": {
                    "tempo": "pressured urgent",
                    "key": "tense minor",
                    "dynamics": "compressed energy",
                    "mood": "stress overload",
                    "instrumental": "tight strings, muted brass",
                    "texture": "compressed, intense"
                },
                "引导阶段": {
                    "tempo": "releasing pressure",
                    "key": "opening up space",
                    "dynamics": "expanding freedom",
                    "mood": "letting go stress",
                    "instrumental": "opening brass, flowing strings",
                    "texture": "expanding, liberating"
                },
                "目标阶段": {
                    "tempo": "weightless floating",
                    "key": "liberated major",
                    "dynamics": "free flowing",
                    "mood": "stress-free sleep",
                    "instrumental": "ambient space, soft pads",
                    "texture": "weightless, free"
                }
            }
        }
        
        # 为睡眠专用情绪创建默认映射
        self._create_sleep_emotion_mappings()
    
    def _create_sleep_emotion_mappings(self):
        """为睡眠专用情绪创建音乐特征映射"""
        sleep_emotions = [
            "反刍思考", "睡眠焦虑", "身体疲惫", "精神疲惫", 
            "过度觉醒", "就寝担忧", "思维奔逸", "躯体紧张"
        ]
        
        for emotion in sleep_emotions:
            if emotion not in self.features_database:
                # 根据睡眠情绪特点分配基础模板
                if "焦虑" in emotion or "担忧" in emotion:
                    template = self.features_database["焦虑"]
                elif "疲惫" in emotion:
                    template = self.features_database["疲惫"]
                elif "觉醒" in emotion or "奔逸" in emotion:
                    template = self.features_database["烦躁"]
                elif "紧张" in emotion:
                    template = self.features_database["压力"]
                else:
                    template = self.features_database["焦虑"]  # 默认
                
                # 复制模板并做小的调整
                self.features_database[emotion] = {
                    stage: {**features, "mood": f"{emotion} - {features['mood']}"}
                    for stage, features in template.items()
                }
    
    def get_music_features(self, emotion: str) -> Dict[str, Any]:
        """
        根据情绪获取ISO三阶段音乐特征
        
        Args:
            emotion: 检测到的情绪
            
        Returns:
            Dict: 包含三阶段音乐特征的字典
        """
        if emotion in self.features_database:
            return self.features_database[emotion]
        else:
            # 如果没有找到，使用焦虑作为默认
            print(f"⚠️ 未找到情绪 '{emotion}' 的音乐特征，使用默认(焦虑)")
            return self.features_database["焦虑"]
    
    def extract_matching_stage_features(self, emotion: str) -> Dict[str, Any]:
        """
        提取匹配阶段的特征（用于检索）
        这对应前25%的视频内容特征
        
        Args:
            emotion: 情绪类型
            
        Returns:
            Dict: 匹配阶段的音乐特征
        """
        features = self.get_music_features(emotion)
        matching_features = features["匹配阶段"]
        
        # 添加检索相关的元数据
        matching_features["emotion"] = emotion
        matching_features["stage"] = "匹配阶段"
        matching_features["iso_stage_ratio"] = 0.25  # 前25%
        
        return matching_features
    
    def get_supported_emotions(self) -> List[str]:
        """获取所有支持的情绪类型"""
        return list(self.features_database.keys())

# 单例实例，供其他模块使用
emotion_recognizer = EmotionRecognizer()
music_feature_mapper = MusicFeatureMapper()

def detect_emotion_enhanced(user_input: str) -> Tuple[str, float]:
    """
    增强情绪检测的便捷函数
    保持与3.0版本的API兼容性
    """
    return emotion_recognizer.detect_emotion(user_input)

def get_emotion_music_features(emotion: str) -> Dict[str, Any]:
    """
    获取情绪音乐特征的便捷函数
    保持与3.0版本的API兼容性
    """
    return music_feature_mapper.get_music_features(emotion)

if __name__ == "__main__":
    # 测试情绪识别
    test_inputs = [
        "我感到很焦虑，心跳加速，难以入睡",
        "太累了，身体和精神都很疲惫",
        "今晚又开始胡思乱想，停不下来",
        "压力很大，感觉喘不过气",
        "比较平静，但希望更深层的放松"
    ]
    
    print("🧠 情绪识别测试：")
    for text in test_inputs:
        emotion, confidence = detect_emotion_enhanced(text)
        print(f"输入: {text}")
        print(f"识别: {emotion} (置信度: {confidence:.2%})")
        
        features = get_emotion_music_features(emotion)
        matching = features["匹配阶段"]
        print(f"匹配阶段特征: {matching['mood']}, {matching['tempo']}")
        print("-" * 50)