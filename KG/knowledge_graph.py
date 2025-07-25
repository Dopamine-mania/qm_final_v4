#!/usr/bin/env python3
"""
知识图谱 (Knowledge Graph) - 情绪驱动音乐治疗系统核心

基于GEMS模型和ISO原则，实现27维情绪向量到音乐参数的智能映射
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicRule:
    """音乐治疗规则类"""
    
    def __init__(self, name: str, conditions: Dict[str, float], 
                 parameters: Dict[str, Any], priority: str = "medium"):
        """
        初始化音乐规则
        
        Args:
            name: 规则名称
            conditions: 情绪条件字典 {"情绪名": 最小阈值}
            parameters: 音乐参数字典
            priority: 规则优先级 (critical/high/medium/low)
        """
        self.name = name
        self.conditions = conditions
        self.parameters = parameters
        self.priority = priority
        
        # 优先级权重
        self.priority_weights = {
            "critical": 4,
            "high": 3, 
            "medium": 2,
            "low": 1
        }
    
    def evaluate(self, emotion_dict: Dict[str, float]) -> Tuple[bool, float]:
        """
        评估规则是否匹配当前情绪状态
        
        Args:
            emotion_dict: 情绪字典 {"情绪名": 强度值}
            
        Returns:
            (是否匹配, 匹配强度)
        """
        total_match_strength = 0.0
        matched_conditions = 0
        
        for emotion_name, threshold in self.conditions.items():
            if emotion_name in emotion_dict:
                emotion_value = emotion_dict[emotion_name]
                if emotion_value >= threshold:
                    # 匹配强度 = 超出阈值的程度
                    match_strength = emotion_value - threshold
                    total_match_strength += match_strength
                    matched_conditions += 1
                else:
                    # 如果有条件不满足，规则不匹配
                    return False, 0.0
        
        if matched_conditions == len(self.conditions):
            # 所有条件都满足，计算总匹配强度
            avg_match_strength = total_match_strength / len(self.conditions)
            # 加上优先级权重
            weighted_strength = avg_match_strength * self.priority_weights[self.priority]
            return True, weighted_strength
        
        return False, 0.0

class KnowledgeGraph:
    """
    情绪-音乐知识图谱
    
    基于GEMS (Geneva Emotional Music Scale) 模型实现
    27维情绪向量到音乐治疗参数的智能映射
    """
    
    def __init__(self):
        """初始化知识图谱"""
        
        # 27个情绪名称 (固定顺序，对应27维向量的索引)
        self.emotion_names = [
            "钦佩", "崇拜", "审美欣赏", "娱乐", "愤怒", "焦虑", "敬畏", "尴尬",
            "无聊", "平静", "困惑", "蔑视", "渴望", "失望", "厌恶", "同情",
            "入迷", "嫉妒", "兴奋", "恐惧", "内疚", "恐怖", "兴趣", "快乐",
            "怀旧", "浪漫", "悲伤"
        ]
        
        # 默认/中性音乐参数 (治疗起始点)
        self.default_music_parameters = {
            'tempo': 80.0,                    # BPM，中等节拍
            'mode': 0.5,                      # 0=小调, 1=大调, 0.5=中性
            'dynamics': 0.5,                  # 0=很轻, 1=很响, 0.5=适中
            'harmony_consonance': 0.5,        # 0=不协和, 1=协和, 0.5=中性
            'timbre_preference': 'neutral_pad', # 音色偏好
            'pitch_register': 0.5,            # 0=低音, 1=高音, 0.5=中音
            'density': 0.5,                   # 0=稀疏, 1=密集, 0.5=适中
            'emotional_envelope_direction': 'neutral'  # 情绪包络方向
        }
        
        # 建立基于GEMS模型的规则系统
        self.rules = []
        self._initialize_gems_rules()
        
        logger.info("✅ 知识图谱初始化完成")
        logger.info(f"   情绪维度: {len(self.emotion_names)}")
        logger.info(f"   规则数量: {len(self.rules)}")
    
    def _initialize_gems_rules(self):
        """初始化基于GEMS模型的音乐治疗规则"""
        
        # Critical级别规则 (极端情绪状态，优先处理)
        
        # 1. 极度焦虑 → 缓慢稳定音乐
        self.rules.append(MusicRule(
            name="极度焦虑缓解",
            conditions={"焦虑": 0.8},
            parameters={
                'tempo': 60.0,              # 慢节拍降低焦虑
                'mode': 0.7,                # 偏大调，温暖感
                'dynamics': 0.3,            # 轻柔音量
                'harmony_consonance': 0.8,  # 高协和度，安全感
                'timbre_preference': 'warm_pad',
                'pitch_register': 0.3,      # 低音域，稳定感
                'density': 0.2,             # 稀疏，不增加压力
                'emotional_envelope_direction': 'calming'
            },
            priority="critical"
        ))
        
        # 2. 极度愤怒 → 渐进式释放
        self.rules.append(MusicRule(
            name="愤怒情绪疏导",
            conditions={"愤怒": 0.8},
            parameters={
                'tempo': 90.0,              # 中等节拍，避免激化
                'mode': 0.4,                # 偏小调，符合情绪
                'dynamics': 0.6,            # 中等音量，有表达空间
                'harmony_consonance': 0.3,  # 适度不协和，情绪释放
                'timbre_preference': 'expressive_strings',
                'pitch_register': 0.6,      # 中高音域
                'density': 0.7,             # 较密集，情绪密度
                'emotional_envelope_direction': 'descending'  # 递减，情绪缓解
            },
            priority="critical"
        ))
        
        # 3. 极度恐惧 → 安全包容
        self.rules.append(MusicRule(
            name="恐惧安抚",
            conditions={"恐惧": 0.8},
            parameters={
                'tempo': 55.0,              # 非常慢，安全感
                'mode': 0.8,                # 大调，积极暗示
                'dynamics': 0.2,            # 很轻，非威胁性
                'harmony_consonance': 0.9,  # 极度协和，安全感
                'timbre_preference': 'soft_choir',
                'pitch_register': 0.4,      # 偏低音，包容感
                'density': 0.1,             # 极稀疏，不增加刺激
                'emotional_envelope_direction': 'steady'
            },
            priority="critical"
        ))
        
        # High级别规则 (显著情绪状态)
        
        # 4. 高度平静维持
        self.rules.append(MusicRule(
            name="平静状态维持",
            conditions={"平静": 0.7},
            parameters={
                'tempo': 65.0,
                'mode': 0.6,
                'dynamics': 0.4,
                'harmony_consonance': 0.7,
                'timbre_preference': 'nature_sounds',
                'pitch_register': 0.4,
                'density': 0.3,
                'emotional_envelope_direction': 'steady'
            },
            priority="high"
        ))
        
        # 5. 高度悲伤 → 情感支持
        self.rules.append(MusicRule(
            name="悲伤情感支持",
            conditions={"悲伤": 0.7},
            parameters={
                'tempo': 70.0,
                'mode': 0.3,                # 小调，情感共鸣
                'dynamics': 0.4,
                'harmony_consonance': 0.6,
                'timbre_preference': 'gentle_piano',
                'pitch_register': 0.3,
                'density': 0.4,
                'emotional_envelope_direction': 'gentle_rise'  # 轻缓上升，希望感
            },
            priority="high"
        ))
        
        # 6. 高度快乐 → 能量维持
        self.rules.append(MusicRule(
            name="快乐能量维持",
            conditions={"快乐": 0.7},
            parameters={
                'tempo': 100.0,
                'mode': 0.8,                # 明亮大调
                'dynamics': 0.7,
                'harmony_consonance': 0.8,
                'timbre_preference': 'bright_ensemble',
                'pitch_register': 0.7,      # 高音域，明亮感
                'density': 0.6,
                'emotional_envelope_direction': 'uplifting'
            },
            priority="high"
        ))
        
        # Medium级别规则 (中等情绪状态)
        
        # 7. 中度焦虑 → 渐进放松
        self.rules.append(MusicRule(
            name="中度焦虑缓解",
            conditions={"焦虑": 0.5},
            parameters={
                'tempo': 75.0,
                'mode': 0.6,
                'dynamics': 0.4,
                'harmony_consonance': 0.7,
                'timbre_preference': 'ambient_pad',
                'pitch_register': 0.4,
                'density': 0.3,
                'emotional_envelope_direction': 'calming'
            },
            priority="medium"
        ))
        
        # 8. 兴奋+快乐组合 → 积极能量
        self.rules.append(MusicRule(
            name="积极兴奋状态",
            conditions={"兴奋": 0.6, "快乐": 0.5},
            parameters={
                'tempo': 110.0,
                'mode': 0.8,
                'dynamics': 0.7,
                'harmony_consonance': 0.7,
                'timbre_preference': 'energetic_mix',
                'pitch_register': 0.7,
                'density': 0.7,
                'emotional_envelope_direction': 'energizing'
            },
            priority="medium"
        ))
        
        # 9. 怀旧情感 → 温暖回忆
        self.rules.append(MusicRule(
            name="怀旧情感抚慰",
            conditions={"怀旧": 0.6},
            parameters={
                'tempo': 85.0,
                'mode': 0.5,                # 中性调式，复杂情感
                'dynamics': 0.5,
                'harmony_consonance': 0.6,
                'timbre_preference': 'vintage_warmth',
                'pitch_register': 0.5,
                'density': 0.4,
                'emotional_envelope_direction': 'nostalgic_wave'
            },
            priority="medium"
        ))
        
        # Low级别规则 (轻微情绪状态)
        
        # 10. 轻微无聊 → 兴趣激发
        self.rules.append(MusicRule(
            name="兴趣激发",
            conditions={"无聊": 0.4},
            parameters={
                'tempo': 95.0,
                'mode': 0.6,
                'dynamics': 0.6,
                'harmony_consonance': 0.6,
                'timbre_preference': 'interesting_textures',
                'pitch_register': 0.6,
                'density': 0.5,
                'emotional_envelope_direction': 'engaging'
            },
            priority="low"
        ))
        
        logger.info(f"📚 GEMS规则系统加载完成: {len(self.rules)} 条规则")
    
    def _vector_to_emotion_dict(self, emotion_vector: np.ndarray) -> Dict[str, float]:
        """
        将27维情绪向量转换为情绪字典
        
        Args:
            emotion_vector: 长度为27的numpy数组
            
        Returns:
            情绪字典 {"情绪名": 强度值}
        """
        if len(emotion_vector) != 27:
            raise ValueError(f"情绪向量维度必须为27，当前为{len(emotion_vector)}")
        
        emotion_dict = {}
        for i, emotion_name in enumerate(self.emotion_names):
            emotion_dict[emotion_name] = float(emotion_vector[i])
        
        return emotion_dict
    
    def _evaluate_condition(self, emotion_dict: Dict[str, float], 
                          condition: Dict[str, float]) -> bool:
        """
        评估情绪条件是否满足
        
        Args:
            emotion_dict: 当前情绪状态
            condition: 规则条件
            
        Returns:
            是否满足条件
        """
        for emotion_name, threshold in condition.items():
            if emotion_name not in emotion_dict:
                return False
            if emotion_dict[emotion_name] < threshold:
                return False
        return True
    
    def get_initial_music_parameters(self, emotion_vector: np.ndarray) -> Dict[str, Any]:
        """
        根据27维情绪向量获取初始音乐参数
        
        这是知识图谱的核心方法，基于GEMS模型和ISO原则
        
        Args:
            emotion_vector: 长度为27的numpy数组，取值范围[0,1]
            
        Returns:
            音乐参数字典，包含：
            - tempo: 节拍 (BPM)
            - mode: 调式 (0=小调, 1=大调)
            - dynamics: 音量动态 (0=轻, 1=响)
            - harmony_consonance: 和声协和度 (0=不协和, 1=协和)
            - timbre_preference: 音色偏好
            - pitch_register: 音域 (0=低, 1=高)
            - density: 密度 (0=稀疏, 1=密集)
            - emotional_envelope_direction: 情绪包络方向
        """
        try:
            # 验证输入
            if not isinstance(emotion_vector, np.ndarray):
                emotion_vector = np.array(emotion_vector)
            
            if emotion_vector.shape[0] != 27:
                raise ValueError(f"情绪向量维度错误: 期望27维，实际{emotion_vector.shape[0]}维")
            
            # 数值范围检查
            if np.any(emotion_vector < 0) or np.any(emotion_vector > 1):
                logger.warning("⚠️  情绪向量值超出[0,1]范围，将进行裁剪")
                emotion_vector = np.clip(emotion_vector, 0, 1)
            
            # 转换为情绪字典
            emotion_dict = self._vector_to_emotion_dict(emotion_vector)
            
            # 分析主要情绪
            primary_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"🧠 主要情绪状态: {[(name, f'{value:.3f}') for name, value in primary_emotions]}")
            
            # 评估所有规则并找到最佳匹配
            best_rule = None
            best_match_strength = 0.0
            matched_rules = []
            
            for rule in self.rules:
                is_match, match_strength = rule.evaluate(emotion_dict)
                if is_match:
                    matched_rules.append((rule, match_strength))
                    if match_strength > best_match_strength:
                        best_match_strength = match_strength
                        best_rule = rule
            
            # 开始构建音乐参数
            music_params = self.default_music_parameters.copy()
            
            if best_rule:
                # 应用最佳匹配规则
                logger.info(f"🎯 匹配规则: {best_rule.name} (优先级: {best_rule.priority}, 强度: {best_match_strength:.3f})")
                
                for param_name, param_value in best_rule.parameters.items():
                    music_params[param_name] = param_value
                
                # 记录所有匹配的规则
                if len(matched_rules) > 1:
                    logger.info(f"📋 其他匹配规则: {[(rule.name, f'{strength:.3f}') for rule, strength in matched_rules if rule != best_rule]}")
            
            else:
                logger.info("🔍 未找到匹配规则，使用默认参数")
                
                # 基于情绪强度进行基础调整
                max_emotion_name, max_emotion_value = primary_emotions[0]
                
                if max_emotion_value > 0.3:  # 有明显情绪
                    # 基础情绪调整逻辑
                    if max_emotion_name in ["焦虑", "恐惧", "恐怖"]:
                        music_params['tempo'] = max(50, music_params['tempo'] - 20)
                        music_params['harmony_consonance'] = 0.8
                        music_params['dynamics'] = 0.3
                    elif max_emotion_name in ["快乐", "兴奋", "娱乐"]:
                        music_params['tempo'] = min(120, music_params['tempo'] + 20) 
                        music_params['mode'] = 0.8
                        music_params['dynamics'] = 0.7
                    elif max_emotion_name in ["悲伤", "失望", "怀旧"]:
                        music_params['tempo'] = max(60, music_params['tempo'] - 10)
                        music_params['mode'] = 0.3
                        
                    logger.info(f"🔧 基于{max_emotion_name}({max_emotion_value:.3f})进行基础调整")
            
            # 确保参数在合理范围内
            music_params['tempo'] = max(40, min(160, music_params['tempo']))
            music_params['mode'] = max(0, min(1, music_params['mode']))
            music_params['dynamics'] = max(0, min(1, music_params['dynamics']))
            music_params['harmony_consonance'] = max(0, min(1, music_params['harmony_consonance']))
            music_params['pitch_register'] = max(0, min(1, music_params['pitch_register']))
            music_params['density'] = max(0, min(1, music_params['density']))
            
            logger.info(f"🎵 最终音乐参数: Tempo={music_params['tempo']:.1f}, Mode={music_params['mode']:.2f}, Dynamics={music_params['dynamics']:.2f}")
            
            return music_params
            
        except Exception as e:
            logger.error(f"❌ 获取音乐参数失败: {e}")
            logger.info("🔄 返回默认参数")
            return self.default_music_parameters.copy()
    
    def get_music_search_parameters(self, emotion_vector: np.ndarray) -> Dict[str, Any]:
        """
        获取适用于MI_retrieve模块的搜索参数
        
        Args:
            emotion_vector: 27维情绪向量
            
        Returns:
            包含文本描述、结构化参数和情绪上下文的字典
        """
        try:
            # 获取音乐参数
            music_params = self.get_initial_music_parameters(emotion_vector)
            
            # 分析情绪上下文
            emotion_dict = self._vector_to_emotion_dict(emotion_vector)
            primary_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            primary_emotions = [(name, value) for name, value in primary_emotions if value > 0.1]
            
            # 计算情绪强度
            emotion_intensity = np.mean([value for _, value in primary_emotions]) if primary_emotions else 0.0
            
            # 生成文本描述
            text_description = self._generate_text_description(music_params)
            
            # 构建结构化参数
            structured_params = {
                "tempo": music_params['tempo'],
                "mode": "major" if music_params['mode'] > 0.6 else "minor" if music_params['mode'] < 0.4 else "neutral",
                "dynamics": "loud" if music_params['dynamics'] > 0.7 else "soft" if music_params['dynamics'] < 0.3 else "medium",
                "harmony": "consonant" if music_params['harmony_consonance'] > 0.7 else "dissonant" if music_params['harmony_consonance'] < 0.3 else "mixed",
                "timbre": music_params['timbre_preference'],
                "register": "high" if music_params['pitch_register'] > 0.7 else "low" if music_params['pitch_register'] < 0.3 else "medium",
                "density": "dense" if music_params['density'] > 0.7 else "sparse" if music_params['density'] < 0.3 else "medium"
            }
            
            return {
                "text_description": text_description,
                "structured_params": structured_params,
                "emotion_context": {
                    "primary_emotions": [name for name, _ in primary_emotions],
                    "emotion_intensity": emotion_intensity,
                    "emotion_details": dict(primary_emotions)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 生成搜索参数失败: {e}")
            return {
                "text_description": "节拍适中，调式中性，音色温和，适合放松",
                "structured_params": {"tempo": 80, "mode": "neutral", "dynamics": "medium"},
                "emotion_context": {"primary_emotions": [], "emotion_intensity": 0.0}
            }
    
    def _generate_text_description(self, music_params: Dict[str, Any]) -> str:
        """
        根据音乐参数生成自然语言描述
        
        Args:
            music_params: 音乐参数字典
            
        Returns:
            自然语言描述字符串
        """
        # 节拍描述
        tempo = music_params['tempo']
        if tempo < 60:
            tempo_desc = f"建议初始节拍为 {tempo:.0f} BPM，节奏非常缓慢"
        elif tempo < 80:
            tempo_desc = f"建议初始节拍为 {tempo:.0f} BPM，节奏缓慢放松"
        elif tempo < 100:
            tempo_desc = f"建议初始节拍为 {tempo:.0f} BPM，节奏适中稳定"
        elif tempo < 120:
            tempo_desc = f"建议初始节拍为 {tempo:.0f} BPM，节奏明快活泼"
        else:
            tempo_desc = f"建议初始节拍为 {tempo:.0f} BPM，节奏快速有力"
        
        # 调式描述
        mode = music_params['mode']
        if mode > 0.7:
            mode_desc = "调式倾向大调，明亮积极"
        elif mode < 0.3:
            mode_desc = "调式倾向小调，深沉内敛"
        else:
            mode_desc = "调式中性或调式变化丰富"
        
        # 和声描述
        harmony = music_params['harmony_consonance']
        if harmony > 0.8:
            harmony_desc = "和声高度协和，纯净安全"
        elif harmony > 0.6:
            harmony_desc = "和声相对协和，温暖稳定"
        elif harmony < 0.3:
            harmony_desc = "和声包含不协和，表现力强"
        else:
            harmony_desc = "和声复杂多变，层次丰富"
        
        # 音色描述
        timbre = music_params['timbre_preference']
        timbre_mapping = {
            'neutral_pad': '音色中性柔和',
            'warm_pad': '音色温暖包容',
            'soft_choir': '音色轻柔如歌',
            'gentle_piano': '音色清雅如钢琴',
            'nature_sounds': '音色自然清新',
            'ambient_pad': '音色环境化氛围',
            'bright_ensemble': '音色明亮丰富',
            'energetic_mix': '音色充满活力',
            'expressive_strings': '音色表现力强',
            'vintage_warmth': '音色怀旧温暖',
            'interesting_textures': '音色富有质感'
        }
        timbre_desc = timbre_mapping.get(timbre, '音色特色鲜明')
        
        # 动态描述
        dynamics = music_params['dynamics']
        if dynamics > 0.8:
            dynamics_desc = "音量饱满有力"
        elif dynamics > 0.6:
            dynamics_desc = "音量适中清晰"
        elif dynamics < 0.3:
            dynamics_desc = "音量轻柔细腻"
        else:
            dynamics_desc = "音量变化自然"
        
        # 音域描述
        register = music_params['pitch_register']
        if register > 0.7:
            register_desc = "音域偏高，明亮清透"
        elif register < 0.3:
            register_desc = "音域偏低，深沉稳重"
        else:
            register_desc = "音域适中，平衡舒适"
        
        # 组合描述
        description = f"{tempo_desc}，{mode_desc}，{harmony_desc}，{timbre_desc}，{dynamics_desc}，{register_desc}"
        
        return description
    
    def analyze_emotion_vector(self, emotion_vector: np.ndarray) -> Dict[str, Any]:
        """
        分析情绪向量的详细信息
        
        Args:
            emotion_vector: 27维情绪向量
            
        Returns:
            情绪分析结果
        """
        emotion_dict = self._vector_to_emotion_dict(emotion_vector)
        
        # 排序并分析
        sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 找出显著情绪 (> 0.3)
        significant_emotions = [(name, value) for name, value in sorted_emotions if value > 0.3]
        
        # 情绪分类
        positive_emotions = ["快乐", "兴奋", "娱乐", "钦佩", "崇拜", "审美欣赏", "敬畏", "入迷", "兴趣", "浪漫"]
        negative_emotions = ["愤怒", "焦虑", "悲伤", "恐惧", "内疚", "恐怖", "失望", "厌恶", "嫉妒", "蔑视"]
        neutral_emotions = ["平静", "无聊", "困惑", "尴尬", "同情", "渴望", "怀旧"]
        
        positive_score = sum(emotion_dict[e] for e in positive_emotions if e in emotion_dict)
        negative_score = sum(emotion_dict[e] for e in negative_emotions if e in emotion_dict)
        neutral_score = sum(emotion_dict[e] for e in neutral_emotions if e in emotion_dict)
        
        return {
            "top_emotions": sorted_emotions[:5],
            "significant_emotions": significant_emotions,
            "emotion_balance": {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            },
            "overall_intensity": np.mean(emotion_vector),
            "max_emotion": sorted_emotions[0],
            "emotion_diversity": len(significant_emotions)
        }

def main():
    """演示知识图谱的使用"""
    print("🧠 情绪-音乐知识图谱演示")
    print("=" * 50)
    
    # 初始化知识图谱
    kg = KnowledgeGraph()
    
    # 测试场景1: 高焦虑状态
    print("\n🔍 测试场景1: 高焦虑状态")
    anxiety_vector = np.zeros(27)
    anxiety_vector[5] = 0.8  # 焦虑
    anxiety_vector[9] = 0.1  # 平静
    
    result1 = kg.get_music_search_parameters(anxiety_vector)
    print("情绪分析:", result1["emotion_context"])
    print("音乐描述:", result1["text_description"])
    
    # 测试场景2: 高平静状态
    print("\n🔍 测试场景2: 高平静状态")
    calm_vector = np.zeros(27)
    calm_vector[9] = 0.9   # 平静
    
    result2 = kg.get_music_search_parameters(calm_vector)
    print("情绪分析:", result2["emotion_context"])
    print("音乐描述:", result2["text_description"])
    
    # 测试场景3: 快乐兴奋混合
    print("\n🔍 测试场景3: 快乐兴奋状态")
    happy_vector = np.zeros(27)
    happy_vector[23] = 0.8  # 快乐
    happy_vector[18] = 0.6  # 兴奋
    
    result3 = kg.get_music_search_parameters(happy_vector)
    print("情绪分析:", result3["emotion_context"])
    print("音乐描述:", result3["text_description"])
    
    print("\n✅ 知识图谱演示完成!")

if __name__ == "__main__":
    main()