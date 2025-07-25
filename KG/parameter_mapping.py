#!/usr/bin/env python3
"""
参数映射工具 (Parameter Mapping)

处理KG模块和MI_retrieve模块之间的参数格式转换
提供音乐参数的标准化映射和文本描述生成
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParameterMapper:
    """
    音乐参数映射器
    
    负责在不同模块间转换音乐参数格式
    支持结构化参数、自然语言描述、数值参数之间的转换
    """
    
    def __init__(self):
        """初始化参数映射器"""
        
        # 节拍范围映射
        self.tempo_mapping = {
            "very_slow": (40, 60),
            "slow": (60, 80),
            "medium": (80, 100), 
            "fast": (100, 120),
            "very_fast": (120, 160)
        }
        
        # 调式映射
        self.mode_mapping = {
            "minor": (0.0, 0.4),      # 小调
            "neutral": (0.4, 0.6),    # 中性
            "major": (0.6, 1.0)       # 大调
        }
        
        # 动态映射
        self.dynamics_mapping = {
            "very_soft": (0.0, 0.2),
            "soft": (0.2, 0.4),
            "medium": (0.4, 0.6),
            "loud": (0.6, 0.8),
            "very_loud": (0.8, 1.0)
        }
        
        # 和声映射
        self.harmony_mapping = {
            "dissonant": (0.0, 0.3),
            "mixed": (0.3, 0.7),
            "consonant": (0.7, 1.0)
        }
        
        # 音域映射
        self.register_mapping = {
            "low": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "high": (0.7, 1.0)
        }
        
        # 密度映射
        self.density_mapping = {
            "sparse": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "dense": (0.7, 1.0)
        }
        
        # 音色描述映射
        self.timbre_descriptions = {
            'neutral_pad': '音色中性柔和，适合各种情绪状态',
            'warm_pad': '音色温暖包容，营造安全感',
            'soft_choir': '音色轻柔如人声合唱，抚慰心灵',
            'gentle_piano': '音色清雅如钢琴，纯净透明',
            'nature_sounds': '音色自然清新，如山水鸟鸣',
            'ambient_pad': '音色环境化氛围，空间感强',
            'bright_ensemble': '音色明亮丰富，层次感强',
            'energetic_mix': '音色充满活力，振奋精神',
            'expressive_strings': '音色表现力强，情感丰富',
            'vintage_warmth': '音色怀旧温暖，nostalgic质感',
            'interesting_textures': '音色富有质感，引人入胜',
            'healing_bells': '音色治愈钟声，净化心灵',
            'meditative_drone': '音色冥想持续音，深度放松',
            'crystalline_shimmer': '音色水晶般闪烁，清透明亮'
        }
        
        # 情绪包络方向描述
        self.envelope_descriptions = {
            'steady': '情绪包络平稳持续',
            'ascending': '情绪包络逐步上升',
            'descending': '情绪包络逐渐下降',
            'calming': '情绪包络渐趋平静',
            'energizing': '情绪包络逐步活跃',
            'gentle_rise': '情绪包络轻缓上升',
            'uplifting': '情绪包络提升振奋',
            'nostalgic_wave': '情绪包络怀旧波动',
            'engaging': '情绪包络吸引注意',
            'neutral': '情绪包络保持中性'
        }
        
        logger.info("🗺️  参数映射器初始化完成")
    
    def kg_to_text_description(self, kg_params: Dict[str, Any]) -> str:
        """
        将KG参数转换为自然语言描述
        
        Args:
            kg_params: KG模块的音乐参数
            
        Returns:
            自然语言描述字符串
        """
        try:
            descriptions = []
            
            # 节拍描述
            tempo = kg_params.get('tempo', 80)
            tempo_desc = self._tempo_to_description(tempo)
            descriptions.append(tempo_desc)
            
            # 调式描述
            mode = kg_params.get('mode', 0.5)
            mode_desc = self._mode_to_description(mode)
            descriptions.append(mode_desc)
            
            # 和声描述
            harmony = kg_params.get('harmony_consonance', 0.5)
            harmony_desc = self._harmony_to_description(harmony)
            descriptions.append(harmony_desc)
            
            # 音色描述
            timbre = kg_params.get('timbre_preference', 'neutral_pad')
            timbre_desc = self.timbre_descriptions.get(timbre, '音色特色鲜明')
            descriptions.append(timbre_desc)
            
            # 动态描述
            dynamics = kg_params.get('dynamics', 0.5)
            dynamics_desc = self._dynamics_to_description(dynamics)
            descriptions.append(dynamics_desc)
            
            # 音域描述
            register = kg_params.get('pitch_register', 0.5)
            register_desc = self._register_to_description(register)
            descriptions.append(register_desc)
            
            # 密度描述 (可选)
            density = kg_params.get('density', 0.5)
            if density < 0.3 or density > 0.7:  # 只在显著时添加
                density_desc = self._density_to_description(density)
                descriptions.append(density_desc)
            
            # 组合描述
            result = "，".join(descriptions)
            
            logger.info(f"📝 参数转换完成: {len(descriptions)} 个特征")
            return result
            
        except Exception as e:
            logger.error(f"❌ 参数转文本失败: {e}")
            return "节拍适中，调式中性，音色温和，适合放松"
    
    def kg_to_structured_params(self, kg_params: Dict[str, Any]) -> Dict[str, str]:
        """
        将KG参数转换为结构化参数
        
        Args:
            kg_params: KG模块的音乐参数
            
        Returns:
            结构化参数字典
        """
        try:
            structured = {}
            
            # 节拍分类
            tempo = kg_params.get('tempo', 80)
            structured['tempo'] = self._classify_tempo(tempo)
            structured['tempo_bpm'] = f"{tempo:.0f}"
            
            # 调式分类
            mode = kg_params.get('mode', 0.5)
            structured['mode'] = self._classify_mode(mode)
            
            # 动态分类
            dynamics = kg_params.get('dynamics', 0.5)
            structured['dynamics'] = self._classify_dynamics(dynamics)
            
            # 和声分类
            harmony = kg_params.get('harmony_consonance', 0.5)
            structured['harmony'] = self._classify_harmony(harmony)
            
            # 音域分类
            register = kg_params.get('pitch_register', 0.5)
            structured['register'] = self._classify_register(register)
            
            # 密度分类
            density = kg_params.get('density', 0.5)
            structured['density'] = self._classify_density(density)
            
            # 音色保持原样
            structured['timbre'] = kg_params.get('timbre_preference', 'neutral_pad')
            
            # 情绪包络
            envelope = kg_params.get('emotional_envelope_direction', 'neutral')
            structured['envelope'] = envelope
            
            return structured
            
        except Exception as e:
            logger.error(f"❌ 结构化参数转换失败: {e}")
            return {
                'tempo': 'medium',
                'mode': 'neutral', 
                'dynamics': 'medium',
                'harmony': 'mixed',
                'timbre': 'neutral_pad'
            }
    
    def text_to_kg_params(self, text_description: str) -> Dict[str, Any]:
        """
        从文本描述反向推导KG参数 (实验性功能)
        
        Args:
            text_description: 文本描述
            
        Returns:
            KG参数字典
        """
        # 这是一个简化的实现，实际应用中可能需要更复杂的NLP处理
        kg_params = {
            'tempo': 80.0,
            'mode': 0.5,
            'dynamics': 0.5,
            'harmony_consonance': 0.5,
            'timbre_preference': 'neutral_pad',
            'pitch_register': 0.5,
            'density': 0.5,
            'emotional_envelope_direction': 'neutral'
        }
        
        text_lower = text_description.lower()
        
        # 节拍识别
        if 'bpm' in text_lower:
            import re
            bpm_match = re.search(r'(\d+)\s*bpm', text_lower)
            if bpm_match:
                kg_params['tempo'] = float(bpm_match.group(1))
        
        # 调式识别
        if '大调' in text_description or 'major' in text_lower:
            kg_params['mode'] = 0.8
        elif '小调' in text_description or 'minor' in text_lower:
            kg_params['mode'] = 0.2
        
        # 和声识别
        if '协和' in text_description or 'consonant' in text_lower:
            kg_params['harmony_consonance'] = 0.8
        elif '不协和' in text_description or 'dissonant' in text_lower:
            kg_params['harmony_consonance'] = 0.2
        
        # 动态识别
        if '轻柔' in text_description or 'soft' in text_lower:
            kg_params['dynamics'] = 0.3
        elif '响亮' in text_description or 'loud' in text_lower:
            kg_params['dynamics'] = 0.8
        
        return kg_params
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证参数的有效性
        
        Args:
            params: 待验证的参数
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必需参数
        required_params = ['tempo', 'mode', 'dynamics', 'harmony_consonance']
        for param in required_params:
            if param not in params:
                errors.append(f"缺少必需参数: {param}")
        
        # 检查数值范围
        if 'tempo' in params:
            tempo = params['tempo']
            if not isinstance(tempo, (int, float)) or tempo < 30 or tempo > 200:
                errors.append(f"tempo 值超出合理范围 [30, 200]: {tempo}")
        
        for param in ['mode', 'dynamics', 'harmony_consonance', 'pitch_register', 'density']:
            if param in params:
                value = params[param]
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    errors.append(f"{param} 值必须在 [0, 1] 范围内: {value}")
        
        return len(errors) == 0, errors
    
    def _tempo_to_description(self, tempo: float) -> str:
        """节拍数值转描述"""
        if tempo < 50:
            return f"建议初始节拍为 {tempo:.0f} BPM，节奏极其缓慢，深度冥想"
        elif tempo < 70:
            return f"建议初始节拍为 {tempo:.0f} BPM，节奏非常缓慢，安抚镇静"
        elif tempo < 85:
            return f"建议初始节拍为 {tempo:.0f} BPM，节奏缓慢放松"
        elif tempo < 100:
            return f"建议初始节拍为 {tempo:.0f} BPM，节奏适中稳定"
        elif tempo < 115:
            return f"建议初始节拍为 {tempo:.0f} BPM，节奏明快轻松"
        elif tempo < 130:
            return f"建议初始节拍为 {tempo:.0f} BPM，节奏活泼有力"
        else:
            return f"建议初始节拍为 {tempo:.0f} BPM，节奏快速激励"
    
    def _mode_to_description(self, mode: float) -> str:
        """调式数值转描述"""
        if mode > 0.8:
            return "调式明确大调，明亮积极向上"
        elif mode > 0.6:
            return "调式倾向大调，温暖乐观"
        elif mode < 0.2:
            return "调式明确小调，深沉内省"
        elif mode < 0.4:
            return "调式倾向小调，略显忧郁"
        else:
            return "调式中性平衡或调式变化丰富"
    
    def _harmony_to_description(self, harmony: float) -> str:
        """和声数值转描述"""
        if harmony > 0.8:
            return "和声高度协和，纯净安全"
        elif harmony > 0.6:
            return "和声相对协和，温暖稳定"
        elif harmony < 0.3:
            return "和声包含不协和，张力表现"
        elif harmony < 0.5:
            return "和声略显不协和，情感色彩"
        else:
            return "和声复杂多变，层次丰富"
    
    def _dynamics_to_description(self, dynamics: float) -> str:
        """动态数值转描述"""
        if dynamics > 0.8:
            return "音量饱满有力"
        elif dynamics > 0.6:
            return "音量适中清晰"
        elif dynamics < 0.3:
            return "音量轻柔细腻"
        elif dynamics < 0.5:
            return "音量偏轻，温和亲切"
        else:
            return "音量变化自然"
    
    def _register_to_description(self, register: float) -> str:
        """音域数值转描述"""
        if register > 0.8:
            return "音域偏高，明亮清透"
        elif register > 0.6:
            return "音域中高，清晰明朗"
        elif register < 0.2:
            return "音域很低，深沉厚重"
        elif register < 0.4:
            return "音域偏低，温暖稳定"
        else:
            return "音域适中，平衡舒适"
    
    def _density_to_description(self, density: float) -> str:
        """密度数值转描述"""
        if density > 0.8:
            return "音乐密度很高，丰富饱满"
        elif density > 0.6:
            return "音乐密度较高，层次丰富"
        elif density < 0.2:
            return "音乐密度很低，简约纯净"
        elif density < 0.4:
            return "音乐密度较低，空间感强"
        else:
            return "音乐密度适中"
    
    def _classify_tempo(self, tempo: float) -> str:
        """节拍分类"""
        for category, (min_val, max_val) in self.tempo_mapping.items():
            if min_val <= tempo < max_val:
                return category
        return "medium"
    
    def _classify_mode(self, mode: float) -> str:
        """调式分类"""
        for category, (min_val, max_val) in self.mode_mapping.items():
            if min_val <= mode <= max_val:
                return category
        return "neutral"
    
    def _classify_dynamics(self, dynamics: float) -> str:
        """动态分类"""
        for category, (min_val, max_val) in self.dynamics_mapping.items():
            if min_val <= dynamics <= max_val:
                return category
        return "medium"
    
    def _classify_harmony(self, harmony: float) -> str:
        """和声分类"""
        for category, (min_val, max_val) in self.harmony_mapping.items():
            if min_val <= harmony <= max_val:
                return category
        return "mixed"
    
    def _classify_register(self, register: float) -> str:
        """音域分类"""
        for category, (min_val, max_val) in self.register_mapping.items():
            if min_val <= register <= max_val:
                return category
        return "medium"
    
    def _classify_density(self, density: float) -> str:
        """密度分类"""
        for category, (min_val, max_val) in self.density_mapping.items():
            if min_val <= density <= max_val:
                return category
        return "medium"
    
    def get_mapping_info(self) -> Dict[str, Any]:
        """获取映射信息"""
        return {
            "tempo_ranges": self.tempo_mapping,
            "mode_ranges": self.mode_mapping,
            "dynamics_ranges": self.dynamics_mapping,
            "harmony_ranges": self.harmony_mapping,
            "register_ranges": self.register_mapping,
            "density_ranges": self.density_mapping,
            "timbre_options": list(self.timbre_descriptions.keys()),
            "envelope_options": list(self.envelope_descriptions.keys())
        }

def main():
    """演示参数映射器"""
    print("🗺️  参数映射器演示")
    print("=" * 50)
    
    # 初始化映射器
    mapper = ParameterMapper()
    
    # 测试KG参数
    test_kg_params = {
        'tempo': 95.0,
        'mode': 0.3,  # 小调
        'dynamics': 0.4,  # 偏轻
        'harmony_consonance': 0.7,  # 较协和
        'timbre_preference': 'gentle_piano',
        'pitch_register': 0.6,  # 偏高
        'density': 0.3,  # 较稀疏
        'emotional_envelope_direction': 'calming'
    }
    
    print("🧪 测试KG参数:")
    for key, value in test_kg_params.items():
        print(f"   {key}: {value}")
    
    # 转换为文本描述
    print(f"\n📝 文本描述:")
    text_desc = mapper.kg_to_text_description(test_kg_params)
    print(f"   {text_desc}")
    
    # 转换为结构化参数
    print(f"\n🏗️  结构化参数:")
    structured = mapper.kg_to_structured_params(test_kg_params)
    for key, value in structured.items():
        print(f"   {key}: {value}")
    
    # 参数验证
    print(f"\n✅ 参数验证:")
    is_valid, errors = mapper.validate_parameters(test_kg_params)
    print(f"   有效性: {is_valid}")
    if errors:
        for error in errors:
            print(f"   错误: {error}")
    
    # 显示映射信息
    print(f"\n📊 映射范围信息:")
    mapping_info = mapper.get_mapping_info()
    print(f"   节拍范围: {len(mapping_info['tempo_ranges'])} 个类别")
    print(f"   音色选项: {len(mapping_info['timbre_options'])} 种")
    print(f"   包络选项: {len(mapping_info['envelope_options'])} 种")
    
    print(f"\n✅ 参数映射器演示完成!")

if __name__ == "__main__":
    main()