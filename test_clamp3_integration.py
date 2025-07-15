#!/usr/bin/env python3
"""
CLAMP3集成测试脚本
验证CLAMP3特征提取器是否正常工作
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.video_processor import VideoProcessor
from core.feature_extractor import CLAMP3FeatureExtractor, AudioFeatureExtractor
from core.retrieval_engine import VideoRetrievalEngine
from core.emotion_mapper import detect_emotion_enhanced

# 设置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_clamp3_availability():
    """测试CLAMP3是否可用"""
    print("🔍 测试CLAMP3可用性...")
    
    try:
        extractor = CLAMP3FeatureExtractor()
        print("✅ CLAMP3特征提取器初始化成功")
        return True
    except Exception as e:
        print(f"❌ CLAMP3特征提取器初始化失败: {e}")
        return False

def test_video_processing():
    """测试视频处理功能"""
    print("\n🎬 测试视频处理...")
    
    processor = VideoProcessor()
    
    # 检查ffmpeg
    if not processor.check_ffmpeg_availability():
        print("❌ ffmpeg不可用，跳过视频处理测试")
        return False
    
    # 扫描视频文件
    videos = processor.scan_source_videos()
    if not videos:
        print("❌ 未找到视频文件")
        return False
    
    print(f"✅ 找到 {len(videos)} 个视频文件")
    for video in videos:
        print(f"   - {video['file_name']}: {video['duration_formatted']}")
    
    # 尝试切分视频（仅处理第一个视频的第一个片段）
    print("\n🔪 测试视频切分...")
    segments = processor.segment_videos(extract_intro_only=True)
    
    if not segments:
        print("❌ 视频切分失败")
        return False
    
    # 统计切分结果
    total_segments = sum(len(seg_list) for seg_list in segments.values())
    print(f"✅ 视频切分成功，共生成 {total_segments} 个片段")
    
    return True

def test_clamp3_feature_extraction():
    """测试CLAMP3特征提取"""
    print("\n🎵 测试CLAMP3特征提取...")
    
    try:
        extractor = CLAMP3FeatureExtractor()
        
        # 查找测试视频片段
        segments_dir = Path("materials/segments")
        test_video = None
        
        for duration_dir in segments_dir.iterdir():
            if duration_dir.is_dir():
                video_files = list(duration_dir.glob("*.mp4"))
                if video_files:
                    test_video = video_files[0]
                    break
        
        if not test_video:
            print("❌ 未找到测试视频片段")
            return False
        
        print(f"📁 测试视频: {test_video}")
        
        # 提取特征
        features = extractor.extract_video_features(str(test_video))
        
        if features:
            print("✅ CLAMP3特征提取成功")
            print(f"   特征向量形状: {features['clamp3_features'].shape}")
            print(f"   特征向量维度: {features['clamp3_features'].size}")
            print(f"   提取版本: {features['extractor_version']}")
            print(f"   模型类型: {features['model_type']}")
            return True
        else:
            print("❌ CLAMP3特征提取失败")
            return False
            
    except Exception as e:
        print(f"❌ CLAMP3特征提取异常: {e}")
        return False

def test_audio_feature_extractor():
    """测试音频特征提取器（包含CLAMP3后端）"""
    print("\n🎶 测试音频特征提取器...")
    
    try:
        extractor = AudioFeatureExtractor()
        
        # 查找测试视频片段
        segments_dir = Path("materials/segments")
        test_video = None
        
        for duration_dir in segments_dir.iterdir():
            if duration_dir.is_dir():
                video_files = list(duration_dir.glob("*.mp4"))
                if video_files:
                    test_video = video_files[0]
                    break
        
        if not test_video:
            print("❌ 未找到测试视频片段")
            return False
        
        print(f"📁 测试视频: {test_video}")
        
        # 提取特征
        features = extractor.extract_video_features(str(test_video))
        
        if features:
            print("✅ 音频特征提取成功")
            
            # 检查使用的后端
            if 'clamp3_features' in features:
                print("   后端: CLAMP3")
                print(f"   CLAMP3特征形状: {features['clamp3_features'].shape}")
            else:
                print("   后端: 传统特征提取")
                print(f"   传统特征数量: {len([k for k in features.keys() if isinstance(features[k], (int, float))])}")
            
            print(f"   提取版本: {features['extractor_version']}")
            return True
        else:
            print("❌ 音频特征提取失败")
            return False
            
    except Exception as e:
        print(f"❌ 音频特征提取异常: {e}")
        return False

def test_retrieval_engine():
    """测试检索引擎"""
    print("\n🔍 测试检索引擎...")
    
    try:
        engine = VideoRetrievalEngine()
        
        # 测试情绪检测
        test_emotion = "悲伤"
        emotion, confidence = detect_emotion_enhanced("我今天心情不好，很沮丧")
        print(f"情绪检测结果: {emotion} (置信度: {confidence:.3f})")
        
        # 尝试检索
        results = engine.retrieve_videos(emotion, top_k=3)
        
        if results:
            print(f"✅ 检索成功，找到 {len(results)} 个匹配视频")
            for i, (video_path, similarity, video_info) in enumerate(results, 1):
                print(f"   {i}. {Path(video_path).name} - 相似度: {similarity:.3f}")
                
                # 检查特征类型
                if 'clamp3_features' in video_info:
                    print(f"      特征类型: CLAMP3")
                else:
                    print(f"      特征类型: 传统")
            
            return True
        else:
            print("❌ 检索失败或无匹配结果")
            return False
            
    except Exception as e:
        print(f"❌ 检索引擎异常: {e}")
        return False

def test_complete_pipeline():
    """测试完整的4.0流水线"""
    print("\n🔄 测试完整流水线...")
    
    try:
        # 1. 视频处理
        processor = VideoProcessor()
        videos = processor.scan_source_videos()
        
        if not videos:
            print("❌ 未找到视频文件")
            return False
        
        # 2. 特征提取
        extractor = AudioFeatureExtractor()
        
        # 获取一些视频片段进行测试
        segments_5min = processor.get_segments_by_duration(5)
        if not segments_5min:
            print("❌ 未找到5分钟视频片段")
            return False
        
        # 提取前3个片段的特征
        test_segments = segments_5min[:3]
        features_db = extractor.extract_batch_features(test_segments)
        
        if not features_db:
            print("❌ 批量特征提取失败")
            return False
        
        print(f"✅ 批量特征提取成功，处理 {len(features_db)} 个视频")
        
        # 3. 检索测试
        engine = VideoRetrievalEngine()
        engine.features_database = features_db
        
        # 测试几种情绪
        test_emotions = ["快乐", "悲伤", "平静", "兴奋"]
        
        for emotion in test_emotions:
            results = engine.retrieve_videos(emotion, top_k=2)
            if results:
                print(f"   {emotion}: 找到 {len(results)} 个匹配")
            else:
                print(f"   {emotion}: 无匹配结果")
        
        print("✅ 完整流水线测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 完整流水线测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始CLAMP3集成测试")
    print("=" * 50)
    
    # 测试列表
    tests = [
        ("CLAMP3可用性", test_clamp3_availability),
        ("视频处理", test_video_processing),
        ("CLAMP3特征提取", test_clamp3_feature_extraction),
        ("音频特征提取器", test_audio_feature_extractor),
        ("检索引擎", test_retrieval_engine),
        ("完整流水线", test_complete_pipeline)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                failed += 1
                print(f"❌ {test_name} 失败")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} 异常: {e}")
        
        print("-" * 30)
    
    print("=" * 50)
    print(f"测试总结: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过！CLAMP3集成成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查日志")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)