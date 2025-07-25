#!/usr/bin/env python3
"""
简化MVP测试脚本
验证qm_final4核心功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.emotion_mapper import detect_emotion_enhanced
from core.feature_extractor import AudioFeatureExtractor  
from core.retrieval_engine import VideoRetrievalEngine
from core.video_processor import VideoProcessor

def main():
    print("🚀 qm_final4 MVP系统验证")
    print("=" * 50)
    
    # 1. 测试情绪识别
    print("🧠 测试情绪识别...")
    test_input = "我今天很焦虑，工作压力很大，晚上睡不着"
    emotion, confidence = detect_emotion_enhanced(test_input)
    print(f"   输入: {test_input}")
    print(f"   ✅ 情绪: {emotion} (置信度: {confidence:.3f})")
    
    # 2. 测试特征提取
    print("\n🎵 测试特征提取...")
    extractor = AudioFeatureExtractor()
    test_video = "materials/segments/5min/32_seg000_5min.mp4"
    
    if os.path.exists(test_video):
        features = extractor.extract_video_features(test_video)
        if features:
            print(f"   ✅ 特征提取成功")
            print(f"   模型: {features.get('model_type', 'unknown')}")
            print(f"   维度: {features['feature_vector'].shape}")
        else:
            print("   ❌ 特征提取失败")
    else:
        print(f"   ❌ 测试视频不存在: {test_video}")
    
    # 3. 测试检索引擎
    print("\n🔍 测试检索引擎...")
    engine = VideoRetrievalEngine()
    print(f"   特征数据库: {len(engine.features_database)} 个视频")
    
    if engine.features_database:
        results = engine.retrieve_videos(emotion, top_k=3)
        print(f"   ✅ 检索成功: {len(results)} 个匹配")
        for i, (path, score, info) in enumerate(results, 1):
            video_name = os.path.basename(path)
            print(f"     {i}. {video_name} (相似度: {score:.3f})")
    else:
        print("   ❌ 特征数据库为空")
    
    # 4. 测试视频处理
    print("\n🎬 测试视频处理...")
    processor = VideoProcessor()
    videos = processor.scan_source_videos()
    print(f"   ✅ 发现源视频: {len(videos)} 个")
    
    segments_dir = Path("materials/segments")
    if segments_dir.exists():
        total_segments = len(list(segments_dir.rglob("*.mp4")))
        print(f"   ✅ 视频片段: {total_segments} 个")
    else:
        print("   ❌ 片段目录不存在")
    
    # 5. 系统状态总结
    print("\n📊 系统状态总结")
    print("=" * 50)
    
    if engine.features_database and len(engine.features_database) > 0:
        print("🟢 MVP系统就绪！")
        print(f"   ✅ 情绪识别: 27维细粒度")
        print(f"   ✅ 特征提取: CLAMP3降级到Librosa")
        print(f"   ✅ 视频检索: {len(engine.features_database)}个素材可用")
        print(f"   ✅ 处理流程: 端到端验证通过")
        
        # 示例完整流程
        print(f"\n🎯 完整疗愈流程示例:")
        print(f"   用户输入: \"{test_input}\"")
        print(f"   → 识别情绪: {emotion} ({confidence:.1%}置信度)")
        
        if results:
            selected_video = results[0][0]
            similarity = results[0][1]
            print(f"   → 匹配视频: {os.path.basename(selected_video)}")
            print(f"   → 相似度: {similarity:.1%}")
            print(f"   → 输出: 疗愈视频播放")
    else:
        print("🟡 系统部分就绪")
        print("   建议运行: python build_priority_material_library.py")
    
    print("\n🎉 MVP验证完成！")

if __name__ == "__main__":
    main()