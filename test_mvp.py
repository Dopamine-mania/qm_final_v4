#!/usr/bin/env python3
"""
4.0版本MVP测试脚本
测试所有核心组件的基本功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from core.emotion_mapper import detect_emotion_enhanced, get_emotion_music_features
from core.video_processor import VideoProcessor
from core.feature_extractor import AudioFeatureExtractor
from core.retrieval_engine import VideoRetrievalEngine, TherapyVideoSelector

def test_emotion_recognition():
    """测试情绪识别"""
    print("🧠 测试情绪识别模块...")
    
    test_cases = [
        "我感到很焦虑，心跳加速，难以入睡",
        "太累了，身体和精神都很疲惫", 
        "今晚又开始胡思乱想，停不下来",
        "压力很大，感觉喘不过气",
        "比较平静，但希望更深层的放松"
    ]
    
    for i, text in enumerate(test_cases, 1):
        emotion, confidence = detect_emotion_enhanced(text)
        features = get_emotion_music_features(emotion)
        
        print(f"{i}. 输入: {text}")
        print(f"   情绪: {emotion} (置信度: {confidence:.1%})")
        print(f"   匹配阶段: {features['匹配阶段']['mood']}")
        print()
    
    print("✅ 情绪识别测试完成\\n")

def test_video_processor():
    """测试视频处理器"""
    print("🎬 测试视频处理模块...")
    
    processor = VideoProcessor()
    
    # 检查ffmpeg
    if not processor.check_ffmpeg_availability():
        print("❌ ffmpeg不可用，跳过视频处理测试")
        return False
    
    # 扫描视频
    videos = processor.scan_source_videos()
    print(f"发现视频文件: {len(videos)} 个")
    
    for video in videos:
        print(f"  - {video['file_name']}: {video['duration_formatted']}")
    
    if not videos:
        print("⚠️ 未找到视频文件，请确保视频位于 materials/video/ 目录下")
        return False
    
    print("✅ 视频处理测试完成\\n")
    return True

def test_feature_extractor():
    """测试特征提取器"""
    print("🎵 测试特征提取模块...")
    
    extractor = AudioFeatureExtractor()
    
    # 查找测试视频
    test_video = None
    video_dir = Path("materials/video")
    
    if video_dir.exists():
        for video_file in video_dir.glob("*.mp4"):
            test_video = str(video_file)
            break
    
    if not test_video or not os.path.exists(test_video):
        print("⚠️ 未找到测试视频文件，跳过特征提取测试")
        return False
    
    print(f"测试视频: {Path(test_video).name}")
    
    # 提取特征
    features = extractor.extract_video_features(test_video, extract_ratio=0.25)
    
    if features:
        print("✅ 特征提取成功")
        print(f"   时长: {features.get('duration', 0):.1f}秒")
        print(f"   RMS能量: {features.get('rms_energy', 0):.4f}")
        print(f"   频谱质心: {features.get('spectral_centroid', 0):.1f}Hz")
        print(f"   估计节拍: {features.get('tempo_estimate', 0):.1f}BPM")
        print(f"   亮度: {features.get('brightness', 0):.3f}")
        print(f"   温暖度: {features.get('warmth', 0):.3f}")
    else:
        print("❌ 特征提取失败")
        return False
    
    print("✅ 特征提取测试完成\\n")
    return True

def test_retrieval_engine():
    """测试检索引擎"""
    print("🔍 测试检索引擎模块...")
    
    engine = VideoRetrievalEngine()
    
    print(f"特征数据库: {len(engine.features_database)} 个视频")
    print(f"情绪数据库: {len(engine.emotion_database)} 种情绪")
    
    if len(engine.features_database) == 0:
        print("⚠️ 特征数据库为空，需要先运行特征提取")
        return False
    
    # 测试检索
    test_emotion = "焦虑"
    results = engine.retrieve_videos(test_emotion, top_k=3)
    
    print(f"\\n检索测试 - 情绪: {test_emotion}")
    if results:
        for i, (path, score, _) in enumerate(results, 1):
            print(f"  {i}. {Path(path).name}: {score:.3f}")
        print("✅ 检索测试成功")
    else:
        print("❌ 检索测试失败")
        return False
    
    print("✅ 检索引擎测试完成\\n")
    return True

def test_therapy_selector():
    """测试疗愈选择器"""
    print("🌙 测试疗愈选择器...")
    
    engine = VideoRetrievalEngine()
    selector = TherapyVideoSelector(engine)
    
    test_inputs = [
        "我感到很焦虑，心跳加速",
        "太累了，身体和精神都很疲惫"
    ]
    
    success_count = 0
    
    for user_input in test_inputs:
        result = selector.select_therapy_video(user_input)
        
        if result:
            print(f"输入: {user_input}")
            print(f"推荐: {result['video_name']}")
            print(f"情绪: {result['detected_emotion']} | 相似度: {result['similarity_score']:.3f}")
            print()
            success_count += 1
        else:
            print(f"❌ 选择失败: {user_input}")
    
    if success_count > 0:
        print(f"✅ 疗愈选择器测试完成 ({success_count}/{len(test_inputs)} 成功)\\n")
        return True
    else:
        print("❌ 疗愈选择器测试失败\\n")
        return False

def run_full_test():
    """运行完整测试"""
    print("=" * 60)
    print("🚀 音乐疗愈AI系统 4.0版本 - MVP测试")
    print("=" * 60)
    print()
    
    test_results = []
    
    # 1. 情绪识别测试
    try:
        test_emotion_recognition()
        test_results.append(("情绪识别", True))
    except Exception as e:
        print(f"❌ 情绪识别测试失败: {e}\\n")
        test_results.append(("情绪识别", False))
    
    # 2. 视频处理测试
    try:
        video_ok = test_video_processor()
        test_results.append(("视频处理", video_ok))
    except Exception as e:
        print(f"❌ 视频处理测试失败: {e}\\n")
        test_results.append(("视频处理", False))
    
    # 3. 特征提取测试
    try:
        feature_ok = test_feature_extractor()
        test_results.append(("特征提取", feature_ok))
    except Exception as e:
        print(f"❌ 特征提取测试失败: {e}\\n")
        test_results.append(("特征提取", False))
    
    # 4. 检索引擎测试
    try:
        retrieval_ok = test_retrieval_engine()
        test_results.append(("检索引擎", retrieval_ok))
    except Exception as e:
        print(f"❌ 检索引擎测试失败: {e}\\n")
        test_results.append(("检索引擎", False))
    
    # 5. 疗愈选择器测试
    try:
        selector_ok = test_therapy_selector()
        test_results.append(("疗愈选择器", selector_ok))
    except Exception as e:
        print(f"❌ 疗愈选择器测试失败: {e}\\n")
        test_results.append(("疗愈选择器", False))
    
    # 测试总结
    print("=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    success_count = 0
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:15} {status}")
        if success:
            success_count += 1
    
    print(f"\\n总体结果: {success_count}/{len(test_results)} 个测试通过")
    
    if success_count == len(test_results):
        print("🎉 所有测试通过！MVP已就绪！")
        return True
    elif success_count >= 3:
        print("⚠️ 大部分测试通过，MVP基本可用")
        return True
    else:
        print("❌ 多个测试失败，需要修复问题")
        return False

if __name__ == "__main__":
    success = run_full_test()
    
    if success:
        print("\\n🚀 启动建议:")
        print("1. 运行: python gradio_retrieval_4.0.py")
        print("2. 访问Web界面")
        print("3. 点击'初始化系统'")
        print("4. 开始体验智能疗愈视频推荐！")
    else:
        print("\\n🔧 修复建议:")
        print("1. 确保ffmpeg已安装")
        print("2. 确保视频文件位于 materials/video/ 目录")
        print("3. 安装所需依赖: pip install -r requirements.txt")
        print("4. 检查错误信息并修复问题")
    
    sys.exit(0 if success else 1)