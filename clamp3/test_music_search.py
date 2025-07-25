#!/usr/bin/env python3
"""
音乐检索系统测试脚本
"""

import os
import sys
import json
import time
import random
from music_search_api import MusicSearchAPI

def test_basic_search():
    """基础搜索测试"""
    print("🧪 基础搜索测试")
    print("="*50)
    
    api = MusicSearchAPI()
    
    # 测试音频文件
    test_files = [
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_1min/32_1min_01.mp4",
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_3min/56_3min_05.mp4"
    ]
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  测试文件不存在: {test_file}")
            continue
            
        print(f"\n📁 测试文件: {os.path.basename(test_file)}")
        
        # 测试在不同版本中搜索
        for duration in ["1min", "3min"]:
            print(f"\n🔍 在 {duration} 版本中搜索...")
            
            start_time = time.time()
            result = api.search_by_video_file(test_file, duration, top_k=3)
            search_time = time.time() - start_time
            
            if result["success"]:
                print(f"✅ 搜索成功 (用时: {search_time:.2f}秒)")
                print(f"📊 结果数量: {result['total_results']}")
                
                for i, item in enumerate(result["results"], 1):
                    print(f"   {i}. {item['video_name']} - 相似度: {item['similarity']}")
            else:
                print(f"❌ 搜索失败: {result['error']}")

def test_partial_vs_full_audio():
    """测试部分音频 vs 完整音频的搜索效果"""
    print("\n🧪 部分音频 vs 完整音频测试")
    print("="*50)
    
    api = MusicSearchAPI()
    
    test_file = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_3min/32_3min_01.mp4"
    
    if not os.path.exists(test_file):
        print(f"⚠️  测试文件不存在: {test_file}")
        return
    
    print(f"📁 测试文件: {os.path.basename(test_file)}")
    
    # 测试部分音频
    print(f"\n🔍 使用前25%音频搜索...")
    result_partial = api.search_by_video_file(test_file, "3min", top_k=5, use_partial=True)
    
    # 测试完整音频
    print(f"\n🔍 使用完整音频搜索...")
    result_full = api.search_by_video_file(test_file, "3min", top_k=5, use_partial=False)
    
    # 比较结果
    print(f"\n📊 结果比较:")
    print(f"部分音频搜索结果:")
    if result_partial["success"]:
        for i, item in enumerate(result_partial["results"], 1):
            print(f"   {i}. {item['video_name']} - 相似度: {item['similarity']}")
    
    print(f"\n完整音频搜索结果:")
    if result_full["success"]:
        for i, item in enumerate(result_full["results"], 1):
            print(f"   {i}. {item['video_name']} - 相似度: {item['similarity']}")

def test_cross_duration_search():
    """测试跨时长版本搜索"""
    print("\n🧪 跨时长版本搜索测试")
    print("="*50)
    
    api = MusicSearchAPI()
    
    # 使用1分钟版本的音频在3分钟版本中搜索
    test_1min = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_1min/32_1min_01.mp4"
    
    if os.path.exists(test_1min):
        print(f"📁 使用1分钟版本音频: {os.path.basename(test_1min)}")
        print(f"🔍 在3分钟版本中搜索...")
        
        result = api.search_by_video_file(test_1min, "3min", top_k=3)
        
        if result["success"]:
            print(f"✅ 搜索成功")
            print(f"📊 找到的相似音乐:")
            for i, item in enumerate(result["results"], 1):
                print(f"   {i}. {item['video_name']} - 相似度: {item['similarity']}")
                
                # 检查是否找到了相同编号的音乐
                if "32_3min_01" in item["video_name"]:
                    print(f"   🎯 找到了对应的3分钟版本! 相似度: {item['similarity']}")
        else:
            print(f"❌ 搜索失败: {result['error']}")

def test_performance():
    """性能测试"""
    print("\n🧪 性能测试")
    print("="*50)
    
    api = MusicSearchAPI()
    
    # 随机选择几个测试文件
    test_files = [
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_1min/32_1min_02.mp4",
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_1min/56_1min_03.mp4",
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_3min/32_3min_04.mp4"
    ]
    
    available_files = [f for f in test_files if os.path.exists(f)]
    
    if not available_files:
        print("⚠️  没有可用的测试文件")
        return
    
    print(f"📁 测试文件数量: {len(available_files)}")
    
    total_time = 0
    success_count = 0
    
    for i, test_file in enumerate(available_files, 1):
        print(f"\n🔍 性能测试 {i}/{len(available_files)}: {os.path.basename(test_file)}")
        
        start_time = time.time()
        result = api.search_by_video_file(test_file, "3min", top_k=3)
        search_time = time.time() - start_time
        
        total_time += search_time
        
        if result["success"]:
            success_count += 1
            print(f"✅ 搜索成功 (用时: {search_time:.2f}秒)")
        else:
            print(f"❌ 搜索失败: {result['error']}")
    
    avg_time = total_time / len(available_files)
    success_rate = success_count / len(available_files) * 100
    
    print(f"\n📊 性能统计:")
    print(f"   总用时: {total_time:.2f}秒")
    print(f"   平均用时: {avg_time:.2f}秒/次")
    print(f"   成功率: {success_rate:.1f}%")

def test_feature_library_stats():
    """测试特征库统计信息"""
    print("\n🧪 特征库统计信息测试")
    print("="*50)
    
    api = MusicSearchAPI()
    
    stats_result = api.get_feature_library_stats()
    
    if stats_result["success"]:
        stats = stats_result["stats"]
        print(f"📊 特征库统计:")
        print(f"   总特征数: {stats['total_features']}")
        
        for duration, info in stats["by_duration"].items():
            print(f"   {duration}: {info['count']} 个特征")
            print(f"     文件列表: {', '.join(info['videos'][:5])}{'...' if len(info['videos']) > 5 else ''}")
    else:
        print(f"❌ 获取统计信息失败: {stats_result['error']}")

def main():
    """主函数"""
    print("🎵 音乐检索系统综合测试")
    print("="*60)
    
    try:
        # 1. 特征库统计信息测试
        test_feature_library_stats()
        
        # 2. 基础搜索测试
        test_basic_search()
        
        # 3. 部分音频 vs 完整音频测试
        test_partial_vs_full_audio()
        
        # 4. 跨时长版本搜索测试
        test_cross_duration_search()
        
        # 5. 性能测试
        test_performance()
        
        print(f"\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()