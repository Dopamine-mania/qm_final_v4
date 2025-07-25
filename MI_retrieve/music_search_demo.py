#!/usr/bin/env python3
"""
音乐检索系统使用示例
"""

import os
import sys
from music_search_api import MusicSearchAPI

def demo_search_by_video():
    """演示通过视频文件搜索音乐"""
    print("🎵 音乐检索系统使用示例")
    print("="*50)
    
    # 初始化API
    api = MusicSearchAPI()
    
    # 示例1: 使用1分钟视频在3分钟版本中搜索
    print("\n📺 示例1: 跨时长版本搜索")
    print("-" * 30)
    
    video_1min = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/materials/retrieve_libraries/segments_1min/32_1min_01.mp4"
    
    if os.path.exists(video_1min):
        print(f"🔍 使用1分钟视频搜索3分钟版本的相似音乐")
        print(f"输入: {os.path.basename(video_1min)}")
        
        result = api.search_by_video_file(
            video_path=video_1min,
            duration="3min",
            top_k=3,
            use_partial=True  # 使用前25%音频
        )
        
        if result["success"]:
            print(f"\n✅ 搜索成功! 找到 {result['total_results']} 个相似音乐:")
            
            for i, item in enumerate(result["results"], 1):
                print(f"  {i}. {item['video_name']}")
                print(f"     相似度: {item['similarity']:.4f}")
                print(f"     文件路径: {item['video_path']}")
                print()
        else:
            print(f"❌ 搜索失败: {result['error']}")
    
    # 示例2: 比较部分音频vs完整音频的搜索效果
    print("\n📺 示例2: 部分音频 vs 完整音频对比")
    print("-" * 30)
    
    video_3min = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/materials/retrieve_libraries/segments_3min/56_3min_05.mp4"
    
    if os.path.exists(video_3min):
        print(f"🔍 使用同一个3分钟视频进行对比测试")
        print(f"输入: {os.path.basename(video_3min)}")
        
        # 使用前25%音频搜索
        print(f"\n🔹 使用前25%音频搜索:")
        result_partial = api.search_by_video_file(
            video_path=video_3min,
            duration="3min",
            top_k=3,
            use_partial=True
        )
        
        if result_partial["success"]:
            for i, item in enumerate(result_partial["results"], 1):
                print(f"  {i}. {item['video_name']} - 相似度: {item['similarity']:.4f}")
        
        # 使用完整音频搜索
        print(f"\n🔹 使用完整音频搜索:")
        result_full = api.search_by_video_file(
            video_path=video_3min,
            duration="3min",
            top_k=3,
            use_partial=False
        )
        
        if result_full["success"]:
            for i, item in enumerate(result_full["results"], 1):
                print(f"  {i}. {item['video_name']} - 相似度: {item['similarity']:.4f}")
    
    # 示例3: 查看特征库统计信息
    print("\n📊 示例3: 特征库统计信息")
    print("-" * 30)
    
    stats_result = api.get_feature_library_stats()
    if stats_result["success"]:
        stats = stats_result["stats"]
        print(f"📁 特征库概况:")
        print(f"  总特征数: {stats['total_features']}")
        
        for duration, info in stats["by_duration"].items():
            print(f"  {duration} 版本: {info['count']} 个音乐特征")

def interactive_demo():
    """交互式演示"""
    print("\n🎮 交互式音乐搜索")
    print("="*50)
    
    api = MusicSearchAPI()
    
    # 列出可用的测试文件
    test_files_1min = [
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/materials/retrieve_libraries/segments_1min/32_1min_01.mp4",
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/materials/retrieve_libraries/segments_1min/56_1min_03.mp4"
    ]
    
    test_files_3min = [
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/materials/retrieve_libraries/segments_3min/32_3min_02.mp4",
        "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/materials/retrieve_libraries/segments_3min/56_3min_05.mp4"
    ]
    
    all_files = [(f, "1min") for f in test_files_1min if os.path.exists(f)] + \
                [(f, "3min") for f in test_files_3min if os.path.exists(f)]
    
    if not all_files:
        print("❌ 没有找到可用的测试文件")
        return
    
    print("📁 可用的测试文件:")
    for i, (file_path, duration) in enumerate(all_files, 1):
        print(f"  {i}. {os.path.basename(file_path)} ({duration})")
    
    try:
        choice = input(f"\n请选择一个文件 (1-{len(all_files)}): ")
        choice_idx = int(choice) - 1
        
        if 0 <= choice_idx < len(all_files):
            selected_file, file_duration = all_files[choice_idx]
            
            print(f"\n🎯 您选择了: {os.path.basename(selected_file)}")
            
            # 选择搜索版本
            search_duration = input("搜索版本 (1min/3min, 默认3min): ").strip() or "3min"
            if search_duration not in ["1min", "3min"]:
                search_duration = "3min"
            
            # 选择是否使用部分音频
            use_partial_input = input("使用前25%音频? (y/n, 默认y): ").strip().lower()
            use_partial = use_partial_input != 'n'
            
            print(f"\n🔍 开始搜索...")
            print(f"  文件: {os.path.basename(selected_file)}")
            print(f"  搜索版本: {search_duration}")
            print(f"  使用音频: {'前25%' if use_partial else '完整'}")
            
            result = api.search_by_video_file(
                video_path=selected_file,
                duration=search_duration,
                top_k=5,
                use_partial=use_partial
            )
            
            if result["success"]:
                print(f"\n🎉 搜索完成! 找到 {result['total_results']} 个相似音乐:")
                for i, item in enumerate(result["results"], 1):
                    print(f"  {i}. {item['video_name']} - 相似度: {item['similarity']:.4f}")
            else:
                print(f"❌ 搜索失败: {result['error']}")
        else:
            print("❌ 无效的选择")
            
    except (ValueError, KeyboardInterrupt):
        print("\n👋 退出交互式演示")

def main():
    """主函数"""
    print("🎵 音乐检索系统演示程序")
    print("🎯 展示基于CLAMP3的音乐相似度搜索功能")
    print("📊 当前支持1分钟和3分钟版本，共40个音乐特征")
    print("="*60)
    
    try:
        # 运行基础演示
        demo_search_by_video()
        
        # 询问是否运行交互式演示
        if input("\n🎮 是否运行交互式演示? (y/n): ").strip().lower() == 'y':
            interactive_demo()
        
        print("\n🎉 演示完成!")
        print("\n💡 使用说明:")
        print("  1. 支持mp4视频和wav/mp3音频文件")
        print("  2. 自动提取前25%音频用于快速搜索")
        print("  3. 返回相似度最高的前3个结果")
        print("  4. 支持跨时长版本搜索")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()