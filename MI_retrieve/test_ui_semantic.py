#!/usr/bin/env python3
"""
测试UI界面的语义检索集成
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from music_retrieval_ui_simple import MusicRetrievalUI

def test_ui_semantic_integration():
    """测试UI界面的语义检索集成"""
    print("🧪 测试UI语义检索集成...")
    
    # 创建UI实例
    ui = MusicRetrievalUI()
    
    # 初始化系统
    print("\n🔄 初始化系统...")
    init_result = ui.initialize_system()
    print(init_result)
    
    if not ui.is_initialized:
        print("❌ 系统初始化失败")
        return
    
    # 测试语义检索功能
    print("\n🔍 测试语义检索功能...")
    
    test_descriptions = [
        "轻松愉悦的大调音乐，适合工作",
        "节奏缓慢的小调音乐，适合冥想放松",
        "tempo 120 BPM活泼明快"
    ]
    
    for i, description in enumerate(test_descriptions, 1):
        print(f"\n--- 测试 {i}: {description} ---")
        
        # 使用3分钟版本测试
        report, video_path = ui.search_by_description(description, "3min", 3)
        
        print("检索报告:")
        print(report[:500] + "..." if len(report) > 500 else report)
        
        if video_path:
            print(f"\n选中的音乐: {os.path.basename(video_path)}")
            print(f"文件存在: {os.path.exists(video_path)}")
        else:
            print("❌ 没有返回音乐文件")
        
        # 测试重新选择功能
        if ui.last_search_results:
            print(f"\n🔄 测试重新选择功能...")
            reselect_report, reselect_path = ui.reselect_music()
            
            if reselect_path:
                print(f"重新选择的音乐: {os.path.basename(reselect_path)}")
            else:
                print("❌ 重新选择失败")
    
    print("\n✅ UI语义检索集成测试完成!")

if __name__ == "__main__":
    test_ui_semantic_integration()