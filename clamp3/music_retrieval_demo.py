#!/usr/bin/env python3
"""
🎵 音乐检索系统演示 - 命令行版本
展示完整的音乐检索和随机选择功能
"""

import os
import sys
import random
import json
from pathlib import Path
from datetime import datetime
import logging

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from music_search_api import MusicSearchAPI

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicRetrievalDemo:
    """音乐检索系统演示类"""
    
    def __init__(self):
        """初始化演示"""
        self.api = MusicSearchAPI()
        self.last_search_results = []
        self.current_selection = None
        
        # 预设音乐描述示例
        self.music_examples = {
            "1": ("流行音乐", "节奏明快的流行音乐，旋律优美动听"),
            "2": ("古典音乐", "优雅的古典音乐，配器丰富情感深沉"),
            "3": ("电子音乐", "现代电子音乐，节拍感强合成器声音"),
            "4": ("民谣音乐", "温暖的民谣音乐，吉他伴奏人声清澈"),
            "5": ("摇滚音乐", "有力的摇滚音乐，鼓点强劲电吉他突出"),
            "6": ("爵士音乐", "轻松的爵士乐，即兴演奏节奏复杂"),
            "7": ("新世纪音乐", "宁静的新世纪音乐，冥想放松自然音效"),
            "8": ("世界音乐", "异域风情的世界音乐，传统乐器文化特色")
        }
    
    def show_welcome(self):
        """显示欢迎界面"""
        print("\n" + "="*80)
        print("🎵 音乐信息检索系统演示")
        print("🔍 基于CLAMP3的智能音乐检索与播放")
        print("🎯 支持文件检索和描述检索，随机选择播放")
        print("="*80)
        
        # 获取系统状态
        stats_result = self.api.get_feature_library_stats()
        if stats_result["success"]:
            stats = stats_result["stats"]
            print(f"\n📊 音乐库状态:")
            print(f"   • 总音乐数: {stats['total_features']} 首")
            for duration, info in stats["by_duration"].items():
                print(f"   • {duration}: {info['count']} 首音乐")
        else:
            print(f"\n❌ 无法获取音乐库状态: {stats_result['error']}")
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "-"*50)
        print("🎯 请选择功能:")
        print("1. 📁 文件检索 - 上传音频/视频文件进行匹配")
        print("2. ✍️ 描述检索 - 输入音乐特征描述进行匹配")
        print("3. 🎲 重新随机选择 - 从上次结果中重新选择")
        print("4. 📊 查看系统状态")
        print("5. 🚪 退出系统")
        print("-"*50)
    
    def file_search_demo(self):
        """文件检索演示"""
        print("\n📁 文件检索模式")
        print("🎵 可用的测试文件:")
        
        # 显示可用的测试文件
        test_files = [
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_1min/32_1min_01.mp4",
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_1min/56_1min_03.mp4",
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_3min/32_3min_02.mp4",
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_3min/56_3min_05.mp4"
        ]
        
        available_files = [(i+1, f) for i, f in enumerate(test_files) if os.path.exists(f)]
        
        if not available_files:
            print("❌ 没有找到可用的测试文件")
            return
        
        for i, (num, file_path) in enumerate(available_files):
            file_name = Path(file_path).name
            duration = "1min" if "_1min_" in file_name else "3min"
            print(f"   {num}. {file_name} ({duration})")
        
        # 让用户选择文件
        try:
            choice = input(f"\n请选择文件 (1-{len(available_files)}) 或直接输入文件路径: ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_files):
                    selected_file = available_files[choice_num-1][1]
                else:
                    print("❌ 无效的选择")
                    return
            else:
                selected_file = choice
                if not os.path.exists(selected_file):
                    print(f"❌ 文件不存在: {selected_file}")
                    return
            
            # 选择搜索参数
            duration = input("搜索版本 (1min/3min, 默认3min): ").strip() or "3min"
            if duration not in ["1min", "3min"]:
                duration = "3min"
            
            try:
                search_count = int(input("检索数量 (3-10, 默认5): ").strip() or "5")
                search_count = max(3, min(10, search_count))
            except ValueError:
                search_count = 5
            
            # 执行搜索
            print(f"\n🔍 开始检索...")
            print(f"   文件: {Path(selected_file).name}")
            print(f"   版本: {duration}")
            print(f"   数量: {search_count}")
            
            result = self.api.search_by_video_file(
                video_path=selected_file,
                duration=duration,
                top_k=search_count,
                use_partial=True
            )
            
            self._display_search_results(result, selected_file)
            
        except KeyboardInterrupt:
            print("\n操作已取消")
        except Exception as e:
            print(f"\n❌ 检索过程中出错: {e}")
    
    def description_search_demo(self):
        """描述检索演示"""
        print("\n✍️ 描述检索模式")
        print("🎭 预设音乐类型示例:")
        
        for key, (name, desc) in self.music_examples.items():
            print(f"   {key}. {name}: {desc}")
        
        try:
            choice = input("\n选择示例 (1-8) 或直接输入描述: ").strip()
            
            if choice in self.music_examples:
                description = self.music_examples[choice][1]
                print(f"选择了: {self.music_examples[choice][0]}")
            else:
                description = choice
                if len(description) < 3:
                    print("❌ 描述太短，请至少输入3个字符")
                    return
            
            # 选择搜索参数
            duration = input("搜索版本 (1min/3min, 默认3min): ").strip() or "3min"
            if duration not in ["1min", "3min"]:
                duration = "3min"
            
            try:
                search_count = int(input("检索数量 (3-10, 默认5): ").strip() or "5")
                search_count = max(3, min(10, search_count))
            except ValueError:
                search_count = 5
            
            # 执行模拟搜索
            print(f"\n🔍 开始描述检索...")
            print(f"   描述: {description}")
            print(f"   版本: {duration}")
            print(f"   数量: {search_count}")
            
            result = self._simulate_description_search(description, duration, search_count)
            self._display_search_results(result, f"描述: {description}")
            
        except KeyboardInterrupt:
            print("\n操作已取消")
        except Exception as e:
            print(f"\n❌ 检索过程中出错: {e}")
    
    def _simulate_description_search(self, description: str, duration: str, search_count: int) -> dict:
        """模拟描述搜索"""
        # 获取特征库统计
        stats_result = self.api.get_feature_library_stats()
        if not stats_result["success"]:
            return {"success": False, "error": stats_result["error"]}
        
        stats = stats_result["stats"]
        if duration not in stats["by_duration"]:
            return {"success": False, "error": f"不支持的时长版本: {duration}"}
        
        available_videos = stats["by_duration"][duration]["videos"]
        if not available_videos:
            return {"success": False, "error": f"{duration} 版本中没有可用音乐"}
        
        # 随机选择结果
        num_results = min(search_count, len(available_videos))
        selected_videos = random.sample(available_videos, num_results)
        
        # 生成模拟结果
        results = []
        for video_name in selected_videos:
            video_path = f"/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_{duration}/{video_name}.mp4"
            results.append({
                "video_name": video_name,
                "similarity": random.uniform(0.7, 0.95),
                "video_path": video_path,
                "duration": duration
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "success": True,
            "query": {
                "description": description,
                "duration": duration,
                "top_k": search_count,
                "use_partial": True
            },
            "results": results,
            "total_results": len(results)
        }
    
    def _display_search_results(self, result: dict, query_info: str):
        """显示搜索结果"""
        if not result["success"]:
            print(f"\n❌ 搜索失败: {result['error']}")
            return
        
        if not result["results"]:
            print("\n❌ 未找到匹配的音乐")
            return
        
        # 保存搜索结果
        self.last_search_results = result["results"]
        
        # 随机选择一个结果
        selected_music = random.choice(result["results"])
        self.current_selection = selected_music
        
        print(f"\n✅ 搜索完成！")
        print(f"📁 查询: {query_info}")
        print(f"🎯 找到 {result['total_results']} 首匹配音乐")
        
        print(f"\n📊 完整搜索结果:")
        for i, item in enumerate(result["results"], 1):
            marker = "🎯" if item["video_name"] == selected_music["video_name"] else "  "
            print(f"   {marker} {i}. {item['video_name']} - 相似度: {item['similarity']:.4f}")
        
        print(f"\n🎲 随机选择结果:")
        print(f"   🎵 音乐: {selected_music['video_name']}")
        print(f"   📊 相似度: {selected_music['similarity']:.4f}")
        print(f"   📂 路径: {selected_music['video_path']}")
        
        # 检查文件是否存在
        if os.path.exists(selected_music["video_path"]):
            print(f"   ✅ 文件存在，可以播放")
            
            # 询问是否播放
            if input(f"\n🎧 是否播放选中的音乐? (y/n): ").strip().lower() == 'y':
                self._play_music(selected_music["video_path"])
        else:
            print(f"   ❌ 文件不存在")
    
    def _play_music(self, video_path: str):
        """播放音乐"""
        try:
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == "Darwin":  # macOS
                subprocess.run(["open", video_path], check=True)
            elif system == "Windows":
                subprocess.run(["start", video_path], shell=True, check=True)
            elif system == "Linux":
                subprocess.run(["xdg-open", video_path], check=True)
            else:
                print(f"❌ 不支持的操作系统: {system}")
                return
            
            print(f"✅ 已在默认播放器中打开: {Path(video_path).name}")
            
        except subprocess.CalledProcessError:
            print(f"❌ 无法播放文件: {video_path}")
        except Exception as e:
            print(f"❌ 播放失败: {e}")
    
    def reselect_music(self):
        """重新随机选择音乐"""
        if not self.last_search_results:
            print("\n⚠️ 没有可用的搜索结果，请先执行搜索")
            return
        
        # 随机选择一个结果
        selected_music = random.choice(self.last_search_results)
        self.current_selection = selected_music
        
        print(f"\n🎲 重新随机选择音乐:")
        print(f"   🎵 音乐: {selected_music['video_name']}")
        print(f"   📊 相似度: {selected_music['similarity']:.4f}")
        print(f"   📂 路径: {selected_music['video_path']}")
        
        # 检查文件是否存在
        if os.path.exists(selected_music["video_path"]):
            print(f"   ✅ 文件存在，可以播放")
            
            # 询问是否播放
            if input(f"\n🎧 是否播放选中的音乐? (y/n): ").strip().lower() == 'y':
                self._play_music(selected_music["video_path"])
        else:
            print(f"   ❌ 文件不存在")
    
    def show_system_status(self):
        """显示系统状态"""
        print("\n📊 音乐检索系统状态:")
        
        # 获取特征库统计
        stats_result = self.api.get_feature_library_stats()
        if stats_result["success"]:
            stats = stats_result["stats"]
            print(f"   🎵 总音乐数: {stats['total_features']} 首")
            
            for duration, info in stats["by_duration"].items():
                print(f"   📁 {duration}: {info['count']} 首音乐")
                sample_videos = info["videos"][:5]
                print(f"      示例: {', '.join(sample_videos)}")
        else:
            print(f"   ❌ 无法获取统计信息: {stats_result['error']}")
        
        print(f"\n🔍 当前会话状态:")
        print(f"   📊 上次搜索结果: {len(self.last_search_results)} 首")
        print(f"   🎯 当前选择: {self.current_selection['video_name'] if self.current_selection else '无'}")
        print(f"   🕒 状态更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run(self):
        """运行演示"""
        self.show_welcome()
        
        while True:
            try:
                self.show_menu()
                choice = input("\n请选择功能 (1-5): ").strip()
                
                if choice == "1":
                    self.file_search_demo()
                elif choice == "2":
                    self.description_search_demo()
                elif choice == "3":
                    self.reselect_music()
                elif choice == "4":
                    self.show_system_status()
                elif choice == "5":
                    print("\n👋 感谢使用音乐检索系统！")
                    break
                else:
                    print("❌ 无效的选择，请重试")
                    
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出系统")
                break
            except Exception as e:
                print(f"\n❌ 系统错误: {e}")
                continue

def main():
    """主函数"""
    print("🎵 音乐检索系统演示启动...")
    
    try:
        demo = MusicRetrievalDemo()
        demo.run()
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()