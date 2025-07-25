#!/usr/bin/env python3
"""
音乐检索API - 为外部系统提供简单的音乐搜索接口
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any
from music_search_system import MusicSearchSystem

class MusicSearchAPI:
    """音乐检索API类"""
    
    def __init__(self):
        """初始化API"""
        self.search_system = MusicSearchSystem()
        print("✅ 音乐检索API初始化完成")
    
    def search_by_audio_file(self, audio_path: str, duration: str = "3min", 
                           top_k: int = 3, use_partial: bool = True) -> Dict[str, Any]:
        """
        通过音频文件搜索相似音乐
        
        Args:
            audio_path: 音频文件路径
            duration: 搜索版本 ("1min" 或 "3min")
            top_k: 返回结果数量
            use_partial: 是否只使用前25%音频
            
        Returns:
            搜索结果字典
        """
        try:
            # 验证参数
            if not os.path.exists(audio_path):
                return {
                    "success": False,
                    "error": f"音频文件不存在: {audio_path}",
                    "results": []
                }
            
            supported_durations = ["1min", "3min", "5min", "10min", "20min", "30min"]
            if duration not in supported_durations:
                return {
                    "success": False,
                    "error": f"不支持的时长版本: {duration}，仅支持 {', '.join(supported_durations)}",
                    "results": []
                }
            
            # 执行搜索
            results = self.search_system.search_music_by_file(
                audio_path=audio_path,
                duration=duration,
                top_k=top_k,
                use_partial=use_partial
            )
            
            # 格式化结果
            formatted_results = []
            for video_name, similarity in results:
                video_path = self.search_system.get_video_path(video_name, duration)
                formatted_results.append({
                    "video_name": video_name,
                    "similarity": round(similarity, 4),
                    "video_path": video_path,
                    "duration": duration
                })
            
            return {
                "success": True,
                "query": {
                    "audio_path": audio_path,
                    "duration": duration,
                    "top_k": top_k,
                    "use_partial": use_partial
                },
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def search_by_video_file(self, video_path: str, duration: str = "3min", 
                           top_k: int = 3, use_partial: bool = True) -> Dict[str, Any]:
        """
        通过视频文件搜索相似音乐（自动提取音频）
        
        Args:
            video_path: 视频文件路径
            duration: 搜索版本
            top_k: 返回结果数量
            use_partial: 是否只使用前25%音频
            
        Returns:
            搜索结果字典
        """
        try:
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"视频文件不存在: {video_path}",
                    "results": []
                }
            
            # 直接使用视频文件路径（MusicSearchSystem会自动提取音频）
            return self.search_by_audio_file(video_path, duration, top_k, use_partial)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def search_by_description(self, description: str, duration: str = "3min", 
                            top_k: int = 5) -> Dict[str, Any]:
        """
        通过文本描述搜索相似音乐 (真正的语义检索)
        
        Args:
            description: 音乐特征描述
            duration: 搜索版本
            top_k: 返回结果数量
            
        Returns:
            搜索结果字典
        """
        try:
            # 验证参数
            if not description or len(description.strip()) < 3:
                return {
                    "success": False,
                    "error": "描述内容过短，请提供至少3个字符的描述",
                    "results": []
                }
            
            supported_durations = ["1min", "3min", "5min", "10min", "20min", "30min"]
            if duration not in supported_durations:
                return {
                    "success": False,
                    "error": f"不支持的时长版本: {duration}，仅支持 {', '.join(supported_durations)}",
                    "results": []
                }
            
            # 执行语义检索
            results = self.search_system.search_music_by_text(
                text_description=description,
                duration=duration,
                top_k=top_k
            )
            
            # 格式化结果
            formatted_results = []
            for video_name, similarity in results:
                video_path = self.search_system.get_video_path(video_name, duration)
                formatted_results.append({
                    "video_name": video_name,
                    "similarity": round(similarity, 4),
                    "video_path": video_path,
                    "duration": duration
                })
            
            return {
                "success": True,
                "query": {
                    "description": description,
                    "duration": duration,
                    "top_k": top_k,
                    "method": "semantic_search"
                },
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def get_feature_library_stats(self) -> Dict[str, Any]:
        """获取特征库统计信息"""
        try:
            stats = self.search_system.get_statistics()
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stats": {}
            }
    
    def batch_search(self, audio_files: List[str], duration: str = "3min", 
                    top_k: int = 3, use_partial: bool = True) -> List[Dict[str, Any]]:
        """
        批量搜索音乐
        
        Args:
            audio_files: 音频文件路径列表
            duration: 搜索版本
            top_k: 返回结果数量
            use_partial: 是否只使用前25%音频
            
        Returns:
            搜索结果列表
        """
        results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n📁 批量搜索进度: {i}/{len(audio_files)}")
            result = self.search_by_audio_file(audio_file, duration, top_k, use_partial)
            results.append(result)
        
        return results

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="音乐检索API命令行工具")
    
    parser.add_argument("--audio", "-a", type=str, help="音频文件路径")
    parser.add_argument("--video", "-v", type=str, help="视频文件路径")
    parser.add_argument("--text", "-t", type=str, help="文本描述")
    parser.add_argument("--duration", "-d", type=str, default="5min", 
                       choices=["1min", "3min", "5min", "10min", "20min", "30min"], help="搜索版本")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="返回结果数量")
    parser.add_argument("--full-audio", action="store_true", help="使用完整音频（默认使用前25%）")
    parser.add_argument("--stats", action="store_true", help="显示特征库统计信息")
    parser.add_argument("--output", "-o", type=str, help="输出结果到JSON文件")
    
    args = parser.parse_args()
    
    # 初始化API
    api = MusicSearchAPI()
    
    # 显示统计信息
    if args.stats:
        stats_result = api.get_feature_library_stats()
        print("\n📊 特征库统计信息:")
        print(json.dumps(stats_result, indent=2, ensure_ascii=False))
        return
    
    # 执行搜索
    result = None
    use_partial = not args.full_audio
    
    if args.audio:
        print(f"\n🔍 搜索音频: {args.audio}")
        result = api.search_by_audio_file(args.audio, args.duration, args.top_k, use_partial)
    elif args.video:
        print(f"\n🔍 搜索视频: {args.video}")
        result = api.search_by_video_file(args.video, args.duration, args.top_k, use_partial)
    elif args.text:
        print(f"\n🔍 语义搜索: {args.text}")
        result = api.search_by_description(args.text, args.duration, args.top_k)
    else:
        print("❌ 请指定音频文件 (--audio)、视频文件 (--video) 或文本描述 (--text)")
        return
    
    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果已保存到: {args.output}")
    else:
        print("\n🎯 搜索结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()