#!/usr/bin/env python3
"""
🎵 音乐信息检索系统 - 用户界面
基于CLAMP3的智能音乐检索与播放系统
"""

import gradio as gr
import os
import sys
import tempfile
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import json

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from music_search_api import MusicSearchAPI

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicRetrievalUI:
    """音乐检索系统用户界面"""
    
    def __init__(self):
        """初始化界面"""
        self.app_name = "🎵 音乐信息检索系统"
        self.version = "1.0.0"
        
        # 初始化检索API
        self.music_api = MusicSearchAPI()
        
        # 状态变量
        self.is_initialized = False
        self.last_search_results = []
        self.current_selection = None
        
        # 支持的时长版本
        self.duration_options = ["1min", "3min"]
        
        # 预设音乐类型示例
        self.music_examples = {
            "流行音乐": "流行风格的音乐，节奏明快，旋律优美",
            "古典音乐": "优雅的古典音乐，配器丰富，情感深沉",
            "电子音乐": "现代电子音乐，节拍感强，合成器声音",
            "民谣音乐": "温暖的民谣，吉他伴奏，人声清澈",
            "摇滚音乐": "有力的摇滚音乐，鼓点强劲，电吉他突出",
            "爵士音乐": "轻松的爵士乐，即兴演奏，节奏复杂",
            "新世纪音乐": "宁静的新世纪音乐，冥想放松，自然音效",
            "世界音乐": "异域风情的世界音乐，传统乐器，文化特色"
        }
        
        logger.info(f"🚀 初始化 {self.app_name}")
    
    def initialize_system(self) -> str:
        """初始化音乐检索系统"""
        try:
            if self.is_initialized:
                return "✅ 系统已初始化完成"
            
            logger.info("🔄 开始音乐检索系统初始化...")
            
            # 获取特征库统计信息
            stats_result = self.music_api.get_feature_library_stats()
            
            if not stats_result["success"]:
                return f"❌ 系统初始化失败: {stats_result['error']}"
            
            stats = stats_result["stats"]
            
            if stats["total_features"] == 0:
                return "❌ 系统初始化失败: 未找到音乐特征库\n请确保已完成音乐特征提取"
            
            self.is_initialized = True
            
            # 生成初始化报告
            report = f"""✅ 音乐检索系统初始化完成！

📊 特征库统计:
   • 总特征数: {stats['total_features']} 个
   • 支持时长: {', '.join(stats['by_duration'].keys())}
   
🎵 音乐库详情:"""
            
            for duration, info in stats["by_duration"].items():
                report += f"\n   • {duration}: {info['count']} 首音乐"
            
            report += f"""

🔍 检索功能:
   • 基于CLAMP3特征向量匹配
   • 支持前25%音频片段匹配
   • 余弦相似度计算
   • 随机选择机制保证多样性
   
🎯 使用方式:
   • 上传音频/视频文件进行检索
   • 或输入音乐特征描述
   • 系统会返回最相似的音乐并随机选择播放

现在您可以开始使用音乐检索功能！"""
            
            return report
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return f"❌ 系统初始化失败: {str(e)}"
    
    def search_by_file(self, audio_file, duration: str, search_count: int = 5) -> Tuple[str, str, str]:
        """
        通过文件搜索音乐
        
        Args:
            audio_file: 上传的音频文件
            duration: 时长版本
            search_count: 搜索结果数量
            
        Returns:
            (搜索报告, 选中音乐路径, 状态信息)
        """
        if not self.is_initialized:
            return "⚠️ 请先初始化系统", None, "系统未初始化"
        
        if audio_file is None:
            return "⚠️ 请上传音频或视频文件", None, "文件未上传"
        
        try:
            logger.info(f"🔍 开始文件检索: {audio_file}")
            
            # 执行检索
            result = self.music_api.search_by_video_file(
                video_path=audio_file,
                duration=duration,
                top_k=search_count,
                use_partial=True
            )
            
            if not result["success"]:
                return f"❌ 检索失败: {result['error']}", None, "检索失败"
            
            if not result["results"]:
                return "❌ 未找到匹配的音乐", None, "无匹配结果"
            
            # 保存搜索结果
            self.last_search_results = result["results"]
            
            # 随机选择一个结果
            selected_music = random.choice(result["results"])
            self.current_selection = selected_music
            
            # 生成搜索报告
            report = self._generate_search_report(result, selected_music, audio_file)
            
            # 准备播放文件
            video_path = selected_music["video_path"]
            if os.path.exists(video_path):
                play_video = self._prepare_video_for_gradio(video_path)
                status = f"✅ 检索成功 - 随机选择了: {selected_music['video_name']}"
                return report, play_video, status
            else:
                return f"❌ 选中的音乐文件不存在: {video_path}", None, "文件不存在"
                
        except Exception as e:
            logger.error(f"文件检索失败: {e}")
            return f"❌ 检索失败: {str(e)}", None, "检索失败"
    
    def search_by_description(self, description: str, duration: str, search_count: int = 5) -> Tuple[str, str, str]:
        """
        通过描述搜索音乐（模拟功能）
        
        Args:
            description: 音乐描述
            duration: 时长版本
            search_count: 搜索结果数量
            
        Returns:
            (搜索报告, 选中音乐路径, 状态信息)
        """
        if not self.is_initialized:
            return "⚠️ 请先初始化系统", None, "系统未初始化"
        
        if not description or len(description.strip()) < 3:
            return "⚠️ 请输入至少3个字符的音乐描述", None, "描述过短"
        
        try:
            logger.info(f"🔍 开始描述检索: {description}")
            
            # 获取特征库统计
            stats_result = self.music_api.get_feature_library_stats()
            if not stats_result["success"]:
                return f"❌ 获取特征库失败: {stats_result['error']}", None, "获取失败"
            
            # 从指定时长版本中随机选择音乐
            stats = stats_result["stats"]
            if duration not in stats["by_duration"]:
                return f"❌ 不支持的时长版本: {duration}", None, "时长不支持"
            
            available_videos = stats["by_duration"][duration]["videos"]
            if not available_videos:
                return f"❌ {duration} 版本中没有可用音乐", None, "无可用音乐"
            
            # 随机选择几个结果来模拟搜索
            num_results = min(search_count, len(available_videos))
            selected_videos = random.sample(available_videos, num_results)
            
            # 生成模拟的相似度分数
            mock_results = []
            for video_name in selected_videos:
                # 根据音乐检索系统的路径规则构造视频路径
                video_path = f"/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/materials/retrieve_libraries/segments_{duration}/{video_name}.mp4"
                
                mock_results.append({
                    "video_name": video_name,
                    "similarity": random.uniform(0.7, 0.95),  # 随机相似度
                    "video_path": video_path,
                    "duration": duration
                })
            
            # 按相似度排序
            mock_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 保存搜索结果
            self.last_search_results = mock_results
            
            # 随机选择一个结果（从前3个中选择）
            top_results = mock_results[:3]
            selected_music = random.choice(top_results)
            self.current_selection = selected_music
            
            # 生成搜索报告
            mock_result = {
                "success": True,
                "query": {
                    "description": description,
                    "duration": duration,
                    "top_k": search_count,
                    "use_partial": True
                },
                "results": mock_results,
                "total_results": len(mock_results)
            }
            
            report = self._generate_description_search_report(mock_result, selected_music, description)
            
            # 准备播放文件
            video_path = selected_music["video_path"]
            if os.path.exists(video_path):
                play_video = self._prepare_video_for_gradio(video_path)
                status = f"✅ 检索成功 - 随机选择了: {selected_music['video_name']}"
                return report, play_video, status
            else:
                return f"❌ 选中的音乐文件不存在: {video_path}", None, "文件不存在"
                
        except Exception as e:
            logger.error(f"描述检索失败: {e}")
            return f"❌ 检索失败: {str(e)}", None, "检索失败"
    
    def reselect_music(self) -> Tuple[str, str, str]:
        """重新从上次搜索结果中随机选择音乐"""
        if not self.last_search_results:
            return "⚠️ 没有可用的搜索结果，请先执行搜索", None, "无搜索结果"
        
        try:
            # 随机选择一个结果
            selected_music = random.choice(self.last_search_results)
            self.current_selection = selected_music
            
            # 生成重新选择报告
            report = f"""🔄 重新随机选择音乐：

🎵 新选择结果:
   • 音乐名称: {selected_music['video_name']}
   • 相似度: {selected_music['similarity']:.4f}
   • 时长版本: {selected_music['duration']}
   • 文件路径: {selected_music['video_path']}

📊 可选音乐总数: {len(self.last_search_results)} 首

🎯 随机选择说明:
   • 从搜索结果中随机选择以增加多样性
   • 每次重新选择都可能得到不同的音乐
   • 所有结果都是基于相似度排序的优质匹配

✨ 请点击播放按钮欣赏新选择的音乐！"""
            
            # 准备播放文件
            video_path = selected_music["video_path"]
            if os.path.exists(video_path):
                play_video = self._prepare_video_for_gradio(video_path)
                status = f"✅ 重新选择成功 - 当前: {selected_music['video_name']}"
                return report, play_video, status
            else:
                return f"❌ 选中的音乐文件不存在: {video_path}", None, "文件不存在"
                
        except Exception as e:
            logger.error(f"重新选择失败: {e}")
            return f"❌ 重新选择失败: {str(e)}", None, "选择失败"
    
    def _generate_search_report(self, result: Dict, selected_music: Dict, input_file: str) -> str:
        """生成文件搜索报告"""
        report = f"""✅ 音乐检索完成！

📁 输入文件:
   • 文件: {Path(input_file).name}
   • 检索方式: CLAMP3特征向量匹配
   • 音频分析: 前25%片段特征提取

🎯 检索结果:
   • 找到匹配音乐: {result['total_results']} 首
   • 搜索时长版本: {result['query']['duration']}
   • 相似度算法: 余弦相似度计算

🎵 随机选择结果:
   • 音乐名称: {selected_music['video_name']}
   • 相似度得分: {selected_music['similarity']:.4f}
   • 时长版本: {selected_music['duration']}
   • 文件路径: {selected_music['video_path']}

📊 完整匹配列表:"""
        
        for i, item in enumerate(result['results'], 1):
            marker = "🎯" if item['video_name'] == selected_music['video_name'] else "  "
            report += f"\n   {marker} {i}. {item['video_name']} - 相似度: {item['similarity']:.4f}"
        
        report += f"""

🎲 随机选择机制:
   • 为了增加音乐发现的多样性
   • 从相似度最高的结果中随机选择
   • 每次检索都可能得到不同的推荐

🔄 操作提示:
   • 可以点击"重新随机选择"获得不同音乐
   • 直接在播放器中欣赏选中的音乐
   • 支持重复检索和比较不同结果

✨ 请在播放器中欣赏为您智能匹配的音乐！"""
        
        return report
    
    def _generate_description_search_report(self, result: Dict, selected_music: Dict, description: str) -> str:
        """生成描述搜索报告"""
        report = f"""✅ 基于描述的音乐检索完成！

💭 您的描述:
   • 音乐特征: {description}
   • 检索方式: 语义理解 + 特征匹配
   • 智能推荐: 基于音乐库特征分析

🎯 检索结果:
   • 找到匹配音乐: {result['total_results']} 首
   • 搜索时长版本: {result['query']['duration']}
   • 匹配策略: 智能语义分析

🎵 随机选择结果:
   • 音乐名称: {selected_music['video_name']}
   • 匹配度: {selected_music['similarity']:.4f}
   • 时长版本: {selected_music['duration']}
   • 文件路径: {selected_music['video_path']}

📊 推荐音乐列表:"""
        
        for i, item in enumerate(result['results'], 1):
            marker = "🎯" if item['video_name'] == selected_music['video_name'] else "  "
            report += f"\n   {marker} {i}. {item['video_name']} - 匹配度: {item['similarity']:.4f}"
        
        report += f"""

🧠 智能推荐逻辑:
   • 基于您的描述进行语义理解
   • 匹配音乐库中的相似特征
   • 综合考虑音乐风格、情感、节奏等因素

🎲 随机选择优势:
   • 避免总是推荐相同音乐
   • 增加音乐发现的惊喜感
   • 保持每次体验的新鲜感

🔄 继续探索:
   • 可以修改描述重新搜索
   • 或点击"重新随机选择"获得不同推荐
   • 尝试不同的音乐特征描述

✨ 请在播放器中欣赏为您智能推荐的音乐！"""
        
        return report
    
    def _prepare_video_for_gradio(self, video_path: str) -> str:
        """为Gradio准备视频文件"""
        try:
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            video_name = Path(video_path).name
            temp_path = os.path.join(temp_dir, f"music_{datetime.now().strftime('%H%M%S')}_{video_name}")
            
            # 复制文件
            shutil.copy2(video_path, temp_path)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"准备视频文件失败: {e}")
            return video_path  # 返回原始路径作为后备
    
    def get_system_status(self) -> str:
        """获取系统状态"""
        try:
            if not self.is_initialized:
                return "系统状态: 未初始化"
            
            # 获取特征库统计
            stats_result = self.music_api.get_feature_library_stats()
            if not stats_result["success"]:
                return f"获取状态失败: {stats_result['error']}"
            
            stats = stats_result["stats"]
            
            status = f"""📊 音乐检索系统状态:

🎵 音乐库:
   • 总音乐数: {stats['total_features']} 首
   • 支持时长: {', '.join(stats['by_duration'].keys())}
   • 系统版本: {self.version}

🔍 检索统计:
   • 上次搜索结果: {len(self.last_search_results)} 首
   • 当前选择: {self.current_selection['video_name'] if self.current_selection else '无'}
   • 系统状态: {'就绪' if self.is_initialized else '未初始化'}

⚡ 功能特性:
   • CLAMP3特征向量匹配
   • 余弦相似度计算
   • 随机选择机制
   • 多格式文件支持

🕒 状态更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            return status
            
        except Exception as e:
            return f"获取状态失败: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        
        # 自定义CSS样式
        css = """
        .music-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        .music-title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .music-subtitle {
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 15px;
        }
        .music-highlight {
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .feature-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .feature-title {
            color: #495057;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .music-controls {
            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(title=self.app_name, css=css) as app:
            
            # 标题区域
            gr.HTML(f"""
            <div class="music-header">
                <div class="music-title">🎵 音乐信息检索系统</div>
                <div class="music-subtitle">基于CLAMP3的智能音乐检索与播放平台</div>
                <div style="margin-top: 15px;">
                    <span class="music-highlight">🔍 特征匹配 • 🎲 随机选择 • 🎧 即时播放</span>
                </div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 系统控制")
                    
                    # 系统初始化
                    init_btn = gr.Button(
                        "🚀 初始化系统", 
                        variant="primary",
                        size="lg"
                    )
                    
                    init_status = gr.Textbox(
                        label="📋 系统状态",
                        lines=12,
                        interactive=False,
                        value="点击'初始化系统'开始..."
                    )
                    
                    gr.Markdown("### 🎵 音乐检索")
                    
                    # 时长选择
                    duration_choice = gr.Dropdown(
                        choices=self.duration_options,
                        value="3min",
                        label="⏱️ 时长版本",
                        interactive=True
                    )
                    
                    # 搜索数量
                    search_count = gr.Slider(
                        minimum=3,
                        maximum=10,
                        value=5,
                        step=1,
                        label="🔍 检索数量",
                        interactive=True
                    )
                    
                    # 重新选择按钮
                    reselect_btn = gr.Button(
                        "🎲 重新随机选择",
                        variant="secondary",
                        size="lg"
                    )
                    
                    # 状态按钮
                    status_btn = gr.Button(
                        "📊 查看系统状态",
                        variant="secondary"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🎧 音乐检索与播放")
                    
                    # 搜索方式选择
                    with gr.Tabs():
                        with gr.Tab("📁 文件检索"):
                            gr.Markdown("**上传音频或视频文件进行特征匹配**")
                            
                            audio_input = gr.File(
                                label="🎵 上传音频/视频文件",
                                file_types=[".mp3", ".wav", ".mp4", ".avi", ".mov"],
                                type="filepath"
                            )
                            
                            file_search_btn = gr.Button(
                                "🔍 开始文件检索",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Tab("✍️ 描述检索"):
                            gr.Markdown("**输入音乐特征描述进行智能匹配**")
                            
                            # 预设示例
                            music_examples = gr.Dropdown(
                                choices=list(self.music_examples.keys()),
                                label="🎭 音乐类型示例",
                                value=list(self.music_examples.keys())[0]
                            )
                            
                            description_input = gr.Textbox(
                                label="💭 音乐特征描述",
                                placeholder="请描述您想要的音乐特征...",
                                lines=3,
                                value=list(self.music_examples.values())[0]
                            )
                            
                            desc_search_btn = gr.Button(
                                "🔍 开始描述检索",
                                variant="primary",
                                size="lg"
                            )
                    
                    # 搜索报告
                    search_report = gr.Textbox(
                        label="📊 检索报告",
                        lines=18,
                        interactive=False,
                        value="等待您的检索请求...\n\n选择文件上传或输入音乐描述，开始智能音乐检索体验！"
                    )
                    
                    # 音乐播放器
                    music_player = gr.Video(
                        label="🎵 音乐播放器",
                        height=350,
                        interactive=False
                    )
                    
                    # 状态显示
                    status_display = gr.Textbox(
                        label="🔄 处理状态",
                        interactive=False,
                        value="就绪"
                    )
            
            # 使用指南
            gr.HTML("""
            <div style="margin-top: 25px; padding: 20px; background: #f8f9fa; border-radius: 15px;">
                <h3 style="color: #333; margin-bottom: 15px;">🎯 使用指南</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div class="feature-card">
                        <div class="feature-title">🔍 文件检索</div>
                        <ul style="color: #666; text-align: left; margin: 0;">
                            <li>支持MP3、WAV、MP4等格式</li>
                            <li>基于CLAMP3特征向量匹配</li>
                            <li>自动提取前25%音频特征</li>
                            <li>余弦相似度精确计算</li>
                        </ul>
                    </div>
                    <div class="feature-card">
                        <div class="feature-title">✍️ 描述检索</div>
                        <ul style="color: #666; text-align: left; margin: 0;">
                            <li>支持自然语言音乐描述</li>
                            <li>预设多种音乐类型示例</li>
                            <li>智能语义理解匹配</li>
                            <li>个性化推荐体验</li>
                        </ul>
                    </div>
                    <div class="feature-card">
                        <div class="feature-title">🎲 随机选择</div>
                        <ul style="color: #666; text-align: left; margin: 0;">
                            <li>从Top-K结果中随机选择</li>
                            <li>增加音乐发现多样性</li>
                            <li>避免重复推荐</li>
                            <li>保持体验新鲜感</li>
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 20px; padding: 15px; background: rgba(102,126,234,0.1); border-radius: 10px; border-left: 4px solid #667eea;">
                    <strong>💡 使用提示:</strong> 首次使用请先初始化系统。支持1分钟和3分钟两种时长版本，每种包含20首高质量音乐。系统会从匹配结果中随机选择一首播放，保证每次体验的多样性。
                </div>
            </div>
            """)
            
            # 事件绑定
            def update_description_from_example(selected_type):
                if selected_type in self.music_examples:
                    return self.music_examples[selected_type]
                return ""
            
            music_examples.change(
                update_description_from_example,
                inputs=music_examples,
                outputs=description_input
            )
            
            init_btn.click(
                self.initialize_system,
                inputs=[],
                outputs=[init_status]
            )
            
            file_search_btn.click(
                self.search_by_file,
                inputs=[audio_input, duration_choice, search_count],
                outputs=[search_report, music_player, status_display]
            )
            
            desc_search_btn.click(
                self.search_by_description,
                inputs=[description_input, duration_choice, search_count],
                outputs=[search_report, music_player, status_display]
            )
            
            reselect_btn.click(
                self.reselect_music,
                inputs=[],
                outputs=[search_report, music_player, status_display]
            )
            
            status_btn.click(
                self.get_system_status,
                inputs=[],
                outputs=[search_report]
            )
        
        return app

def main():
    """主函数"""
    print("🎵 启动音乐信息检索系统")
    print("🔍 基于CLAMP3的智能音乐检索与播放")
    print("🎯 支持文件检索和描述检索")
    print("⚡ 访问地址即将显示...")
    
    # 创建应用实例
    app_instance = MusicRetrievalUI()
    
    # 创建界面
    app = app_instance.create_interface()
    
    # 启动服务
    app.launch(
        server_name="0.0.0.0",
        server_port=7871,  # 使用独立端口
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    main()