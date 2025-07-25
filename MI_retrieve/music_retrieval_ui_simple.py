#!/usr/bin/env python3
"""
🎵 音乐信息检索系统 - 简化版界面
基于CLAMP3的智能音乐检索与播放系统
"""

import gradio as gr
import os
import sys
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

try:
    from music_search_api import MusicSearchAPI
except ImportError:
    print("❌ 无法导入 music_search_api，请确保文件存在")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicRetrievalUI:
    """音乐检索系统用户界面"""
    
    def __init__(self):
        """初始化界面"""
        self.app_name = "🎵 音乐疗愈检索系统"
        self.version = "1.0.0"
        
        # 初始化检索API
        try:
            self.music_api = MusicSearchAPI()
        except Exception as e:
            logger.error(f"初始化MusicSearchAPI失败: {e}")
            self.music_api = None
        
        # 状态变量
        self.is_initialized = False
        self.last_search_results = []
        self.current_selection = None
        
        # 支持的时长版本 (目前只有1min和3min有特征文件)
        self.duration_options = ["1min", "3min"]
        
        # 预设音乐类型示例
        self.music_examples = {
            "疗愈音乐(示例)": "建议初始节奏为 90-110 BPM，调式倾向小调或不确定性，和声包含轻微不协和，音色可能略显尖锐",
            "放松冥想": "节奏缓慢平稳，60-80 BPM，大调为主，和声简单纯净，音色温暖柔和，适合深度放松",
            "专注工作": "节奏中等稳定，100-120 BPM，调式中性，和声不过于复杂，音色清晰不分散注意力",
            "情绪疏导": "节奏渐进变化，80-100 BPM，小调转大调，和声层次丰富，音色表现力强",
            "睡前安眠": "节奏极慢，50-70 BPM，大调柔和，和声简单，音色轻柔如耳语",
            "焦虑缓解": "节奏规律稳定，70-90 BPM，避免突然变化，和声稳定，音色温暖包容",
            "活力提升": "节奏明快活泼，120-140 BPM，大调明亮，和声丰富，音色清新有活力",
            "深度思考": "节奏慢而深沉，60-80 BPM，小调神秘，和声复杂层次，音色富有内涵"
        }
        
        logger.info(f"🚀 初始化 {self.app_name}")
    
    def initialize_system(self) -> str:
        """初始化音乐检索系统"""
        try:
            if self.is_initialized:
                return "✅ 系统已初始化完成"
            
            if self.music_api is None:
                return "❌ 系统初始化失败: MusicSearchAPI 未能正确初始化"
            
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
   • 基于CLAMP3跨模态语义理解
   • 真正的文本-音频特征匹配
   • 768维向量空间余弦相似度计算
   • 智能降级机制确保稳定性
   • 随机选择机制保证多样性
   
🎯 使用方式:
   • 输入音乐特征描述（支持中英文混合）
   • 系统进行语义理解并匹配最相似音乐
   • 自动从高相似度结果中随机选择播放

🧠 语义检索技术:
   • CLAMP3多模态模型：文本特征提取
   • 跨模态相似度计算：文本↔音频语义匹配
   • 非关键词匹配：真正理解音乐语义描述

现在您可以开始使用真正的语义音乐检索功能！"""
            
            return report
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return f"❌ 系统初始化失败: {str(e)}"
    
    def search_by_description(self, description: str, duration: str, search_count: int = 5) -> Tuple[str, Optional[str]]:
        """通过描述搜索音乐 - 使用真正的语义检索"""
        if not self.is_initialized:
            return "⚠️ 请先初始化系统", None
        
        if not description or len(description.strip()) < 3:
            return "⚠️ 请输入至少3个字符的音乐描述", None
        
        if self.music_api is None:
            return "❌ 音乐检索API未初始化", None
        
        try:
            logger.info(f"🔍 开始语义检索: {description}")
            
            # 使用真正的语义检索API
            result = self.music_api.search_by_description(
                description=description,
                duration=duration,
                top_k=search_count
            )
            
            if not result["success"]:
                return f"❌ 语义检索失败: {result['error']}", None
            
            if not result["results"]:
                return f"❌ 没有找到匹配的音乐，请尝试其他描述", None
            
            # 转换结果格式并添加正确的路径
            semantic_results = []
            for item in result["results"]:
                video_path = f"/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/retrieve_libraries/segments_{duration}/{item['video_name']}.mp4"
                
                semantic_results.append({
                    "video_name": item["video_name"],
                    "similarity": item["similarity"],
                    "video_path": video_path,
                    "duration": duration,
                    "method": "semantic_search"
                })
            
            # 保存搜索结果
            self.last_search_results = semantic_results
            
            # 从前3个结果中随机选择一个
            top_results = semantic_results[:min(3, len(semantic_results))]
            selected_music = random.choice(top_results)
            self.current_selection = selected_music
            
            # 生成搜索报告
            search_method = "CLAMP3语义检索" if result["query"].get("method") == "semantic_search" else "简化版语义检索"
            
            report = f"""✅ 语义音乐检索完成！

💭 您的描述:
   • 音乐特征: {description}
   • 检索方式: {search_method}
   • 智能推荐: 基于跨模态特征匹配

🎯 检索结果:
   • 找到匹配音乐: {len(semantic_results)} 首
   • 搜索时长版本: {duration}
   • 匹配策略: 真正的语义理解 + 特征向量相似度

🎵 随机选择结果:
   • 音乐名称: {selected_music['video_name']}
   • 相似度: {selected_music['similarity']:.4f}
   • 时长版本: {selected_music['duration']}
   • 检索方法: {selected_music.get('method', 'semantic_search')}

📊 语义匹配列表:"""
            
            for i, item in enumerate(semantic_results, 1):
                marker = "🎯" if item['video_name'] == selected_music['video_name'] else "  "
                report += f"\n   {marker} {i}. {item['video_name']} - 相似度: {item['similarity']:.4f}"
            
            report += f"""

🧠 语义检索技术:
   • 基于CLAMP3多模态模型的文本特征提取
   • 与音频特征向量进行余弦相似度计算
   • 跨模态语义理解，非关键词匹配
   • 智能降级机制确保稳定性

✨ 请在播放器中欣赏为您语义匹配的音乐！"""
            
            # 准备播放文件
            video_path = selected_music["video_path"]
            if os.path.exists(video_path):
                return report, video_path
            else:
                return f"❌ 选中的音乐文件不存在: {video_path}", None
                
        except Exception as e:
            logger.error(f"描述检索失败: {e}")
            return f"❌ 检索失败: {str(e)}", None
    
    
    def reselect_music(self) -> Tuple[str, Optional[str]]:
        """重新从上次搜索结果中随机选择音乐"""
        if not self.last_search_results:
            return "⚠️ 没有可用的搜索结果，请先执行搜索", None
        
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
                return report, video_path
            else:
                return f"❌ 选中的音乐文件不存在: {video_path}", None
                
        except Exception as e:
            logger.error(f"重新选择失败: {e}")
            return f"❌ 重新选择失败: {str(e)}", None
    
    def create_interface(self) -> gr.Blocks:
        """创建简化版Gradio界面"""
        
        with gr.Blocks(title=self.app_name, theme=gr.themes.Soft()) as app:
            
            # 标题
            gr.Markdown(f"""
            # 🎵 音乐疗愈检索系统
            
            基于CLAMP3的智能音乐检索与播放平台 | 专为疗愈音乐体验设计
            
            🧠 语义理解 • 🔍 跨模态匹配 • 🎲 智能随机 • 🎧 即时播放 • 🧘 疗愈体验
            """)
            
            # 系统初始化
            with gr.Row():
                init_btn = gr.Button("🚀 初始化系统", variant="primary")
                
            init_status = gr.Textbox(
                label="📋 系统状态",
                lines=15,
                interactive=False,
                value="点击'初始化系统'开始..."
            )
            
            # 音乐检索界面
            with gr.Row():
                with gr.Column(scale=1):
                    # 时长选择
                    duration_choice = gr.Dropdown(
                        choices=self.duration_options,
                        value="3min",
                        label="⏱️ 时长版本"
                    )
                    
                    # 预设示例
                    music_examples = gr.Dropdown(
                        choices=list(self.music_examples.keys()),
                        label="🎭 疗愈音乐类型示例",
                        value=list(self.music_examples.keys())[0]
                    )
                    
                    # 重新选择按钮
                    reselect_btn = gr.Button("🎲 重新随机选择", variant="secondary")
                
                with gr.Column(scale=2):
                    # 音乐特征描述输入
                    description_input = gr.Textbox(
                        label="💭 音乐特征描述",
                        placeholder="例如：建议初始节奏为 90-110 BPM，调式倾向小调或不确定性，和声包含轻微不协和，音色可能略显尖锐",
                        lines=4,
                        value=list(self.music_examples.values())[0]
                    )
                    
                    # 搜索按钮
                    search_btn = gr.Button("🔍 开始音乐检索", variant="primary")
            
            # 搜索报告
            search_report = gr.Textbox(
                label="📊 检索报告",
                lines=20,
                interactive=False,
                value="等待您的检索请求..."
            )
            
            # 音乐播放器
            music_player = gr.Video(
                label="🎵 音乐播放器",
                height=400
            )
            
            # 使用指南
            gr.Markdown("""
            ## 🎯 使用指南
            
            ### 🚀 快速开始
            1. **初始化系统**: 点击"初始化系统"按钮
            2. **选择时长**: 选择1分钟或3分钟版本
            3. **输入描述**: 选择预设示例或自定义描述
            4. **开始检索**: 点击"开始音乐检索"
            5. **欣赏音乐**: 在播放器中直接播放
            
            ### 🎲 智能随机机制
            - 系统通过语义检索找到最相似的音乐，从前3个结果中随机选择
            - 既保证语义相似度，又增加发现新音乐的惊喜感
            - 可点击"重新随机选择"体验不同音乐
            
            ### 🧠 语义检索优势
            - **真正理解**: 不是关键词匹配，而是理解音乐语义
            - **跨模态**: 文本描述直接与音频特征匹配
            - **智能降级**: CLAMP3优先，依赖冲突时自动切换简化版
            - **稳定可靠**: 无论何种情况都能提供语义检索
            
            **💡 提示:** 目前支持1分钟和3分钟两种时长版本，每个版本包含20首精选疗愈音乐。现在支持真正的语义检索！
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
            
            search_btn.click(
                self.search_by_description,
                inputs=[description_input, duration_choice],
                outputs=[search_report, music_player]
            )
            
            reselect_btn.click(
                self.reselect_music,
                inputs=[],
                outputs=[search_report, music_player]
            )
        
        return app

def main():
    """主函数"""
    print("🎵 启动音乐疗愈检索系统")
    print("🔍 基于CLAMP3的智能音乐检索与播放")
    print("🎯 支持描述检索和智能匹配")
    
    # 创建应用实例
    app_instance = MusicRetrievalUI()
    
    # 创建界面
    app = app_instance.create_interface()
    
    # 启动服务
    app.launch(
        server_name="0.0.0.0",
        server_port=7873,
        share=True,  # 开启公网分享
        show_error=True,
        debug=False,
        inbrowser=True,  # 自动打开浏览器
        allowed_paths=[
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/retrieve_libraries/segments_1min",
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/retrieve_libraries/segments_3min",
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/retrieve_libraries/segments_5min",
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/retrieve_libraries/segments_10min",
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/retrieve_libraries/segments_20min",
            "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/retrieve_libraries/segments_30min"
        ]
    )

if __name__ == "__main__":
    main()