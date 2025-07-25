#!/usr/bin/env python3
"""
🌙 音乐疗愈AI系统 4.0版本 - 检索驱动界面
基于预制疗愈视频素材库的智能检索系统
"""

import gradio as gr
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

# 添加core模块到路径
sys.path.append(str(Path(__file__).parent))

# 注意：此文件依赖外部core模块，如需独立运行，请使用music_retrieval_ui_simple.py
# from core.emotion_mapper import detect_emotion_enhanced, get_emotion_music_features
# from core.video_processor import VideoProcessor
# from core.feature_extractor import AudioFeatureExtractor
# from core.retrieval_engine import VideoRetrievalEngine, TherapyVideoSelector

print("⚠️  gradio_retrieval_4.0.py 需要外部core模块支持")
print("💡 如需独立运行，请使用: python music_retrieval_ui_simple.py")
sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapyApp4:
    """
    4.0版本疗愈应用
    基于检索的智能疗愈视频推荐系统
    """
    
    def __init__(self):
        """初始化应用"""
        self.app_name = "🌙 音乐疗愈AI系统 4.0 - 检索版"
        self.version = "4.0.0-MVP"
        
        # 初始化组件
        self.video_processor = VideoProcessor()
        self.feature_extractor = AudioFeatureExtractor()
        self.retrieval_engine = VideoRetrievalEngine()
        self.video_selector = TherapyVideoSelector(self.retrieval_engine)
        
        # 状态变量
        self.is_initialized = False
        self.processing_status = "就绪"
        
        logger.info(f"🚀 初始化 {self.app_name}")
    
    def initialize_system(self) -> str:
        """
        初始化系统：扫描视频、提取特征、构建索引
        
        Returns:
            str: 初始化状态报告
        """
        if self.is_initialized:
            return "✅ 系统已初始化完成"
        
        try:
            logger.info("🔄 开始系统初始化...")
            
            # 1. 检查ffmpeg
            if not self.video_processor.check_ffmpeg_availability():
                return "❌ 系统初始化失败：需要安装ffmpeg"
            
            # 2. 扫描视频文件
            videos = self.video_processor.scan_source_videos()
            if not videos:
                return "❌ 系统初始化失败：未找到视频文件\\n请确保视频文件位于 materials/video/ 目录下"
            
            # 3. 尝试加载现有索引
            if self.video_processor.load_segment_index():
                logger.info("✅ 加载现有片段索引")
            else:
                logger.info("🔪 开始视频切分...")
                segments = self.video_processor.segment_videos(extract_intro_only=True)
                if not segments:
                    return "❌ 视频切分失败"
            
            # 4. 加载或提取特征
            features_db = self.feature_extractor.load_features_database()
            if not features_db:
                logger.info("🎵 开始特征提取...")
                intro_segments = self.video_processor.get_intro_segments(duration_min=5)
                if intro_segments:
                    features_db = self.feature_extractor.extract_batch_features(intro_segments)
                else:
                    return "❌ 没有找到可用的视频片段"
            
            # 5. 重新加载检索引擎
            self.retrieval_engine.load_databases()
            
            # 6. 生成初始化报告
            video_count = len(videos)
            segment_count = sum(len(segs) for segs in self.video_processor.segment_index.values())
            feature_count = len(features_db) if features_db else 0
            
            self.is_initialized = True
            
            return f"""✅ 系统初始化完成！
            
🎬 视频素材库:
   • 原始视频: {video_count} 个
   • 切分片段: {segment_count} 个
   • 特征提取: {feature_count} 个
   
🧠 情绪识别:
   • 支持情绪: {len(self.retrieval_engine.emotion_database)} 种
   • 检索算法: ISO三阶段匹配
   
🔍 检索引擎:
   • 状态: 就绪
   • 匹配策略: Top-5随机选择
   
现在您可以输入情绪描述来体验智能疗愈视频推荐！"""
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return f"❌ 系统初始化失败: {str(e)}"
    
    def process_therapy_request(self, user_input: str) -> tuple:
        """
        处理疗愈请求
        
        Args:
            user_input: 用户情绪输入
            
        Returns:
            tuple: (报告, 视频文件, 状态)
        """
        if not self.is_initialized:
            return "⚠️ 请先点击'初始化系统'按钮", None, "系统未初始化"
        
        if not user_input or len(user_input.strip()) < 3:
            return "⚠️ 请输入至少3个字符描述您的情绪状态", None, "输入过短"
        
        try:
            logger.info(f"🔍 处理疗愈请求: {user_input}")
            
            # 1. 选择疗愈视频
            video_info = self.video_selector.select_therapy_video(user_input)
            
            if not video_info:
                return "❌ 未找到匹配的疗愈视频", None, "检索失败"
            
            # 2. 准备视频文件
            video_path = video_info['video_path']
            if not os.path.exists(video_path):
                return f"❌ 视频文件不存在: {video_path}", None, "文件不存在"
            
            # 3. 生成详细报告
            report = self._generate_therapy_report(video_info)
            
            # 4. 复制到临时位置（Gradio需要）
            temp_video = self._prepare_video_for_gradio(video_path)
            
            logger.info(f"✅ 疗愈视频推荐完成: {video_info['video_name']}")
            
            return report, temp_video, f"✅ 成功推荐 - {video_info['detected_emotion']}疗愈"
            
        except Exception as e:
            logger.error(f"处理疗愈请求失败: {e}")
            return f"❌ 处理失败: {str(e)}", None, "处理失败"
    
    def _generate_therapy_report(self, video_info: dict) -> str:
        """生成疗愈报告"""
        emotion = video_info['detected_emotion']
        confidence = video_info['emotion_confidence']
        similarity = video_info['similarity_score']
        video_name = video_info['video_name']
        duration = video_info['therapy_duration']
        iso_features = video_info.get('iso_features', {})
        
        # 获取匹配阶段特征
        matching_stage = iso_features.get('匹配阶段', {})
        
        report = f"""✅ 智能疗愈视频推荐完成！

🧠 情绪识别结果:
   • 检测情绪: {emotion}
   • 置信度: {confidence:.1%}
   • 用户输入: {video_info['user_input']}

🎯 视频匹配结果:
   • 推荐视频: {video_name}
   • 相似度得分: {similarity:.3f}
   • 视频时长: {duration:.1f}秒
   • 疗愈阶段: ISO {video_info['therapy_stage']}

🎼 ISO三阶段疗愈设计:
   • 匹配阶段特征: {matching_stage.get('mood', '情绪同步')}
     └─ 节拍: {matching_stage.get('tempo', '适中')}
     └─ 调性: {matching_stage.get('key', '和谐')}
     └─ 动态: {matching_stage.get('dynamics', '自然')}
   
   • 引导阶段: 逐步过渡到平静状态
   • 目标阶段: 深度放松准备入睡

🔍 检索技术:
   • 算法: 音乐特征向量相似度匹配
   • 策略: Top-5结果中随机选择
   • 基础: 前25%音频特征分析（对应ISO匹配阶段）

🌙 使用建议:
   • 在安静环境中观看
   • 调节到舒适的音量
   • 跟随视频内容放松身心
   • 专注感受三阶段情绪转换

✨ 这是基于您的情绪状态智能检索的个性化疗愈视频！"""
        
        return report
    
    def _prepare_video_for_gradio(self, video_path: str) -> str:
        """为Gradio准备视频文件"""
        try:
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            video_name = Path(video_path).name
            temp_path = os.path.join(temp_dir, f"therapy_{datetime.now().strftime('%H%M%S')}_{video_name}")
            
            # 复制文件
            shutil.copy2(video_path, temp_path)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"准备视频文件失败: {e}")
            return video_path  # 返回原始路径作为后备
    
    def get_system_stats(self) -> str:
        """获取系统统计信息"""
        if not self.is_initialized:
            return "系统未初始化"
        
        try:
            # 获取处理总结
            processor_summary = self.video_processor.get_processing_summary()
            
            # 获取检索统计
            retrieval_stats = self.retrieval_engine.get_retrieval_stats()
            
            # 获取选择历史
            history = self.video_selector.get_selection_history()
            
            stats = f"""📊 系统状态统计:

🎬 视频处理:
   • 原始视频: {processor_summary['source_videos']} 个
   • 总片段数: {processor_summary['total_segments']} 个
   • Intro片段: {processor_summary['intro_segments']} 个
   • 磁盘使用: {processor_summary['total_disk_usage_mb']} MB

🧠 情绪检索:
   • 支持情绪: {retrieval_stats['supported_emotions']} 种
   • 特征数据库: {retrieval_stats['total_videos']} 个视频
   • 检索历史: {len(history)} 次

⚡ 系统性能:
   • 版本: {self.version}
   • 状态: {'就绪' if self.is_initialized else '未初始化'}
   • 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 支持的情绪类型:
{', '.join(retrieval_stats['emotion_list'])}"""
            
            return stats
            
        except Exception as e:
            return f"获取统计信息失败: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        
        # 自定义CSS
        css = \"\"\"
        .therapy-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        .therapy-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .therapy-subtitle {
            font-size: 16px;
            opacity: 0.9;
        }
        .therapy-highlight {
            background: #ffeb3b;
            color: #333;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        \"\"\"
        
        with gr.Blocks(title=self.app_name, css=css) as app:
            
            # 标题区域
            gr.HTML(f"""
            <div class="therapy-header">
                <div class="therapy-title">{self.app_name}</div>
                <div class="therapy-subtitle">基于预制疗愈素材库的智能检索推荐</div>
                <div style="margin-top: 10px;">
                    <span class="therapy-highlight">🔍 从生成到检索 • ISO三阶段匹配 • Top-K智能选择</span>
                </div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 系统控制")
                    
                    # 系统初始化按钮
                    init_btn = gr.Button(
                        "🚀 初始化系统",
                        variant="primary",
                        size="lg"
                    )
                    
                    # 初始化状态显示
                    init_status = gr.Textbox(
                        label="📋 初始化状态",
                        lines=15,
                        interactive=False,
                        value="点击'初始化系统'开始..."
                    )
                    
                    gr.Markdown("### 💭 情绪输入")
                    
                    # 快速情绪选择
                    emotion_examples = gr.Dropdown(
                        choices=[
                            "😰 我感到很焦虑，心跳加速，难以入睡",
                            "😴 我很疲惫，但大脑还在活跃，无法放松",
                            "😤 我感到烦躁不安，容易被小事影响",
                            "😌 我比较平静，但希望更深层的放松",
                            "🤯 最近压力很大，总是感到紧张",
                            "💭 脑子里总是胡思乱想，停不下来"
                        ],
                        label="🎭 快速选择情绪",
                        value="😰 我感到很焦虑，心跳加速，难以入睡"
                    )
                    
                    # 详细情绪描述
                    emotion_input = gr.Textbox(
                        label="✍️ 详细描述您的情绪状态",
                        placeholder="请详细描述您当前的情绪感受...",
                        lines=3,
                        value="我感到很焦虑，心跳加速，难以入睡"
                    )
                    
                    # 获取疗愈视频按钮
                    therapy_btn = gr.Button(
                        "🌊 获取疗愈视频推荐",
                        variant="primary",
                        size="lg"
                    )
                    
                    # 系统统计按钮
                    stats_btn = gr.Button(
                        "📊 查看系统统计",
                        variant="secondary"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🎬 疗愈体验")
                    
                    # 疗愈报告
                    therapy_report = gr.Textbox(
                        label="📊 智能推荐报告",
                        lines=20,
                        interactive=False,
                        value="等待您的情绪输入，开始个性化疗愈视频推荐..."
                    )
                    
                    # 视频播放器
                    video_output = gr.Video(
                        label="🎵 疗愈视频",
                        height=400,
                        interactive=False
                    )
                    
                    # 状态显示
                    status_output = gr.Textbox(
                        label="🔄 处理状态",
                        interactive=False,
                        value="就绪"
                    )
            
            # 使用指南
            gr.HTML(\"\"\"
            <div style="margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <h3 style="color: #333;">🎯 4.0版本使用指南</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">
                    <div>
                        <h4 style="color: #555;">🔍 检索逻辑</h4>
                        <ul style="color: #666; text-align: left;">
                            <li>预制疗愈视频素材库（约3.5小时×2）</li>
                            <li>基于ISO原则的三阶段设计</li>
                            <li>提取前25%音频特征（匹配阶段）</li>
                            <li>智能特征向量相似度计算</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #555;">🎵 推荐策略</h4>
                        <ul style="color: #666; text-align: left;">
                            <li>27维情绪精确识别</li>
                            <li>Top-5相似度匹配</li>
                            <li>随机选择增加多样性</li>
                            <li>个性化疗愈体验</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #555;">🌟 技术特色</h4>
                        <ul style="color: #666; text-align: left;">
                            <li>从音乐生成到智能检索</li>
                            <li>成本效率大幅提升</li>
                            <li>响应速度显著加快</li>
                            <li>疗愈效果更加稳定</li>
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 15px; padding: 10px; background: rgba(255,193,7,0.1); border-radius: 5px;">
                    <strong>💡 首次使用:</strong> 请先点击"初始化系统"按钮，系统会自动扫描视频、提取特征并构建检索索引。初始化完成后即可开始使用。
                </div>
            </div>
            \"\"\")
            
            # 事件绑定
            def update_input_from_dropdown(selected):
                if " " in selected:
                    return selected.split(" ", 1)[1]
                return selected
            
            emotion_examples.change(
                update_input_from_dropdown,
                inputs=emotion_examples,
                outputs=emotion_input
            )
            
            init_btn.click(
                self.initialize_system,
                inputs=[],
                outputs=[init_status]
            )
            
            therapy_btn.click(
                self.process_therapy_request,
                inputs=[emotion_input],
                outputs=[therapy_report, video_output, status_output]
            )
            
            stats_btn.click(
                self.get_system_stats,
                inputs=[],
                outputs=[therapy_report]
            )
        
        return app

def main():
    """主函数"""
    print("🚀 启动音乐疗愈AI系统 4.0版本 - 检索驱动")
    print("🔍 从生成到检索的重大架构调整")
    print("🎯 基于预制疗愈视频素材库的智能推荐")
    print("⚡ 访问地址即将显示...")
    
    # 创建应用实例
    app_instance = TherapyApp4()
    
    # 创建界面
    app = app_instance.create_interface()
    
    # 启动服务
    app.launch(
        server_name="0.0.0.0",
        server_port=7870,  # 使用不同端口避免与3.0版本冲突
        share=True,
        show_error=True
    )

if __name__ == "__main__":
    main()