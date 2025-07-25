#!/usr/bin/env python3
"""
音乐检索系统 - 基于CLAMP3特征的相似度搜索
支持1分钟和3分钟版本的音乐特征检索
"""

import os
import sys
import numpy as np
import json
import time
import glob
import subprocess
import shutil
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import tempfile

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicSearchSystem:
    """音乐检索系统核心类"""
    
    def __init__(self, features_base_dir: str = None):
        """
        初始化音乐检索系统
        
        Args:
            features_base_dir: 特征文件基础目录
        """
        if features_base_dir is None:
            features_base_dir = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/music_features"
        
        self.features_base_dir = features_base_dir
        self.supported_durations = ["1min", "3min", "5min", "10min", "20min", "30min"]
        self.feature_cache = {}
        
        # 加载所有特征文件
        self._load_features()
    
    def _load_features(self):
        """加载所有可用的特征文件"""
        print("🔄 加载音乐特征库...")
        
        for duration in self.supported_durations:
            features_dir = os.path.join(self.features_base_dir, f"features_{duration}")
            
            if not os.path.exists(features_dir):
                print(f"⚠️  {duration} 特征目录不存在: {features_dir}")
                continue
            
            # 加载该时长的所有特征
            feature_files = glob.glob(os.path.join(features_dir, "*.npy"))
            
            if not feature_files:
                print(f"⚠️  {duration} 目录中没有特征文件")
                continue
            
            duration_features = {}
            for feature_file in feature_files:
                try:
                    # 从文件名提取视频名称
                    video_name = os.path.splitext(os.path.basename(feature_file))[0]
                    
                    # 加载特征向量
                    feature_vector = np.load(feature_file)
                    
                    # 确保特征向量是正确的维度 (1, 768)
                    if feature_vector.shape != (1, 768):
                        print(f"⚠️  {video_name} 特征维度异常: {feature_vector.shape}")
                        continue
                    
                    duration_features[video_name] = feature_vector.flatten()  # 转为1D数组
                    
                except Exception as e:
                    print(f"❌ 加载特征文件失败 {feature_file}: {e}")
                    continue
            
            self.feature_cache[duration] = duration_features
            print(f"✅ {duration}: 加载了 {len(duration_features)} 个特征文件")
        
        total_features = sum(len(features) for features in self.feature_cache.values())
        print(f"🎉 特征库加载完成，总计: {total_features} 个音乐特征")
    
    def extract_target_features(self, audio_path: str, use_partial: bool = True) -> np.ndarray:
        """
        提取目标音乐的特征
        
        Args:
            audio_path: 音频文件路径
            use_partial: 是否只使用前25%的音频
            
        Returns:
            768维特征向量
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_dir = os.path.join(temp_dir, "audio")
            features_dir = os.path.join(temp_dir, "features")
            os.makedirs(audio_dir)
            
            # 文件名处理
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            temp_audio_path = os.path.join(audio_dir, f"{audio_name}.wav")
            
            try:
                if use_partial:
                    # 只提取前25%的音频
                    print(f"🔄 提取音频前25%部分...")
                    
                    # 先获取音频时长
                    duration_cmd = [
                        'ffprobe', '-v', 'quiet', '-show_entries', 
                        'format=duration', '-of', 'csv=p=0', audio_path
                    ]
                    result = subprocess.run(duration_cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise Exception(f"无法获取音频时长: {result.stderr}")
                    
                    total_duration = float(result.stdout.strip())
                    partial_duration = total_duration * 0.25  # 前25%
                    
                    # 提取前25%的音频
                    cmd = [
                        'ffmpeg', '-i', audio_path, '-t', str(partial_duration),
                        '-q:a', '0', '-map', 'a', '-y', temp_audio_path
                    ]
                else:
                    # 提取完整音频
                    cmd = [
                        'ffmpeg', '-i', audio_path, '-q:a', '0', 
                        '-map', 'a', '-y', temp_audio_path
                    ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"音频提取失败: {result.stderr}")
                
                if use_partial:
                    print(f"✅ 已提取前25%音频 ({partial_duration:.1f}秒)")
                else:
                    print(f"✅ 已提取完整音频")
                
                # 使用CLAMP3提取特征
                print("🔄 提取CLAMP3特征...")
                
                result = subprocess.run([
                    'python', 'clamp3_embd.py', 
                    audio_dir, features_dir, '--get_global'
                ], capture_output=True, text=True, cwd=os.getcwd())
                
                if result.returncode != 0:
                    raise Exception(f"CLAMP3特征提取失败: {result.stderr}")
                
                # 加载提取的特征
                feature_file = os.path.join(features_dir, f"{audio_name}.npy")
                if not os.path.exists(feature_file):
                    raise Exception("特征文件未生成")
                
                feature_vector = np.load(feature_file)
                
                if feature_vector.shape != (1, 768):
                    raise Exception(f"特征维度异常: {feature_vector.shape}")
                
                print("✅ 目标音乐特征提取完成")
                return feature_vector.flatten()
                
            except Exception as e:
                print(f"❌ 目标特征提取失败: {e}")
                raise
    
    def calculate_similarity(self, target_features: np.ndarray, reference_features: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度
        
        Args:
            target_features: 目标特征向量
            reference_features: 参考特征向量
            
        Returns:
            相似度分数 (0-1)
        """
        # 计算余弦相似度
        dot_product = np.dot(target_features, reference_features)
        norm_target = np.linalg.norm(target_features)
        norm_reference = np.linalg.norm(reference_features)
        
        if norm_target == 0 or norm_reference == 0:
            return 0.0
        
        similarity = dot_product / (norm_target * norm_reference)
        
        # 将相似度从[-1, 1]映射到[0, 1]
        return (similarity + 1) / 2
    
    def search_similar_music(self, target_features: np.ndarray, duration: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        搜索最相似的音乐
        
        Args:
            target_features: 目标音乐特征
            duration: 搜索的时长版本 ("1min" 或 "3min")
            top_k: 返回最相似的前k个结果
            
        Returns:
            [(视频名称, 相似度分数), ...] 按相似度降序排列
        """
        if duration not in self.feature_cache:
            raise ValueError(f"不支持的时长版本: {duration}")
        
        if duration not in self.supported_durations:
            raise ValueError(f"当前只支持: {self.supported_durations}")
        
        duration_features = self.feature_cache[duration]
        
        if not duration_features:
            raise ValueError(f"{duration} 版本没有可用的特征")
        
        print(f"🔍 在 {duration} 版本中搜索相似音乐...")
        
        # 计算与所有音乐的相似度
        similarities = []
        
        for video_name, reference_features in duration_features.items():
            similarity = self.calculate_similarity(target_features, reference_features)
            similarities.append((video_name, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个结果
        return similarities[:top_k]
    
    def search_music_by_file(self, audio_path: str, duration: str, top_k: int = 3, use_partial: bool = True) -> List[Tuple[str, float]]:
        """
        通过音频文件搜索相似音乐
        
        Args:
            audio_path: 目标音频文件路径
            duration: 搜索的时长版本
            top_k: 返回前k个结果
            use_partial: 是否只使用前25%的音频
            
        Returns:
            搜索结果列表
        """
        print(f"🎯 开始音乐搜索任务...")
        print(f"   目标音频: {audio_path}")
        print(f"   搜索版本: {duration}")
        print(f"   返回数量: {top_k}")
        print(f"   使用部分音频: {'前25%' if use_partial else '完整音频'}")
        
        start_time = time.time()
        
        try:
            # 1. 提取目标音乐特征
            target_features = self.extract_target_features(audio_path, use_partial)
            
            # 2. 搜索相似音乐
            results = self.search_similar_music(target_features, duration, top_k)
            
            search_time = time.time() - start_time
            
            # 3. 显示结果
            print(f"\n🎉 搜索完成 (用时: {search_time:.2f}秒)")
            print(f"📊 搜索结果 (前{len(results)}个):")
            
            for i, (video_name, similarity) in enumerate(results, 1):
                print(f"   {i}. {video_name} - 相似度: {similarity:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            raise
    
    def get_video_path(self, video_name: str, duration: str) -> Optional[str]:
        """
        获取视频文件的完整路径
        
        Args:
            video_name: 视频名称 (不含扩展名)
            duration: 时长版本
            
        Returns:
            视频文件路径或None
        """
        video_dir = f"/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/materials/retrieve_libraries/segments_{duration}"
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        
        if os.path.exists(video_path):
            return video_path
        
        return None
    
    def get_statistics(self) -> Dict:
        """获取特征库统计信息"""
        stats = {
            "total_features": sum(len(features) for features in self.feature_cache.values()),
            "by_duration": {}
        }
        
        for duration, features in self.feature_cache.items():
            stats["by_duration"][duration] = {
                "count": len(features),
                "videos": list(features.keys())
            }
        
        return stats
    
    def search_music_by_text(self, text_description: str, duration: str = "3min", 
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        通过文本描述搜索相似音乐 (语义检索)
        
        Args:
            text_description: 文本描述
            duration: 搜索版本
            top_k: 返回结果数量
            
        Returns:
            [(视频名称, 相似度分数), ...] 列表，按相似度降序排列
        """
        if duration not in self.feature_cache:
            logger.warning(f"⚠️  不支持的时长版本: {duration}")
            return []
        
        if not self.feature_cache[duration]:
            logger.warning(f"⚠️  {duration} 版本没有特征数据")
            return []
        
        try:
            # 尝试使用CLAMP3语义提取器，失败则使用简化版
            try:
                from semantic_text_extractor import SemanticTextExtractor
                
                # 初始化文本特征提取器
                if not hasattr(self, 'text_extractor'):
                    logger.info("🔄 初始化CLAMP3文本特征提取器...")
                    self.text_extractor = SemanticTextExtractor()
                
                # 提取文本特征
                logger.info(f"🔄 提取文本特征: {text_description}")
                text_features = self.text_extractor.extract_single_text_feature(text_description)
                
                # 计算与所有音乐的相似度
                similarities = []
                for video_name, audio_features in self.feature_cache[duration].items():
                    similarity = self._compute_cosine_similarity(text_features, audio_features)
                    similarities.append((video_name, similarity))
                
                logger.info("✅ 使用CLAMP3语义检索")
                
            except Exception as clamp_error:
                logger.warning(f"⚠️  CLAMP3检索失败: {clamp_error}")
                logger.info("🔄 切换到简化版语义检索...")
                
                # 使用简化版语义检索
                from simple_semantic_search import SimpleSemanticSearcher
                
                if not hasattr(self, 'simple_searcher'):
                    self.simple_searcher = SimpleSemanticSearcher()
                
                # 计算与所有音乐的相似度
                similarities = []
                for video_name, audio_features in self.feature_cache[duration].items():
                    similarity = self.simple_searcher.compute_text_audio_similarity(
                        text_description, audio_features)
                    similarities.append((video_name, similarity))
                
                logger.info("✅ 使用简化版语义检索")
            
            # 按相似度降序排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前top_k个结果
            results = similarities[:top_k]
            
            logger.info(f"✅ 文本检索完成，返回 {len(results)} 个结果")
            for video_name, similarity in results:
                logger.info(f"   {video_name}: {similarity:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 文本检索失败: {e}")
            return []
    
    def _compute_cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个特征向量的余弦相似度
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 确保特征向量是一维的
            if features1.ndim > 1:
                features1 = features1.flatten()
            if features2.ndim > 1:
                features2 = features2.flatten()
            
            # 计算余弦相似度
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # 将相似度从 [-1, 1] 范围映射到 [0, 1]
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ 相似度计算失败: {e}")
            return 0.0

def main():
    """主函数 - 演示使用"""
    print("🎵 音乐检索系统演示")
    print("="*50)
    
    # 初始化检索系统
    search_system = MusicSearchSystem()
    
    # 显示统计信息
    stats = search_system.get_statistics()
    print(f"\n📊 特征库统计:")
    print(f"   总特征数: {stats['total_features']}")
    for duration, info in stats["by_duration"].items():
        print(f"   {duration}: {info['count']} 个特征")
    
    # 示例：使用现有的一个音乐文件作为查询
    # 这里可以替换为实际的音频文件路径
    sample_video = "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4/MI_retrieve/materials/retrieve_libraries/segments_1min/32_1min_01.mp4"
    
    if os.path.exists(sample_video):
        print(f"\n🔍 示例搜索:")
        print(f"使用 {sample_video} 作为查询音频")
        
        try:
            # 在3分钟版本中搜索
            results = search_system.search_music_by_file(
                audio_path=sample_video,
                duration="3min",
                top_k=3,
                use_partial=True
            )
            
            print(f"\n📁 对应的视频文件路径:")
            for video_name, similarity in results:
                video_path = search_system.get_video_path(video_name, "3min")
                print(f"   {video_name}: {video_path}")
                
        except Exception as e:
            print(f"❌ 示例搜索失败: {e}")
    
    print("\n✅ 音乐检索系统就绪！")

if __name__ == "__main__":
    main()