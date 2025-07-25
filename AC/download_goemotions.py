#!/usr/bin/env python3
"""
下载和预处理GoEmotions数据集

从Google Research GitHub仓库下载GoEmotions数据
并转换为适合训练的格式
"""

import os
import requests
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
import json

from config import DATA_PATHS, GOEMOTIONS_LABELS, GOEMOTIONS_TO_CK_MAPPING, COWEN_KELTNER_EMOTIONS

logger = logging.getLogger(__name__)

class GoEmotionsDownloader:
    """GoEmotions数据集下载器"""
    
    def __init__(self):
        """初始化下载器"""
        self.base_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # 文件映射
        self.files = {
            "train": "train.tsv",
            "dev": "dev.tsv", 
            "test": "test.tsv",
            "emotions": "emotions.txt"
        }
        
        logger.info("✅ GoEmotions下载器初始化完成")
    
    def download_files(self) -> bool:
        """
        下载GoEmotions数据文件
        
        Returns:
            bool: 是否下载成功
        """
        try:
            logger.info("📥 开始下载GoEmotions数据集...")
            
            for split, filename in self.files.items():
                url = self.base_url + filename
                local_path = self.data_dir / filename
                
                if local_path.exists():
                    logger.info(f"   ✓ {filename} 已存在，跳过下载")
                    continue
                
                logger.info(f"   📥 下载 {filename}...")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"   ✅ {filename} 下载完成 ({len(response.content)/1024:.1f} KB)")
            
            logger.info("✅ 所有文件下载完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 下载失败: {e}")
            return False
    
    def parse_tsv_data(self, file_path: Path) -> pd.DataFrame:
        """
        解析TSV格式的GoEmotions数据
        
        Args:
            file_path: TSV文件路径
            
        Returns:
            解析后的DataFrame
        """
        try:
            logger.info(f"📊 解析数据文件: {file_path.name}")
            
            # GoEmotions TSV格式: [text, emotion_ids, id]
            df = pd.read_csv(file_path, sep='\t', header=None, 
                           names=['text', 'emotion_ids', 'id'])
            
            logger.info(f"   原始数据: {len(df)} 条记录")
            
            # 清理数据
            df = df.dropna(subset=['text', 'emotion_ids'])
            df['text'] = df['text'].astype(str).str.strip()
            df = df[df['text'].str.len() > 0]
            
            logger.info(f"   清理后数据: {len(df)} 条记录")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 数据解析失败: {e}")
            return pd.DataFrame()
    
    def convert_to_multilabel_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将GoEmotions数据转换为多标签格式
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            多标签格式的DataFrame
        """
        try:
            logger.info("🔄 转换为多标签格式...")
            
            # 创建结果DataFrame
            result_data = []
            
            for idx, row in df.iterrows():
                text = row['text']
                emotion_ids = row['emotion_ids']
                
                # 解析情绪ID
                if pd.isna(emotion_ids) or emotion_ids == '':
                    continue
                
                # 处理情绪ID列表
                try:
                    if isinstance(emotion_ids, str):
                        # 格式可能是 "1,5,12" 或 "[1,5,12]"
                        emotion_ids = emotion_ids.strip('[]')
                        if emotion_ids:
                            emotion_id_list = [int(x.strip()) for x in emotion_ids.split(',') if x.strip()]
                        else:
                            continue
                    else:
                        emotion_id_list = [int(emotion_ids)]
                except (ValueError, TypeError):
                    logger.warning(f"⚠️  无法解析情绪ID: {emotion_ids}")
                    continue
                
                # 创建多标签向量 (27维GoEmotions标签)
                label_vector = [0.0] * len(GOEMOTIONS_LABELS)
                
                for emotion_id in emotion_id_list:
                    if 0 <= emotion_id < len(GOEMOTIONS_LABELS):
                        label_vector[emotion_id] = 1.0
                
                # 构建数据行
                data_row = {'text': text}
                for i, label in enumerate(GOEMOTIONS_LABELS):
                    data_row[label] = label_vector[i]
                
                result_data.append(data_row)
            
            result_df = pd.DataFrame(result_data)
            logger.info(f"✅ 多标签转换完成: {len(result_df)} 条记录")
            
            # 统计标签分布
            label_counts = {}
            for label in GOEMOTIONS_LABELS:
                if label in result_df.columns:
                    label_counts[label] = int(result_df[label].sum())
            
            logger.info("📊 标签分布统计:")
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_labels[:10]:  # 显示前10个
                logger.info(f"   {label}: {count}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ 多标签转换失败: {e}")
            return pd.DataFrame()
    
    def convert_to_ck_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将GoEmotions格式转换为Cowen & Keltner 27维格式
        
        Args:
            df: GoEmotions多标签格式DataFrame
            
        Returns:
            C&K 27维格式DataFrame
        """
        try:
            logger.info("🔄 转换为C&K 27维格式...")
            
            # 导入映射器
            from emotion_mapper import GoEmotionsMapper
            mapper = GoEmotionsMapper()
            
            result_data = []
            conversion_stats = {ck_emotion: 0 for ck_emotion in COWEN_KELTNER_EMOTIONS}
            
            for idx, row in df.iterrows():
                text = row['text']
                
                # 提取GoEmotions分数
                ge_scores = {}
                for label in GOEMOTIONS_LABELS:
                    if label in row:
                        ge_scores[label] = float(row[label])
                
                # 映射到C&K向量
                ck_vector = mapper.map_goemotions_to_ck_vector(ge_scores)
                
                # 构建数据行
                data_row = {'text': text}
                
                # 添加C&K情绪列
                for i, emotion in enumerate(COWEN_KELTNER_EMOTIONS):
                    data_row[emotion] = float(ck_vector[i])
                    if ck_vector[i] > 0:
                        conversion_stats[emotion] += 1
                
                # 添加元数据
                active_ge_labels = [label for label, score in ge_scores.items() if score > 0]
                data_row['original_goemotions'] = ','.join(active_ge_labels)
                data_row['max_emotion'] = COWEN_KELTNER_EMOTIONS[np.argmax(ck_vector)]
                data_row['emotion_intensity'] = float(np.max(ck_vector))
                data_row['total_intensity'] = float(np.sum(ck_vector))
                
                result_data.append(data_row)
                
                if (idx + 1) % 1000 == 0:
                    logger.info(f"   转换进度: {idx + 1}/{len(df)}")
            
            result_df = pd.DataFrame(result_data)
            logger.info(f"✅ C&K格式转换完成: {len(result_df)} 条记录")
            
            # 统计C&K情绪分布
            logger.info("📊 C&K情绪分布统计:")
            sorted_ck = sorted(conversion_stats.items(), key=lambda x: x[1], reverse=True)
            for emotion, count in sorted_ck[:10]:
                logger.info(f"   {emotion}: {count}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ C&K格式转换失败: {e}")
            return pd.DataFrame()
    
    def process_and_save_data(self) -> bool:
        """
        处理并保存所有数据
        
        Returns:
            bool: 是否处理成功
        """
        try:
            logger.info("🔄 开始处理和保存数据...")
            
            splits = ['train', 'dev', 'test']
            
            for split in splits:
                logger.info(f"\n📊 处理 {split} 数据集...")
                
                # 读取原始TSV文件
                tsv_file = self.data_dir / f"{split}.tsv"
                if not tsv_file.exists():
                    logger.error(f"❌ 文件不存在: {tsv_file}")
                    continue
                
                # 解析TSV数据
                df = self.parse_tsv_data(tsv_file)
                if df.empty:
                    logger.error(f"❌ {split} 数据解析失败")
                    continue
                
                # 转换为多标签格式
                multilabel_df = self.convert_to_multilabel_format(df)
                if multilabel_df.empty:
                    logger.error(f"❌ {split} 多标签转换失败")
                    continue
                
                # 保存GoEmotions格式
                ge_output_path = self.data_dir / f"goemotions_{split}.csv"
                multilabel_df.to_csv(ge_output_path, index=False, encoding='utf-8')
                logger.info(f"✅ GoEmotions格式保存: {ge_output_path}")
                
                # 转换为C&K格式
                ck_df = self.convert_to_ck_format(multilabel_df)
                if ck_df.empty:
                    logger.error(f"❌ {split} C&K转换失败")
                    continue
                
                # 保存C&K格式
                ck_output_path = self.data_dir / f"processed_{split}.csv"
                ck_df.to_csv(ck_output_path, index=False, encoding='utf-8')
                logger.info(f"✅ C&K格式保存: {ck_output_path}")
            
            logger.info("\n✅ 所有数据处理完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据处理失败: {e}")
            return False
    
    def generate_dataset_summary(self) -> Dict:
        """
        生成数据集摘要信息
        
        Returns:
            数据集摘要字典
        """
        try:
            summary = {
                "dataset_name": "GoEmotions",
                "source": "Google Research",
                "emotion_taxonomy": "Cowen & Keltner (2017) 27 dimensions",
                "splits": {},
                "mapping_info": {
                    "original_labels": len(GOEMOTIONS_LABELS),
                    "target_emotions": len(COWEN_KELTNER_EMOTIONS),
                    "mapping_coverage": len(GOEMOTIONS_TO_CK_MAPPING) / len(GOEMOTIONS_LABELS)
                }
            }
            
            # 统计各分割的信息
            for split in ['train', 'dev', 'test']:
                ck_file = self.data_dir / f"processed_{split}.csv"
                if ck_file.exists():
                    df = pd.read_csv(ck_file)
                    
                    # 计算统计信息
                    emotion_columns = [col for col in df.columns if col in COWEN_KELTNER_EMOTIONS]
                    emotion_matrix = df[emotion_columns].values
                    
                    summary["splits"][split] = {
                        "samples": len(df),
                        "avg_emotions_per_sample": float(np.mean(np.sum(emotion_matrix > 0, axis=1))),
                        "avg_total_intensity": float(np.mean(np.sum(emotion_matrix, axis=1))),
                        "most_common_emotion": emotion_columns[np.argmax(np.sum(emotion_matrix, axis=0))]
                    }
            
            # 保存摘要
            summary_path = self.data_dir / "dataset_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 数据集摘要保存: {summary_path}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ 生成摘要失败: {e}")
            return {}

def main():
    """主函数"""
    print("🚀 GoEmotions数据集下载和预处理")
    print("=" * 50)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 初始化下载器
    downloader = GoEmotionsDownloader()
    
    # 下载文件
    if not downloader.download_files():
        print("❌ 数据下载失败")
        return False
    
    # 处理数据
    if not downloader.process_and_save_data():
        print("❌ 数据处理失败")
        return False
    
    # 生成摘要
    summary = downloader.generate_dataset_summary()
    if summary:
        print("\n📊 数据集摘要:")
        for split, info in summary.get("splits", {}).items():
            print(f"   {split}: {info['samples']} 样本, 平均{info['avg_emotions_per_sample']:.1f}个情绪")
    
    print("\n🎉 GoEmotions数据集准备完成!")
    print("现在可以开始训练xlm-roberta模型了！")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)