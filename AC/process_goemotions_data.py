#!/usr/bin/env python3
"""
处理GoEmotions数据并转换为C&K格式

修复导入问题并完成数据转换
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from emotion_mapper import GoEmotionsMapper
from config import COWEN_KELTNER_EMOTIONS, GOEMOTIONS_LABELS

logger = logging.getLogger(__name__)

def process_goemotions_to_ck():
    """处理GoEmotions数据并转换为C&K格式"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    data_dir = Path(__file__).parent / "data"
    mapper = GoEmotionsMapper()
    
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        logger.info(f"\n🔄 处理 {split} 数据集...")
        
        # 读取GoEmotions格式数据
        ge_file = data_dir / f"goemotions_{split}.csv"
        if not ge_file.exists():
            logger.error(f"❌ 文件不存在: {ge_file}")
            continue
        
        logger.info(f"📂 读取文件: {ge_file}")
        df = pd.read_csv(ge_file)
        logger.info(f"   数据量: {len(df)} 条记录")
        
        # 转换为C&K格式
        logger.info("🔄 转换为C&K 27维格式...")
        
        result_data = []
        conversion_stats = {ck_emotion: 0 for ck_emotion in COWEN_KELTNER_EMOTIONS}
        
        for idx, row in df.iterrows():
            try:
                text = row['text']
                
                # 提取GoEmotions分数
                ge_scores = {}
                for label in GOEMOTIONS_LABELS:
                    if label in df.columns:
                        ge_scores[label] = float(row[label]) if pd.notna(row[label]) else 0.0
                
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
                
                if (idx + 1) % 5000 == 0:
                    logger.info(f"   转换进度: {idx + 1}/{len(df)}")
                    
            except Exception as e:
                logger.warning(f"⚠️  第{idx}行转换失败: {e}")
                continue
        
        # 保存结果
        result_df = pd.DataFrame(result_data)
        output_path = data_dir / f"processed_{split}.csv"
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"✅ C&K格式保存: {output_path}")
        logger.info(f"   转换成功: {len(result_df)} 条记录")
        
        # 统计C&K情绪分布
        logger.info("📊 C&K情绪分布统计:")
        sorted_ck = sorted(conversion_stats.items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_ck[:10]:
            logger.info(f"   {emotion}: {count}")
    
    logger.info("\n✅ 所有数据转换完成!")

def verify_processed_data():
    """验证处理后的数据"""
    logger.info("\n🔍 验证处理后的数据...")
    
    data_dir = Path(__file__).parent / "data"
    splits = ['train', 'dev', 'test']
    
    summary = {}
    
    for split in splits:
        ck_file = data_dir / f"processed_{split}.csv"
        if ck_file.exists():
            df = pd.read_csv(ck_file)
            
            # 基础统计
            emotion_columns = [col for col in df.columns if col in COWEN_KELTNER_EMOTIONS]
            emotion_matrix = df[emotion_columns].values
            
            stats = {
                "samples": len(df),
                "emotion_dimensions": len(emotion_columns),
                "avg_emotions_per_sample": float(np.mean(np.sum(emotion_matrix > 0, axis=1))),
                "avg_total_intensity": float(np.mean(np.sum(emotion_matrix, axis=1))),
                "most_common_emotion": emotion_columns[np.argmax(np.sum(emotion_matrix, axis=0))] if len(emotion_columns) > 0 else "N/A"
            }
            
            summary[split] = stats
            
            logger.info(f"📊 {split} 数据集:")
            logger.info(f"   样本数: {stats['samples']}")
            logger.info(f"   情绪维度: {stats['emotion_dimensions']}")
            logger.info(f"   平均活跃情绪: {stats['avg_emotions_per_sample']:.2f}")
            logger.info(f"   平均总强度: {stats['avg_total_intensity']:.3f}")
            logger.info(f"   最常见情绪: {stats['most_common_emotion']}")
            
            # 检查数据质量
            text_lengths = df['text'].str.len()
            logger.info(f"   文本长度: 平均{text_lengths.mean():.1f}, 最大{text_lengths.max()}")
            
            # 检查是否有无效数据
            invalid_rows = df[df[emotion_columns].sum(axis=1) == 0]
            if len(invalid_rows) > 0:
                logger.warning(f"⚠️  发现 {len(invalid_rows)} 条无情绪标签的数据")
    
    # 保存验证报告
    import json
    report_path = data_dir / "data_validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ 验证报告保存: {report_path}")
    return summary

def main():
    """主函数"""
    print("🔄 处理GoEmotions数据并转换为C&K格式")
    print("=" * 50)
    
    try:
        # 处理数据
        process_goemotions_to_ck()
        
        # 验证数据
        summary = verify_processed_data()
        
        # 显示总结
        print("\n📋 数据处理总结:")
        total_samples = sum(info['samples'] for info in summary.values())
        print(f"总样本数: {total_samples}")
        print(f"数据分割: {list(summary.keys())}")
        
        print("\n🎉 数据处理完成！现在可以开始训练模型了。")
        return True
        
    except Exception as e:
        print(f"❌ 数据处理失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)