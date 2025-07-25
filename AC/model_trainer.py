#!/usr/bin/env python3
"""
xlm-roberta模型微调训练器

基于GoEmotions数据集微调xlm-roberta模型
支持多标签情感分类和C&K情绪体系转换
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

from config import (
    MODEL_CONFIG, 
    MODEL_PATHS, 
    DATA_PATHS,
    COWEN_KELTNER_EMOTIONS,
    GOEMOTIONS_TO_CK_MAPPING
)
from emotion_mapper import GoEmotionsMapper

logger = logging.getLogger(__name__)

class EmotionDataset(Dataset):
    """情感数据集类"""
    
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签矩阵 (N, 27)
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

class ModelTrainer:
    """xlm-roberta模型训练器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config or MODEL_CONFIG
        self.mapper = GoEmotionsMapper()
        self.device = self._detect_device()
        
        logger.info("✅ 模型训练器初始化完成")
        logger.info(f"   使用设备: {self.device}")
        logger.info(f"   模型配置: {self.config['model_name']}")
    
    def _detect_device(self) -> str:
        """检测可用设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  
        else:
            return "cpu"
    
    def prepare_data(self, data_path: str) -> Tuple[List[str], np.ndarray]:
        """
        准备训练数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            (texts, labels): 文本列表和标签矩阵
        """
        try:
            logger.info(f"📂 准备训练数据: {data_path}")
            
            # 读取数据
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            else:
                raise ValueError(f"不支持的数据格式: {data_path}")
            
            logger.info(f"   原始数据: {len(df)} 条样本")
            
            # 检查是否已经是C&K格式
            if all(emotion in df.columns for emotion in COWEN_KELTNER_EMOTIONS):
                logger.info("   检测到C&K格式数据，直接使用")
                texts = df['text'].tolist()
                labels = df[COWEN_KELTNER_EMOTIONS].values.astype(np.float32)
            else:
                # GoEmotions格式，需要转换
                logger.info("   检测到GoEmotions格式，开始转换")
                texts = df['text'].tolist()
                
                # 转换标签
                labels_list = []
                for idx, row in df.iterrows():
                    ge_scores = {label: row[label] for label in self.mapper.goemotions_labels if label in row}
                    ck_vector = self.mapper.map_goemotions_to_ck_vector(ge_scores)
                    labels_list.append(ck_vector)
                
                labels = np.array(labels_list, dtype=np.float32)
            
            # 数据验证
            assert len(texts) == len(labels), "文本和标签数量不匹配"
            assert labels.shape[1] == 27, f"标签维度错误: 期望27维，实际{labels.shape[1]}维"
            
            # 统计信息
            active_labels = np.sum(labels > 0, axis=1)
            logger.info(f"   处理后数据: {len(texts)} 条样本")
            logger.info(f"   平均活跃情绪数: {np.mean(active_labels):.2f}")
            logger.info(f"   标签分布: min={np.min(labels):.3f}, max={np.max(labels):.3f}")
            
            return texts, labels
            
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            raise
    
    def create_datasets(self, texts: List[str], labels: np.ndarray, 
                       test_size: float = 0.2, val_size: float = 0.1) -> Tuple[EmotionDataset, EmotionDataset, EmotionDataset]:
        """
        创建训练、验证和测试数据集
        
        Args:
            texts: 文本列表
            labels: 标签矩阵
            test_size: 测试集比例
            val_size: 验证集比例
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        try:
            logger.info("🔄 创建数据集分割")
            
            # 初始化tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"],
                cache_dir=MODEL_PATHS["pretrained_cache"]
            )
            
            # 第一次分割: 训练+验证 vs 测试
            X_temp, X_test, y_temp, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=None
            )
            
            # 第二次分割: 训练 vs 验证
            val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=None
            )
            
            # 创建数据集对象
            train_dataset = EmotionDataset(
                X_train, y_train, tokenizer, self.config["max_length"]
            )
            val_dataset = EmotionDataset(
                X_val, y_val, tokenizer, self.config["max_length"]
            )
            test_dataset = EmotionDataset(
                X_test, y_test, tokenizer, self.config["max_length"]
            )
            
            logger.info(f"✅ 数据集创建完成:")
            logger.info(f"   训练集: {len(train_dataset)} 样本")
            logger.info(f"   验证集: {len(val_dataset)} 样本") 
            logger.info(f"   测试集: {len(test_dataset)} 样本")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"❌ 数据集创建失败: {e}")
            raise
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            eval_pred: 评估预测结果
            
        Returns:
            评估指标字典
        """
        predictions, labels = eval_pred
        
        # 应用sigmoid激活
        predictions = 1 / (1 + np.exp(-predictions))  # sigmoid
        
        # 二值化预测 (阈值=0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        binary_labels = (labels > 0.5).astype(int)
        
        # 计算指标
        metrics = {}
        
        # Macro F1
        f1_macro = f1_score(binary_labels, binary_predictions, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_macro
        
        # Micro F1  
        f1_micro = f1_score(binary_labels, binary_predictions, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_micro
        
        # 每个情绪的F1分数
        f1_per_emotion = f1_score(binary_labels, binary_predictions, average=None, zero_division=0)
        for i, emotion in enumerate(COWEN_KELTNER_EMOTIONS):
            metrics[f'f1_{emotion}'] = f1_per_emotion[i]
        
        # 准确率 (Hamming accuracy)
        hamming_acc = accuracy_score(binary_labels, binary_predictions)
        metrics['hamming_accuracy'] = hamming_acc
        
        # 完全匹配准确率
        exact_match = np.all(binary_labels == binary_predictions, axis=1).mean()
        metrics['exact_match_accuracy'] = exact_match
        
        return metrics
    
    def train_model(self, train_dataset: EmotionDataset, val_dataset: EmotionDataset,
                   output_dir: str = None) -> None:
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集  
            output_dir: 输出目录
        """
        try:
            output_dir = output_dir or str(MODEL_PATHS["finetuned_model"])
            logger.info(f"🚀 开始模型训练")
            logger.info(f"   输出目录: {output_dir}")
            
            # 初始化模型配置
            config = AutoConfig.from_pretrained(
                self.config["model_name"],
                num_labels=27,
                problem_type="multi_label_classification",
                cache_dir=MODEL_PATHS["pretrained_cache"]
            )
            
            # 初始化模型
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config["model_name"],
                config=config,
                cache_dir=MODEL_PATHS["pretrained_cache"]
            )
            
            # 初始化tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"], 
                cache_dir=MODEL_PATHS["pretrained_cache"]
            )
            
            # 数据整理器
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config["num_epochs"],
                per_device_train_batch_size=self.config["batch_size"],
                per_device_eval_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                warmup_steps=self.config["warmup_steps"],
                weight_decay=self.config["weight_decay"],
                learning_rate=self.config["learning_rate"],
                logging_steps=100,
                eval_steps=500,
                save_steps=500,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,
                report_to=None,  # 禁用wandb等日志
                push_to_hub=False,
                dataloader_num_workers=0,  # 避免多进程问题
            )
            
            # 初始化Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )
            
            # 开始训练
            logger.info("🔄 开始训练...")
            train_result = trainer.train()
            
            # 保存模型
            logger.info("💾 保存训练后的模型...")
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # 训练总结
            logger.info("✅ 模型训练完成!")
            logger.info(f"   训练损失: {train_result.training_loss:.4f}")
            logger.info(f"   训练步数: {train_result.global_step}")
            
        except Exception as e:
            logger.error(f"❌ 模型训练失败: {e}")
            raise
    
    def evaluate_model(self, test_dataset: EmotionDataset, model_path: str = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_dataset: 测试数据集
            model_path: 模型路径
            
        Returns:
            评估结果字典
        """
        try:
            model_path = model_path or str(MODEL_PATHS["finetuned_model"])
            logger.info(f"📊 开始模型评估: {model_path}")
            
            # 加载模型和tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 数据整理器
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # 评估参数
            eval_args = TrainingArguments(
                output_dir="./temp_eval",
                per_device_eval_batch_size=self.config["batch_size"],
                dataloader_num_workers=0,
                report_to=None
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=eval_args,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )
            
            # 评估
            eval_results = trainer.evaluate()
            
            logger.info("✅ 模型评估完成!")
            logger.info(f"   F1 Macro: {eval_results['eval_f1_macro']:.4f}")
            logger.info(f"   F1 Micro: {eval_results['eval_f1_micro']:.4f}")
            logger.info(f"   Hamming Accuracy: {eval_results['eval_hamming_accuracy']:.4f}")
            logger.info(f"   Exact Match: {eval_results['eval_exact_match_accuracy']:.4f}")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"❌ 模型评估失败: {e}")
            return {}

def main():
    """训练脚本主函数"""
    print("🚀 开始xlm-roberta情感分类模型训练")
    print("=" * 60)
    
    # 初始化训练器
    trainer = ModelTrainer()
    
    # 检查数据路径
    train_data_path = DATA_PATHS["goemotions_train"]
    if not train_data_path.exists():
        logger.error(f"❌ 训练数据不存在: {train_data_path}")
        logger.info("💡 请先下载并准备GoEmotions数据集")
        return
    
    try:
        # 准备数据
        texts, labels = trainer.prepare_data(str(train_data_path))
        
        # 创建数据集
        train_dataset, val_dataset, test_dataset = trainer.create_datasets(texts, labels)
        
        # 训练模型  
        trainer.train_model(train_dataset, val_dataset)
        
        # 评估模型
        eval_results = trainer.evaluate_model(test_dataset)
        
        print(f"\n🎉 训练流程完成!")
        print(f"模型已保存到: {MODEL_PATHS['finetuned_model']}")
        
    except Exception as e:
        logger.error(f"❌ 训练流程失败: {e}")
        print(f"❌ 训练失败: {e}")

if __name__ == "__main__":
    main()