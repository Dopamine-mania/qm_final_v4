#!/usr/bin/env python3
"""
简化版情感模型训练器
避开transformers复杂依赖，直接使用PyTorch
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmotionModel(nn.Module):
    """简化的情感分类模型 - 基于LSTM"""
    
    def __init__(self, vocab_size=50000, embed_dim=256, hidden_dim=512, num_emotions=27, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_emotions)
        
    def forward(self, input_ids, attention_mask=None):
        # 嵌入层
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)
        
        # 使用注意力掩码获取最后有效位置的输出
        if attention_mask is not None:
            # 计算每个序列的有效长度
            lengths = attention_mask.sum(dim=1).long() - 1  # 减1因为索引从0开始
            batch_size = lstm_out.size(0)
            # 获取每个序列最后一个有效位置的输出
            last_outputs = lstm_out[range(batch_size), lengths]
        else:
            # 如果没有掩码，使用最后一个时间步
            last_outputs = lstm_out[:, -1, :]
        
        # 分类层
        output = self.dropout(last_outputs)
        logits = self.classifier(output)
        
        return logits

class SimpleEmotionDataset(Dataset):
    """简化的情感数据集"""
    
    def __init__(self, texts, labels, vocab_dict=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # 构建词汇表
        if vocab_dict is None:
            self.vocab_dict = self._build_vocab(texts)
        else:
            self.vocab_dict = vocab_dict
            
    def _build_vocab(self, texts):
        """构建词汇表"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        word_count = {}
        
        # 统计词频
        for text in texts:
            for word in str(text).split():
                word_count[word] = word_count.get(word, 0) + 1
        
        # 按频率排序，只保留前48000个词（留2个位置给特殊符号）
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:48000]
        
        for word, _ in sorted_words:
            vocab[word] = len(vocab)
            
        logger.info(f"📚 构建词汇表完成，大小: {len(vocab)}")
        return vocab
    
    def _text_to_ids(self, text):
        """将文本转换为ID序列"""
        words = str(text).split()[:self.max_length]
        ids = [self.vocab_dict.get(word, 1) for word in words]  # 1是<UNK>
        
        # 填充或截断到固定长度
        if len(ids) < self.max_length:
            attention_mask = [1] * len(ids) + [0] * (self.max_length - len(ids))
            ids.extend([0] * (self.max_length - len(ids)))  # 0是<PAD>
        else:
            attention_mask = [1] * self.max_length
            
        return ids, attention_mask
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        input_ids, attention_mask = self._text_to_ids(text)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }

class EmotionModelTrainer:
    """情感模型训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 使用设备: {self.device}")
        
        # 情绪标签（与处理后的数据保持一致）
        self.emotion_columns = [
            '钦佩', '崇拜', '审美欣赏', '娱乐', '愤怒', '焦虑', '敬畏', '尴尬',
            '无聊', '平静', '困惑', '蔑视', '渴望', '失望', '厌恶', '同情',
            '入迷', '嫉妒', '兴奋', '恐惧', '内疚', '恐怖', '兴趣', '快乐',
            '怀旧', '浪漫', '悲伤'
        ]
        
    def load_data(self):
        """加载预处理后的数据"""
        logger.info("📂 加载训练数据...")
        
        data_dir = Path("./data")
        
        # 加载训练和验证数据
        train_df = pd.read_csv(data_dir / "processed_train.csv")
        dev_df = pd.read_csv(data_dir / "processed_dev.csv")
        
        logger.info(f"📊 训练数据: {len(train_df)} 样本")
        logger.info(f"📊 验证数据: {len(dev_df)} 样本")
        
        # 提取文本和标签
        train_texts = train_df['text'].values
        train_labels = train_df[self.emotion_columns].values
        
        dev_texts = dev_df['text'].values
        dev_labels = dev_df[self.emotion_columns].values
        
        return train_texts, train_labels, dev_texts, dev_labels
    
    def train_and_save(self):
        """训练并保存模型"""
        try:
            # 加载数据
            train_texts, train_labels, dev_texts, dev_labels = self.load_data()
            
            # 创建数据集
            logger.info("🏗️ 创建数据集...")
            train_dataset = SimpleEmotionDataset(train_texts, train_labels)
            dev_dataset = SimpleEmotionDataset(dev_texts, dev_labels, 
                                             vocab_dict=train_dataset.vocab_dict)
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
            
            # 初始化模型
            logger.info("🤖 初始化模型...")
            vocab_size = len(train_dataset.vocab_dict)
            model = SimpleEmotionModel(vocab_size=vocab_size).to(self.device)
            
            # 损失函数和优化器
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
            
            # 训练循环
            logger.info("📈 开始训练...")
            num_epochs = 5
            best_f1 = 0.0
            
            for epoch in range(num_epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    # 移动到设备
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 前向传播
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    if batch_idx % 500 == 0:
                        logger.info(f"   Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # 验证阶段
                model.eval()
                dev_loss = 0.0
                all_predictions = []
                all_labels = []
                
                with torch.no_grad():
                    for batch in dev_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                        dev_loss += loss.item()
                        
                        # 收集预测结果
                        predictions = torch.sigmoid(outputs).cpu().numpy()
                        all_predictions.append(predictions)
                        all_labels.append(labels.cpu().numpy())
                
                # 计算指标
                all_predictions = np.vstack(all_predictions)
                all_labels = np.vstack(all_labels)
                
                # 使用0.5阈值进行二值化
                binary_predictions = (all_predictions > 0.5).astype(int)
                
                f1_macro = f1_score(all_labels, binary_predictions, average='macro', zero_division=0)
                f1_micro = f1_score(all_labels, binary_predictions, average='micro', zero_division=0)
                
                avg_train_loss = train_loss / train_batches
                avg_dev_loss = dev_loss / len(dev_loader)
                
                logger.info(f"✅ Epoch {epoch+1}/{num_epochs} 完成:")
                logger.info(f"   训练损失: {avg_train_loss:.4f}")
                logger.info(f"   验证损失: {avg_dev_loss:.4f}")
                logger.info(f"   F1-macro: {f1_macro:.4f}")
                logger.info(f"   F1-micro: {f1_micro:.4f}")
                
                # 保存最佳模型
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    self._save_model(model, train_dataset.vocab_dict, f1_macro)
                    logger.info(f"💾 保存新的最佳模型 (F1: {f1_macro:.4f})")
                
                scheduler.step()
            
            logger.info(f"🎉 训练完成! 最佳F1分数: {best_f1:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_model(self, model, vocab_dict, f1_score):
        """保存模型和相关配置"""
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        
        save_path = models_dir / "simple_emotion_model"
        save_path.mkdir(exist_ok=True)
        
        # 保存模型权重
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'vocab_size': len(vocab_dict),
                'embed_dim': 256,
                'hidden_dim': 512,
                'num_emotions': 27,
                'dropout': 0.3
            },
            'f1_score': f1_score,
            'emotion_columns': self.emotion_columns
        }, save_path / "model.pth")
        
        # 保存词汇表
        with open(save_path / "vocab.json", 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 模型已保存到: {save_path}")

def main():
    """主函数"""
    logger.info("🚀 启动简化版情感模型训练...")
    
    trainer = EmotionModelTrainer()
    success = trainer.train_and_save()
    
    if success:
        logger.info("✅ 训练成功完成!")
        logger.info("🎯 现在你的AC模块可以将文本转换为27维情绪向量了!")
    else:
        logger.error("❌ 训练失败")
    
    return success

if __name__ == "__main__":
    main()