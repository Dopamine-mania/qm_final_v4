#!/usr/bin/env python3
"""
测试简化版训练完成的情感模型
"""

import torch
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmotionModel(torch.nn.Module):
    """简化的情感分类模型 - 与训练时保持一致"""
    
    def __init__(self, vocab_size=50000, embed_dim=256, hidden_dim=512, num_emotions=27, dropout=0.3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, 
                                 dropout=dropout, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim * 2, num_emotions)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long() - 1
            batch_size = lstm_out.size(0)
            last_outputs = lstm_out[range(batch_size), lengths]
        else:
            last_outputs = lstm_out[:, -1, :]
        
        output = self.dropout(last_outputs)
        logits = self.classifier(output)
        return logits

class SimpleEmotionInference:
    """简化版情感推理类"""
    
    def __init__(self, model_path="./models/simple_emotion_model"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # 加载模型和配置
        self._load_model()
        
    def _load_model(self):
        """加载训练好的模型"""
        logger.info(f"📂 加载模型: {self.model_path}")
        
        # 加载模型权重和配置
        checkpoint = torch.load(self.model_path / "model.pth", map_location=self.device)
        
        self.model_config = checkpoint['model_config']
        self.emotion_columns = checkpoint['emotion_columns']
        
        # 初始化模型
        self.model = SimpleEmotionModel(**self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 加载词汇表
        with open(self.model_path / "vocab.json", 'r', encoding='utf-8') as f:
            self.vocab_dict = json.load(f)
        
        logger.info(f"✅ 模型加载成功!")
        logger.info(f"   词汇表大小: {len(self.vocab_dict)}")
        logger.info(f"   情绪维度: {len(self.emotion_columns)}")
        logger.info(f"   训练F1分数: {checkpoint.get('f1_score', 'N/A'):.4f}")
        
    def _text_to_ids(self, text, max_length=128):
        """将文本转换为ID序列"""
        words = str(text).split()[:max_length]
        ids = [self.vocab_dict.get(word, 1) for word in words]  # 1是<UNK>
        
        # 填充到固定长度
        if len(ids) < max_length:
            attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            ids.extend([0] * (max_length - len(ids)))  # 0是<PAD>
        else:
            attention_mask = [1] * max_length
            
        return ids, attention_mask
    
    def predict_emotion_vector(self, text):
        """预测文本的27维情绪向量"""
        # 预处理文本
        input_ids, attention_mask = self._text_to_ids(text)
        
        # 转换为tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        return probabilities
    
    def analyze_emotion(self, text):
        """分析文本情绪并返回详细结果"""
        vector = self.predict_emotion_vector(text)
        
        # 分析结果
        emotion_dict = dict(zip(self.emotion_columns, vector))
        
        # 排序找出最强的情绪
        sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'text': text,
            'emotion_vector': vector,
            'emotion_dict': emotion_dict,
            'top_emotions': sorted_emotions[:5],
            'max_emotion': sorted_emotions[0][0],
            'max_intensity': sorted_emotions[0][1],
            'total_intensity': float(np.sum(vector)),
            'active_emotions': int(np.sum(vector > 0.1))
        }

def test_trained_model():
    """测试训练好的模型"""
    print("🧪 测试简化版训练完成的情感模型...")
    
    try:
        # 检查模型是否存在
        model_path = Path("./models/simple_emotion_model")
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            print("💡 请先运行训练脚本: python start_training.py")
            return False
        
        # 初始化推理器
        inference = SimpleEmotionInference()
        
        # 测试用例
        test_texts = [
            "我今天非常开心，天气很好！",
            "这首歌让我很感动，想起了童年时光。", 
            "我对这件事感到很愤怒和失望。",
            "看到这个消息我很震惊，简直不敢相信。",
            "他的表现让我感到非常钦佩。",
            "这个故事太有趣了，让我捧腹大笑。",
            "我为自己的行为感到羞耻和内疚。"
        ]
        
        print("\n📊 情感分析测试结果:")
        print("=" * 80)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n🔍 测试 {i}: {text}")
            
            # 分析情绪
            result = inference.analyze_emotion(text)
            
            print(f"   📈 27维向量: [{result['emotion_vector'][0]:.3f}, {result['emotion_vector'][1]:.3f}, {result['emotion_vector'][2]:.3f}, ...]")
            print(f"   🎯 最强情绪: {result['max_emotion']} (强度: {result['max_intensity']:.3f})")
            print(f"   📊 总体情绪强度: {result['total_intensity']:.3f}")
            print(f"   🔢 活跃情绪数量: {result['active_emotions']}")
            
            # 显示前3个最强情绪
            print("   🏆 前3个最强情绪:")
            for emotion, intensity in result['top_emotions'][:3]:
                print(f"      {emotion}: {intensity:.3f}")
        
        print(f"\n✅ 模型测试完成!")
        print("🎉 你的AC模块现在可以将文本转换为27维情绪向量了!")
        print("\n💡 集成到你的系统中:")
        print("   from test_simple_model import SimpleEmotionInference")
        print("   inference = SimpleEmotionInference()")
        print("   vector = inference.predict_emotion_vector('你的文本')")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trained_model()
    if not success:
        exit(1)