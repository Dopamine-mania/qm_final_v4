# SuperClaude音乐特征提取方案

## 项目概述
基于CLAMP3模型的音乐特征提取系统，用于SuperClaude毕设项目中的素材库音乐分析。

## 环境配置
- **环境名称**: qm_final4
- **Python版本**: 3.10.16
- **PyTorch版本**: 2.5.1 (支持MPS加速)
- **系统**: macOS (Apple M2, 16GB RAM)

## CLAMP3模型特性
- **多模态支持**: 文本、音频、MIDI、乐谱、图像
- **音频格式**: .mp3, .wav (最大640秒)
- **特征维度**: 768维全局特征向量
- **预训练权重**: SAAS版本(优化音频处理)

## 素材库音乐特征提取流程

### 1. 数据准备
```bash
# 素材库结构
materials/
├── segments_3min/          # 3分钟音频片段
│   ├── audio1.mp3
│   ├── audio2.wav
│   └── ...
└── features/              # 提取的特征文件
    ├── audio1.npy
    ├── audio2.npy
    └── ...
```

### 2. 特征提取命令
```bash
# 激活环境
conda activate qm_final4

# 批量提取特征
python clamp3_embd.py materials/segments_3min/ materials/features/ --get_global
```

### 3. 特征文件格式
- **输出**: .npy文件
- **维度**: (1, 768)
- **类型**: float32全局语义特征

### 4. 集成到SuperClaude系统
```python
import numpy as np
import torch

def load_music_features(feature_path):
    """加载音乐特征向量"""
    features = np.load(feature_path)
    return torch.tensor(features, dtype=torch.float32)

def music_similarity(feat1, feat2):
    """计算音乐相似度"""
    similarity = torch.cosine_similarity(feat1, feat2, dim=1)
    return similarity.item()
```

## 优化建议

### 性能优化
1. **批量处理**: 一次处理多个音频文件
2. **缓存机制**: 避免重复提取相同文件
3. **MPS加速**: 利用M2芯片GPU加速

### 存储优化
1. **特征压缩**: 使用float16降低存储需求
2. **索引建立**: 为快速检索建立特征索引
3. **元数据存储**: 记录音频文件与特征的映射关系

## 使用示例

### 基本特征提取
```bash
cd /path/to/clamp3
conda activate qm_final4
python clamp3_embd.py input_audio/ output_features/ --get_global
```

### 相似度检索
```bash
python clamp3_search.py query.mp3 reference_music_folder/
```

### 语义相似度计算
```bash
python clamp3_score.py query_folder/ reference_folder/
```

## 注意事项

1. **权重文件**: 确保SAAS版本权重文件完整下载(2.57GB)
2. **内存使用**: 大批量处理时注意内存管理
3. **音频格式**: 支持.mp3/.wav，自动检测格式
4. **处理时间**: 约2秒/音频文件(包含MERT特征提取)

## 后续集成计划

1. **检索系统**: 基于特征向量的音乐检索
2. **推荐算法**: 利用语义相似度推荐相似音乐
3. **可视化**: 特征降维可视化音乐分布
4. **API接口**: 提供RESTful API供其他模块调用

## 技术架构

```
SuperClaude系统
├── 音频输入模块
├── CLAMP3特征提取
├── 特征存储与索引
├── 相似度计算
├── 推荐生成
└── 结果输出
```

---
*创建日期: 2025-07-18*
*环境: qm_final4*
*模型: CLAMP3 SAAS*