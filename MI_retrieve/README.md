# 🎵 音乐信息检索系统 (MI_retrieve)

## 📋 项目简介

基于CLAMP3的智能音乐检索与播放系统，专为疗愈音乐体验设计。

## 🏗️ 项目结构

```
MI_retrieve/
├── music_retrieval_ui_fixed.py     # 完整版Web界面
├── music_retrieval_ui_simple.py    # 简化版Web界面  
├── music_search_api.py              # 统一API接口
├── music_search_system.py           # 核心检索引擎
├── clamp3_embd.py                   # CLAMP3特征提取
├── code/                            # CLAMP3模型核心代码
├── music_features/                  # 音乐特征文件库
├── retrieve_libraries/              # 音乐素材库
├── gradio_retrieval_4.0.py         # Gradio检索系统
├── music_retrieval_demo.py          # 检索演示脚本
├── music_search_demo.py             # 搜索演示脚本
├── test_music_search.py             # 功能测试脚本
├── test_single_extraction.py        # 单文件测试脚本
└── MUSIC_RETRIEVAL_DEVELOPMENT_GUIDE.md  # 完整开发指南
```

## 🚀 快速启动

### 简化版界面（推荐）
```bash
cd MI_retrieve
python music_retrieval_ui_simple.py
```

### 完整版界面
```bash
cd MI_retrieve  
python music_retrieval_ui_fixed.py
```

## 🎯 核心功能

- **智能检索**: 基于音乐特征描述的语义匹配
- **多时长支持**: 1min-30min六种版本
- **随机选择**: 从Top-3结果中随机选择
- **即时播放**: Web界面直接播放
- **公网访问**: 支持Gradio公网分享

## 📖 详细文档

参见 `MUSIC_RETRIEVAL_DEVELOPMENT_GUIDE.md` 获取完整的开发指南和集成方法。

---

*基于CLAMP3的智能音乐检索系统 | 版本: v1.0.0*