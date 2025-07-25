# 🧪 qm_final4 系统测试指南

## 快速开始测试流程

### 第一步：环境准备
```bash
# 进入项目目录
cd "/Users/wanxinchen/Study/AI/Project/Final project/SuperClaude/qm_final4"

# 安装CLAMP3依赖（关键步骤）
pip install -r CLAMP3/requirements.txt

# 验证基础环境
python -c "import numpy, scipy; print('✅ 基础依赖OK')"
ffmpeg -version | head -1  # 验证ffmpeg
```

### 第二步：系统组件测试
```bash
# 运行完整集成测试
python test_clamp3_integration.py
```

**期望结果**:
- ✅ CLAMP3可用性 - 通过
- ✅ 视频处理 - 通过  
- ✅ CLAMP3特征提取 - 通过
- ✅ 音频特征提取器 - 通过
- ✅ 检索引擎 - 通过
- ✅ 完整流水线 - 通过

### 第三步：素材库验证
```bash
# 检查素材库状态
python build_full_material_library.py status

# 如果素材不足，运行优先级构建
python build_priority_material_library.py
```

**期望结果**:
- 至少70个5分钟片段
- 总存储 > 10GB
- 多种时长分布

### 第四步：MVP系统启动
```bash
# 启动完整系统
python quick_start.py

# 或者单独启动gradio界面
python gradio_retrieval_4.0.py
```

**期望结果**:
- Gradio界面在浏览器打开
- 可以输入情绪文本
- 返回疗愈视频

## 🔍 详细测试场景

### 测试场景1：基础功能验证
```bash
# 测试情绪识别
python -c "
from core.emotion_mapper import detect_emotion_enhanced
emotion, confidence = detect_emotion_enhanced('我今天很焦虑，睡不着觉')
print(f'情绪: {emotion}, 置信度: {confidence:.3f}')
"
```

### 测试场景2：特征提取验证
```bash
# 测试CLAMP3特征提取
python -c "
from core.feature_extractor import AudioFeatureExtractor
import os

extractor = AudioFeatureExtractor()
# 找一个5分钟片段测试
test_video = 'materials/segments/5min/32_seg000_5min.mp4'
if os.path.exists(test_video):
    features = extractor.extract_video_features(test_video)
    if features:
        print('✅ CLAMP3特征提取成功')
        print(f'特征维度: {features.get(\"clamp3_features\", \"unknown\")}')
    else:
        print('❌ 特征提取失败')
else:
    print('❌ 测试视频不存在')
"
```

### 测试场景3：检索功能验证
```bash
# 测试视频检索
python -c "
from core.retrieval_engine import VideoRetrievalEngine
from core.emotion_mapper import detect_emotion_enhanced

# 初始化检索引擎
engine = VideoRetrievalEngine()

# 测试情绪检索
test_emotions = ['焦虑', '悲伤', '平静', '快乐']
for emotion in test_emotions:
    results = engine.retrieve_videos(emotion, top_k=3)
    print(f'{emotion}: 找到 {len(results)} 个匹配视频')
    for i, (path, score, info) in enumerate(results[:2], 1):
        print(f'  {i}. 相似度: {score:.3f}')
"
```

### 测试场景4：端到端流程验证
```bash
# 完整流程测试
python test_mvp.py
```

## 🐛 常见问题排查

### 问题1：CLAMP3依赖缺失
**症状**: ModuleNotFoundError: No module named 'accelerate'
```bash
# 解决方案
pip install accelerate transformers torch torchvision torchaudio
pip install -r CLAMP3/requirements.txt
```

### 问题2：ffmpeg不可用
**症状**: ffmpeg: command not found
```bash
# macOS解决方案
brew install ffmpeg

# 或者使用conda
conda install ffmpeg
```

### 问题3：素材库为空
**症状**: 特征数据库为空，请先提取视频特征
```bash
# 解决方案：构建素材库
python build_priority_material_library.py
```

### 问题4：CLAMP3特征提取失败
**症状**: CLAMP3执行失败
```bash
# 检查CLAMP3环境
cd CLAMP3
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from transformers import AutoModel; print('Transformers OK')"

# 手动测试CLAMP3
mkdir -p test_audio
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" test_audio/test.wav
python clamp3_embd.py test_audio test_output --get_global
```

## 📊 性能基准测试

### 基准测试1：特征提取速度
```bash
python -c "
import time
from core.feature_extractor import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
test_video = 'materials/segments/5min/32_seg000_5min.mp4'

start_time = time.time()
features = extractor.extract_video_features(test_video)
end_time = time.time()

if features:
    print(f'✅ 特征提取耗时: {end_time - start_time:.1f}秒')
else:
    print('❌ 特征提取失败')
"
```

### 基准测试2：检索响应时间
```bash
python -c "
import time
from core.retrieval_engine import VideoRetrievalEngine

engine = VideoRetrievalEngine()

start_time = time.time()
results = engine.retrieve_videos('焦虑', top_k=5)
end_time = time.time()

print(f'✅ 检索耗时: {end_time - start_time:.3f}秒')
print(f'返回结果: {len(results)}个')
"
```

## 🎯 MVP演示脚本

### 快速演示流程
```bash
# 1. 启动系统
python gradio_retrieval_4.0.py

# 2. 在浏览器中测试以下输入：
# - "我今天很焦虑，工作压力大，晚上睡不着"
# - "心情很低落，需要一些温暖的音乐"
# - "想要放松一下，听点平静的音乐"
# - "感觉很兴奋，但需要慢慢calm down"

# 3. 观察系统响应：
# - 情绪识别结果
# - 返回的疗愈视频
# - 相似度分数
```

## 🔬 学术验证测试

### 验证ISO原则实现
```bash
python -c "
from core.video_processor import VideoProcessor

processor = VideoProcessor()
segments = processor.get_intro_segments(5)  # 获取5分钟intro片段

print(f'✅ ISO匹配阶段片段: {len(segments)}个')
for seg in segments[:3]:
    print(f'  - {seg[\"segment_name\"]}: intro_ratio={seg[\"intro_ratio\"]}')
"
```

### 验证27维情绪系统
```bash
python -c "
from core.emotion_mapper import emotion_recognizer

test_texts = [
    '我很开心', '感到沮丧', '非常焦虑', 
    '心情平静', '有点烦躁', '深度悲伤'
]

print('✅ 27维情绪识别测试:')
for text in test_texts:
    emotion, confidence = emotion_recognizer.detect_emotion(text)
    print(f'  \"{text}\" → {emotion} ({confidence:.3f})')
"
```

## 📈 系统健康检查

```bash
# 运行完整健康检查
python -c "
import os
from pathlib import Path

print('🏥 系统健康检查')
print('=' * 30)

# 检查核心文件
core_files = [
    'core/emotion_mapper.py',
    'core/feature_extractor.py', 
    'core/retrieval_engine.py',
    'core/video_processor.py'
]

for file in core_files:
    status = '✅' if Path(file).exists() else '❌'
    print(f'{status} {file}')

# 检查CLAMP3
clamp3_status = '✅' if Path('CLAMP3/clamp3_embd.py').exists() else '❌'
print(f'{clamp3_status} CLAMP3模型')

# 检查素材库
segments_count = len(list(Path('materials/segments').rglob('*.mp4')))
print(f'📹 视频片段: {segments_count}个')

print('\n🎯 系统状态: ', end='')
if segments_count > 50:
    print('🟢 就绪')
else:
    print('🟡 需要构建素材库')
"
```

## 🚀 生产就绪检查

```bash
# 最终就绪检查
python -c "
def check_system_readiness():
    import subprocess
    import sys
    from pathlib import Path
    
    checks = []
    
    # 检查Python依赖
    try:
        import numpy, scipy, gradio
        checks.append(('Python依赖', True))
    except ImportError as e:
        checks.append(('Python依赖', False, str(e)))
    
    # 检查ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        checks.append(('ffmpeg', result.returncode == 0))
    except:
        checks.append(('ffmpeg', False))
    
    # 检查CLAMP3
    clamp3_ready = Path('CLAMP3/clamp3_embd.py').exists()
    checks.append(('CLAMP3', clamp3_ready))
    
    # 检查素材库
    segments = len(list(Path('materials/segments').rglob('*.mp4')))
    checks.append(('素材库', segments > 50, f'{segments}个片段'))
    
    print('🚀 生产就绪检查')
    print('=' * 25)
    
    all_passed = True
    for check in checks:
        name = check[0]
        passed = check[1]
        detail = check[2] if len(check) > 2 else ''
        
        status = '✅' if passed else '❌'
        print(f'{status} {name:<12} {detail}')
        
        if not passed:
            all_passed = False
    
    print('\n🎯 整体状态:', '🟢 系统就绪' if all_passed else '🟡 需要修复')
    return all_passed

check_system_readiness()
"
```

---

## 💡 测试建议

1. **按顺序执行**: 先基础环境，再组件测试，最后系统测试
2. **保存日志**: 出现问题时保存错误信息
3. **性能监控**: 关注特征提取和检索的响应时间
4. **用户体验**: 测试多种情绪输入，验证结果的相关性

**祝您测试顺利！🎊**