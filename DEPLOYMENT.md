# 部署文档 - 音乐疗愈AI系统 4.0版本

## 📋 部署概览

本文档提供音乐疗愈AI系统4.0版本的完整部署指南，包括环境配置、依赖安装、系统初始化和运行维护。

### 系统信息
- **项目名称**: 音乐疗愈AI系统 4.0版本
- **架构类型**: 检索驱动架构
- **部署难度**: ⭐⭐⭐ (中等)
- **硬件要求**: CPU密集型，无GPU依赖
- **部署时间**: 约15-30分钟

## 🔧 环境要求

### 基础环境
```bash
# Python版本
Python 3.7+

# 操作系统支持
✅ Linux (推荐 Ubuntu 18.04+)
✅ macOS 10.14+
✅ Windows 10+

# 硬件配置
CPU: 4核心以上 (推荐8核心)
内存: 4GB+ (推荐8GB+)
存储: 20GB+ 可用空间
网络: 稳定互联网连接
```

### 系统依赖
```bash
# ffmpeg (必需)
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# CentOS/RHEL
sudo yum install epel-release
sudo yum install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 下载ffmpeg并添加到系统PATH
```

## 📦 安装部署

### 第一步：获取代码
```bash
# 克隆仓库
git clone https://github.com/Dopamine-mania/qm_final_v4.git
cd qm_final_v4

# 检查文件完整性
ls -la
# 应该看到 core/, materials/, *.py 等文件
```

### 第二步：Python环境配置
```bash
# 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 第三步：验证依赖安装
```bash
# 验证核心依赖
python -c "import gradio; print('Gradio:', gradio.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import scipy; print('SciPy:', scipy.__version__)"

# 验证ffmpeg
ffmpeg -version | head -1
```

### 第四步：视频素材准备
```bash
# 创建素材目录结构
mkdir -p materials/video materials/segments materials/features

# 放置疗愈视频文件
# 将 32.mp4 和 56.mp4 放置到 materials/video/ 目录
# 每个文件约3.5小时时长

# 验证视频文件
ls -lh materials/video/
# 应该看到两个大型MP4文件
```

## 🚀 系统启动

### 快速启动 (推荐)
```bash
# 一键启动脚本
python quick_start.py

# 脚本会自动执行：
# 1. 环境检查
# 2. 视频文件验证
# 3. MVP测试 (可选)
# 4. Web界面启动
```

### 手动启动
```bash
# 1. 运行系统测试
python test_mvp.py

# 2. 启动Web界面
python gradio_retrieval_4.0.py

# 3. 访问本地地址
# 浏览器打开: http://localhost:7870
```

### 生产环境启动
```bash
# 使用nohup后台运行
nohup python gradio_retrieval_4.0.py > app.log 2>&1 &

# 或使用screen会话
screen -S therapy_app
python gradio_retrieval_4.0.py
# Ctrl+A, D 分离会话
```

## 🔍 健康检查

### 系统状态检查
```bash
# 检查服务运行状态
ps aux | grep gradio_retrieval

# 检查端口占用
netstat -tlnp | grep 7870

# 检查日志文件
tail -f app.log
```

### 功能验证测试
```bash
# 运行完整MVP测试
python test_mvp.py

# 预期输出应包含：
# ✅ 情绪识别测试完成
# ✅ 视频处理测试完成  
# ✅ 特征提取测试完成
# ✅ 检索引擎测试完成
# ✅ 疗愈选择器测试完成
```

### Web界面验证
```bash
# 访问健康检查端点
curl http://localhost:7870/

# 验证界面功能
# 1. 打开浏览器访问 http://localhost:7870
# 2. 点击"初始化系统"按钮
# 3. 输入测试情绪："我感到很焦虑"
# 4. 点击"获取疗愈视频推荐"
# 5. 验证视频播放功能
```

## 🛠️ 配置选项

### 端口配置
```python
# 修改 gradio_retrieval_4.0.py 中的端口设置
app.launch(
    server_port=7870,  # 修改为所需端口
    share=True         # 设置为False禁用公共链接
)
```

### 性能优化配置
```python
# 在 core/feature_extractor.py 中调整
BATCH_SIZE = 32        # 批处理大小
CACHE_ENABLED = True   # 启用特征缓存
MAX_WORKERS = 4        # 最大工作线程数
```

### 日志级别配置
```python
# 在各模块中调整日志级别
logging.basicConfig(level=logging.INFO)  # INFO, DEBUG, WARNING, ERROR
```

## 📊 监控维护

### 性能监控
```bash
# 系统资源监控
top -p $(pgrep -f gradio_retrieval)

# 内存使用监控
free -h

# 磁盘空间监控
df -h
```

### 日志监控
```bash
# 实时日志监控
tail -f app.log

# 错误日志过滤
grep -i error app.log

# 性能日志分析
grep -i "处理时间\|响应时间" app.log
```

### 定期维护
```bash
# 清理临时文件
find /tmp -name "therapy_*" -mtime +7 -delete

# 特征缓存清理
rm -rf materials/features/*.tmp

# 日志轮转
logrotate /path/to/logrotate.conf
```

## 🚨 故障排除

### 常见问题

**1. ffmpeg not found**
```bash
# 解决方案：安装ffmpeg
# 参考"系统依赖"部分的安装命令
```

**2. 端口被占用**
```bash
# 查找占用进程
sudo lsof -i :7870

# 杀死占用进程
sudo kill -9 <PID>

# 或修改应用端口
```

**3. 视频文件缺失**
```bash
# 检查视频文件
ls -la materials/video/

# 确保文件名正确：32.mp4, 56.mp4
# 文件大小应该较大（GB级别）
```

**4. 内存不足**
```bash
# 监控内存使用
free -h

# 增加系统swap
sudo dd if=/dev/zero of=/swapfile bs=1G count=4
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 错误代码参考

| 错误代码 | 描述 | 解决方案 |
|----------|------|----------|
| E001 | ffmpeg未安装 | 安装ffmpeg |
| E002 | 视频文件缺失 | 检查materials/video/目录 |
| E003 | 端口占用 | 修改端口或杀死占用进程 |
| E004 | 依赖包缺失 | 重新安装requirements.txt |
| E005 | 权限不足 | 检查文件权限 |

## 🔒 安全建议

### 网络安全
```bash
# 防火墙配置
sudo ufw allow 7870/tcp

# 反向代理配置 (nginx)
server {
    listen 80;
    server_name your-domain.com;
    location / {
        proxy_pass http://localhost:7870;
    }
}
```

### 文件权限
```bash
# 设置适当的文件权限
chmod 755 *.py
chmod 644 *.md *.txt
chmod -R 755 core/
chmod -R 644 materials/
```

## 📈 性能基准

### 预期性能指标
- **系统启动时间**: <30秒
- **初始化时间**: 2-5分钟（首次）
- **情绪识别延迟**: <100ms
- **视频推荐延迟**: <500ms  
- **并发用户数**: 10-50用户
- **内存占用**: 2-4GB
- **CPU占用**: 20-40%（空闲时）

### 压力测试
```bash
# 简单并发测试
for i in {1..10}; do
    curl -X POST http://localhost:7870/api/predict \
         -d '{"data": ["我感到很焦虑"]}' &
done
wait
```

## 🆘 技术支持

### 获取帮助
- **GitHub Issues**: https://github.com/Dopamine-mania/qm_final_v4/issues
- **项目文档**: README.md
- **技术规格**: 查看core/模块注释

### 收集诊断信息
```bash
# 生成诊断报告
python -c "
import sys, platform, subprocess
print('Python:', sys.version)
print('Platform:', platform.platform())
print('ffmpeg:', subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\\n')[0])
"

# 检查文件结构
find . -name "*.py" | head -10
```

---

**部署完成后，您的音乐疗愈AI系统4.0版本应该已经成功运行！**

*最后更新: 2024年7月*