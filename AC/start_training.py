#!/usr/bin/env python3
"""
AC情感计算模块完整训练启动脚本
支持xlm-roberta模型微调和C&K情绪体系
"""

import os
import sys
import logging
import torch
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境和依赖"""
    logger.info("🔍 检查训练环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        logger.error(f"❌ Python版本过低: {python_version}, 需要Python >= 3.8")
        return False
    
    # 检查必要包
    required_packages = [
        'torch', 'transformers', 'pandas', 'numpy', 
        'sklearn', 'datasets', 'accelerate'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}: 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}: 未安装")
    
    if missing_packages:
        logger.error(f"❌ 缺少依赖包: {missing_packages}")
        logger.info("💡 请运行: pip install " + " ".join(missing_packages))
        return False
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        logger.info(f"🚀 GPU可用: {torch.cuda.get_device_name()}")
        logger.info(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        logger.info("💻 使用CPU训练 (建议使用GPU加速)")
    
    # 检查数据文件
    data_dir = Path("./data")
    required_files = ["processed_train.csv", "processed_dev.csv", "processed_test.csv"]
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            logger.info(f"✅ 数据文件: {file_name}")
        else:
            logger.error(f"❌ 缺少数据文件: {file_name}")
            return False
    
    return True

def start_training():
    """启动模型训练"""
    logger.info("🚀 开始AC情感模型训练...")
    
    try:
        # 导入xlm-roberta训练模块 (依赖问题已解决)
        from model_trainer import ModelTrainer
        from pathlib import Path
        
        # 创建训练器
        trainer = ModelTrainer()
        
        # 开始训练
        logger.info("📈 启动xlm-roberta情感模型微调...")
        
        # 准备数据
        logger.info("📂 准备训练数据...")
        data_path = "./data/processed_train.csv"
        texts, labels = trainer.prepare_data(data_path)
        
        # 创建数据集
        train_dataset, val_dataset, test_dataset = trainer.create_datasets(texts, labels)
        
        # 训练模型
        output_dir = "./models/finetuned_xlm_roberta"
        trainer.train_model(train_dataset, val_dataset, output_dir)
        
        # 评估模型
        logger.info("📊 评估训练完成的模型...")
        eval_results = trainer.evaluate_model(test_dataset, output_dir)
        
        logger.info("🎯 评估结果:")
        for metric, value in eval_results.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        success = True
        
        if success:
            logger.info("🎉 模型训练成功完成!")
            logger.info("📍 模型保存位置: ./models/finetuned_xlm_roberta/")
            logger.info("🔧 现在可以使用训练好的模型进行情感分析了!")
            return True
        else:
            logger.error("❌ 模型训练失败")
            return False
            
    except ImportError as e:
        logger.error(f"❌ 导入训练模块失败: {e}")
        logger.info("💡 请确保所有依赖已正确安装")
        return False
    except Exception as e:
        logger.error(f"❌ 训练过程出错: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 AC情感计算模块 - xlm-roberta模型训练")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        return False
    
    print("\n✅ 环境检查通过!")
    
    # 确认开始训练
    response = input("\n🤖 是否开始训练? (y/n): ").strip().lower()
    if response != 'y':
        print("⏸️ 训练已取消")
        return False
    
    # 开始训练
    success = start_training()
    
    if success:
        print("\n🎉 训练完成! 你的AC模块现在可以:")
        print("   • 输入中文/英文文本")
        print("   • 输出27维C&K情绪向量") 
        print("   • 与KG模块完美集成")
        print("\n📝 接下来可以:")
        print("   1. 测试模型效果: python test_model.py")
        print("   2. 集成到主系统: python integration_test.py")
    else:
        print("\n❌ 训练失败，请检查日志文件 training.log")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)