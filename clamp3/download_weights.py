#!/usr/bin/env python3
"""
CLAMP3权重文件下载脚本
下载SAAS版本的预训练权重用于音频特征提取
"""
import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """下载大文件并显示进度条"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def main():
    # CLAMP3 SAAS权重文件URL
    weights_url = "https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
    
    # 目标文件路径
    weights_path = "code/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
    
    if os.path.exists(weights_path):
        print(f"权重文件已存在: {weights_path}")
        return
    
    print("开始下载CLAMP3 SAAS权重文件...")
    print(f"URL: {weights_url}")
    print(f"目标: {weights_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    try:
        download_file(weights_url, weights_path)
        print(f"✅ 权重文件下载完成: {weights_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(weights_path)
        print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if os.path.exists(weights_path):
            os.remove(weights_path)

if __name__ == "__main__":
    main()