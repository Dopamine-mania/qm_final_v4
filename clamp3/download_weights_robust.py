#!/usr/bin/env python3
"""
CLAMP3权重文件稳定下载脚本
支持断点续传和网络重连
"""
import os
import requests
import time
from tqdm import tqdm

def download_file_with_resume(url, filename, max_retries=5, chunk_size=8192):
    """带断点续传的文件下载"""
    headers = {}
    initial_pos = 0
    
    # 检查是否有未完成的下载
    if os.path.exists(filename):
        initial_pos = os.path.getsize(filename)
        headers['Range'] = f'bytes={initial_pos}-'
        print(f"断点续传，从 {initial_pos / 1024 / 1024:.1f} MB 开始")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # 获取文件总大小
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])
            else:
                total_size = int(response.headers.get('content-length', 0)) + initial_pos
            
            mode = 'ab' if initial_pos > 0 else 'wb'
            
            with open(filename, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit='B',
                    unit_scale=True,
                    desc=os.path.basename(filename)
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ 下载完成: {filename}")
            return True
            
        except (requests.exceptions.RequestException, KeyboardInterrupt) as e:
            print(f"❌ 下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"等待 {2 ** attempt} 秒后重试...")
                time.sleep(2 ** attempt)
            else:
                print("达到最大重试次数，下载失败")
                return False
    
    return False

def verify_file_size(filename, expected_size=None):
    """验证文件大小"""
    if not os.path.exists(filename):
        return False
    
    actual_size = os.path.getsize(filename)
    print(f"文件大小: {actual_size / 1024 / 1024:.1f} MB")
    
    if expected_size and actual_size != expected_size:
        print(f"⚠️  文件大小不匹配，期望: {expected_size / 1024 / 1024:.1f} MB")
        return False
    
    return True

def main():
    # CLAMP3 SAAS权重文件
    weights_url = "https://huggingface.co/sander-wood/clamp3/resolve/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
    weights_path = "code/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
    
    print("开始下载CLAMP3 SAAS权重文件...")
    print(f"目标文件: {weights_path}")
    print("支持断点续传，按 Ctrl+C 暂停，重新运行继续下载")
    print("-" * 60)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    # 下载文件
    success = download_file_with_resume(weights_url, weights_path)
    
    if success:
        print("✅ 权重文件下载成功！")
        verify_file_size(weights_path)
        
        # 测试PyTorch加载
        try:
            import torch
            print("🔍 验证权重文件...")
            checkpoint = torch.load(weights_path, map_location='cpu')
            print(f"✅ 权重文件验证通过，包含 {len(checkpoint)} 个模块")
        except Exception as e:
            print(f"⚠️  权重文件验证失败: {e}")
    else:
        print("❌ 权重文件下载失败")
        return False
    
    return True

if __name__ == "__main__":
    main()