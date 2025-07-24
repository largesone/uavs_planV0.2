# -*- coding: utf-8 -*-
# 文件名: install_rllib.py
# 描述: 安装Ray RLlib和相关依赖

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ 安装 {package} 失败")
        return False

def main():
    """安装RLlib相关依赖"""
    print("=== 安装Ray RLlib依赖 ===")
    
    # 必需的包列表
    required_packages = [
        "ray[rllib]>=2.0.0",
        "gymnasium>=0.28.0",
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
    ]
    
    # 可选但推荐的包
    optional_packages = [
        "tensorboard>=2.8.0",
        "wandb>=0.12.0",
        "optuna>=3.0.0",
    ]
    
    print("安装必需的包...")
    success_count = 0
    for package in required_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n必需包安装完成: {success_count}/{len(required_packages)}")
    
    if success_count < len(required_packages):
        print("警告: 部分必需包安装失败，可能影响功能")
    
    print("\n安装可选的包...")
    optional_success = 0
    for package in optional_packages:
        if install_package(package):
            optional_success += 1
    
    print(f"可选包安装完成: {optional_success}/{len(optional_packages)}")
    
    print("\n=== 安装完成 ===")
    print("现在可以运行RLlib训练了!")
    print("\n使用示例:")
    print("python main_rllib.py --scenario simple --episodes 1000")
    print("python main_rllib.py --mode compare --scenario complex")

if __name__ == "__main__":
    main() 