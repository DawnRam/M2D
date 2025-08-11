#!/usr/bin/env python3
"""
项目完整性检查脚本
在运行训练前检查所有组件是否正确配置
"""

import os
import sys
import importlib
from pathlib import Path

def check_python_environment():
    """检查Python环境"""
    print("🔍 检查Python环境...")
    
    # Python版本
    version = sys.version_info
    print(f"  Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python版本太低，需要Python 3.8+")
        return False
    else:
        print("  ✅ Python版本满足要求")
    
    return True


def check_dependencies():
    """检查关键依赖"""
    print("\n🔍 检查关键依赖...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'transformers': 'HuggingFace Transformers',
        'diffusers': 'Diffusers',
        'accelerate': 'Accelerate',
        'wandb': 'Weights & Biases',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA不可用，将使用CPU（训练会很慢）")
    except:
        pass
    
    return True


def check_project_structure():
    """检查项目结构"""
    print("\n🔍 检查项目结构...")
    
    required_dirs = [
        'src',
        'src/data', 
        'src/models',
        'src/training',
        'src/utils',
        'configs',
        'scripts'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'configs/config.py',
        'src/models/panderm_extractor.py',
        'src/models/vae.py', 
        'src/models/unet.py',
        'src/training/trainer.py',
        'src/training/losses.py',
        'src/training/visualization.py',
        'src/data/isic_dataset.py',
        'scripts/train.py',
        'scripts/generate.py'
    ]
    
    all_good = True
    
    # 检查目录
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ 不存在")
            all_good = False
    
    # 检查文件
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} 不存在")
            all_good = False
    
    return all_good


def check_data_setup():
    """检查数据设置"""
    print("\n🔍 检查数据设置...")
    
    try:
        from configs.model_paths import DATA_ROOT, OUTPUT_DIRS
        print(f"  ✅ 配置文件加载成功")
        print(f"  数据根目录: {DATA_ROOT}")
        
        # 检查数据目录
        if os.path.exists(DATA_ROOT):
            print(f"  ✅ 数据根目录存在")
            
            images_dir = os.path.join(DATA_ROOT, "images")
            if os.path.exists(images_dir):
                image_files = [f for f in os.listdir(images_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  ✅ 图像目录: {len(image_files)} 张图像")
                
                if len(image_files) < 10:
                    print("  ⚠ 图像数量很少，建议运行: python scripts/setup_data.py --create-mock")
                    return False
            else:
                print(f"  ❌ 图像目录不存在: {images_dir}")
                return False
        else:
            print(f"  ❌ 数据根目录不存在: {DATA_ROOT}")
            print("  请运行: python scripts/setup_data.py --all")
            return False
        
        # 检查输出目录
        for name, path in OUTPUT_DIRS.items():
            if os.path.exists(path):
                print(f"  ✅ {name}: {path}")
            else:
                print(f"  ❌ {name}目录不存在: {path}")
                os.makedirs(path, exist_ok=True)
                print(f"  ✅ 已创建: {path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置检查失败: {e}")
        return False


def check_model_imports():
    """检查模型导入"""
    print("\n🔍 检查模型导入...")
    
    try:
        from src.models import PanDermFeatureExtractor, VAE, UNet2D
        print("  ✅ 模型类导入成功")
        
        from src.training import PanDermDiffusionTrainer
        print("  ✅ 训练器导入成功")
        
        from src.data import create_dataloaders
        print("  ✅ 数据加载器导入成功")
        
        from src.training.visualization import WandBVisualizer
        print("  ✅ 可视化工具导入成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型导入失败: {e}")
        return False


def check_config_validity():
    """检查配置有效性"""
    print("\n🔍 检查配置有效性...")
    
    try:
        from configs.config import Config
        config = Config()
        print("  ✅ 配置文件加载成功")
        
        # 检查关键配置
        print(f"  批次大小: {config.data.batch_size}")
        print(f"  学习率: {config.training.learning_rate}")
        print(f"  训练步数: {config.training.num_diffusion_steps}")
        print(f"  特征融合类型: {config.model.fusion_type}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置检查失败: {e}")
        return False


def run_basic_model_test():
    """运行基础模型测试"""
    print("\n🔍 运行基础模型测试...")
    
    try:
        import torch
        from configs.config import Config
        from src.models import PanDermFeatureExtractor, VAE, UNet2D
        
        device = torch.device("cpu")  # 使用CPU避免内存问题
        config = Config()
        
        # 测试PanDerm特征提取器
        extractor = PanDermFeatureExtractor(
            model_name="panderm-large",
            freeze_backbone=True,
            feature_dim=256  # 减小维度
        ).to(device)
        
        test_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            features = extractor(test_input)
            print(f"  ✅ PanDerm特征提取器: 输出形状 {features['global'].shape}")
        
        # 测试VAE
        vae = VAE(
            in_channels=3,
            latent_channels=4,
            base_channels=32  # 减小参数量
        ).to(device)
        
        with torch.no_grad():
            results = vae(test_input)
            print(f"  ✅ VAE: 重构形状 {results['reconstruction'].shape}")
        
        # 测试UNet
        unet = UNet2D(
            in_channels=4,
            out_channels=4,
            model_channels=32,  # 减小参数量
            context_dim=256
        ).to(device)
        
        latent_input = torch.randn(2, 4, 56, 56).to(device)  # 小尺寸
        timesteps = torch.randint(0, 1000, (2,)).to(device)
        panderm_feat = torch.randn(2, 256).to(device)
        
        with torch.no_grad():
            output = unet(latent_input, timesteps, panderm_feat)
            print(f"  ✅ UNet: 输出形状 {output['sample'].shape}")
        
        print("  ✅ 所有模型组件测试通过")
        return True
        
    except Exception as e:
        print(f"  ❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主检查函数"""
    print("🔎 PanDerm-Guided Diffusion 项目完整性检查")
    print("=" * 50)
    
    checks = [
        ("Python环境", check_python_environment),
        ("关键依赖", check_dependencies), 
        ("项目结构", check_project_structure),
        ("数据设置", check_data_setup),
        ("模型导入", check_model_imports),
        ("配置文件", check_config_validity),
        ("模型测试", run_basic_model_test)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"  ❌ {check_name}检查时发生错误: {e}")
            results[check_name] = False
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 检查结果总结:")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 所有检查通过！系统已准备好进行训练")
        print("\n下一步:")
        print("1. 快速测试: python scripts/train.py --epochs 5 --batch_size 4")
        print("2. 完整训练: python scripts/train.py --epochs 50 --use_wandb")
        print("3. 一键启动: run_training.bat (Windows) 或 ./run_training.sh (Linux/Mac)")
        return True
    else:
        print("⚠ 存在问题，请根据上述检查结果进行修复")
        print("\n建议解决步骤:")
        if not results.get("关键依赖", True):
            print("- 安装依赖: pip install -r requirements.txt")
        if not results.get("数据设置", True):
            print("- 设置数据: python scripts/setup_data.py --all")
        if not results.get("项目结构", True):
            print("- 检查项目文件是否完整")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠ 用户中断检查")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 检查过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)