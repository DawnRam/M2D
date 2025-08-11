"""
实验管理工具
管理实验目录、版本控制和文件组织
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path


class ExperimentManager:
    """实验管理器"""
    
    def __init__(
        self, 
        experiment_root: str = "/nfs/scratch/eechengyang/Code/logs",
        experiment_name: Optional[str] = None,
        auto_create: bool = True
    ):
        """
        Args:
            experiment_root: 实验根目录
            experiment_name: 实验名称，如果为None则自动生成
            auto_create: 是否自动创建目录
        """
        self.experiment_root = Path(experiment_root)
        
        # 生成带时间戳的实验名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            self.experiment_name = f"panderm_diffusion_{timestamp}"
        else:
            # 如果提供了实验名称，在后面添加时间戳
            self.experiment_name = f"{experiment_name}_{timestamp}"
        
        # 实验目录：/logs/experiment_name_timestamp/
        self.experiment_dir = self.experiment_root / self.experiment_name
        
        # 创建实验目录结构
        if auto_create:
            self.create_experiment_dirs()
    
    def create_experiment_dirs(self):
        """创建实验目录结构"""
        
        # 首先创建实验根目录
        self.experiment_root.mkdir(parents=True, exist_ok=True)
        
        # 然后创建带时间戳的实验目录
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建实验主目录: {self.experiment_dir}")
        
        # 在实验目录下创建子目录
        subdirs_to_create = [
            "checkpoints",      # 模型检查点
            "logs",            # 训练日志
            "outputs",         # 输出文件
            "generated_images", # 生成的图像
            "cache",           # 缓存文件
            "wandb",           # WandB日志
            "configs",         # 配置文件备份
            "code_backup",     # 代码备份
            "metrics",         # 评估指标
            "visualizations"   # 可视化结果
        ]
        
        for subdir_name in subdirs_to_create:
            subdir_path = self.experiment_dir / subdir_name
            subdir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 创建子目录: {subdir_name}")
        
        print(f"✓ 实验目录结构创建完成: {self.experiment_dir}")
        return self.experiment_dir
    
    def get_dir(self, dir_name: str) -> Path:
        """获取指定目录路径"""
        return self.experiment_dir / dir_name
    
    def get_checkpoint_dir(self) -> Path:
        """获取检查点目录"""
        return self.get_dir("checkpoints")
    
    def get_log_dir(self) -> Path:
        """获取日志目录"""
        return self.get_dir("logs")
    
    def get_output_dir(self) -> Path:
        """获取输出目录"""
        return self.get_dir("outputs")
    
    def get_cache_dir(self) -> Path:
        """获取缓存目录"""
        return self.get_dir("cache")
    
    def save_config(self, config: Dict[str, Any], filename: str = "config.json"):
        """保存配置文件"""
        config_path = self.get_dir("configs") / filename
        
        # 确保配置可序列化
        serializable_config = self._make_serializable(config)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 配置已保存: {config_path}")
        return config_path
    
    def save_experiment_info(self, info: Dict[str, Any]):
        """保存实验信息"""
        info.update({
            "experiment_name": self.experiment_name,
            "experiment_dir": str(self.experiment_dir),
            "created_time": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
            "python_env": self._get_python_env()
        })
        
        info_path = self.experiment_dir / "experiment_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 实验信息已保存: {info_path}")
        return info_path
    
    def backup_code(self, source_dir: Optional[str] = None):
        """备份代码"""
        if source_dir is None:
            # 自动检测项目根目录
            current_file = Path(__file__)
            source_dir = current_file.parent.parent.parent
        
        source_dir = Path(source_dir)
        backup_dir = self.get_dir("code_backup")
        
        # 只备份重要的代码文件
        important_patterns = [
            "src/**/*.py",
            "scripts/**/*.py", 
            "configs/**/*.py",
            "main.py",
            "requirements.txt",
            "README.md"
        ]
        
        import shutil
        
        for pattern in important_patterns:
            for file_path in source_dir.glob(pattern):
                if file_path.is_file():
                    # 保持目录结构
                    relative_path = file_path.relative_to(source_dir)
                    target_path = backup_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, target_path)
        
        print(f"✓ 代码已备份: {backup_dir}")
        return backup_dir
    
    def create_symlinks(self, project_root: str):
        """在项目根目录创建到实验目录的符号链接"""
        project_root = Path(project_root)
        
        # 创建符号链接
        links_to_create = {
            "checkpoints": self.get_checkpoint_dir(),
            "logs": self.get_log_dir(), 
            "outputs": self.get_output_dir(),
            "generated_images": self.get_dir("generated_images"),
            "cache": self.get_cache_dir()
        }
        
        for link_name, target_path in links_to_create.items():
            link_path = project_root / link_name
            
            # 删除已存在的链接或目录
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_symlink():
                    link_path.unlink()
                elif link_path.is_dir():
                    import shutil
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()
            
            # 创建符号链接
            try:
                link_path.symlink_to(target_path, target_is_directory=True)
                print(f"✓ 创建符号链接: {link_name} -> {target_path}")
            except OSError as e:
                print(f"⚠ 创建符号链接失败: {link_name} -> {target_path}, 错误: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """使对象可序列化"""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _get_git_commit(self) -> Optional[str]:
        """获取当前Git提交哈希"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                cwd=Path(__file__).parent.parent.parent
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_python_env(self) -> Dict[str, str]:
        """获取Python环境信息"""
        import sys
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown")
        }
    
    def list_experiments(self) -> list:
        """列出所有实验"""
        if not self.experiment_root.exists():
            return []
        
        experiments = []
        for exp_dir in self.experiment_root.iterdir():
            if exp_dir.is_dir():
                info_file = exp_dir / "experiment_info.json"
                if info_file.exists():
                    try:
                        with open(info_file, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        experiments.append(info)
                    except Exception:
                        pass
        
        return sorted(experiments, key=lambda x: x.get('created_time', ''), reverse=True)
    
    def cleanup_old_experiments(self, keep_recent: int = 10):
        """清理旧实验（保留最近的几个）"""
        experiments = self.list_experiments()
        
        if len(experiments) <= keep_recent:
            print(f"实验数量({len(experiments)})未超过保留数量({keep_recent})，无需清理")
            return
        
        to_remove = experiments[keep_recent:]
        
        for exp_info in to_remove:
            exp_dir = Path(exp_info['experiment_dir'])
            if exp_dir.exists():
                import shutil
                shutil.rmtree(exp_dir)
                print(f"✓ 已删除旧实验: {exp_info['experiment_name']}")
        
        print(f"✓ 清理完成，保留了最近的 {keep_recent} 个实验")


def create_experiment_manager(
    experiment_name: Optional[str] = None,
    project_root: Optional[str] = None
) -> ExperimentManager:
    """创建实验管理器的便捷函数"""
    
    # 自动检测项目根目录
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    # 创建实验管理器
    exp_manager = ExperimentManager(experiment_name=experiment_name)
    
    # 保存实验信息
    exp_manager.save_experiment_info({
        "description": "PanDerm-Guided Diffusion训练实验",
        "project_root": str(project_root)
    })
    
    # 备份代码
    exp_manager.backup_code(project_root)
    
    # 创建符号链接（可选）
    try:
        exp_manager.create_symlinks(project_root)
    except Exception as e:
        print(f"⚠ 创建符号链接失败: {e}")
    
    return exp_manager


if __name__ == "__main__":
    # 测试实验管理器
    exp_manager = create_experiment_manager("test_experiment")
    print(f"实验目录: {exp_manager.experiment_dir}")
    
    # 列出实验
    experiments = exp_manager.list_experiments()
    print(f"找到 {len(experiments)} 个实验")
    for exp in experiments[:3]:  # 显示最近3个
        print(f"  - {exp['experiment_name']} ({exp.get('created_time', 'unknown')})")