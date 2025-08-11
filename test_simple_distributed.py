#!/usr/bin/env python3
"""
最简单的分布式训练测试
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def test_distributed(rank, world_size):
    """分布式进程测试函数"""
    print(f"进程 {rank}/{world_size} 启动")
    
    # 初始化进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    try:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 设置当前进程使用的GPU
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        print(f"进程 {rank}: 使用设备 {device}")
        
        # 创建测试张量
        tensor = torch.randn(10, device=device) * (rank + 1)
        
        # 执行all_reduce操作
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        print(f"进程 {rank}: all_reduce完成, tensor sum = {tensor.sum().item():.2f}")
        
        # 清理
        dist.destroy_process_group()
        print(f"进程 {rank}: 完成")
        
    except Exception as e:
        print(f"进程 {rank}: 错误 - {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("测试简单分布式通信")
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    world_size = min(2, torch.cuda.device_count())  # 使用2个GPU
    print(f"使用 {world_size} 个GPU进行测试")
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(world_size)))
    
    try:
        # 启动多进程
        mp.spawn(
            test_distributed,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        print("✅ 分布式测试成功")
        
    except Exception as e:
        print(f"❌ 分布式测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()