import math
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List


class LoRALayer(nn.Module):
    """
    LoRA适配层：对权重矩阵添加低秩分解
    """
    def __init__(self, original_layer: nn.Module, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # 获取原始权重的形状 (out_dim, in_dim)（注意顺序！）
        out_dim, in_dim = original_layer.weight.shape
        
        # 初始化LoRA权重矩阵
        self.A = nn.Parameter(torch.zeros(in_dim, rank, device=original_layer.weight.device)) # (in_dim, rank)
        self.B = nn.Parameter(torch.zeros(rank, out_dim, device=original_layer.weight.device)) # (rank, out_dim)

        # 使用Kaiming初始化A，B初始化为零
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original(x)
        # 计算LoRA增量: x @ A @ B * (alpha / rank)
        lora_output = x @ self.A @ self.B * (self.alpha / self.rank)
        return original_output + lora_output


def apply_lora(
    model: nn.Module, 
    lora_targets: List[str], 
    rank: int = 8,
    alpha: float = 16.0
) -> nn.Module:
    """
    将LoRA应用到所有名称包含lora_targets且是nn.Linear的层上
    """
    for name, module in model.named_modules():
        if any(key in name for key in lora_targets) and isinstance(module, nn.Linear):
            # 创建LoRA层
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            
            # 找到父模块并替换
            *parent_name, target_name = name.rsplit('.', 1)
            parent_module = model
            if parent_name:
                parent_module = model.get_submodule('.'.join(parent_name))
            
            # 替换原始模块
            setattr(parent_module, target_name, lora_layer)
    
    # 冻结所有不在LoRA目标中的参数
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # 解冻LoRA层的参数
    for name, param in model.named_parameters():
        if any(key in name for key in lora_targets) and ('.A' in name or '.B' in name):
            param.requires_grad = True
    
    print("\n应用LoRA后参数梯度状态:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
   
    return model



# 测试用例
if __name__ == "__main__":
    # 定义一个测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)  # weight.shape = (20, 10)
            self.target_dense = nn.Linear(20, 5)  # weight.shape = (5, 20)（会被LoRA替换）
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.target_dense(x)
            return x

    # 创建模型实例
    model = TestModel()
    print("原始模型结构:")
    print(model)
    
    # 应用LoRA，匹配包含'target'或'dense'的线性层
    lora_targets = ['target', 'dense']
    lora_model = apply_lora(model, lora_targets=lora_targets, rank=4, alpha=8.0)
    
    print("\n应用LoRA后的模型结构:")
    print(lora_model)
    
    # 检查参数是否被正确冻结和解冻
    print("\n参数梯度状态:")
    for name, param in lora_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    # 测试前向传播
    input_tensor = torch.randn(5, 10)  # (batch_size=5, in_dim=10)
    output = lora_model(input_tensor)
    print("\n测试前向传播成功，输出形状:", output.shape)  # 应为 (5, 5)


    '''
    1、保存lora与原模型的模型权重
        尽量保留lora相关参数：
            torch.save({
                'epoch': e+1,
                'step': total_step,
                'num_edges' : args.num_neighbors,
                'noise_level': args.backbone_noise,
                "lora":{"is_lora": True if args.mode == "lora_finetune" else False, "rank":args.lora_rank, "alpha":args.lora_alpha, "targets": args.lora_targets},
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, checkpoint_filename_last)
    
    2、应用lora权重
        # 1. 先加载原始模型
        model = your_model(**)
        model.to(device)

        # 2. lora 化
        lora_ = checkpoint["lora"]
        if lora_["is_lora"]:
            model = apply_lora(model, lora_["targets"], lora_["rank"], lora_["alpha"])
        
        # 3. load 全量权重
        model.load_state_dict(checkpoint['model_state_dict'])
    
    '''