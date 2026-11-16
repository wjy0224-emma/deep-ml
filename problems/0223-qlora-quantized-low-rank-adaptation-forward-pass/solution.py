import torch

def qlora_forward(
    x: torch.Tensor,
    quantized_W: torch.Tensor,
    scale: float,
    zero_point: float,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    QLoRA forward pass with quantized frozen weights using PyTorch.
    
    Args:
        x: Input tensor (batch_size x in_features)
        quantized_W: 4-bit quantized weights as integers (in_features x out_features)
        scale: Quantization scale factor
        zero_point: Quantization zero point
        A: LoRA matrix A (rank x out_features)
        B: LoRA matrix B (in_features x rank)
        alpha: LoRA scaling factor
        
    Returns:
        Output tensor (batch_size x out_features)
    """
    # Your code here
    # 1. 反量化：将整数权重恢复为近似浮点权重
    W = quantized_W.to(dtype=x.dtype) *scale +zero_point

    # 2. 原始线性层
    base = x @ W

    # 3. LoRA 低秩更新
    delta = x @ B @ A

    # 4. LoRA 缩放
    r = A.shape[0]
    delta = delta * (alpha / r)

    # 5. 合并输出
    return base + delta
