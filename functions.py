import torch
import torch.nn.functional as F

def tmpr_loss(output: torch.Tensor, target: torch.Tensor, mem: list[torch.Tensor], step: int, 
                          lamb: float = 1.0) -> torch.Tensor:
    ce_loss = F.cross_entropy(output, target)
    mem_loss = 0
    N = len(mem)
    for t in range(step):
        for n in range(N):
            mem_loss += (lamb / (t + 1)) * F.mse_loss(mem[n][t], torch.zeros_like(mem[n][t], device=mem[n][t].device)).to(ce_loss.device)
    mem_loss /= (step * N)
    return mem_loss + ce_loss