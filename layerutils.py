import torch.nn as nn
import torch
import torch.nn.functional as F
class RMSNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-10):
        super(RMSNorm, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim = 1, keepdim = True) + self.epsilon)
        return self.w * (x / rms)

class MoELayer(nn.Module):
    def __init__(self, d, num_experts=16, top_k=2):
        super().__init__()
        self.dim =d
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d*4),
                          nn.SiLU(),
                          nn.Linear(d*4, d)
            ) for i in range(num_experts)
        ])
        #add swiglu when its merged into torch since im too lazy to code it https://github.com/pytorch/pytorch/issues/128712

        self.router = nn.Linear(d, num_experts)
        self.norm = RMSNorm(d)

    def forward(self, x):
        res = x
        x = self.norm(x)
        route_probs = self.router(x)
        top_k_probs, top_k_idxs = torch.topk(route_probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim =-1)
        out = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_idxs[..., i]
            expert_w = top_k_probs[..., i:i+1]
            expert_out = torch.stack([
                self.experts[expert_idx](x_i) for x_i, expert_idx in zip(x, expert_idx)
            ])
            out += expert_out * expert_w
        return out + res
if __name__ == "__main__":
    pass
