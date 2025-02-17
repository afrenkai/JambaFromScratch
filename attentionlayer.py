import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layerutils import RMSNorm
class MHA(nn.Module):
    def __init__(self, d, num_heads=8):
        super().__init__()
        self.dim = d
        self.num_heads = num_heads
        self.heads_dim = d // num_heads

        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.norm = RMSNorm(d)

    def forward(self, x, mask= None):
        res = x
        B, L, D = x.shape # batch, layer, dim
        q = self.q(x).reshape(B, L, self.num_heads, self.heads_dim)
        k = self.k(x).reshape(B, L, self.num_heads, self.heads_dim)
        v = self.v(x).reshape(B, L, self.num_heads, self.heads_dim)
        q = q.transpose(1, 2)
        k= k.transpose(1, 2)
        v = v.transpose (1,2)

        scores = torch.matmul(q, k.transpose(-2, -1)) // math.sqrt(self.heads_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) # replace padded 0s with neg inf as is standard
        attn = F.softmax(scores, dim = -1) # row wise, not mat wise
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.o(x)
        return x + res

