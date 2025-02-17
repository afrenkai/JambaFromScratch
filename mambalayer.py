import torch
import torch.nn as nn
from layerutils import RMSNorm, MoELayer


class Mamba(nn.Module):
    def __init__(self, d, d_state = 16, conv_d = 4, expansion =2, usemoe = False):
        # still not sure whether I want to keep MoE usage as a bool in the init
        super().__init__()
        self.dim = d
        self.d_state = d_state
        self.conv_d = conv_d
        self.in_lin_proj = nn.Linear(d, d*expansion) # higher dim lin proj as per Gu
        self.conv  = nn.Conv1d(
            in_channels = d * expansion, 
            out_channels = d * expansion,
            kernel_size = conv_d,
            padding  = conv_d -1, 
            groups = d * expansion
        )
        #trainable A, B, and C. D is a skip connection
        self.A = nn.Parameter(torch.randn(d * expansion, d_state, d_state))
        self.B = nn.Parameter(torch.randn(d * expansion, d_state, 1))
        self.C = nn.Parameter(torch.randn(d * expansion, 1, d_state))
        if not usemoe:
            self.out = nn.Linear(d * expansion, d) # MLP 
        else: 
            self.out = MoELayer(d = d * expansion) # MOE 
        self.norm = RMSNorm(d)

    def forward(self, x):
        #in should be B x seq_len x dim
        res = x
        x = self.norm(x) #DONT FORGET THAT I ALR DID THIS HERE. IN BLOCK, OMIT 1ST RMSNORM
        x  = self.in_lin_proj(x)
        x = x.transpose(1, 2)
        x = self.conv(x)[..., :x.shape[-1]]
        x = x.transpose(1, 2)

        B, L, D = x.shape
        h = torch.zeros(B, D, self.d_state, device = x.device)

        outputs = []
        for t in range(L):
            h = torch.einsum('bds, bsd->bsd', h, self.A.expand(B, -1, -1, -1))
            h += torch.einsum('bd, bds->bsd', x[:, t], self.B.expand(B, -1, -1))
            out = torch.einsum('bsd, bds->bd', h, self.C.expand(B, -1, -1))
            outputs.append(out)

        x = torch.stack(outputs, dim = 1)
        x = self.out(x)
        return x + res

