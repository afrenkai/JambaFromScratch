import torch.nn as nn
from hparams import L, A, M, E, N, K
from attentionlayer import MHA
from mambalayer import Mamba
class JambaBlock(nn.Module):

    def __init__(self, d, num_layers = L, atom = (A,M), moe_freq = E, num_experts = N, top_k = K):
        super().__init__()
        self.dim = d
        total_ratio = atom[0] + atom[1]
        num_attn = (num_layers * atom[0]) // total_ratio
        num_mamba = num_layers-num_attn

        self.layers = nn.ModuleList()
        mamba_count =0
        attn_count = 0
        for i in range(num_layers):
            if attn_count < num_attn and (mamba_count >= num_mamba or i % total_ratio < atom[0]):
                self.layers.append(MHA(d))
                attn_count += 1
            else:
                self.layers.append(Mamba(d))
                mamba_count += 1
            #add check for moe and layers and stuff later

    def forward(self, x, mask=None):
        for layer in self.layers:
            if isinstance(layer, MHA):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x

if __name__ == "__main__":
    d = 64
    num_layers = 8
    atom = (1, 7)
    moe_freq = 2
    num_experts = 16
    top_k = 2
    model = JambaBlock(d, num_layers=num_layers, atom=atom, moe_freq=moe_freq, num_experts=num_experts, top_k=top_k)
    mha_count = sum(1 for layer in model.layers if isinstance(layer, MHA))
    mamba_count = sum(1 for layer in model.layers if isinstance(layer, Mamba))
    total_ratio = atom[0] + atom[1]
    expected_mha = (num_layers * atom[0]) // total_ratio
    expected_mamba = num_layers - expected_mha
    print(f'all the layers in the block: {len(model.layers)} (what it was supposed to be: {num_layers})')
    print(f'current mha layers: {mha_count} (what we were supposed to get: {expected_mha})')
    print(f'current mamba layers: {mamba_count} (what we were supposed to get: {expected_mamba})')
    assert len(model.layers) == num_layers, "complete layer count wrong"
    assert mha_count == expected_mha, "incorrect MHA layers"
    assert mamba_count == expected_mamba, "incorrect Mamba layers"

    print("we good")

