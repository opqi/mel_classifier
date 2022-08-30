import torch
import numpy as np

from torch import nn
from archs.msa import MSA


def get_pos_embed(seq_len: int, d: int) -> torch.tensor:
    result = torch.ones(seq_len, d)
    for i in range(seq_len):
        for j in range(d):
            if j % 2 == 0:
                result[i][j] = np.sin(i / 10000**(j/d))
            else:
                result[i][j] = np.cos(i / 10000**((j-1)/d))
    return result


class ViT(nn.Module):
    def __init__(self, input_shape: tuple, device: str, n_patches: int = 10,
                 hidden_d: int = 32, n_heads: int = 8, out_dim: int = 2):
        super(ViT, self).__init__()
        self.input_shape = input_shape
        self.n_patches = n_patches
        self.device = device

        assert input_shape[1] % n_patches == 0
        assert input_shape[2] % n_patches == 0

        self.patch_size = (
            input_shape[1] / n_patches, input_shape[2] / n_patches)
        self.input_d = int(
            input_shape[0] * self.patch_size[0] * self.patch_size[1])

        self.hidden_d = hidden_d

        self.lin_mapper = nn.Linear(self.input_d, self.hidden_d)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        self.ln = nn.LayerNorm((self.n_patches**2 + 1, self.hidden_d))
        self.msa = MSA(self.hidden_d, n_heads)

        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d),
            nn.ReLU(inplace=False))

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_dim),
            nn.Softmax(dim=-1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        n, c, h, w = x.shape
        patches = x.reshape(n, self.n_patches**2, self.input_d)

        tokens = self.lin_mapper(patches)
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i]))
                             for i in range(len(tokens))])

        tokens += get_pos_embed(self.n_patches**2 + 1,
                                self.hidden_d).repeat(n, 1, 1).to(self.device)

        output = tokens + self.msa(self.ln(tokens))
        output += self.encoder_mlp(self.ln(output))

        output = output[:, 0]
        return self.mlp(output)
