import torch
from torch import nn


class MSA(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8):
        super(MSA, self).__init__()

        self.dim = dim
        self.n_heads = n_heads

        assert dim % n_heads == 0

        d_head = int(dim / n_heads)
        self.d_head = d_head
        self.q_mapping = [nn.Linear(d_head, d_head).to('cuda')
                          for _ in range(self.n_heads)]
        self.k_mapping = [nn.Linear(d_head, d_head).to('cuda')
                          for _ in range(self.n_heads)]
        self.v_mapping = [nn.Linear(d_head, d_head).to('cuda')
                          for _ in range(self.n_heads)]

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        result = list()
        for s in x:
            seq_result = list()
            for h in range(self.n_heads):
                q_mapping = self.q_mapping[h]
                k_mapping = self.k_mapping[h]
                v_mapping = self.v_mapping[h]

                seq = s[:, h * self.d_head:(h + 1) * self.d_head]

                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))

                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
