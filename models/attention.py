import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """Additive attention over image regions."""

    def __init__(self, image_dim: int, hidden_dim: int, attn_dim: int = 512):
        super().__init__()
        self.W1 = nn.Linear(image_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, image_regions: torch.Tensor, decoder_hidden: torch.Tensor):
        # image_regions: (B, N, D)
        # decoder_hidden: (B, H)
        w1 = self.W1(image_regions)
        w2 = self.W2(decoder_hidden).unsqueeze(1)
        scores = self.v(torch.tanh(w1 + w2)).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * image_regions, dim=1)
        return context, attn_weights
