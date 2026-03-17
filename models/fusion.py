import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """Concatenate image and question features then project to decoder hidden size."""

    def __init__(self, image_dim: int, question_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(image_dim + question_dim, hidden_dim)

    def forward(self, image_feat: torch.Tensor, question_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([image_feat, question_feat], dim=1)
        return torch.tanh(self.fc(fused))
