import random
from typing import Optional, Tuple
import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    """LSTM decoder with optional attention over image regions."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        image_dim: int,
        padding_idx: int = 0,
        attention=None,
        embeddings=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.lstm = nn.LSTM(embedding_dim + image_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention = attention
        self.image_dim = image_dim

    def _get_context(self, image_global: torch.Tensor, image_regions: Optional[torch.Tensor], hidden: torch.Tensor):
        if self.attention is not None and image_regions is not None:
            context, attn_weights = self.attention(image_regions, hidden)
            return context, attn_weights
        return image_global, None

    def step(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        image_global: torch.Tensor,
        image_regions: Optional[torch.Tensor],
    ):
        embed = self.embedding(input_token)
        context, attn = self._get_context(image_global, image_regions, hidden[0][-1])
        lstm_input = torch.cat([embed, context], dim=1).unsqueeze(1)
        output, hidden = self.lstm(lstm_input, hidden)
        logits = self.fc(output.squeeze(1))
        return logits, hidden, attn

    def forward(
        self,
        answers_input: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        image_global: torch.Tensor,
        image_regions: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ):
        batch_size, max_len = answers_input.size()
        outputs = []
        input_token = answers_input[:, 0]

        for t in range(max_len):
            logits, hidden, _ = self.step(input_token, hidden, image_global, image_regions)
            outputs.append(logits)
            predicted = logits.argmax(dim=1)

            use_teacher = random.random() < teacher_forcing_ratio
            if t + 1 < max_len and use_teacher:
                input_token = answers_input[:, t + 1]
            else:
                input_token = predicted

        return torch.stack(outputs, dim=1)
