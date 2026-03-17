import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    """LSTM encoder for Vietnamese questions."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int = 512, padding_idx: int = 0, embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

    def forward(self, questions: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(questions)
        _, (h_n, _) = self.lstm(embeds)
        return h_n[-1]
