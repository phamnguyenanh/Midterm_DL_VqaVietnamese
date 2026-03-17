from typing import Optional, Tuple
import torch
import torch.nn as nn

from utils.beam_search import beam_search_decode


class VQAModel(nn.Module):
    """Full VQA model combining encoders, fusion, and decoder."""

    def __init__(
        self,
        image_encoder: nn.Module,
        question_encoder: nn.Module,
        fusion: nn.Module,
        decoder: nn.Module,
        use_attention: bool = False,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.question_encoder = question_encoder
        self.fusion = fusion
        self.decoder = decoder
        self.use_attention = use_attention

    def encode(self, images: torch.Tensor, questions: torch.Tensor):
        image_global, image_regions = self.image_encoder(images)
        question_feat = self.question_encoder(questions)
        fused = self.fusion(image_global, question_feat)
        h0 = fused.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        return image_global, image_regions, (h0, c0)

    def forward(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
        answers_input: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ):
        image_global, image_regions, hidden = self.encode(images, questions)
        regions = image_regions if self.use_attention else None
        outputs = self.decoder(
            answers_input,
            hidden,
            image_global,
            regions,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return outputs

    def generate(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_width: int = 3,
        max_len: int = 15,
    ):
        image_global, image_regions, hidden = self.encode(images, questions)
        regions = image_regions if self.use_attention else None
        sequences = []
        for i in range(images.size(0)):
            seq = beam_search_decode(
                self.decoder,
                image_global[i : i + 1],
                None if regions is None else regions[i : i + 1],
                (hidden[0][:, i : i + 1, :], hidden[1][:, i : i + 1, :]),
                start_idx=start_idx,
                end_idx=end_idx,
                beam_width=beam_width,
                max_len=max_len,
                device=images.device,
            )
            sequences.append(seq)
        return sequences
