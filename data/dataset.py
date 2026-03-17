import os
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.tokenizer import tokenize, normalize_text
from data.preprocessing import load_dataset, parse_annotations, build_image_id_map


class VQADataset(Dataset):
    """OpenViVQA dataset for sequence-to-sequence VQA."""

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        vocab,
        max_question_len: int = 20,
        max_answer_len: int = 15,
        transform=None,
        cache_tokenization: bool = True,
    ):
        self.json_path = json_path
        self.images_dir = images_dir
        self.vocab = vocab
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.transform = transform
        self.cache_tokenization = cache_tokenization

        data = load_dataset(json_path)
        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])
        self.image_id_map = build_image_id_map(self.images)
        self.samples = parse_annotations(data)

        self.cached = None
        if self.cache_tokenization:
            self.cached = []
            for _, question, answer in self.samples:
                q_ids, a_input, a_target = self._encode_qa(question, answer)
                self.cached.append((q_ids, a_input, a_target))

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_sequence(self, seq: List[int], max_len: int) -> List[int]:
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [self.vocab.pad_idx] * (max_len - len(seq))

    def _encode_qa(self, question: str, answer: str):
        q_tokens = tokenize(normalize_text(question))
        a_tokens = tokenize(normalize_text(answer))

        q_ids = self._pad_sequence(self.vocab.encode(q_tokens), self.max_question_len)

        a_input = [self.vocab.start_idx] + self.vocab.encode(a_tokens)
        a_target = self.vocab.encode(a_tokens) + [self.vocab.end_idx]

        a_input = self._pad_sequence(a_input, self.max_answer_len)
        a_target = self._pad_sequence(a_target, self.max_answer_len)

        return q_ids, a_input, a_target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_id, question, answer = self.samples[idx]
        image_filename = self.image_id_map[image_id]
        image_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.cache_tokenization and self.cached is not None:
            q_ids, a_input, a_target = self.cached[idx]
        else:
            q_ids, a_input, a_target = self._encode_qa(question, answer)

        return (
            image,
            torch.tensor(q_ids, dtype=torch.long),
            torch.tensor(a_input, dtype=torch.long),
            torch.tensor(a_target, dtype=torch.long),
        )
