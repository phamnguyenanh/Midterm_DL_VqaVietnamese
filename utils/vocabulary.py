import json
from collections import Counter
from typing import List, Dict


class Vocabulary:
    """Vocabulary with special tokens and utilities for encoding/decoding."""

    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for token in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]

    def __len__(self) -> int:
        return len(self.token2idx)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def start_idx(self) -> int:
        return self.token2idx[self.START_TOKEN]

    @property
    def end_idx(self) -> int:
        return self.token2idx[self.END_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    def build(self, token_lists: List[List[str]], max_size: int = 8000, min_freq: int = 1):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        # Keep special tokens at the front
        most_common = [t for t, c in counter.most_common() if c >= min_freq]
        for token in most_common[: max(0, max_size - len(self))]:
            self.add_token(token)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices: List[int], stop_at_end: bool = True) -> List[str]:
        tokens = []
        for idx in indices:
            token = self.idx2token.get(int(idx), self.UNK_TOKEN)
            if stop_at_end and token == self.END_TOKEN:
                break
            if token not in [self.PAD_TOKEN, self.START_TOKEN]:
                tokens.append(token)
        return tokens

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2idx": self.token2idx}, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        return vocab
