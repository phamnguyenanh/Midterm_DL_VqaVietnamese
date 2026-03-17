import json
import os
from typing import List, Dict, Tuple
import numpy as np

from utils.tokenizer import tokenize, normalize_text
from utils.vocabulary import Vocabulary


def load_dataset(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_id_map(images: List[Dict]) -> Dict[int, str]:
    id_map = {}
    for item in images:
        image_id = item.get("id")
        filename = item.get("file_name") or item.get("filename") or item.get("image")
        if image_id is not None and filename is not None:
            id_map[int(image_id)] = filename
    return id_map


def parse_annotations(data: Dict) -> List[Tuple[int, str, str]]:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    id_map = build_image_id_map(images)
    samples = []
    for ann in annotations:
        image_id = ann.get("image_id")
        question = ann.get("question", "")
        answers = ann.get("answers", [])
        answer = answers[0] if answers else ""
        if image_id in id_map:
            samples.append((int(image_id), question, answer))
    return samples


def build_vocab_from_json(json_paths: List[str], max_size: int = 8000, min_freq: int = 1) -> Vocabulary:
    token_lists = []
    for path in json_paths:
        data = load_dataset(path)
        for _, question, answer in parse_annotations(data):
            token_lists.append(tokenize(normalize_text(question)))
            token_lists.append(tokenize(normalize_text(answer)))
    vocab = Vocabulary()
    vocab.build(token_lists, max_size=max_size, min_freq=min_freq)
    return vocab


def load_fasttext_embeddings(vec_path: str, vocab: Vocabulary, embedding_dim: int = 300) -> np.ndarray:
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
    embeddings[vocab.pad_idx] = np.zeros(embedding_dim, dtype=np.float32)

    if not os.path.exists(vec_path):
        return embeddings

    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().rstrip().split()
        if len(first_line) == 2:
            pass
        else:
            token = first_line[0]
            if token in vocab.token2idx and len(first_line[1:]) == embedding_dim:
                embeddings[vocab.token2idx[token]] = np.asarray(first_line[1:], dtype=np.float32)
        for line in f:
            parts = line.rstrip().split()
            if len(parts) <= embedding_dim:
                continue
            token = parts[0]
            if token in vocab.token2idx:
                vector = np.asarray(parts[1:1 + embedding_dim], dtype=np.float32)
                embeddings[vocab.token2idx[token]] = vector
    return embeddings
