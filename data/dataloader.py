import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import VQADataset


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _loader_kwargs(config: dict):
    num_workers = config["training"]["num_workers"]
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = config["training"].get("prefetch_factor", 2)
        kwargs["persistent_workers"] = config["training"].get("persistent_workers", True)
    return kwargs


def build_dataloaders(config: dict, vocab):
    data_cfg = config["data"]
    train_path = data_cfg["train_json"]
    dev_path = data_cfg["dev_json"]
    test_path = data_cfg["test_json"]
    images_dir = data_cfg["images_dir"]
    cache_tokenization = data_cfg.get("cache_tokenization", True)

    transform = build_transforms()

    train_set = VQADataset(
        train_path,
        images_dir,
        vocab,
        max_question_len=data_cfg["max_question_len"],
        max_answer_len=data_cfg["max_answer_len"],
        transform=transform,
        cache_tokenization=cache_tokenization,
    )
    dev_set = VQADataset(
        dev_path,
        images_dir,
        vocab,
        max_question_len=data_cfg["max_question_len"],
        max_answer_len=data_cfg["max_answer_len"],
        transform=transform,
        cache_tokenization=cache_tokenization,
    )
    test_set = VQADataset(
        test_path,
        images_dir,
        vocab,
        max_question_len=data_cfg["max_question_len"],
        max_answer_len=data_cfg["max_answer_len"],
        transform=transform,
        cache_tokenization=cache_tokenization,
    )

    loader_kwargs = _loader_kwargs(config)

    train_loader = DataLoader(
        train_set,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, dev_loader, test_loader
