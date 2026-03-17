import torch.nn as nn


def get_loss_fn(pad_idx: int):
    return nn.CrossEntropyLoss(ignore_index=pad_idx)
