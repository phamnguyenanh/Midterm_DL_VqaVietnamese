import os
import random
from typing import Dict
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from training.metrics import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class Trainer:
    """Training and evaluation utilities for VQA models."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        vocab,
        device,
        use_amp: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.vocab = vocab
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

    def train_epoch(self, loader, teacher_forcing_ratio: float, grad_clip: float = 5.0):
        self.model.train()
        total_loss = 0.0
        progress = tqdm(loader, desc="Training", leave=False)

        for images, questions, answers_input, answers_target in progress:
            images = images.to(self.device)
            questions = questions.to(self.device)
            answers_input = answers_input.to(self.device)
            answers_target = answers_target.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images, questions, answers_input, teacher_forcing_ratio)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), answers_target.view(-1))

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(1, len(loader))

    def _decode_batch(self, sequences):
        texts = []
        for seq in sequences:
            tokens = self.vocab.decode(seq, stop_at_end=True)
            texts.append(" ".join(tokens))
        return texts

    def evaluate(self, loader, beam_width: int = 1, max_len: int = 15):
        self.model.eval()
        total_loss = 0.0
        references = []
        predictions = []

        with torch.no_grad():
            for images, questions, answers_input, answers_target in tqdm(loader, desc="Validation", leave=False):
                images = images.to(self.device)
                questions = questions.to(self.device)
                answers_input = answers_input.to(self.device)
                answers_target = answers_target.to(self.device)

                logits = self.model(images, questions, answers_input, teacher_forcing_ratio=0.0)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), answers_target.view(-1))
                total_loss += loss.item()

                seqs = self.model.generate(
                    images,
                    questions,
                    start_idx=self.vocab.start_idx,
                    end_idx=self.vocab.end_idx,
                    beam_width=beam_width,
                    max_len=max_len,
                )
                predictions.extend(self._decode_batch(seqs))

                for target in answers_target.cpu().tolist():
                    tokens = self.vocab.decode(target, stop_at_end=True)
                    references.append(" ".join(tokens))

        metrics = compute_metrics(references, predictions)
        return total_loss / max(1, len(loader)), metrics

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        teacher_forcing_ratio: float,
        teacher_forcing_min: float,
        teacher_forcing_decay: float,
        grad_clip: float,
        beam_width: int,
        max_len: int,
        checkpoint_path: str,
        resume_path: str = None,
        latest_path: str = None,
    ):
        best_bleu4 = -1.0
        start_epoch = 1

        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            start_epoch, best_bleu4 = self._load_checkpoint(resume_path)
            start_epoch += 1
            print(f"Resumed at epoch {start_epoch} with best BLEU-4 {best_bleu4:.4f}")

        if latest_path is None:
            base, ext = os.path.splitext(checkpoint_path)
            latest_path = f"{base}_latest{ext}"

        for epoch in range(start_epoch, num_epochs + 1):
            current_tf = max(teacher_forcing_min, teacher_forcing_ratio - (epoch - 1) * teacher_forcing_decay)

            train_loss = self.train_epoch(train_loader, current_tf, grad_clip=grad_clip)
            val_loss, metrics = self.evaluate(val_loader, beam_width=beam_width, max_len=max_len)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {val_loss:.4f}")
            print(
                f"BLEU-1: {metrics['bleu1']:.4f} | BLEU-4: {metrics['bleu4']:.4f} | "
                f"ROUGE-L: {metrics['rouge_l']:.4f} | METEOR: {metrics['meteor']:.4f} | "
                f"Exact Match: {metrics['exact_match']:.4f}"
            )

            os.makedirs(os.path.dirname(latest_path), exist_ok=True)
            self._save_checkpoint(latest_path, epoch, best_bleu4)

            if metrics["bleu4"] > best_bleu4:
                best_bleu4 = metrics["bleu4"]
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                self._save_checkpoint(checkpoint_path, epoch, best_bleu4)
                print(f"Saved checkpoint to {checkpoint_path}")

    def _save_checkpoint(self, path: str, epoch: int, best_bleu4: float):
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "epoch": epoch,
            "best_bleu4": best_bleu4,
        }
        torch.save(state, path)

    def _load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        if self.optimizer is not None and state.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler is not None and state.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if self.scaler is not None and state.get("scaler_state") is not None:
            self.scaler.load_state_dict(state["scaler_state"])
        return state.get("epoch", 0), state.get("best_bleu4", -1.0)

