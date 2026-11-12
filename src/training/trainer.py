"""
src/training/trainer.py
Single Responsibility: owns the training loop.
Open/Closed: new loss terms can be added by extending Trainer.train() without
             touching the CLI entrypoint (train.py).
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_frac: float = 0.1,
        device: str = "cpu",
        blunder_loss_weight: float = 0.5,
    ):
        self.model = model.to(device)
        self.device = device
        self.blunder_loss_weight = blunder_loss_weight

        val_size = max(1, int(len(dataset) * val_frac))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.policy_criterion = nn.CrossEntropyLoss()
        self.blunder_criterion = nn.BCEWithLogitsLoss()

        print(f"Trainer ready | train={train_size} val={val_size} | device={device}")

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, total_correct, total_n = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                board = batch["board"].to(self.device)
                move_idx = batch["move_idx"].to(self.device)
                history = batch["history"].to(self.device)
                blunder_label = batch["blunder"].to(self.device)

                logits, blunder_logit = self.model(board, history_tensor=history)
                policy_loss = self.policy_criterion(logits, move_idx)
                blunder_loss = (
                    self.blunder_criterion(blunder_logit, blunder_label)
                    if blunder_logit is not None else torch.tensor(0.0)
                )
                loss = policy_loss + self.blunder_loss_weight * blunder_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * board.size(0)
                total_correct += (logits.argmax(dim=1) == move_idx).sum().item()
                total_n += board.size(0)

        return total_loss / total_n, total_correct / total_n

    def train(self, epochs: int) -> list[dict]:
        history = []
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            dt = time.time() - t0
            record = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                          val_loss=val_loss, val_acc=val_acc)
            history.append(record)
            print(
                f"epoch {epoch}/{epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({dt:.1f}s)"
            )
        return history
