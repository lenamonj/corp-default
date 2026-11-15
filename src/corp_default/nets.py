"""Networks: seeding, firm-history sequences, two small models, and one training loop."""

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch, and make PyTorch repeat itself.

    One thread, because a sum split over several threads can round differently on another machine.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def build_sequences(
    frame: pd.DataFrame, cols: list[str], window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's last `window` calendar years of `cols`, oldest first, and a mask of the years filed.

    Returns values of shape (rows, window, cols) and a mask of shape (rows, window). A year the
    company did not file, or a year before its first, is a masked step holding zeros. Steps are
    found by (company_name, year), never by position, so a window never crosses a company.
    """
    keys = ["company_name", "year"]
    if frame.duplicated(keys).any():
        raise ValueError("build_sequences needs one row per (company_name, year)")
    by_key = frame.set_index(keys)[cols]
    values = np.zeros((len(frame), window, len(cols)), dtype=np.float32)
    mask = np.zeros((len(frame), window), dtype=bool)
    for step in range(window):
        years_back = window - 1 - step
        earlier = pd.MultiIndex.from_arrays([frame["company_name"], frame["year"] - years_back])
        filed = earlier.isin(by_key.index)
        mask[:, step] = filed
        values[filed, step] = by_key.reindex(earlier[filed]).to_numpy()
    return values, mask


class TabularNet(nn.Module):
    """Two hidden layers with dropout, and one logit for each row."""

    def __init__(self, n_inputs: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class SequenceHazardNet(nn.Module):
    """A GRU over a company's years, pooled over the years it filed, with one hazard logit a horizon."""

    def __init__(self, n_inputs: int, hidden: int = 32, horizons: int = 3, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_inputs, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizons)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        filed = mask.unsqueeze(-1).to(x.dtype)
        states, _ = self.gru(x * filed)
        pooled = (states * filed).sum(dim=1) / filed.sum(dim=1)
        return self.head(self.dropout(pooled))


def hazard_loss(
    logits: torch.Tensor, targets: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    """Mean binary cross-entropy over the observed cells. An unobserved cell has zero weight."""
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * observed).sum() / observed.sum()


def train_net(
    model: nn.Module,
    train: tuple[torch.Tensor, ...],
    valid: tuple[torch.Tensor, ...] | None,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    batch_size: int = 256,
) -> dict:
    """Fit `model` with AdamW, stopping when the loss on `valid` has not improved for `patience` epochs.

    `train` and `valid` are tuples of tensors: the model's inputs, then targets, then observed.
    With `valid` None every epoch runs and the last state is the best one.
    The model is left holding its best state. Returns that state, its epoch, and the loss history.
    """
    *inputs, targets, observed = train
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(targets))
        total = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            loss = hazard_loss(
                model(*(part[rows] for part in inputs)), targets[rows], observed[rows]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * observed[rows].sum().item()
        valid_loss = float("nan")
        if valid is not None:
            model.eval()
            with torch.no_grad():
                valid_loss = hazard_loss(model(*valid[:-2]), *valid[-2:]).item()
        history.append(
            {"epoch": epoch, "train_loss": total / observed.sum().item(), "valid_loss": valid_loss}
        )
        if valid is None or valid_loss < best_loss:
            best_loss, best_epoch, best_state = valid_loss, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return {"state": best_state, "best_epoch": best_epoch, "history": pd.DataFrame(history)}


def predict(model: nn.Module, *inputs: torch.Tensor) -> np.ndarray:
    """The model's probabilities for `inputs`, with dropout off."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(*inputs)).numpy()


def cumulative_pd(hazards: np.ndarray) -> np.ndarray:
    """Probability of failing within each horizon, from the hazard of each year given survival to it."""
    return 1 - np.cumprod(1 - hazards, axis=-1)
