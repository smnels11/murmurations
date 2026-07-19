"""
PyTorch models and training utilities for a_p feature vectors: datasets and
loaders, an MLP and a 1-D CNN over the prime axis, and a training loop with
early stopping.

MPS gotchas handled here:
    * MPS does NOT support float64. All tensors are float32.
    * DataLoader(pin_memory=True) only helps CUDA; it's a no-op/warn on MPS,
      so we leave it False when the device is mps.
    * Move both the model and each batch to the device inside the loop.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


def get_device(prefer: str | None = None) -> torch.device:
    """
    Return the best available device: mps (Apple Silicon) > cuda > cpu.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ApDataset(Dataset):
    """
    Wraps an (X, y) pair as float32 features / long labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # float32 is required for MPS; standardize upstream if desired.
        # np.ascontiguousarray gives a writable, contiguous buffer (torch warns
        # otherwise).
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    val_frac: float = 0.2,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Train/val DataLoaders. pin_memory is enabled only for CUDA.
    """
    ds = ApDataset(X, y)
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    pin = device is not None and device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin
    )
    return train_loader, val_loader


def make_loaders_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Train/val DataLoaders from a pre-made split.

    Use this when preprocessing must be fit on the training rows only (e.g.
    StandardScaler): split first, fit/transform upstream, then build loaders.
    ``make_loaders`` splits internally and is fine when preprocessing is
    per-row.
    """
    pin = device is not None and device.type == "cuda"
    train_loader = DataLoader(
        ApDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        ApDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
    )
    return train_loader, val_loader


class MLP(nn.Module):
    """
    A small feed-forward classifier built with nn.Sequential.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int = 2,
        hidden=(256, 128),
        p_drop: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p_drop),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ApCNN(nn.Module):
    """
    1-D CNN over the prime axis, input shape (batch, channels, n_primes).

    Motivation: logistic regression and the MLP treat the primes as an
    unordered bag -- permute the columns and they are unchanged. The CNN
    brings two distinct priors, worth keeping separate:

    * WEIGHT SHARING: one kernel computes the same local statistic at every
      prime; an MLP must rediscover it at every coordinate. Notebook 06 shows
      this alone lets a single-channel CNN find the synthetic skew signal an
      MLP never finds.
    * LOCALITY: pooling neighbours raises SNR only if adjacent primes carry
      related signal. The prime-shuffle control in notebook 06 tests this
      separately from weight sharing.

    A single Hasse-normalized channel (features.moment_channels[:, :1, :]) is
    the default, honest input; the 3-moment-channel variant is an ablation --
    the extra channels add little on either synthetic or real data
    (Notebooks 05, 06).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 2,
        widths=(32, 64, 128),
        kernel_size: int = 5,
        pool: int = 4,
        p_drop: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for w in widths:
            layers += [
                nn.Conv1d(
                    prev, w, kernel_size=kernel_size, padding=kernel_size // 2
                ),
                nn.BatchNorm1d(w),
                nn.ReLU(),
                nn.MaxPool1d(pool),
            ]
            prev = w
        self.features = nn.Sequential(*layers)
        # Global average pool -> the model summarises the whole prime axis,
        # so it works for any n_primes without reshaping the head.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(prev, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int | None = None,
    scheduler: str | None = None,  # None or "plateau"
    verbose: bool = True,
) -> History:
    """
    Autograd training loop with Adam and cross-entropy.

    patience  : if set, stop when val accuracy hasn't improved for this many
               epochs and restore the best weights (early stopping).
    scheduler : "plateau" halves the LR when val loss stalls.
    """

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    sched = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3
        )
        if scheduler == "plateau"
        else None
    )
    history = History()

    best_acc, best_state, since_improved = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)  # move batch to MPS/CUDA/CPU
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()  # autograd
            optimizer.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        tr_loss = running / seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.train_loss.append(tr_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        if sched is not None:
            sched.step(val_loss)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"epoch {epoch:3d}  train {tr_loss:.4f}  val {val_loss:.4f}  "
                f"acc {val_acc:.4f}"
            )

        if val_acc > best_acc + 1e-4:
            best_acc, since_improved = val_acc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_improved += 1
            if patience is not None and since_improved >= patience:
                if verbose:
                    print(
                        f"early stop at epoch {epoch}; "
                        f"best val acc {best_acc:.4f}"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # restore best weights
    return history
