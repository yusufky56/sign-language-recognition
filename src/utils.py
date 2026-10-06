"""Yardımcı fonksiyonlar: seed, device, checkpoint, plotting."""
import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


def set_seed(seed: int) -> None:
    """Reproducibility için tüm rastgelelik kaynaklarını sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic mode — küçük hız kaybı ama güvenilir
    torch.backends.cudnn.deterministic = False  # True yaparsan daha yavaş
    torch.backends.cudnn.benchmark = True       # Input size sabit olunca hızlandırır


def get_device() -> torch.device:
    """GPU varsa CUDA, yoksa CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        dev = torch.device("cpu")
        print("[device] GPU yok — CPU kullanılıyor (çok yavaş olacak)")
    return dev


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_history_csv(history: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def plot_curves(history: dict, save_path: Path, model_name: str) -> None:
    """Training/validation loss & accuracy eğrileri."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val",   linewidth=2)
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(history["val_acc"],   label="Val",   linewidth=2)
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Faz ayracı
    if "phase_switch_epoch" in history and history["phase_switch_epoch"]:
        for ax in axes:
            ax.axvline(history["phase_switch_epoch"], color="red",
                       linestyle="--", alpha=0.5, label="Fine-tune start")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
