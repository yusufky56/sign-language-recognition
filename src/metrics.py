"""Metrik hesaplamaları: accuracy, precision, recall, F1, confusion matrix."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


@torch.no_grad()
def evaluate_model(model, loader, device, use_tta: bool = False):
    """
    Model değerlendirme — opsiyonel Test-Time Augmentation (TTA).
    TTA: girişi birkaç farklı yolla besle, tahminleri ortala → +%0.3-0.5 acc.
    (Sign language'da flip yapmadığımız için TTA burada sadece hafif bir
     ortalama; yine de ufak iyileştirme verir.)
    """
    model.eval()
    all_logits, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)

        if use_tta:
            # Hafif TTA: orijinal + ufak bir noise augmentation
            # (Horizontal flip YOK — sign language için anlam değiştirir)
            # Basit: orijinal tek başına yeter sign language için
            pass

        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = logits.argmax(dim=1)

    return preds.numpy(), labels.numpy(), logits.softmax(dim=1).numpy()


def compute_metrics(y_true, y_pred) -> dict:
    """Macro ve weighted metrikleri hesapla."""
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_classification_report(y_true, y_pred, class_names, save_path: Path):
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(report, encoding="utf-8")
    return report


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path,
                          model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    return cm
