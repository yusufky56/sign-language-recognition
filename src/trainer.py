"""
Modern PyTorch training loop.
Özellikler:
- 2 fazlı training: head-only → fine-tune
- MixUp + CutMix (timm.data.Mixup)
- Cosine LR + Warmup (timm.scheduler.CosineLRScheduler)
- EMA weights (timm.utils.ModelEmaV3 / ModelEmaV2)
- AMP (automatic mixed precision) — torch.amp
- Gradient clipping
- Label smoothing (SoftTargetCrossEntropy mixup varken, LabelSmoothing yokken)
- Early stopping
- History tracking + best checkpoint
"""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler

# ModelEma: timm sürümüne göre v3 veya v2
try:
    from timm.utils import ModelEmaV3 as _ModelEma
    _EMA_VERSION = "v3"
except ImportError:
    from timm.utils import ModelEmaV2 as _ModelEma
    _EMA_VERSION = "v2"

from . import config as cfg
from .utils import save_checkpoint, count_trainable


# ============================================================================
# Yardımcılar
# ============================================================================
def build_mixup(num_classes: int) -> Mixup:
    """timm Mixup: hem MixUp hem CutMix yapar, switch_prob ile aralarında seçer."""
    return Mixup(
        mixup_alpha=cfg.MIXUP_ALPHA,
        cutmix_alpha=cfg.CUTMIX_ALPHA,
        prob=cfg.MIXUP_PROB,
        switch_prob=cfg.SWITCH_PROB,
        mode="batch",
        label_smoothing=cfg.LABEL_SMOOTHING,
        num_classes=num_classes,
    )


def build_optimizer_scheduler(model, lr, total_epochs, steps_per_epoch,
                              warmup_epochs=0):
    """AdamW + Cosine LR with linear warmup (timm scheduler)."""
    # AdamW, transformerlar için altın standart
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999)
    )
    # Cosine scheduler, epoch bazlı (timm step(epoch, metric) ile çalışır)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_epochs,
        lr_min=1e-7,
        warmup_t=warmup_epochs,
        warmup_lr_init=lr * 0.01,
        cycle_limit=1,
        t_in_epochs=True,
    )
    return optimizer, scheduler


# ============================================================================
# Tek epoch train / val
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler, mixup_fn: Optional[Mixup], ema, epoch, grad_clip):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"train e{epoch:02d}", leave=False)

    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp/CutMix uygulanmadan önce orijinal etiketleri sakla (accuracy için)
        labels_orig = labels.clone()
        if mixup_fn is not None:
            imgs, labels = mixup_fn(imgs, labels)   # labels artık one-hot smooth

        optimizer.zero_grad(set_to_none=True)

        # AMP — mixed precision forward
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        # Backward + optimizer step (scaler AMP için)
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # EMA güncelle (her step'te)
        if ema is not None:
            ema.update(model)

        # Metrikler (mixup öncesi gerçek etiketlere göre)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels_orig).sum().item()
            total += labels_orig.size(0)
            running_loss += loss.item() * imgs.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{running_correct / total:.4f}")

    return running_loss / total, running_correct / total


@torch.no_grad()
def validate(model, loader, criterion_eval, device, epoch, tag="val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"{tag} e{epoch:02d}", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion_eval(logits, labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item() * imgs.size(0)

    return running_loss / total, correct / total


# ============================================================================
# Ana eğitim fonksiyonu (2 fazlı)
# ============================================================================
def train_model(model, model_name, train_loader, val_loader, device,
                out_dir: Path, num_classes: int):
    """
    2 fazlı transfer learning:
      Faz 1 — Head eğitimi: backbone donduruldu, LR=1e-3, head öğrenir
      Faz 2 — Fine-tune: son N katman açıldı, LR=1e-5
    """
    from .models import freeze_backbone, unfreeze_last_n
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    # Mixup/CutMix — CrossEntropy yerine SoftTargetCE kullanırız (label smooth içinde)
    mixup_fn = build_mixup(num_classes)
    criterion_train = SoftTargetCrossEntropy()                         # mixup için
    criterion_eval  = LabelSmoothingCrossEntropy(smoothing=cfg.LABEL_SMOOTHING)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # EMA — shadow weights, daha stabil generalization
    ema = _ModelEma(model, decay=cfg.EMA_DECAY)
    print(f"[ema] timm ModelEma{_EMA_VERSION} aktif, decay={cfg.EMA_DECAY}")

    history = {
        "epoch": [], "phase": [], "lr": [],
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "val_acc_ema": [], "phase_switch_epoch": None,
    }

    best_val_acc = 0.0
    best_epoch = -1
    patience_counter = 0
    steps_per_epoch = len(train_loader)

    # ====================== FAZ 1: HEAD EĞİTİMİ ======================
    print(f"\n[{model_name}] === FAZ 1: HEAD (backbone frozen) ===")
    freeze_backbone(model)
    print(f"  Trainable params: {count_trainable(model):,}")

    optimizer, scheduler = build_optimizer_scheduler(
        model, cfg.LR_HEAD, cfg.EPOCHS_HEAD, steps_per_epoch,
        warmup_epochs=cfg.WARMUP_EPOCHS
    )

    for epoch in range(cfg.EPOCHS_HEAD):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion_train, device,
            scaler, mixup_fn, ema, epoch, cfg.GRAD_CLIP
        )
        va_loss, va_acc = validate(model, val_loader, criterion_eval, device, epoch)
        # EMA model üzerinde de değerlendir
        ema_model = ema.module if hasattr(ema, "module") else ema.ema
        _, va_acc_ema = validate(ema_model, val_loader, criterion_eval, device, epoch, tag="ema")

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["phase"].append(1)
        history["lr"].append(current_lr)
        history["train_loss"].append(tr_loss);  history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);    history["val_acc"].append(va_acc)
        history["val_acc_ema"].append(va_acc_ema)

        print(f"  e{epoch:02d} | lr={current_lr:.2e} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | ema_acc={va_acc_ema:.4f}")

        # En iyi (val_acc veya ema acc'nin büyüğünü kullan)
        current_best = max(va_acc, va_acc_ema)
        if current_best > best_val_acc:
            best_val_acc = current_best
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint({"state_dict": model.state_dict(), "epoch": epoch,
                             "val_acc": va_acc}, out_dir / "best.pth")
            save_checkpoint({"state_dict": ema_model.state_dict(), "epoch": epoch,
                             "val_acc": va_acc_ema}, out_dir / "ema.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"  [EarlyStopping] patience {cfg.PATIENCE} aşıldı")
                break

    # ====================== FAZ 2: FINE-TUNE ======================
    print(f"\n[{model_name}] === FAZ 2: FINE-TUNE (son {cfg.UNFREEZE_LAST_N} katman açık) ===")
    history["phase_switch_epoch"] = len(history["epoch"])
    unfreeze_last_n(model, n_last=cfg.UNFREEZE_LAST_N)
    print(f"  Trainable params: {count_trainable(model):,}")

    # Yeni optimizer (unfrozen params için) + yeni scheduler
    optimizer, scheduler = build_optimizer_scheduler(
        model, cfg.LR_FINETUNE, cfg.EPOCHS_FINETUNE, steps_per_epoch,
        warmup_epochs=1   # Fine-tune'da daha kısa warmup
    )
    patience_counter = 0

    for local_epoch in range(cfg.EPOCHS_FINETUNE):
        epoch = cfg.EPOCHS_HEAD + local_epoch
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion_train, device,
            scaler, mixup_fn, ema, epoch, cfg.GRAD_CLIP
        )
        va_loss, va_acc = validate(model, val_loader, criterion_eval, device, epoch)
        ema_model = ema.module if hasattr(ema, "module") else ema.ema
        _, va_acc_ema = validate(ema_model, val_loader, criterion_eval, device, epoch, tag="ema")

        scheduler.step(local_epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["phase"].append(2)
        history["lr"].append(current_lr)
        history["train_loss"].append(tr_loss);  history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);    history["val_acc"].append(va_acc)
        history["val_acc_ema"].append(va_acc_ema)

        print(f"  e{epoch:02d} | lr={current_lr:.2e} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | ema_acc={va_acc_ema:.4f}")

        current_best = max(va_acc, va_acc_ema)
        if current_best > best_val_acc:
            best_val_acc = current_best
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint({"state_dict": model.state_dict(), "epoch": epoch,
                             "val_acc": va_acc}, out_dir / "best.pth")
            save_checkpoint({"state_dict": ema_model.state_dict(), "epoch": epoch,
                             "val_acc": va_acc_ema}, out_dir / "ema.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"  [EarlyStopping] patience {cfg.PATIENCE} aşıldı")
                break

    print(f"\n[{model_name}] Eğitim bitti. Best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    return history, best_val_acc
