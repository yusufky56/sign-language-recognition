"""
Ana çalıştırma scripti.

Kullanım:
    python main.py --model convnextv2_tiny            # Tek model
    python main.py --all                              # Tüm 10 model
    python main.py --quick                            # 3 hızlı model
    python main.py --model mobilevit_s --epochs-head 3 --epochs-finetune 3
    python main.py --gradcam --model convnextv2_tiny  # Sadece Grad-CAM
"""
import argparse
import gc
import time
from pathlib import Path

import torch

from src import config as cfg
from src.utils import set_seed, get_device, save_json, save_history_csv, plot_curves
from src.dataset import build_dataloaders
from src.models import build_model
from src.trainer import train_model
from src.metrics import (evaluate_model, compute_metrics,
                         save_classification_report, plot_confusion_matrix)
from src.explain import generate_gradcam


def parse_args():
    p = argparse.ArgumentParser(description="Sign Language MNIST — Transfer Learning")
    p.add_argument("--model", type=str, default=None,
                   help="Tek model adı (timm formatı, örn: convnextv2_tiny)")
    p.add_argument("--all",   action="store_true", help="Tüm 10 modeli sırayla eğit")
    p.add_argument("--quick", action="store_true", help="Sadece 3 hızlı model")
    p.add_argument("--gradcam", action="store_true", help="Eğitimi atla, sadece Grad-CAM üret")
    p.add_argument("--epochs-head",     type=int, default=None)
    p.add_argument("--epochs-finetune", type=int, default=None)
    p.add_argument("--batch-size",      type=int, default=None)
    p.add_argument("--img-size",        type=int, default=None)
    return p.parse_args()


def run_one_model(model_name: str, device, loaders, class_names):
    """Bir modeli eğit, değerlendir, Grad-CAM üret, sonuçları kaydet."""
    train_loader, val_loader, test_loader, (X_test_raw, y_test_raw) = loaders
    out_dir = cfg.RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\n  MODEL: {model_name}\n{'=' * 78}")

    t0 = time.time()

    # ---- 1) Model kur ----
    # img_size: ViT türevleri için kritik (DeiT3, ViT vb. aksi halde 224 dayatır)
    model = build_model(
        model_name, num_classes=cfg.NUM_CLASSES,
        drop_rate=cfg.DROP_RATE, drop_path_rate=cfg.DROP_PATH_RATE,
        pretrained=True, img_size=cfg.IMG_SIZE,
    )

    # ---- 2) Eğit ----
    history, best_val = train_model(
        model, model_name, train_loader, val_loader, device,
        out_dir, num_classes=cfg.NUM_CLASSES
    )

    # ---- 3) En iyi checkpoint'i yükle ve test et ----
    # Hem standart hem EMA checkpoint'i değerlendir, daha iyisini al
    results = {}
    for ckpt_name in ["best.pth", "ema.pth"]:
        ckpt_path = out_dir / ckpt_name
        if not ckpt_path.exists():
            continue
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        y_pred, y_true, _ = evaluate_model(model, test_loader, device)
        results[ckpt_name] = compute_metrics(y_true, y_pred)
        print(f"  [{ckpt_name}] test acc={results[ckpt_name]['accuracy']:.4f} "
              f"f1={results[ckpt_name]['f1']:.4f}")

    # En iyi checkpoint (acc'ye göre)
    best_ckpt = max(results, key=lambda k: results[k]["accuracy"])
    best_metrics = results[best_ckpt]
    best_metrics["model"]         = model_name
    best_metrics["best_ckpt"]     = best_ckpt
    best_metrics["train_time_s"]  = round(time.time() - t0, 1)
    best_metrics["params"]        = sum(p.numel() for p in model.parameters())
    best_metrics["best_val_acc"]  = best_val

    # En iyi checkpoint'i yeniden yükle (sonraki Grad-CAM için)
    state = torch.load(out_dir / best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    y_pred, y_true, _ = evaluate_model(model, test_loader, device)

    # ---- 4) Çıktıları kaydet ----
    save_history_csv(history, out_dir / "history.csv")
    save_json(best_metrics, out_dir / "metrics.json")
    save_json(history,      out_dir / "history.json")
    plot_curves(history, out_dir / "curves.png", model_name)
    save_classification_report(y_true, y_pred, class_names,
                               out_dir / "classification_report.txt")
    plot_confusion_matrix(y_true, y_pred, class_names,
                          out_dir / "confusion_matrix.png", model_name)

    # ---- 5) Grad-CAM ----
    # Her sınıftan 1 örnek seç
    import numpy as np
    sample_idx = [int(np.where(y_true == c)[0][0]) for c in range(min(12, cfg.NUM_CLASSES))
                  if (y_true == c).any()]
    # Preprocess edilmiş tensorları test_loader'dan al
    all_imgs, all_labels = [], []
    for imgs, lbls in test_loader:
        all_imgs.append(imgs); all_labels.append(lbls)
    all_imgs = torch.cat(all_imgs); all_labels = torch.cat(all_labels).numpy()
    # Her sınıftan ilk örneğe ait indeksleri bul
    picks = []
    for c in range(cfg.NUM_CLASSES):
        hits = np.where(all_labels == c)[0]
        if len(hits) > 0:
            picks.append(int(hits[0]))
        if len(picks) >= 12:
            break
    gc_imgs = all_imgs[picks]
    gc_lbls = all_labels[picks]

    try:
        generate_gradcam(model, model_name, cfg.IMG_SIZE,
                         gc_imgs, gc_lbls, class_names, device,
                         out_dir / "gradcam.png", n_samples=len(picks))
    except Exception as e:
        print(f"[gradcam] HATA: {e}")

    print(f"\n[{model_name}] DONE — acc={best_metrics['accuracy']:.4f} "
          f"f1={best_metrics['f1']:.4f} time={best_metrics['train_time_s']}s")
    return best_metrics


def main():
    args = parse_args()

    # CLI override'ları
    if args.epochs_head     is not None: cfg.EPOCHS_HEAD     = args.epochs_head
    if args.epochs_finetune is not None: cfg.EPOCHS_FINETUNE = args.epochs_finetune
    if args.batch_size      is not None: cfg.BATCH_SIZE      = args.batch_size
    if args.img_size        is not None: cfg.IMG_SIZE        = args.img_size

    set_seed(cfg.SEED)
    device = get_device()

    # Veri yükle (bir kez)
    loaders = build_dataloaders(
        cfg.TRAIN_CSV, cfg.TEST_CSV, cfg.IMG_SIZE, cfg.BATCH_SIZE,
        val_split=cfg.VAL_SPLIT, seed=cfg.SEED, num_workers=cfg.NUM_WORKERS,
    )
    class_names = list(cfg.ALPHABET)

    # Hangi modelleri çalıştıralım?
    if args.all:
        models_to_run = cfg.MODELS
    elif args.quick:
        models_to_run = cfg.MODELS_QUICK
    elif args.model:
        models_to_run = [args.model]
    else:
        print("En az bir tanesini ver: --model X, --all, --quick")
        return

    all_results = []
    for name in models_to_run:
        try:
            metrics = run_one_model(name, device, loaders, class_names)
            all_results.append(metrics)
        except Exception as e:
            print(f"\n[HATA] {name}: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"model": name, "error": str(e)})
        finally:
            # Bellek temizliği — 10 model arka arkaya eğitirken kritik
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Tüm sonuçları özet dosyaya yaz
    save_json({"results": all_results}, cfg.RESULTS_DIR / "all_results.json")
    print("\n" + "=" * 78)
    print("TAMAMLANDI — Karşılaştırma tablosu için: python compare.py")


if __name__ == "__main__":
    main()