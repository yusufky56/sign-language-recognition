"""
Tüm model sonuçlarını oku, karşılaştırma tablosu + grafikler üret.
Bildiri için LaTeX tablosu da oluşturur.

Kullanım: python compare.py
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config as cfg


def main():
    results = []
    for model_dir in sorted(cfg.RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        metrics_file = model_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    if not results:
        print("[compare] Hiç sonuç bulunamadı. Önce main.py ile model eğit.")
        return

    df = pd.DataFrame(results)
    # Accuracy'e göre sırala
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df["params_M"] = (df["params"] / 1e6).round(2)

    cols = ["model", "accuracy", "precision", "recall", "f1", "weighted_f1",
            "params_M", "train_time_s"]
    out = df[cols].round(4)

    # ---- CSV ----
    csv_path = cfg.RESULTS_DIR / "comparison.csv"
    out.to_csv(csv_path, index=False)
    print(f"[compare] CSV: {csv_path}")

    # ---- LaTeX ----
    latex_path = cfg.RESULTS_DIR / "paper_table.tex"
    out.to_latex(latex_path, index=False, float_format="%.4f",
                 caption="Sign Language MNIST — 10 Transfer Learning Modeli Karşılaştırması",
                 label="tab:comparison")
    print(f"[compare] LaTeX: {latex_path}")

    # ---- Terminal output ----
    print("\n" + "=" * 90)
    print(out.to_string(index=False))
    print("=" * 90)
    print(f"\nEn iyi model: {out.iloc[0]['model']} "
          f"(Acc={out.iloc[0]['accuracy']:.4f}, F1={out.iloc[0]['f1']:.4f})")

    # ---- Grafik 1: Bar chart (4 metrik) ----
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    metrics_cols = ["accuracy", "precision", "recall", "f1"]
    df_plot = df.set_index("model")[metrics_cols]
    df_plot.plot(kind="bar", ax=axes[0], colormap="viridis", edgecolor="black",
                 width=0.8)
    axes[0].set_title("Model Karşılaştırması — Test Metrikleri",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Skor"); axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right")
    axes[0].tick_params(axis="x", rotation=45)
    for lbl in axes[0].get_xticklabels():
        lbl.set_ha("right")

    # ---- Grafik 2: Parametre sayısı vs accuracy ----
    axes[1].scatter(df["params_M"], df["accuracy"], s=150, alpha=0.7,
                    edgecolor="black", c=df["f1"], cmap="plasma")
    for _, r in df.iterrows():
        axes[1].annotate(r["model"], (r["params_M"], r["accuracy"]),
                         fontsize=8, xytext=(6, 6), textcoords="offset points")
    axes[1].set_xlabel("Parametre Sayısı (M)")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Model Boyutu vs Accuracy (renk = F1)",
                      fontsize=13, fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = cfg.RESULTS_DIR / "comparison.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[compare] Grafik: {fig_path}")

    # ---- Grafik 3: Tüm modellerin val accuracy eğrileri ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for model_dir in cfg.RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        hist_file = model_dir / "history.json"
        if not hist_file.exists():
            continue
        with open(hist_file, "r", encoding="utf-8") as f:
            h = json.load(f)
        if "val_acc" in h:
            axes[0].plot(h["val_acc"], label=model_dir.name, alpha=0.85, linewidth=1.8)
            axes[1].plot(h["val_loss"], label=model_dir.name, alpha=0.85, linewidth=1.8)

    axes[0].set_title("Validation Accuracy (tüm epoch'lar)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Accuracy")
    axes[0].legend(fontsize=8, loc="lower right"); axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val Loss")
    axes[1].legend(fontsize=8, loc="upper right"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curves_path = cfg.RESULTS_DIR / "all_training_curves.png"
    plt.savefig(curves_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[compare] Eğriler: {curves_path}")


if __name__ == "__main__":
    main()
