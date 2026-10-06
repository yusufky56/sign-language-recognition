"""
Bildiri için 300 DPI yüksek kaliteli şekilleri yeniden üretir.
Mevcut sonuçlardan (results/ klasörü) okur, yayın kalitesinde çıktı verir.

Kullanım: python figures_for_paper.py
Çıktı: results/paper_figures/ klasörüne kaydedilir
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Bildiri için global ayarlar ---
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR = RESULTS_DIR / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)


def load_results():
    """Tüm modellerin metrics.json ve history.json dosyalarını yükle."""
    results, histories = [], {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir() or d.name == "paper_figures":
            continue
        mfile, hfile = d / "metrics.json", d / "history.json"
        if mfile.exists():
            with open(mfile, encoding="utf-8") as f:
                results.append(json.load(f))
        if hfile.exists():
            with open(hfile, encoding="utf-8") as f:
                histories[d.name] = json.load(f)
    return pd.DataFrame(results).sort_values("accuracy", ascending=False), histories


def figure_comparison_bars(df):
    """Şekil 1: Modellerin test metrikleri + parametre-accuracy scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels_tr = ["Doğruluk", "Kesinlik", "Duyarlılık", "F1"]

    df_m = df.set_index("model")[metrics]
    df_m.columns = labels_tr
    df_m.plot(kind="bar", ax=axes[0], edgecolor="black", width=0.82,
              color=["#2b4c7e", "#567ebb", "#606d80", "#dce0d9"])
    axes[0].set_title("(a) Test Kümesi Üzerindeki Metrik Skorları", fontweight="bold")
    axes[0].set_ylabel("Skor"); axes[0].set_xlabel("")
    axes[0].set_ylim(0.6, 1.02)
    axes[0].legend(loc="lower left", ncol=2)
    axes[0].tick_params(axis="x", rotation=40)
    for lbl in axes[0].get_xticklabels():
        lbl.set_ha("right")

    sc = axes[1].scatter(df["params"] / 1e6, df["accuracy"], s=160, alpha=0.85,
                         edgecolor="black", c=df["f1"], cmap="viridis", zorder=3)
    for _, r in df.iterrows():
        axes[1].annotate(r["model"], (r["params"] / 1e6, r["accuracy"]),
                         fontsize=8, xytext=(7, 5), textcoords="offset points")
    axes[1].set_xlabel("Parametre Sayısı (Milyon)")
    axes[1].set_ylabel("Test Doğruluğu")
    axes[1].set_title("(b) Model Boyutu – Doğruluk İlişkisi", fontweight="bold")
    cbar = plt.colorbar(sc, ax=axes[1]); cbar.set_label("F1 Skoru")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_comparison.png", dpi=300)
    plt.savefig(OUT_DIR / "fig1_comparison.pdf")  # PDF de kaydedelim
    plt.close()
    print("[fig1] kaydedildi")


def figure_all_curves(histories):
    """Şekil 2: Tüm modellerin doğrulama eğrileri."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (name, h), c in zip(histories.items(), colors):
        if "val_acc" in h:
            axes[0].plot(h["val_acc"], label=name, color=c, linewidth=1.6, alpha=0.9)
            axes[1].plot(h["val_loss"], label=name, color=c, linewidth=1.6, alpha=0.9)

    axes[0].set_title("(a) Doğrulama Doğruluğu", fontweight="bold")
    axes[0].set_xlabel("Epok"); axes[0].set_ylabel("Doğruluk")
    axes[0].axvline(15, color="red", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].text(15.3, 0.05, "Faz 2 başlangıcı", rotation=90, fontsize=8, color="red", alpha=0.7)

    axes[1].set_title("(b) Doğrulama Kaybı", fontweight="bold")
    axes[1].set_xlabel("Epok"); axes[1].set_ylabel("Çapraz Entropi Kaybı")
    axes[1].axvline(15, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Legend'ı figürün dışına koy
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=5, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig2_all_curves.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig2_all_curves.pdf", bbox_inches="tight")
    plt.close()
    print("[fig2] kaydedildi")


def figure_single_model_curves(history, model_name, out_name, title_label):
    """Tek bir modelin train/val eğitim eğrileri."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="Eğitim", linewidth=2, color="#2b4c7e")
    axes[0].plot(history["val_loss"], label="Doğrulama", linewidth=2, color="#e88a1a")
    axes[0].set_title(f"(a) {title_label} — Kayıp", fontweight="bold")
    axes[0].set_xlabel("Epok"); axes[0].set_ylabel("Kayıp")
    axes[0].axvline(15, color="red", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].text(15.3, axes[0].get_ylim()[1]*0.9, "Faz 2", fontsize=9, color="red", alpha=0.7)
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Eğitim", linewidth=2, color="#2b4c7e")
    axes[1].plot(history["val_acc"], label="Doğrulama", linewidth=2, color="#e88a1a")
    axes[1].set_title(f"(b) {title_label} — Doğruluk", fontweight="bold")
    axes[1].set_xlabel("Epok"); axes[1].set_ylabel("Doğruluk")
    axes[1].axvline(15, color="red", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_name}.png", dpi=300)
    plt.savefig(OUT_DIR / f"{out_name}.pdf")
    plt.close()
    print(f"[{out_name}] kaydedildi")


def figure_confusion_matrix(model_name, out_name, title_label):
    """Modelin confusion matrix'ini bildiri için yeniden çiz."""
    cm_file = RESULTS_DIR / model_name / "confusion_matrix.png"
    # Not: Gerçek confusion matrix datasını elimizde tutmadığımız için,
    # bu fonksiyon var olan PNG'yi kopyalar. Eğer istersen numpy'dan yeniden çizebiliriz
    # ama o zaman predict'i tekrar çalıştırmak lazım.
    import shutil
    if cm_file.exists():
        shutil.copy(cm_file, OUT_DIR / f"{out_name}.png")
        print(f"[{out_name}] kopyalandı (mevcut PNG'den)")
    else:
        print(f"[UYARI] {cm_file} bulunamadı")


def figure_gradcam(model_name, out_name, title_label):
    """Modelin Grad-CAM görselini kopyala."""
    src = RESULTS_DIR / model_name / "gradcam.png"
    import shutil
    if src.exists():
        shutil.copy(src, OUT_DIR / f"{out_name}.png")
        print(f"[{out_name}] kopyalandı")
    else:
        print(f"[UYARI] {src} bulunamadı")


# ============================================================================
# ANA AKIŞ
# ============================================================================
if __name__ == "__main__":
    df, histories = load_results()
    print(f"Yüklenen model sayısı: {len(df)}")

    # Şekil 1: Genel karşılaştırma
    figure_comparison_bars(df)

    # Şekil 2: Tüm modellerin eğitim eğrileri
    figure_all_curves(histories)

    # Şekil 3 & 4: En iyi + en kötü modelin detaylı eğrisi
    if "convnextv2_tiny" in histories:
        figure_single_model_curves(histories["convnextv2_tiny"], "convnextv2_tiny",
                                    "fig3_convnextv2_curves", "ConvNeXtV2-Tiny")
    if "mobilevit_s" in histories:
        figure_single_model_curves(histories["mobilevit_s"], "mobilevit_s",
                                    "fig4_mobilevit_curves", "MobileViT-S")

    # Şekil 5 & 6: Confusion matrix'ler (mevcut PNG'lerden kopyalanır)
    figure_confusion_matrix("convnextv2_tiny", "fig5_convnextv2_cm", "ConvNeXtV2-Tiny")
    figure_confusion_matrix("mobilevit_s", "fig6_mobilevit_cm", "MobileViT-S")

    # Şekil 7 & 8: Grad-CAM
    figure_gradcam("convnextv2_tiny", "fig7_convnextv2_gradcam", "ConvNeXtV2-Tiny")
    figure_gradcam("mobilevit_s", "fig8_mobilevit_gradcam", "MobileViT-S")

    print(f"\nTüm şekiller burada: {OUT_DIR}")