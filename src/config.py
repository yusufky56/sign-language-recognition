"""
Proje konfigürasyonu — tüm hiperparametreler ve yollar tek yerde.
Windows path kullanımı için raw string (r"...") önerilir.
"""
from pathlib import Path

# ============================================================================
# YOLLAR (Windows-friendly)
# ============================================================================
# Repo root; data/ and results/ live next to the code
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "sign_mnist_train.csv"
TEST_CSV  = DATA_DIR / "sign_mnist_test.csv"

# ============================================================================
# VERİ
# ============================================================================
NUM_CLASSES = 24                        # J ve Z yok (hareket gerektirir)
ALPHABET    = "ABCDEFGHIKLMNOPQRSTUVWXY"
IMG_SIZE    = 160                       # 96 (hızlı) / 160 (denge) / 224 (kalite)
BATCH_SIZE  = 64                        # OOM olursa 32'ye düşür
NUM_WORKERS = 0                         # Windows'ta 0 stabil; Linux/Mac'te 4
VAL_SPLIT   = 0.15

# ============================================================================
# EĞİTİM (2 fazlı transfer learning)
# ============================================================================
SEED              = 42
EPOCHS_HEAD       = 15                  # Faz 1: backbone frozen
EPOCHS_FINETUNE   = 15                  # Faz 2: son katmanlar açık
LR_HEAD           = 1e-3
LR_FINETUNE       = 1e-5
WEIGHT_DECAY      = 0.05                # AdamW için standart
WARMUP_EPOCHS     = 3                   # Linear warmup, sonra cosine decay
GRAD_CLIP         = 1.0                 # Transformer'larda kritik
UNFREEZE_LAST_N   = 30                  # Fine-tune'da açılacak katman sayısı
PATIENCE          = 7                   # Early stopping

# ============================================================================
# REGULARIZATION (overfit'i ezer)
# ============================================================================
DROP_RATE         = 0.3
DROP_PATH_RATE    = 0.1                 # Stochastic Depth (transformer'lar için)
LABEL_SMOOTHING   = 0.1
MIXUP_ALPHA       = 0.2
CUTMIX_ALPHA      = 1.0
MIXUP_PROB        = 0.5                 # %50 batch'e uygula (MixUp veya CutMix)
SWITCH_PROB       = 0.5                 # MixUp ve CutMix arasında switch
EMA_DECAY         = 0.9998              # Exponential Moving Average weights

# ============================================================================
# 10 MODERN MODEL (2026 — timm model zoo'dan)
# ----------------------------------------------------------------------------
# Seçim stratejisi: 4 pure CNN + 3 Transformer + 3 Hybrid
# Hepsi ImageNet pretrained, 224x224 native (daha küçüğe interpolate olabilir)
# ============================================================================
MODELS = [
    # ---- Modern CNN'ler ----
    "convnextv2_tiny",                  # 2023, MAE pretrain
    "efficientnetv2_rw_s",              # 2021, modern EfficientNet
    "regnety_032",                      # 2020, NAS-designed
    "resnetv2_50",                      # Güçlü baseline (BiT)
    # ---- Pure Transformers ----
    "vit_base_patch16_224",             # Vanilla ViT (AugReg)
    "deit3_small_patch16_224",          # 2022, data-efficient
    "swin_tiny_patch4_window7_224",     # 2021, hierarchical
    # ---- Hybrid (CNN + Transformer) ----
    "maxvit_tiny_tf_224",               # 2022, multi-axis attention
    "coatnet_0_rw_224",                 # 2021, Google
    "mobilevit_s",                      # 2022, mobile transformer
]

# Hızlı test için küçük subset:
MODELS_QUICK = ["mobilevit_s", "convnextv2_tiny", "deit3_small_patch16_224"]
