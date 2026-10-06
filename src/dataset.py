"""
Sign Language MNIST veri seti yükleme.
- 28x28 grayscale → 3 kanal RGB'ye upscale
- RandAugment + modern augmentation pipeline
- Stratified train/val split
- Mixup/CutMix collate (trainer.py'de uygulanır)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split


class SignMNISTDataset(Dataset):
    """CSV'den yükleyen PyTorch Dataset.

    Sign Language MNIST etiketleri 0-24 arasında ama J (9) ve Z (25) yok.
    Biz 0-23 aralığına remap ediyoruz (model çıkışı ile uyumlu olsun).
    """

    # Orijinal → yeniden haritalanmış etiketler
    UNIQUE_ORIG = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    LABEL_MAP = {old: new for new, old in enumerate(UNIQUE_ORIG)}

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images   # (N, 28, 28) uint8
        self.labels = labels   # (N,) int64 (0..23 remapped)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]                           # (28, 28) uint8
        img = np.stack([img, img, img], axis=-1)         # (28, 28, 3) — RGB
        # NOT: to PIL gerekmeden doğrudan transform uygulanabilir (v2 API)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    y = df["label"].values.astype(np.int64)
    X = df.drop(columns=["label"]).values.reshape(-1, 28, 28).astype(np.uint8)
    # Etiketleri remap et
    y = np.array([SignMNISTDataset.LABEL_MAP[v] for v in y], dtype=np.int64)
    return X, y


def build_transforms(img_size: int, training: bool):
    """
    Training: RandAugment + geometric (FLIP YOK!) + color jitter + random erasing
    Eval: sadece resize + normalize
    NOT: Sign language'da horizontal flip anlam değiştirir → kullanma!
    """
    # ImageNet istatistikleri (tüm pretrained modeller için uygun)
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    if training:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size), antialias=True),
            # RandAugment: otomatik politika, manuel aug'dan üstün
            T.RandAugment(num_ops=2, magnitude=9),
            # Hafif ekstra geometrik (flip YOK)
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])


def build_dataloaders(train_csv, test_csv, img_size, batch_size,
                      val_split=0.15, seed=42, num_workers=0):
    """Train/Val/Test DataLoader'ları oluşturur."""
    # Veri yükle
    X_train_full, y_train_full = load_csv(train_csv)
    X_test, y_test             = load_csv(test_csv)

    # Stratified split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_full, y_train_full,
        test_size=val_split, stratify=y_train_full, random_state=seed
    )

    # Dataset'ler
    train_ds = SignMNISTDataset(X_tr, y_tr, transform=build_transforms(img_size, training=True))
    val_ds   = SignMNISTDataset(X_va, y_va, transform=build_transforms(img_size, training=False))
    test_ds  = SignMNISTDataset(X_test, y_test, transform=build_transforms(img_size, training=False))

    # DataLoader'lar
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **kw)

    print(f"[data] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    return train_loader, val_loader, test_loader, (X_test, y_test)
