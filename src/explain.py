"""
Grad-CAM görselleştirme — v2 (bug fix).
- Hybrid modeller (MobileViT, MaxViT, CoAtNet) artık doğru tanınıyor
- Reshape transform sadece gerçekten pure ViT/Swin için uygulanır
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from . import config as cfg
from .models import get_target_layer_for_gradcam, _is_pure_vit, _is_swin


def reshape_transform_vit(tensor, height, width):
    """ViT/DeiT: (B, 1+N, C) → (B, C, H, W). CLS token drop."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def reshape_transform_swin(tensor, height, width):
    """Swin: (B, H*W, C) → (B, C, H, W)."""
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
    return result.transpose(2, 3).transpose(1, 2)


def get_reshape_transform(model_name: str, img_size: int):
    """
    Sadece PURE ViT ve Swin için reshape gerekir.
    Hybrid modeller (MobileViT, MaxViT, CoAtNet) zaten spatial feature map
    üretir, reshape transform'a ihtiyaç yok.
    """
    if _is_pure_vit(model_name):
        patch = 16
        side = img_size // patch
        return lambda t: reshape_transform_vit(t, side, side)
    if _is_swin(model_name):
        side = max(1, img_size // 32)
        return lambda t: reshape_transform_swin(t, side, side)
    return None


def denormalize_image(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return np.clip(img.permute(1, 2, 0).numpy(), 0, 1)


def generate_gradcam(model, model_name: str, img_size: int,
                     images: torch.Tensor, labels: np.ndarray,
                     class_names: list, device, save_path: Path,
                     n_samples: int = 12):
    """En iyi model için Grad-CAM görselleştirmeleri (her sınıftan 1 örnek)."""
    model.eval().to(device)

    try:
        target_layers = get_target_layer_for_gradcam(model, model_name)
    except Exception as e:
        print(f"[gradcam] target layer hatası - {model_name}: {e} — atlanıyor")
        return

    reshape_transform = get_reshape_transform(model_name, img_size)

    try:
        cam = GradCAM(model=model, target_layers=target_layers,
                      reshape_transform=reshape_transform)
    except Exception as e:
        print(f"[gradcam] GradCAM init hatası - {model_name}: {e} — atlanıyor")
        return

    n = min(n_samples, len(images))
    imgs = images[:n].to(device)
    lbls = labels[:n]

    with torch.no_grad():
        preds = model(imgs).argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 6.5))
    if n == 1:
        axes = axes.reshape(3, 1)

    for i in range(n):
        rgb_img = denormalize_image(imgs[i])
        targets = [ClassifierOutputTarget(int(preds[i]))]
        try:
            grayscale_cam = cam(input_tensor=imgs[i:i+1], targets=targets)[0]
        except Exception:
            grayscale_cam = np.zeros((img_size, img_size), dtype=np.float32)

        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[0, i].imshow(rgb_img);                  axes[0, i].axis("off")
        axes[0, i].set_title(f"GT: {class_names[lbls[i]]}", fontsize=9)
        axes[1, i].imshow(grayscale_cam, cmap="jet"); axes[1, i].axis("off")
        axes[2, i].imshow(overlay);                   axes[2, i].axis("off")
        color = "green" if preds[i] == lbls[i] else "red"
        axes[2, i].set_title(f"Pred: {class_names[preds[i]]}",
                             fontsize=9, color=color)

    plt.suptitle(f"Grad-CAM — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[gradcam] kaydedildi: {save_path}")