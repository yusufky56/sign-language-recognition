"""
timm (PyTorch Image Models) tabanlı model inşası — v2 (bug fix).
- ViT türevleri için img_size parametresi (positional embedding interpolation)
- MobileViT/MaxViT/CoAtNet hybrid modelleri düzgün tanınır
- Robust target layer detection: .stages / .blocks / fallback
"""
import timm
import torch.nn as nn


def build_model(model_name: str, num_classes: int,
                drop_rate: float = 0.3, drop_path_rate: float = 0.1,
                pretrained: bool = True, img_size: int = None) -> nn.Module:
    """
    timm'den pretrained model yükle.
    img_size: ViT/DeiT/Swin gibi transformer'lar için — native size dışında
    çalışırken positional embedding'i otomatik interpolate etmek için gerekli.
    ConvNeXt, EfficientNet gibi tam konvolüsyonel modeller bu kwarg'ı kabul
    etmez, silently düşer.
    """
    kwargs = dict(
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    # ViT-türevleri için img_size geç, diğerlerinde TypeError düşer, atlarız
    if img_size is not None:
        try:
            return timm.create_model(model_name, img_size=img_size, **kwargs)
        except TypeError:
            pass   # Model img_size kwarg'ı kabul etmiyor (CNN'ler)

    return timm.create_model(model_name, **kwargs)


def freeze_backbone(model: nn.Module) -> None:
    """Sadece classifier head train edilebilir; backbone donduruldu."""
    for p in model.parameters():
        p.requires_grad = False
    classifier = model.get_classifier()
    for p in classifier.parameters():
        p.requires_grad = True


def unfreeze_last_n(model: nn.Module, n_last: int = 30) -> None:
    """Fine-tuning için son N parametreyi açar, BatchNorm'u dondurur."""
    for p in model.parameters():
        p.requires_grad = True
    all_params = list(model.named_parameters())
    freeze_until = max(0, len(all_params) - n_last)
    for i, (_, p) in enumerate(all_params):
        if i < freeze_until:
            p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def _is_pure_vit(name: str) -> bool:
    """
    'vit' kelimesi 'mobilevit', 'maxvit', 'coatnet' gibi hybrid isimlerde de
    geçiyor — bunları pure ViT saymayalım.
    """
    n = name.lower()
    hybrid_markers = ("mobilevit", "maxvit", "coatnet", "efficientvit", "tinyvit")
    if any(m in n for m in hybrid_markers):
        return False
    return ("vit" in n) or ("deit" in n)


def _is_swin(name: str) -> bool:
    return "swin" in name.lower()


def get_target_layer_for_gradcam(model: nn.Module, model_name: str):
    """
    Grad-CAM hedef katmanı. timm modelleri genelde .stages veya .blocks kullanır.
    Strateji:
      1. Pure ViT/DeiT → son block'un norm1 (reshape_transform gerekir)
      2. Swin → son layer'ın son block norm1
      3. .stages varsa (ConvNeXt, RegNet, MaxViT, CoAtNet, MobileViT, ByobNet) → son stage
      4. Fallback: modelin son Conv2d katmanı
    """
    name = model_name.lower()

    # 1) Pure ViT
    if _is_pure_vit(name):
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            return [model.blocks[-1].norm1]

    # 2) Swin
    if _is_swin(name):
        if hasattr(model, "layers") and len(model.layers) > 0:
            try:
                return [model.layers[-1].blocks[-1].norm1]
            except (AttributeError, IndexError):
                return [model.layers[-1]]

    # 3) .stages (birçok model)
    if hasattr(model, "stages") and len(model.stages) > 0:
        try:
            # Son stage'deki son block
            return [model.stages[-1].blocks[-1]]
        except (AttributeError, IndexError, TypeError):
            return [model.stages[-1]]

    # EfficientNet family
    if hasattr(model, "conv_head"):
        return [model.conv_head]

    # ResNet
    if hasattr(model, "layer4"):
        return [model.layer4]

    # RegNet
    if hasattr(model, "s4"):
        return [model.s4]

    # 4) Fallback: son Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return [last_conv]

    raise ValueError(f"Grad-CAM için uygun katman bulunamadı: {model_name}")