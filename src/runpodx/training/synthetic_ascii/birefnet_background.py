from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from runpodx.training.synthetic_ascii.background_removal import apply_mask_to_white_background

_MODEL_NAME = "ZhengPeng7/BiRefNet"


@lru_cache(maxsize=1)
def load_birefnet_model():
    model = AutoModelForImageSegmentation.from_pretrained(_MODEL_NAME, trust_remote_code=True)
    model = model.float()
    model.eval()
    return model


def remove_background_with_birefnet(image: Image.Image, *, model=None, target_size: int = 1024) -> Image.Image:
    model = model or load_birefnet_model()
    original = image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(original).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)[-1].sigmoid().cpu()[0].squeeze(0)
    mask = transforms.ToPILImage()(pred)
    mask = mask.resize(original.size, Image.Resampling.LANCZOS)
    return apply_mask_to_white_background(original, mask)
