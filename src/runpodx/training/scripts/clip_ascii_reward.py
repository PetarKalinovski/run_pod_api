"""CLIP-based visual reward for ASCII art GRPO training.

Renders ASCII completions to images (white-on-black monospace),
then scores them against the prompt subject using CLIP.

Usage in grpo_profile_ascii_art.py:
    from clip_ascii_reward import clip_visual_reward, init_clip

Add to build_reward_functions() and call init_clip() at profile load time.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# Lazy-loaded globals
_clip_model = None
_clip_processor = None
_clip_device = None
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

# ── Font loading ─────────────────────────────────────────────────────

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    # RunPod images often have these
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]


def _get_mono_font(size: int = 16) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
        except (OSError, IOError):
            continue
    font = ImageFont.load_default(size=size)
    _font_cache[size] = font
    return font


# ── Rendering ────────────────────────────────────────────────────────

def render_ascii_to_image(
    text: str,
    font_size: int = 16,
    padding: int = 10,
    max_pixel_width: int = 800,
    max_pixel_height: int = 800,
    clip_size: int = 224,
) -> Image.Image:
    """Render ASCII text to a 224x224 PIL Image for CLIP input."""
    lines = text.split("\n")
    # Trim empty leading/trailing
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        lines = [" "]

    # Trim common left indent
    min_indent = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()),
        default=0,
    )
    if min_indent > 0:
        lines = [line[min_indent:] if len(line) > min_indent else "" for line in lines]

    font = _get_mono_font(font_size)
    bbox = font.getbbox("M")
    char_w = bbox[2] - bbox[0]
    char_h = int((bbox[3] - bbox[1]) * 1.3)

    max_cols = max(len(line) for line in lines)
    img_w = min(max_cols * char_w + 2 * padding, max_pixel_width)
    img_h = min(len(lines) * char_h + 2 * padding, max_pixel_height)

    img = Image.new("RGB", (img_w, img_h), "black")
    draw = ImageDraw.Draw(img)
    for row, line in enumerate(lines):
        y = padding + row * char_h
        if y > img_h - padding:
            break
        draw.text((padding, y), line, fill="white", font=font)

    # Resize preserving aspect, center on black 224x224
    img.thumbnail((clip_size, clip_size), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (clip_size, clip_size), "black")
    x_off = (clip_size - img.width) // 2
    y_off = (clip_size - img.height) // 2
    result.paste(img, (x_off, y_off))
    return result


# ── CLIP ─────────────────────────────────────────────────────────────

def init_clip(device: str = "cuda") -> None:
    """Pre-load CLIP model. Call once at training start."""
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is not None:
        return

    import torch
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    print(f"[clip_reward] Loading CLIP: {model_name} on {device}")
    t0 = time.time()
    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model = CLIPModel.from_pretrained(model_name)
    # Use float16 on CUDA to save VRAM (~150MB instead of ~300MB)
    if device == "cuda":
        _clip_model = _clip_model.half()
    _clip_model = _clip_model.to(device).eval()
    _clip_device = device
    print(f"[clip_reward] CLIP loaded in {time.time() - t0:.1f}s ({device})")


def _ensure_clip() -> None:
    if _clip_model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        init_clip(device)


def clip_score_batch(
    images: list[Image.Image],
    text_prompts: list[str],
) -> list[float]:
    """Score a batch of images against corresponding text prompts.
    
    Returns per-pair cosine similarities (not cross-product).
    """
    import torch

    _ensure_clip()

    inputs = _clip_processor(
        text=text_prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(_clip_device) for k, v in inputs.items()}
    if _clip_model.dtype == torch.float16:
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()

    with torch.no_grad():
        image_embeds = _clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_embeds = _clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        # Some transformers versions return BaseModelOutputWithPooling instead of a tensor
        if not isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.pooler_output if hasattr(image_embeds, "pooler_output") else image_embeds[0]
        if not isinstance(text_embeds, torch.Tensor):
            text_embeds = text_embeds.pooler_output if hasattr(text_embeds, "pooler_output") else text_embeds[0]
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # Per-pair cosine similarity
        similarities = (image_embeds * text_embeds).sum(dim=-1)

    return similarities.cpu().float().tolist()


# ── Prompt extraction ────────────────────────────────────────────────

_SUBJECT_PATTERNS = [
    re.compile(r"(?:draw|create|make|generate)\s+(?:an?\s+)?ascii\s+art\s+(?:of\s+)?(?:an?\s+)?(.+)", re.I),
    re.compile(r"ascii\s+art\s+(?:of\s+)?(?:an?\s+)?(.+)", re.I),
    re.compile(r"draw\s+(?:an?\s+)?(.+)", re.I),
]


def extract_prompt_subject(prompt_text: str) -> str:
    """Extract subject from prompt like 'Draw ASCII art of a cat' → 'cat'."""
    # If it's a chat-formatted prompt, extract the user message
    text = str(prompt_text or "")
    # Look for the last user message content
    if "<|im_start|>user" in text:
        parts = text.split("<|im_start|>user")
        text = parts[-1].split("<|im_end|>")[0].strip()
    elif "user\n" in text.lower():
        parts = text.lower().split("user\n")
        text = parts[-1].split("\n")[0].strip()

    text = text.strip().rstrip(".")
    for pat in _SUBJECT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip().rstrip(".")
    # Fallback: return last few words
    words = text.split()
    return " ".join(words[-3:]) if len(words) > 3 else text


# ── GRPO reward function ────────────────────────────────────────────

# CLIP cosine sim range for ASCII art (empirically determined):
#   Random/gibberish: ~0.15-0.20
#   Recognizable art:  ~0.22-0.35
# CLIP cosine similarity ranges for rendered ASCII art vs text prompts:
#   Random/garbage:     ~0.15-0.22
#   Generic ASCII text: ~0.22-0.28 (CLIP reads text in the rendered image)
#   Vaguely relevant:   ~0.28-0.32
#   Good match:         ~0.32-0.38
#   Excellent match:    ~0.38+
#
# Use raw cosine as reward — the natural [-1, 1] range provides
# continuous gradient signal. Centering around typical scores (~0.25)
# so mediocre outputs get near-zero reward and good ones get positive.
CLIP_CENTER = 0.25  # subtract so average output → ~0 reward
CLIP_SCALE = 5.0    # amplify: 0.05 cosine delta → 0.25 reward delta


def clip_visual_reward(completions, prompts=None, prompt=None, **kwargs) -> list[float]:
    """GRPO-compatible reward: render ASCII → CLIP score vs prompt subject.
    
    Returns list of float rewards (unbounded, centered ~0 for average outputs).
    """
    # Handle TRL argument conventions
    prompt_list = prompts or prompt or (["unknown"] * len(completions))
    if isinstance(prompt_list, str):
        prompt_list = [prompt_list] * len(completions)

    # Ensure we have enough prompts (TRL may pass fewer with num_generations)
    while len(prompt_list) < len(completions):
        prompt_list = prompt_list * 2
    prompt_list = prompt_list[: len(completions)]

    images = []
    text_prompts = []
    valid_indices = []

    for idx, (completion, raw_prompt) in enumerate(zip(completions, prompt_list)):
        text = str(completion or "").strip()
        # Skip very short or empty completions
        if len(text) < 10:
            continue
        # Skip obviously non-ASCII (mostly CJK/gibberish)
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio < 0.5:
            continue

        subject = extract_prompt_subject(raw_prompt)
        img = render_ascii_to_image(text)
        images.append(img)
        text_prompts.append(f"ASCII art drawing of a {subject}")
        valid_indices.append(idx)

    # Default scores
    rewards = [0.0] * len(completions)

    if images:
        scores = clip_score_batch(images, text_prompts)
        for idx, score in zip(valid_indices, scores):
            # Center and scale: average output (~0.25 cosine) → ~0 reward
            # Good output (~0.35 cosine) → ~0.5 reward
            # Great output (~0.40 cosine) → ~0.75 reward
            rewards[idx] = float((score - CLIP_CENTER) * CLIP_SCALE)

    return rewards
