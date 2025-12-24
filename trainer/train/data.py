import os
from pathlib import Path
from typing import List, Tuple
from collections import Counter

import torch
from PIL import Image
from torchvision import transforms

from .config import TrainConfig, log

def parse_caption_tags(text: str) -> list[str]:
    parts = [t.strip() for t in text.split(",")]
    return [p for p in parts if p]

def load_dataset(dataset_dir: str, caption_ext: str) -> List[Tuple[str, str]]:
    items = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img_path = os.path.join(dataset_dir, name)
            cap_path = os.path.join(dataset_dir, os.path.splitext(name)[0] + caption_ext)
            if not os.path.exists(cap_path):
                raise RuntimeError(f"Missing caption for {name}")
            items.append((img_path, cap_path))
    if not items:
        raise RuntimeError("Dataset is empty")
    return items

def image_transform(res: int):
    return transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

def pick_bucket_resolution(w: int, h: int, cfg: TrainConfig) -> int:
    base = min(max(max(w, h), cfg.bucket_min_res), cfg.bucket_max_res)
    step = cfg.bucket_step
    return (base // step) * step

def apply_caption_options(text: str, cfg: TrainConfig) -> str:
    text = text.strip()
    if cfg.memorize_first_token:
        first = text.split(",")[0].strip()
        if first:
            text = f"{first}, {text}"
    if cfg.prepend_token:
        text = f"{cfg.prepend_token}, {text}"
    if cfg.append_token:
        text = f"{text}, {cfg.append_token}"
    return text

def build_dataset_buckets_and_tags(cfg: TrainConfig):
    log("STATUS loading_dataset")
    base_dataset = load_dataset(cfg.dataset, cfg.caption_ext)

    if cfg.repeats < 1:
        raise ValueError("repeats must be >= 1")

    dataset = base_dataset * cfg.repeats
    tag_counter = Counter()

    for _, cap_path in dataset:
        text = Path(cap_path).read_text(encoding="utf-8").strip()
        text = apply_caption_options(text, cfg)
        tag_counter.update(parse_caption_tags(text))

    trained_words = ", ".join([t for t, _ in tag_counter.most_common(200)])

    bucket_map: dict[int, list[int]] = {}
    if cfg.bucket_enabled:
        for i, (img_path, _) in enumerate(dataset):
            with Image.open(img_path) as img:
                w, h = img.size
            bucket_res = pick_bucket_resolution(w, h, cfg)
            bucket_map.setdefault(bucket_res, []).append(i)
    else:
        bucket_map[cfg.resolution] = list(range(len(dataset)))

    for res, ids in bucket_map.items():
        log(f"STATUS bucket[{res}] size={len(ids)}")

    if cfg.cache_latents and cfg.bucket_enabled:
        log("STATUS cache_latents=ENABLED (per-bucket)")
    elif cfg.cache_latents:
        log("STATUS cache_latents=ENABLED")

    if cfg.bucket_enabled:
        log(f"STATUS bucket=ENABLED min={cfg.bucket_min_res} max={cfg.bucket_max_res} step={cfg.bucket_step}")
    else:
        log("STATUS bucket=DISABLED")

    return dataset, bucket_map, tag_counter, trained_words

def build_latent_cache(
    *,
    cfg: TrainConfig,
    dataset,
    bucket_map,
    vae,
    device: torch.device,
    dtype: torch.dtype,
    scaling_factor: float,
):
    if not cfg.cache_latents:
        return None

    log("STATUS building latent cache")
    cached_latents_by_bucket: dict[int, dict[int, torch.Tensor]] = {}

    with torch.no_grad():
        for bucket_res, bucket_indices in bucket_map.items():
            log(f"STATUS caching bucket_res={bucket_res} samples={len(bucket_indices)}")
            tfm_bucket = image_transform(bucket_res)
            bucket_cache: dict[int, torch.Tensor] = {}

            for idx in bucket_indices:
                img_path, _ = dataset[idx]
                img = Image.open(img_path).convert("RGB")
                pixel = tfm_bucket(img).unsqueeze(0).to(device=device, dtype=dtype)
                latents = vae.encode(pixel).latent_dist.sample() * scaling_factor
                bucket_cache[idx] = latents.detach().to(torch.float16).cpu()

            cached_latents_by_bucket[bucket_res] = bucket_cache

    total_cached = sum(len(v) for v in cached_latents_by_bucket.values())
    log(f"STATUS cached_latents_total={total_cached}")
    return cached_latents_by_bucket
