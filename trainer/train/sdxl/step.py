from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import TrainConfig
from ..data import image_transform, apply_caption_options
from .inference import make_add_time_ids

@torch.no_grad()
def encode_prompt_sdxl(
    captions,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    dtype,
):
    te1_device = next(text_encoder.parameters()).device
    te2_device = next(text_encoder_2.parameters()).device

    inputs_1 = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(te1_device)

    inputs_2 = tokenizer_2(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_2.model_max_length,
        return_tensors="pt",
    ).to(te2_device)

    emb_1 = text_encoder(**inputs_1).last_hidden_state
    out_2 = text_encoder_2(**inputs_2)
    emb_2 = out_2.last_hidden_state
    pooled = out_2.pooler_output

    prompt_embeds = torch.cat([emb_1, emb_2], dim=-1)
    return prompt_embeds.to(dtype), pooled.to(dtype)

class SDXLTrainStep:
    def __init__(
        self,
        *,
        cfg: TrainConfig,
        dataset,
        cached_latents_by_bucket,
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        vae,
        unet,
        scheduler,
        device: torch.device,
        dtype: torch.dtype,
        scaling_factor: float,
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.cached = cached_latents_by_bucket
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype
        self.scaling_factor = scaling_factor

    def __call__(self, batch_indices: list[int], bucket_res: int) -> torch.Tensor:
        tfm_bucket = image_transform(bucket_res)

        images = []
        captions = []

        for idx in batch_indices:
            img_path, cap_path = self.dataset[idx]
            text = Path(cap_path).read_text(encoding="utf-8").strip()
            text = apply_caption_options(text, self.cfg)
            img = Image.open(img_path).convert("RGB")
            images.append(tfm_bucket(img))
            captions.append(text)

        pixel = torch.stack(images).to(device=self.device, dtype=self.dtype)

        if self.cfg.cache_latents:
            assert self.cached is not None
            bucket_cache = self.cached[bucket_res]
            latents = torch.cat([bucket_cache[i] for i in batch_indices], dim=0).to(device=self.device, dtype=self.dtype)
        else:
            with torch.no_grad():
                latents = self.vae.encode(pixel).latent_dist.sample() * self.scaling_factor

        noise = torch.randn_like(latents)
        t = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.size(0),),
            device=self.device,
        ).long()

        noisy = self.scheduler.add_noise(latents, noise, t)

        with torch.no_grad():
            prompt_embeds, pooled = encode_prompt_sdxl(
                captions,
                self.tokenizer,
                self.tokenizer_2,
                self.text_encoder,
                self.text_encoder_2,
                self.dtype,
            )
            add_time_ids = make_add_time_ids(latents.size(0), bucket_res, self.dtype)

        unet_device = next(self.unet.parameters()).device
        pred = self.unet(
            noisy.to(unet_device),
            t.to(unet_device),
            encoder_hidden_states=prompt_embeds.to(unet_device),
            added_cond_kwargs={
                "text_embeds": pooled.to(unet_device),
                "time_ids": add_time_ids.to(unet_device),
            },
        ).sample

        loss = F.mse_loss(pred.float().to(self.device), noise.float())
        return loss
