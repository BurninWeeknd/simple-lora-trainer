from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

_processor = None
_model = None


def _load(device="cpu"):
    global _processor, _model
    if _model is None:
        _processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        _model.eval()


def generate_caption(image_path, device="cpu") -> str:
    _load(device)

    image = Image.open(image_path).convert("RGB")
    inputs = _processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=40
        )

    return _processor.decode(out[0], skip_special_tokens=True)
