import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

DEFAULT_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
DEFAULT_TE_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_scale = 1.0
        self.dropout = dropout

        device = base.weight.device
        dtype = base.weight.dtype

        self.A = nn.Parameter(torch.randn(base.in_features, rank, device=device, dtype=dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, base.out_features, device=device, dtype=dtype))

    def forward(self, x, *args, **kwargs):
        delta = self.scale * ((x @ self.A) @ self.B)
        if self.training and self.dropout > 0:
            delta = F.dropout(delta, p=self.dropout)
        return self.base(x, *args, **kwargs) + self.lora_scale * delta

def lora_parameters(module: nn.Module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m.A
            yield m.B

def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    targets: list[str],
) -> tuple[int, list[nn.Parameter]]:
    replaced = 0
    params: list[nn.Parameter] = []
    modules = dict(module.named_modules())

    for name, mod in list(modules.items()):
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in targets):
            parent_name, child = name.rsplit(".", 1)
            parent = modules[parent_name]
            lora = LoRALinear(mod, rank, alpha, dropout)
            setattr(parent, child, lora)
            params.extend([lora.A, lora.B])
            replaced += 1

    if replaced == 0:
        raise RuntimeError("Injected 0 LoRA layers")

    return replaced, params

def set_lora_scale(module: nn.Module, scale: float) -> None:
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.lora_scale = float(scale)

def save_lora(
    *,
    unet: nn.Module,
    text_encoder: nn.Module | None,
    path: str,
    metadata: dict | None = None,
):
    tensors: dict[str, torch.Tensor] = {}

    for name, m in unet.named_modules():
        if not isinstance(m, LoRALinear):
            continue
        key = "lora_unet_" + name.replace(".", "_")
        tensors[f"{key}.lora_down.weight"] = m.A.T.detach().float().contiguous().cpu()
        tensors[f"{key}.lora_up.weight"] = m.B.T.detach().float().contiguous().cpu()
        tensors[f"{key}.alpha"] = torch.tensor(m.alpha)

    if text_encoder is not None:
        for name, m in text_encoder.named_modules():
            if not isinstance(m, LoRALinear):
                continue
            key = "lora_te_" + name.replace(".", "_")
            tensors[f"{key}.lora_down.weight"] = m.A.T.detach().float().contiguous().cpu()
            tensors[f"{key}.lora_up.weight"] = m.B.T.detach().float().contiguous().cpu()
            tensors[f"{key}.alpha"] = torch.tensor(m.alpha)

    save_file(tensors, path, metadata=metadata or {})

def save_lora_sdxl(
    *,
    unet: nn.Module,
    text_encoder: nn.Module | None,
    text_encoder_2: nn.Module | None,
    path: str,
    metadata: dict | None = None,
):
    tensors: dict[str, torch.Tensor] = {}

    for name, m in unet.named_modules():
        if not isinstance(m, LoRALinear):
            continue
        key = "lora_unet_" + name.replace(".", "_")
        tensors[f"{key}.lora_down.weight"] = m.A.T.detach().float().contiguous().cpu()
        tensors[f"{key}.lora_up.weight"] = m.B.T.detach().float().contiguous().cpu()
        tensors[f"{key}.alpha"] = torch.tensor(m.alpha)

    if text_encoder is not None:
        for name, m in text_encoder.named_modules():
            if not isinstance(m, LoRALinear):
                continue
            key = "lora_te1_" + name.replace(".", "_")
            tensors[f"{key}.lora_down.weight"] = m.A.T.detach().float().contiguous().cpu()
            tensors[f"{key}.lora_up.weight"] = m.B.T.detach().float().contiguous().cpu()
            tensors[f"{key}.alpha"] = torch.tensor(m.alpha)

    if text_encoder_2 is not None:
        for name, m in text_encoder_2.named_modules():
            if not isinstance(m, LoRALinear):
                continue
            key = "lora_te2_" + name.replace(".", "_")
            tensors[f"{key}.lora_down.weight"] = m.A.T.detach().float().contiguous().cpu()
            tensors[f"{key}.lora_up.weight"] = m.B.T.detach().float().contiguous().cpu()
            tensors[f"{key}.alpha"] = torch.tensor(m.alpha)

    save_file(tensors, path, metadata=metadata or {})
