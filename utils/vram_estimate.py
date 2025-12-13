def estimate_vram(config):
    dataset = config["dataset"]
    training = config["training"]
    precision = config["precision"]
    lora = config["lora"]
    model = config.get("model", {}).get("architecture", "sdxl")

    res = dataset["resolution"]
    batch = dataset["batch_size"]
    rank = lora["rank"]

    # -------------------------------
    # Base model memory (fp16, idle)
    # -------------------------------
    if model == "sd15":
        base_model_gb = 4.0
        ref_activation_gb = 1.0
        ref_resolution = 512
    elif model == "sdxl":
        base_model_gb = 6.5
        ref_activation_gb = 1.5
        ref_resolution = 1024
    else:
        # Unknown model — conservative
        base_model_gb = 6.5
        ref_activation_gb = 1.5
        ref_resolution = 1024

    # -------------------------------
    # Activation memory scaling
    # -------------------------------
    pixels = res * res
    ref_pixels = ref_resolution * ref_resolution
    activation_gb = ref_activation_gb * (pixels / ref_pixels) * batch

    # -------------------------------
    # LoRA overhead (small but real)
    # -------------------------------
    lora_gb = 0.1 * (rank / 64)

    # -------------------------------
    # Precision multiplier
    # -------------------------------
    precision_mult = {
        "fp16": 1.0,
        "bf16": 1.1,
        "fp32": 1.8
    }.get(precision.get("mixed_precision", "fp16"), 1.0)

    # -------------------------------
    # Memory savers
    # -------------------------------
    savings = 0.0
    if precision.get("gradient_checkpointing"):
        savings += 1.5
    if precision.get("xformers"):
        savings += 1.0

    total = (base_model_gb + activation_gb + lora_gb) * precision_mult
    total -= savings

    return max(round(total, 1), 0.0)