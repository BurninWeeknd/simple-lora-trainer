from utils.ensure_fields import parse_int, parse_float


def apply(form, config, issues):
    """
    Apply training-related settings from form to config.
    Mutates config in-place.
    Appends validation issues.
    """

    training = config["training"]

    epochs = parse_int(form, "epochs", issues, min_value=1)
    if epochs is not None:
        training["epochs"] = epochs

    save_every = parse_int(form, "save_every_epochs", issues, min_value=1)
    if save_every is not None:
        training["save_every_epochs"] = save_every

    grad_accum = parse_int(form, "gradient_accumulation", issues, min_value=1)
    if grad_accum is not None:
        training["gradient_accumulation"] = grad_accum

    conditioning = training["conditioning"]

    clip_skip = parse_int(form, "clip_skip", issues, min_value=0)
    if clip_skip is not None:
        conditioning["clip_skip"] = clip_skip

    lrs = training["learning_rates"]

    lr_unet = parse_float(form, "lr_unet", issues, min_value=0)
    if lr_unet is not None:
        lrs["unet"] = lr_unet

    lr_clip = parse_float(form, "lr_clip", issues, min_value=0)
    if lr_clip is not None:
        lrs["clip"] = lr_clip

    lr_t5_raw = form.get("lr_t5", "").strip()
    if lr_t5_raw == "":
        lrs["t5"] = 0
    else:
        lr_t5 = parse_float(form, "lr_t5", issues, min_value=0)
        if lr_t5 is not None:
            lrs["t5"] = lr_t5

    model_arch = config.get("model", {}).get("architecture", "sdxl")
    t5_lr = lrs.get("t5")

    try:
        t5_val = float(t5_lr) if t5_lr is not None else 0.0
    except (TypeError, ValueError):
        t5_val = 0.0

    if model_arch == "sd15" and t5_val != 0.0:
        issues.append({
            "field": "lr_t5",
            "level": "fatal",
            "message": "T5 learning rate is not supported for SD 1.5 (set it to 0 or empty)."
        })
