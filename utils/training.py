from utils.ensure_fields import parse_int, parse_float


def apply(form, config, issues):
    """
    Apply training-related settings from form to config.
    Mutates config in-place.
    Appends validation issues.
    """

    training = config["training"]

    # ---- Core training settings ----

    epochs = parse_int(form, "epochs", issues, min_value=1)
    if epochs is not None:
        training["epochs"] = epochs

    grad_accum = parse_int(form, "gradient_accumulation", issues, min_value=1)
    if grad_accum is not None:
        training["gradient_accumulation"] = grad_accum

    # ---- Conditioning ----

    conditioning = training["conditioning"]

    clip_skip = parse_int(form, "clip_skip", issues, min_value=0, max_value=4)
    if clip_skip is not None:
        conditioning["clip_skip"] = clip_skip

    # ---- Learning rates ----

    lrs = training["learning_rates"]

    lr_unet = parse_float(form, "lr_unet", issues, min_value=0)
    if lr_unet is not None:
        lrs["unet"] = lr_unet

    lr_clip = parse_float(form, "lr_clip", issues, min_value=0)
    if lr_clip is not None:
        lrs["clip"] = lr_clip

    # T5 is optional
    lr_t5_raw = form.get("lr_t5", "").strip()
    if lr_t5_raw == "":
        lrs["t5"] = None
    else:
        lr_t5 = parse_float(form, "lr_t5", issues, min_value=0)
        if lr_t5 is not None:
            lrs["t5"] = lr_t5
