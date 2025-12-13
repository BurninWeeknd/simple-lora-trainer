def analyze_training_risk(training):
    issues = []

    lrs = training.get("learning_rates", {})
    unet = lrs.get("unet")
    clip = lrs.get("clip")
    t5 = lrs.get("t5")

    TYPICAL_MIN, TYPICAL_MAX = 1e-5, 5e-5

    def check_lr(name, value):
        if value is None:
            return []

        issues = []
        if value > TYPICAL_MAX * 10 or value < TYPICAL_MIN / 10:
            issues.append({
                "field": f"lr_{name}",
                "level": "danger",
                "message": f"{name.upper()} LR {value} is far outside typical range."
            })
        elif value > TYPICAL_MAX * 5 or value < TYPICAL_MIN / 5:
            issues.append({
                "field": f"lr_{name}",
                "level": "warn",
                "message": f"{name.upper()} LR {value} is unusual."
            })
        return issues

    issues += check_lr("unet", unet)
    issues += check_lr("clip", clip)
    issues += check_lr("t5", t5)

    # Cross-check (useful, not annoying)
    if unet and clip and clip > unet:
        issues.append({
            "field": "lr_clip",
            "level": "warn",
            "message": "CLIP LR is higher than UNet LR. This is uncommon."
        })

    # Batch size sanity
    core = training.get("core", {})
    batch = core.get("batch_size")
    if batch is not None and batch < 1:
        issues.append({
            "field": "batch_size",
            "level": "danger",
            "message": "Batch size must be >= 1."
        })

    # Epochs sanity
    epochs = core.get("epochs")
    if epochs is not None and epochs <= 0:
        issues.append({
            "field": "epochs",
            "level": "danger",
            "message": "Epochs must be > 0."
        })

    return issues
