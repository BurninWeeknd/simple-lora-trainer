def apply(form, config, issues):
    """
    Apply precision-related settings from form to config.
    Mutates config in-place.
    Appends validation issues.
    """
    precision = config.setdefault("precision", {})

    precision["mixed_precision"] = form.get("mixed_precision", "fp16")

    precision["gradient_checkpointing"] = "gradient_checkpointing" in form
    precision["xformers"] = "xformers" in form
    precision["cpu_offload"] = "cpu_offload" in form

    if precision.get("xformers"):
        try:
            import xformers
        except Exception:
            issues.append({
                "field": "xformers",
                "level": "fatal",
                "message": (
                    "xFormers is enabled but not installed. "
                    "Disable xFormers or install it before training."
                )
            })
