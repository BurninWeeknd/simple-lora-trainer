from utils.ensure_fields import parse_int, parse_float


def apply(form, config, issues):
    """
    Apply LoRA-related settings from form to config.
    Mutates config in-place.
    """

    lora = config["lora"]

    rank = parse_int(form, "lora_rank", issues, min_value=1)
    if rank is not None:
        lora["rank"] = rank

    alpha = parse_int(form, "lora_alpha", issues, min_value=1)
    if alpha is not None:
        lora["alpha"] = alpha

    dropout = parse_float(form, "lora_dropout", issues, min_value=0, max_value=1)
    if dropout is not None:
        lora["dropout"] = dropout

    # Strings — no parsing needed
    if "lora_target_modules" in form:
        lora["target_modules"] = form["lora_target_modules"]

    if "lora_bias" in form:
        lora["bias"] = form["lora_bias"]
