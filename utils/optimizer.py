from utils.ensure_fields import parse_int, parse_float


def apply(form, config, issues):
    """
    Apply optimizer and scheduler settings from form to config.
    Mutates config in-place.
    """

    optimizer = config["optimizer"]
    scheduler = config["scheduler"]

    # ---- Optimizer ----

    if "optimizer_type" in form:
        optimizer["type"] = form["optimizer_type"]

    weight_decay = parse_float(form, "weight_decay", issues, min_value=0)
    if weight_decay is not None:
        optimizer["weight_decay"] = weight_decay

    beta1 = parse_float(form, "beta1", issues, min_value=0, max_value=1)
    beta2 = parse_float(form, "beta2", issues, min_value=0, max_value=1)

    if beta1 is not None and beta2 is not None:
        optimizer["betas"] = [beta1, beta2]

    epsilon = parse_float(form, "epsilon", issues, min_value=0)
    if epsilon is not None:
        optimizer["epsilon"] = epsilon

    # ---- Scheduler ----

    if "scheduler_type" in form:
        scheduler["type"] = form["scheduler_type"]

    warmup = parse_int(form, "warmup_steps", issues, min_value=0)
    if warmup is not None:
        scheduler["warmup_steps"] = warmup

    cycles = parse_int(form, "num_cycles", issues, min_value=1)
    if cycles is not None:
        scheduler["num_cycles"] = cycles
