def apply(form, config, issues):
    """
    Apply precision-related settings from form to config.
    Mutates config in-place.
    """

    precision = config["precision"]

    # Mixed precision (string select)
    if "mixed_precision" in form:
        precision["mixed_precision"] = form["mixed_precision"]

    # Booleans (checkboxes)
    precision["gradient_checkpointing"] = "gradient_checkpointing" in form
    precision["xformers"] = "xformers" in form
    precision["cpu_offload"] = "cpu_offload" in form
