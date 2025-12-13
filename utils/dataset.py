from utils.ensure_fields import parse_int


def apply(form, config, issues):
    """
    Apply dataset-related settings from form to config.
    Mutates config in-place.
    """

    dataset = config["dataset"]

    # Basic fields
    if "dataset_path" in form:
        dataset["path"] = form["dataset_path"].strip()

    resolution = parse_int(form, "resolution", issues, min_value=64)
    if resolution is not None:
        dataset["resolution"] = resolution

    repeats = parse_int(form, "repeats", issues, min_value=1)
    if repeats is not None:
        dataset["repeats"] = repeats

    batch = parse_int(form, "batch_size", issues, min_value=1)
    if batch is not None:
        dataset["batch_size"] = batch

    dataset["shuffle"] = "shuffle" in form
    dataset["cache_latents"] = "cache_latents" in form

    # ---- Captions ----
    captions = dataset["captions"]

    if "caption_extension" in form:
        captions["extension"] = form["caption_extension"].strip()

    captions["first_word_memorize"] = "first_word_memorize" in form

    if "prepend_token" in form:
        val = form["prepend_token"].strip()
        captions["prepend_token"] = val if val else None

    if "append_token" in form:
        val = form["append_token"].strip()
        captions["append_token"] = val if val else None

    # ---- Bucketing ----
    bucket = dataset["bucket"]

    bucket["enabled"] = "bucket_enabled" in form

    min_res = parse_int(form, "bucket_min_res", issues, min_value=64)
    max_res = parse_int(form, "bucket_max_res", issues, min_value=64)
    step = parse_int(form, "bucket_step", issues, min_value=1)

    if min_res is not None:
        bucket["min_res"] = min_res
    if max_res is not None:
        bucket["max_res"] = max_res
    if step is not None:
        bucket["step"] = step
