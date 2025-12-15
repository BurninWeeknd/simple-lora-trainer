def apply(form, config, issues):
    """
    Apply VAE-related settings from form to config.
    Mutates config in-place.
    """

    vae = config["vae"]

    if "vae_path" in form:
        path = form["vae_path"].strip()
        vae["path"] = path if path else None

    vae["bake_into_latents"] = "vae_bake_into_latents" in form
    vae["use_for_sampling"] = "vae_use_for_sampling" in form
