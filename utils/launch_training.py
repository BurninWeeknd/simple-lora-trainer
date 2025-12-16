from pathlib import Path
import subprocess
import shlex
import yaml
import os
from safetensors import safe_open


BASE_DIR = Path.home() / "lora_projects"
SD_SCRIPTS = Path(__file__).resolve().parent.parent / "trainer" / "sd-scripts"


class TrainingConfigError(Exception):
    """Fatal configuration error that should never occur if Save validation is correct."""
    pass


def detect_model_architecture(checkpoint_path: Path) -> str:
    """
    Returns 'sdxl' or 'sd15' based on checkpoint contents.
    """
    with safe_open(checkpoint_path, framework="pt") as f:
        keys = f.keys()

        if any(k.startswith("text_encoder_2.") for k in keys):
            return "sdxl"

        if any("conditioner.embedders" in k for k in keys):
            return "sdxl"

        return "sd15"


def launch_training(project_name: str):
    project_dir = BASE_DIR / project_name
    config_path = project_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    config = yaml.safe_load(config_path.read_text())

    dataset = config["dataset"]
    training = config["training"]
    lora = config["lora"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    precision = config["precision"]
    output = config["output"]
    model = config.get("model", {})
    save_every = training.get("save_every_epochs", 1)
    save_every = max(1, int(save_every))

    model_arch = model.get("architecture", "sdxl")
    raw_checkpoint = model.get("checkpoint")

    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

    if raw_checkpoint:
        cp = Path(raw_checkpoint)
        if not cp.is_absolute():
            cp = MODELS_DIR / cp

        if not cp.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {cp}")

        detected_arch = detect_model_architecture(cp)

        if detected_arch != model_arch:
            raise TrainingConfigError(
                f"Model architecture mismatch:\n"
                f"- Selected architecture: {model_arch.upper()}\n"
                f"- Checkpoint appears to be: {detected_arch.upper()}"
            )

        checkpoint = str(cp)

    else:
        checkpoint = (
            "stabilityai/stable-diffusion-xl-base-1.0"
            if model_arch == "sdxl"
            else "runwayml/stable-diffusion-v1-5"
        )

    script = (
        SD_SCRIPTS / "sdxl_train_network.py"
        if model_arch == "sdxl"
        else SD_SCRIPTS / "train_network.py"
    )

    if not script.exists():
        raise FileNotFoundError(f"Training script not found: {script}")

    train_data_dir = project_dir / dataset["path"]
    if not train_data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {train_data_dir}")

    captions = dataset.get("captions", {})
    caption_ext = captions.get("extension") or ".txt"

    prepend = (captions.get("prepend_token") or "").strip()
    append = (captions.get("append_token") or "").strip()

    cmd = [
        "accelerate", "launch",
        str(script),

        "--pretrained_model_name_or_path", checkpoint,
        "--train_data_dir", str(train_data_dir),

        "--resolution", str(dataset["resolution"]),
        "--train_batch_size", str(dataset["batch_size"]),
        "--num_repeats", str(dataset["repeats"]),
        "--max_train_epochs", str(training["epochs"]),
        "--save_every_n_epochs", str(save_every),

        "--caption_extension", caption_ext,
    ]

    if prepend:
        cmd += ["--caption_prefix", prepend]
    if append:
        cmd += ["--caption_suffix", append]
        
    if captions.get("shuffle"):
        cmd.append("--shuffle_caption")

    cmd += [
        "--network_module", "networks.lora",
        "--network_dim", str(lora["rank"]),
        "--network_alpha", str(lora["alpha"]),

        "--learning_rate", str(training["learning_rates"]["unet"]),
        "--optimizer_type", optimizer["type"],
        "--lr_scheduler", scheduler["type"],

        "--output_dir", str(project_dir / output["output_dir"]),
        "--logging_dir", str(log_dir),

        "--mixed_precision", precision["mixed_precision"],
        "--save_model_as", output["save_format"],
    ]

    if precision.get("gradient_checkpointing"):
        cmd.append("--gradient_checkpointing")

    if precision.get("xformers"):
        cmd.append("--xformers")

    print("[TRAIN] Launching training:")
    print(" ".join(shlex.quote(c) for c in cmd))

    process = subprocess.Popen(
        cmd,
        cwd=SD_SCRIPTS,
        stdout=open(log_dir / "train.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

    pid_file = project_dir / "training.pid"
    pgid = os.getpgid(process.pid)
    pid_file.write_text(str(pgid))

    print(f"[TRAIN] Training started (PGID {pgid})")
