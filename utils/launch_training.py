import subprocess
import shlex
import yaml
import os
import signal
import sys
from pathlib import Path

from utils.paths import (
    project_dir,
    project_dataset_dir,
    project_output_dir,
    TRAINER_DIR,
)

from utils.trainer_cli_adapter import build_train_lora_cli_args
from utils.hf_cache import setup_hf_env

setup_hf_env
env = os.environ.copy()


class TrainingConfigError(Exception):
    """Fatal configuration error that should never occur if Save validation is correct."""
    pass

def launch_training(project_name: str):
    proj_dir = project_dir(project_name)
    config_path = proj_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    log_dir = proj_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    config = yaml.safe_load(config_path.read_text())

    model = config.get("model", {})
    arch = model.get("architecture")

    if not arch:
        raise TrainingConfigError("config.model.architecture is required")

    if arch in ("sd15", "sd1", "sd1.5", "sd"):
        model_type = "sd"
    elif arch == "sdxl":
        model_type = "sdxl"
    else:
        raise TrainingConfigError(
            f"Invalid model architecture '{arch}'. Expected sd or sdxl."
        )

    train_data_dir = project_dataset_dir(project_name)
    if not train_data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {train_data_dir}")

    output_dir = project_output_dir(project_name)
    output_dir.mkdir(exist_ok=True)

    args = build_train_lora_cli_args(config, proj_dir)

    if model_type == "sd":
        trainer_script = TRAINER_DIR / "train_sd.py"
    elif model_type == "sdxl":
        trainer_script = TRAINER_DIR / "train_sdxl.py"
    else:
        raise TrainingConfigError(f"Unsupported model_type: {model_type}")

    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer_script}")

    cmd = ["python", str(trainer_script), *args]

    print("[TRAIN] Launching training:")
    print(" ".join(shlex.quote(c) for c in cmd))

    logfile = open(log_dir / "train.log", "w")

    if sys.platform == "win32":
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            env=env,
            cwd=str(proj_dir),
        )
        pgid = process.pid
    else:
        process = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env,
            cwd=str(proj_dir),
        )
        pgid = os.getpgid(process.pid)

    pid_file = proj_dir / "training.pid"
    pid_file.write_text(str(pgid))

    print(f"[TRAIN] Training started (PGID {pgid})")
