# train.py
from pathlib import Path
import yaml
import sys

BASE_DIR = Path.home() / "lora_projects"

def load_config(project):
    config_path = BASE_DIR / project / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found")

    return yaml.safe_load(config_path.read_text())

def dry_run(project):
    config = load_config(project)

    model = config["model"]["architecture"]  # ← ADD THIS LINE

    dataset = config["dataset"]
    training = config["training"]
    lora = config["lora"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    precision = config["precision"]

    print(f"[TRAIN] Project: {project}")
    print(f"[TRAIN] Model: {model.upper()}")  # ← ADD THIS LINE
    print(f"[TRAIN] Dataset path: {dataset['path']}")
    print(f"[TRAIN] Resolution: {dataset['resolution']}")
    print(f"[TRAIN] Batch size: {dataset['batch_size']}")
    print(f"[TRAIN] Epochs: {training['epochs']}")
    print(f"[TRAIN] UNet LR: {training['learning_rates']['unet']}")
    print(f"[TRAIN] LoRA rank: {lora['rank']}")
    print(f"[TRAIN] Optimizer: {optimizer['type']}")
    print(f"[TRAIN] Scheduler: {scheduler['type']}")
    print(f"[TRAIN] Precision: {precision['mixed_precision']}")

    print("\n[TRAIN] Dry-run successful. No training started.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <project>")
        sys.exit(1)

    dry_run(sys.argv[1])
