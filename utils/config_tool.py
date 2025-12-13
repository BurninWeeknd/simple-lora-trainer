from pathlib import Path
import yaml

project_dir = Path.home() / "lora_projects" / input("Project name: ").strip()
config_path = project_dir / "config.yaml"

if not config_path.exists():
    print("Config not found")
    exit(1)

config = yaml.safe_load(config_path.read_text())

def validate_config(config, project_dir):
    warnings = []

    dataset_path = project_dir / config["dataset"]["path"]
    if not dataset_path.exists():
        warnings.append(f"Dataset path does not exist: {dataset_path}")

    batch = config["training"]["batch_size"]
    if batch < 1:
        warnings.append("Batch size must be >= 1")

    lr = config["training"]["learning_rate"]
    if not (1e-6 <= lr <= 1e-3):
        warnings.append(f"Learning rate looks unusual: {lr}")

    res = config["dataset"]["resolution"]
    if res % 64 != 0:
        warnings.append(f"Resolution {res} is not divisible by 64")

    epochs = config["training"]["epochs"]
    if epochs <= 0:
        warnings.append("Epochs must be > 0")

    return warnings

print("\nLoaded config:")
for section, values in config.items():
    print(f"\n[{section}]")
    for key, value in values.items():
        print(f"{key}: {value}")

warnings = validate_config(config, project_dir)

if warnings:
    print("\nWarnings:")
    for w in warnings:
        print(f"- {w}")
else:
    print("\nNo config warnings")

