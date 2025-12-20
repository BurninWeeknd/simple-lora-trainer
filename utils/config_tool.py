import yaml

from utils.paths import project_dir

project_name = input("Project name: ").strip()
proj_dir = project_dir(project_name)
config_path = proj_dir / "config.yaml"

if not config_path.exists():
    print(f"Config not found: {config_path}")
    raise SystemExit(1)

config = yaml.safe_load(config_path.read_text())

def validate_config(config, proj_dir):
    warnings = []

    dataset_rel = config["dataset"]["path"]
    dataset_path = proj_dir / dataset_rel
    if not dataset_path.exists():
        warnings.append(f"Dataset path does not exist: {dataset_path}")

    batch = config["dataset"]["batch_size"]
    if batch < 1:
        warnings.append("Batch size must be >= 1")

    lr = config["training"]["learning_rates"]["unet"]
    if not (1e-6 <= lr <= 1e-3):
        warnings.append(f"UNet learning rate looks unusual: {lr}")

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

warnings = validate_config(config, proj_dir)

if warnings:
    print("\nWarnings:")
    for w in warnings:
        print(f"- {w}")
else:
    print("\nNo config warnings")
