from pathlib import Path
import toml


def generate_sd_scripts_toml(config: dict, out_path: Path):
    model = config["model"]
    dataset = config["dataset"]
    training = config["training"]
    lora = config["lora"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    precision = config["precision"]
    output = config["output"]

    toml_config = {
        "training": {
            "network_module": "networks.lora",
            "network_dim": lora["rank"],
            "network_alpha": lora["alpha"],
            "learning_rate": training["learning_rates"]["unet"],
            "train_batch_size": dataset["batch_size"],
            "max_train_epochs": training["epochs"],
            "mixed_precision": precision["mixed_precision"],
            "save_every_n_epochs": output["save_every_epochs"],
            "output_dir": output["output_dir"],
            "output_name": config["project"]["name"],
        },
        "dataset": {
            "train_data_dir": dataset["path"],
            "resolution": dataset["resolution"],
        },
        "optimizer": {
            "optimizer_type": optimizer["type"],
            "weight_decay": optimizer["weight_decay"],
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(toml.dumps(toml_config))
