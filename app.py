from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import yaml
import shutil
import subprocess

from utils.risk_analysis import analyze_training_risk
import utils.training as training
import utils.lora as lora
import utils.vae as vae
import utils.precision as precision
import utils.optimizer as optimizer
import utils.dataset as dataset
from utils.create_lora_project import create_project
from utils.vram_estimate import estimate_vram
import utils.model as model


app = Flask(__name__)

BASE_DIR = Path.home() / "lora_projects"

@app.route("/create_project", methods=["POST"])
def create_project_route():
    name = request.form.get("project_name", "").strip()

    if not name:
        return redirect(url_for("index"))

    try:
        create_project(name)
    except FileExistsError:
        pass

    return redirect(url_for("index", project=name))

@app.route("/delete_project", methods=["POST"])
def delete_project_route():
    name = request.form.get("project_name", "").strip()
    confirm = request.form.get("confirm_name", "").strip()

    if not name or name != confirm:
        return redirect(url_for("index", project=name))

    project_dir = BASE_DIR / name

    if project_dir.exists() and project_dir.is_dir():
        shutil.rmtree(project_dir)

    return redirect(url_for("index"))

@app.route("/train/<project>")
def start_training(project):
    project_dir = BASE_DIR / project

    if not project_dir.exists():
        return redirect(url_for("index"))

    # TEMP: placeholder
    print(f"[TRAIN] Starting training for {project}")

    return redirect(url_for("index", project=project))

def load_config(project_name):
    path = BASE_DIR / project_name / "config.yaml"

    if not path.exists():
        return None

    try:
        config = yaml.safe_load(path.read_text())

        config.setdefault("model", {})
        config["model"].setdefault("architecture", "sdxl")
        config["model"].setdefault("checkpoint", None)

        return config

    except yaml.YAMLError as e:
        return {
            "__error__": {
                "type": "yaml_parse",
                "message": str(e)
            }
        }

def save_config(project_name, config):
    path = BASE_DIR / project_name / "config.yaml"
    path.write_text(yaml.dump(config, sort_keys=False))

@app.route("/", methods=["GET", "POST"])
def index():
    projects = [p.name for p in BASE_DIR.iterdir() if p.is_dir()]
    selected = request.values.get("project")

    config = None
    issues = []
    danger_fields = []

    REQUIRED_TOP_LEVEL = {
        "dataset",
        "training",
        "lora",
        "optimizer",
        "scheduler",
        "precision",
        "output",
    }

    if selected:
        config = load_config(selected)

        if config is None:
            issues.append({
                "field": "__global__",
                "level": "danger",
                "message": f"Config file for project '{selected}' no longer exists."
            })
            selected = None

        elif "__error__" in config:
            issues.append({
                "field": "__global__",
                "level": "danger",
                "message": "Config file is corrupted or has invalid YAML syntax."
            })
            config = None

    if request.method == "POST":

        if not config:
            issues.append({
                "field": "__global__",
                "level": "danger",
                "message": "Cannot save because the config file is missing or corrupted."
            })

        else:
            # Check structural integrity
            missing = REQUIRED_TOP_LEVEL - set(config.keys())

            if missing:
                issues.append({
                    "field": "__global__",
                    "level": "danger",
                    "message": (
                        "Config file is missing required sections: "
                        + ", ".join(sorted(missing))
                    )
                })
            else:
                issues = []

                dataset.apply(request.form, config, issues)
                model.apply(request.form, config, issues)
                training.apply(request.form, config, issues)
                lora.apply(request.form, config, issues)
                vae.apply(request.form, config, issues)
                precision.apply(request.form, config, issues)
                optimizer.apply(request.form, config, issues)

                issues += analyze_training_risk(config["training"])
                danger_fields = [i["field"] for i in issues if i["level"] == "danger"]

                action = request.form.get("action", "save")

                if not danger_fields or request.form.get("confirm") == "yes":
                    save_config(selected, config)

                    if action == "train":
                        return redirect(url_for("start_training", project=selected))

                    return redirect(url_for("index", project=selected))


    vram_estimate_value = None
    if config:
        try:
            vram_estimate_value = estimate_vram(config)
        except Exception:
            vram_estimate_value = None

    return render_template(
        "index.html",
        projects=projects,
        selected=selected,
        config=config,
        issues=issues,
        danger_fields=danger_fields,
        vram_estimate=vram_estimate_value
    )

if __name__ == "__main__":
    app.run(debug=False)
