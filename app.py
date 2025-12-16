from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pathlib import Path
import yaml
import shutil
import os
import signal

from utils.risk_analysis import analyze_training_risk
import utils.training as training
import utils.lora as lora
import utils.vae as vae
import utils.precision as precision
import utils.optimizer as optimizer
import utils.dataset as dataset
from utils.create_lora_project import create_project
import utils.model as model
from utils.launch_training import launch_training, TrainingConfigError
from utils.dataset import get_dataset_root, get_repeat_from_dataset_root

app = Flask(__name__)
app.secret_key = "dev-key-change-me"

BASE_DIR = Path.home() / "lora_projects"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

@app.route("/train_logs/<project>")
def train_logs(project):
    log_path = BASE_DIR / project / "logs" / "train.log"
    if not log_path.exists():
        return {"logs": ""}

    return {"logs": log_path.read_text()[-10000:]}

@app.route("/stop/<project>", methods=["POST"])
def stop_training(project):
    pid_file = BASE_DIR / project / "training.pid"
    pid = int(pid_file.read_text().strip())
    print(f"[STOP] Stopping process group {pid} for {project}")

    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception as e:
        print(f"[STOP] SIGTERM failed, forcing kill: {e}")
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception as e:
            print(f"[STOP] SIGKILL failed: {e}")

    pid_file.unlink(missing_ok=True)
    return redirect(url_for("index", project=project))

@app.route("/dataset-repeat", methods=["POST"])
def dataset_repeat():
    data = request.get_json(force=True) or {}
    project_name = (data.get("project_name") or "").strip()

    if not project_name:
        return jsonify({"error": "Missing project_name"}), 400

    dataset_root = get_dataset_root(project_name)
    repeat = get_repeat_from_dataset_root(dataset_root)

    return jsonify({
        "dataset_root": str(dataset_root),
        "repeat": repeat
    })

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

@app.route("/train/<project>")
def start_training(project):
    pid_file = BASE_DIR / project / "training.pid"

    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if is_process_running(pid):
                return redirect(url_for("index", project=project))
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)

    try:
        launch_training(project)

    except TrainingConfigError as e:
        # Fatal: cannot proceed
        session["ui_issues"] = [{
            "field": "model_checkpoint",
            "level": "fatal",
            "message": str(e),
        }]
        return redirect(url_for("index", project=project))

    except Exception:
        # Generic failure
        session["ui_issues"] = [{
            "field": "__global__",
            "level": "danger",
            "message": "Training failed to start. Check Training Logs for details.",
        }]
        return redirect(url_for("index", project=project))

    return redirect(url_for("index", project=project))

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
                "level": "fatal",
                "message": f"Config file for project '{selected}' no longer exists."
            })
            selected = None

        elif "__error__" in config:
            issues.append({
                "field": "__global__",
                "level": "fatal",
                "message": "Config file is corrupted or has invalid YAML syntax."
            })
            config = None

    if request.method == "POST":
        issues = []
        danger_fields = []

        if not config:
            issues.append({
                "field": "__global__",
                "level": "fatal",
                "message": "Cannot save because the config file is missing or corrupted."
            })

        else:
            missing = REQUIRED_TOP_LEVEL - set(config.keys())

            if missing:
                issues.append({
                    "field": "__global__",
                    "level": "fatal",
                    "message": (
                        "Config file is missing required sections: "
                        + ", ".join(sorted(missing))
                    )
                })
            else:
                model.apply(request.form, config, issues)
                dataset.apply(request.form, config, issues)
                training.apply(request.form, config, issues)
                lora.apply(request.form, config, issues)
                vae.apply(request.form, config, issues)
                precision.apply(request.form, config, issues)
                optimizer.apply(request.form, config, issues)

                issues += analyze_training_risk(config["training"])

        danger_fields = [i["field"] for i in issues if i["level"] == "danger"]
        fatal_fields = [i["field"] for i in issues if i["level"] == "fatal"]

        action = request.form.get("action", "save")

        if not fatal_fields and not danger_fields:
            save_config(selected, config)

            if action == "train":
                return redirect(url_for("start_training", project=selected))

            return redirect(url_for("index", project=selected))

    available_models = []
    if MODELS_DIR.exists():
        available_models = sorted(p.name for p in MODELS_DIR.glob("*.safetensors"))

    training_status = "idle"
    pid_file = BASE_DIR / selected / "training.pid" if selected else None
    if pid_file and pid_file.exists():
        training_status = "running"

    return render_template(
        "index.html",
        projects=projects,
        selected=selected,
        config=config,
        issues=issues,
        danger_fields=danger_fields,
        training_status=training_status,
        available_models=available_models,
    )

if __name__ == "__main__":
    app.run(debug=False)
