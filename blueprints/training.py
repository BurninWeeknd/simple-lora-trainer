from flask import Blueprint, redirect, url_for, jsonify
import os
import signal
import sys
import subprocess

from utils.paths import project_dir
from utils.launch_training import launch_training, TrainingConfigError

training_bp = Blueprint("training", __name__)

@training_bp.route("/train/<project>", methods=["POST"])
def start_training(project):
    try:
        launch_training(project)
    except TrainingConfigError as e:
        from flask import session
        session["ui_issues"] = [{
            "field": "__global__",
            "level": "fatal",
            "message": str(e),
        }]
    return redirect(url_for("ui.index", project=project))


@training_bp.route("/stop/<project>", methods=["POST"])
def stop_training(project):
    pid_file = project_dir(project) / "training.pid"

    if not pid_file.exists():
        return redirect(url_for("ui.index", project=project))

    pgid = int(pid_file.read_text())

    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pgid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(pgid, signal.SIGTERM)
    except Exception:
        pass

    pid_file.unlink(missing_ok=True)
    return redirect(url_for("ui.index", project=project))

@training_bp.route("/train_logs/<project>")
def train_logs(project):
    log_path = project_dir(project) / "logs" / "train.log"
    if not log_path.exists():
        return jsonify({"logs": ""})
    return jsonify({"logs": log_path.read_text()[-10000:]})
