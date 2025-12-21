from flask import Blueprint, render_template, jsonify, request, send_file
import torch

from utils.dataset_io import (
    set_dataset_root,
    list_dataset_images,
    resolve_dataset_image,
    read_caption_for_image,
    write_caption_for_image,
)

from utils.paths import (
    PROJECTS_DIR,
    project_dir,
)

from utils.blip import generate_caption

from utils.project_config import (
    load_config,
    save_config,
)

ui_dataset_bp = Blueprint("ui_dataset", __name__)

@ui_dataset_bp.route("/dataset")
def dataset():
    return render_template("dataset.html")

@ui_dataset_bp.route("/api/dataset/image/<path:rel_path>")
def api_dataset_image(rel_path):
    img_path = resolve_dataset_image(rel_path)
    if img_path is None:
        return "Not found", 404

    return send_file(img_path)

@ui_dataset_bp.route("/api/dataset/save", methods=["POST"])
def api_dataset_save():
    data = request.get_json(silent=True) or {}
    images = data.get("images", [])

    if not images:
        return jsonify({
            "status": "nothing_to_save",
            "saved": 0
        })

    saved = 0

    for img in images:
        name = img.get("name")
        caption = img.get("caption", "")

        if not name:
            continue

        if write_caption_for_image(name, caption):
            saved += 1

    return jsonify({
        "status": "ok",
        "saved": saved
    })

@ui_dataset_bp.route("/api/projects")
def api_projects():
    projects = sorted(
        p.name for p in PROJECTS_DIR.iterdir()
        if p.is_dir()
    )
    return {"projects": projects}

@ui_dataset_bp.route("/api/dataset/load", methods=["POST"])
def api_dataset_load():
    data = request.get_json() or {}
    project = data.get("project")
    dataset_path = data.get("dataset_path")

    if not project or not dataset_path:
        return {"error": "Missing project or dataset path"}, 400

    config = load_config(project)
    if not config:
        return {"error": "Project config not found"}, 404

    config["dataset"]["path"] = dataset_path
    save_config(project, config)

    dataset_root = project_dir(project) / dataset_path
    if not dataset_root.exists():
        return {
            "error": "Dataset path does not exist",
            "path": str(dataset_root),
        }, 400

    set_dataset_root(dataset_root)

    images = []
    for img in list_dataset_images():
        images.append({
            "name": img["name"],
            "rel_path": img["rel_path"],
            "caption": read_caption_for_image(img["name"]),
        })

    return {
        "project": project,
        "dataset_path": dataset_path,
        "images": images,
    }

@ui_dataset_bp.route("/api/project/config/<project>")
def api_project_config(project):
    config = load_config(project)
    if not config:
        return {"error": "Config not found"}, 404

    return {
        "dataset_path": config["dataset"]["path"]
    }

@ui_dataset_bp.route("/api/dataset/autocaption", methods=["POST"])
def api_dataset_autocaption():
    data = request.get_json() or {}
    images = data.get("images", [])
    overwrite = bool(data.get("overwrite", False))

    if not images:
        return {"error": "No images provided"}, 400

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for img in images:
        name = img["name"]
        path = resolve_dataset_image(name)
        if not path:
            continue

        if not overwrite:
            existing = read_caption_for_image(name)
            if existing.strip():
                continue

        caption = generate_caption(path, device=device)
        write_caption_for_image(name, caption)

        results.append({
            "name": name,
            "caption": caption
        })

    return {
        "status": "ok",
        "captions": results
    }
