from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

UTILS_DIR     = REPO_ROOT / "utils"
TRAINER_DIR   = REPO_ROOT / "trainer"
BLUEPRINTS_DIR = REPO_ROOT / "blueprints"

MODELS_DIR    = REPO_ROOT / "models"
TEMPLATES_DIR = REPO_ROOT / "templates"
STATIC_DIR    = REPO_ROOT / "static"

BASE_MODELS_DIR = MODELS_DIR / "base"
LORAS_DIR       = MODELS_DIR / "loras"

PROJECTS_DIR = REPO_ROOT / "projects"

def project_dir(project_name: str) -> Path:
    return PROJECTS_DIR / project_name

def project_dataset_dir(project_name: str) -> Path:
    return project_dir(project_name) / "dataset"

def project_output_dir(project_name: str) -> Path:
    return project_dir(project_name) / "output"

def project_config_path(project_name: str) -> Path:
    return project_dir(project_name) / "config.yaml"

def ensure_dirs():
    """
    Create all required base directories.
    Safe to call multiple times.
    """
    for d in [
        MODELS_DIR,
        BASE_MODELS_DIR,
        LORAS_DIR,
        PROJECTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
