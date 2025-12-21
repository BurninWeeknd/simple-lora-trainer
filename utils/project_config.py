import yaml
from utils.paths import project_config_path


def load_config(project):
    path = project_config_path(project)
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def save_config(project, config):
    path = project_config_path(project)
    path.write_text(
        yaml.dump(config, sort_keys=False),
        encoding="utf-8"
    )
