from pathlib import Path

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_dataset_root: Path | None = None

def set_dataset_root(path: str | Path):
    global _dataset_root
    _dataset_root = Path(path).expanduser().resolve()

def list_dataset_images():
    if _dataset_root is None or not _dataset_root.exists():
        return []

    images = []
    for p in sorted(_dataset_root.iterdir()):
        if p.is_file() and p.suffix.lower() in _SUPPORTED_IMAGE_EXTS:
            images.append({
                "name": p.name,
                "rel_path": p.name,
            })

    return images

def resolve_dataset_image(rel_path: str) -> Path | None:
    if _dataset_root is None:
        return None

    p = (_dataset_root / rel_path).resolve()

    if not p.is_file():
        return None

    # traversal protection
    if _dataset_root not in p.parents:
        return None

    return p

def read_caption_for_image(image_name: str) -> str:
    if _dataset_root is None:
        return ""

    txt_path = (_dataset_root / Path(image_name).with_suffix(".txt")).resolve()

    if not txt_path.exists() or not txt_path.is_file():
        return ""

    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def write_caption_for_image(image_name: str, caption: str) -> bool:
    if _dataset_root is None:
        return False

    txt_path = (_dataset_root / Path(image_name).with_suffix(".txt")).resolve()

    try:
        txt_path.write_text(caption.strip() + "\n", encoding="utf-8")
        return True
    except Exception:
        return False
