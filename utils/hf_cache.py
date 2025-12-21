import os
from utils.paths import MODELS_DIR

HF_CACHE_ROOT = MODELS_DIR / "hf_cache"

def setup_hf_env():
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_ROOT / "transformers")
    os.environ["DIFFUSERS_CACHE"] = str(HF_CACHE_ROOT / "diffusers")
    os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_ROOT / "datasets")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
