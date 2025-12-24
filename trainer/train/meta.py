import json
from collections import Counter
from .config import TrainConfig

def build_lora_metadata(cfg: TrainConfig, tag_counter: Counter, trained_words: str) -> dict[str, str]:
    tag_payload = {"dataset": dict(tag_counter)}
    return {
        "ss_tag_frequency": json.dumps(tag_payload, ensure_ascii=False),
        "ss_trained_words": trained_words,
        "ss_network_dim": str(cfg.lora_rank),
        "ss_network_alpha": str(cfg.lora_alpha),
    }
