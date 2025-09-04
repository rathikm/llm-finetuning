from pathlib import Path
import yaml

REQUIRED_TOP_LEVEL = ["base_model", "tokenizer_name"]

def load_config(path: str) -> dict:
    """
    Load a YAML config into a dict and do minimal validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in config: {missing}")

    return cfg
