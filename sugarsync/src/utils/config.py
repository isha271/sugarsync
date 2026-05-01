"""SugarSync — Config Loader"""
import yaml
from pathlib import Path

_CONFIG = None

def load_config(path: str = "config.yaml") -> dict:
    global _CONFIG
    if _CONFIG is None:
        config_path = Path(path)
        if not config_path.exists():
            # Try from project root
            config_path = Path(__file__).parents[2] / "config.yaml"
        with open(config_path) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG
