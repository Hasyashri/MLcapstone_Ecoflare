# %%writefile services/ingestion/config_loader.py
"""
Config Loader - YAML → Python dict for all ingestion services
"""
import yaml
from pathlib import Path
from typing import Any, Dict
from services.management.logging import get_logger

logger = get_logger("ConfigLoader")

class ConfigLoader:
    """Load service-specific configs from YAML"""

    def __init__(self, config_path: str = "config/service_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate YAML config"""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config or {}
        except FileNotFoundError:
            logger.error(f"Config not found: {self.config_path}")
            return {}

    def get(self, *keys: str) -> Any:
        """Nested dict lookup: config.get('ingestion', 'satellite', 'modis_canada')"""
        cfg = self.config
        for key in keys:
            cfg = cfg.get(key, {})
        return cfg
