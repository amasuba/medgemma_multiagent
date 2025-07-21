"""
config.py
MedGemma Multi-AI Agentic System

Advanced configuration management with YAML, environment variables, and validation.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseSettings, Field, root_validator, ValidationError
from pydantic_settings import BaseSettings as PydanticSettings, SettingsConfigDict

class SystemConfig(PydanticSettings):
    name: str = Field(..., description="System name")
    version: str = Field(..., description="System version")
    description: str = Field("", description="System description")
    environment: str = Field("development", description="Runtime environment")
    debug: bool = Field(False, description="Enable debug mode")

    model_name: str = Field(..., description="MedGemma HF model identifier")
    device: str = Field("auto", description="Compute device: auto, cpu, cuda")
    cache_dir: Path = Field(Path("./models/cache"), description="Model cache directory")
    use_hf_api: bool = Field(False, description="Use Hugging Face Inference API")
    api_token: str = Field(..., description="Hugging Face API token")

    # Agent global settings
    timeout: int = Field(300, description="Agent timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    communication_timeout: int = Field(60, description="Inter-agent comm timeout")
    log_level: str = Field("INFO", description="Global log level")

    class Config:
        env_prefix = ""
        env_file = ".env"
        case_sensitive = False

class Config:
    """
    Loads and validates the entire application configuration from YAML,
    with overrides from environment variables.
    """

    def __init__(self, path: str = "config.yaml"):
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Load YAML
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Flatten nested fields for settings where appropriate
        system = raw.get("system", {})
        models = raw.get("models", {}).get("medgemma", {})
        agents_global = raw.get("agents", {}).get("global", {})

        # Merge system, model, and agent-global into env-like dict
        merged: Dict[str, Any] = {
            **system,
            **models,
            **agents_global,
        }

        # Environment variable overrides
        # HUGGINGFACE_API_TOKEN must come from env if set
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if api_token:
            merged["api_token"] = api_token

        try:
            self._system_cfg = SystemConfig(**merged)
        except ValidationError as e:
            raise RuntimeError(f"Configuration validation error:\n{e}")

        # Store full raw config for sub-components
        self.raw = raw

    @property
    def system(self) -> SystemConfig:
        return self._system_cfg

    @property
    def models(self) -> Any:
        return self.raw.get("models", {})

    @property
    def agents(self) -> Any:
        return self.raw.get("agents", {})

    @property
    def data(self) -> Any:
        return self.raw.get("data", {})

    @property
    def retrieval(self) -> Any:
        return self.raw.get("retrieval", {})

    @property
    def evaluation(self) -> Any:
        return self.raw.get("evaluation", {})

    @property
    def logging(self) -> Any:
        return self.raw.get("logging", {})

    @property
    def api(self) -> Any:
        return self.raw.get("api", {})

    @property
    def deployment(self) -> Any:
        return self.raw.get("deployment", {})

    @property
    def security(self) -> Any:
        return self.raw.get("security", {})

    @property
    def experimental(self) -> Any:
        return self.raw.get("experimental", {})

    def __repr__(self):
        return f"<Config env={self._system_cfg.environment} version={self._system_cfg.version}>"
