"""
model_config.py
MedGemma Multi-AI Agentic System

Defines Pydantic-based configuration classes for model settings.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class QuantizationConfig(BaseModel):
    enabled: bool = Field(False, description="Enable model quantization")
    bits: int = Field(8, description="Number of bits for quantization")
    method: str = Field("nf4", description="Quantization method (e.g., nf4, fp4)")


class GenerationConfig(BaseModel):
    max_length: int = Field(2048, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling probability")
    top_k: int = Field(50, description="Top-k sampling")
    do_sample: bool = Field(True, description="Whether to use sampling")
    repetition_penalty: float = Field(1.1, description="Repetition penalty")
    pad_token_id: Optional[int] = Field(0, description="Pad token ID")
    eos_token_id: Optional[int] = Field(2, description="EOS token ID")
    stop: Optional[Any] = Field(None, description="Stop sequences for API")


class ModelConfig(BaseModel):
    model_name: str = Field(..., description="Hugging Face model identifier or local path")
    device: str = Field("auto", description="Device to load model on: auto, cpu, cuda")
    cache_dir: Path = Field(Path("./models/cache"), description="Local cache directory for model files")
    use_hf_api: bool = Field(False, description="Whether to use Hugging Face Inference API")
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    api_token: Optional[str] = Field(None, description="Override HF API token via config")

    @validator("device")
    def validate_device(cls, v: str) -> str:
        if v not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of 'auto', 'cpu', or 'cuda'")
        return v

    @validator("cache_dir", pre=True)
    def expand_cache_dir(cls, v: Any) -> Path:
        return Path(v).expanduser().resolve()


# Example usage:
# from medgemma_multiagent.utils.model_config import ModelConfig
# cfg = ModelConfig(**config_dict["models"]["medgemma"])
# print(cfg.json(indent=2))
