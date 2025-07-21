"""
medgemma_wrapper.py
MedGemma Multi-AI Agentic System

Author: Aaron Masuba
License: MIT
"""

import os
import asyncio
from typing import Any, Dict, Optional, Union
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, StoppingCriteriaList
from huggingface_hub import login, HfApi, HfFolder
from PIL import Image

from loguru import logger

class MedGemmaWrapper:
    """
    Wrapper for Google MedGemma multimodal model supporting both
    local and Hugging Face API-based inference.

    Usage:
        wrapper = MedGemmaWrapper(cfg)
        await wrapper.initialize()
        out = await wrapper.generate(image=img, prompt="...", **gen_kwargs)
        await wrapper.cleanup()
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        cfg keys:
          model_name    : str  — HF repo id or local path
          device        : str  — "cpu", "cuda", or "auto"
          cache_dir     : str  — local model cache directory
          quantization  : dict — {enabled: bool, bits: int, method: str}
          api_token     : Optional[str] — HF API token for remote inference
          generation    : generation kwargs (max_length, temperature, etc.)
        """
        self.cfg = cfg
        self.model_name = cfg["model_name"]
        self.cache_dir = Path(cfg.get("cache_dir", "./models"))
        self.device = cfg.get("device", "auto")
        self.quant_cfg = cfg.get("quantization", {})
        self.gen_cfg = cfg.get("generation", {})
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN", None)
        self._use_api = bool(self.api_token) and cfg.get("use_hf_api", False)

        self.model = None
        self.processor = None
        self.initialized = False
        self.log = logger.bind(module="MedGemmaWrapper")

    async def initialize(self) -> None:
        """Load model & processor, login to HF if using API."""
        # HF API login if needed
        if self._use_api:
            self.log.info("Logging in to Hugging Face API")
            login(self.api_token)

        # Device determination
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log.info(f"Using device: {self.device}")

        # Load config & processor
        self.log.info(f"Loading processor for {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

        # Load model locally if not using API
        if not self._use_api:
            self.log.info(f"Loading model weights from {self.model_name}")
            config = AutoConfig.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            if self.quant_cfg.get("enabled", False):
                self.log.info(
                    f"Quantizing model to {self.quant_cfg['bits']} bits ({self.quant_cfg['method']})"
                )
                # bitsandbytes quantization
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=self.quant_cfg["bits"] == 4,
                    bnb_4bit_quant_type=self.quant_cfg.get("method", "nf4"),
                    llm_int8_threshold=6.0
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                    device_map="auto" if self.device.startswith("cuda") else None,
                    cache_dir=self.cache_dir,
                )
        else:
            self.log.info("Using Hugging Face Inference API (no local weights)")

        self.initialized = True
        self.log.success("MedGemmaWrapper initialization complete")

    async def generate(
        self,
        *,
        image: Union[Image.Image, bytes, str],
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> str:
        """
        Generate text from MedGemma given an image and prompt.
        Selects local or API-based inference automatically.
        Returns generated string.
        """
        if not self.initialized:
            raise RuntimeError("MedGemmaWrapper not initialized")

        # Prepare inputs
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        if not self._use_api:
            # Move tensors to device
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            gen_kwargs = {
                "max_length": max_length or self.gen_cfg.get("max_length"),
                "temperature": temperature or self.gen_cfg.get("temperature"),
                "top_p": top_p or self.gen_cfg.get("top_p"),
                "top_k": top_k or self.gen_cfg.get("top_k"),
                "do_sample": do_sample if do_sample is not None else self.gen_cfg.get("do_sample"),
                "repetition_penalty": repetition_penalty or self.gen_cfg.get("repetition_penalty"),
                "pad_token_id": pad_token_id or self.gen_cfg.get("pad_token_id"),
                "eos_token_id": eos_token_id or self.gen_cfg.get("eos_token_id"),
            }
            self.log.debug(f"Generating with kwargs: {gen_kwargs}")
            output_ids = self.model.generate(**inputs, **gen_kwargs)
            text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return text.strip()
        else:
            # Hugging Face text-generation inference API
            from huggingface_hub import InferenceClient
            client = InferenceClient()
            payload = {"inputs": {"image": image, "prompt": prompt}, "parameters": {
                "max_new_tokens": max_length or self.gen_cfg.get("max_length"),
                "temperature": temperature or self.gen_cfg.get("temperature"),
                "top_p": top_p or self.gen_cfg.get("top_p"),
                "top_k": top_k or self.gen_cfg.get("top_k"),
                "stop": self.gen_cfg.get("stop", None),
            }}
            self.log.debug("Calling Hugging Face Inference API")
            result = client.text_generation(model=self.model_name, **payload)
            return result.generated_text.strip()

    async def cleanup(self) -> None:
        """Free GPU memory and other resources."""
        if self.model is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.initialized = False
        self.log.info("MedGemmaWrapper cleaned up")
