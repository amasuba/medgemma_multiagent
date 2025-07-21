"""
vision_agent.py
MedGemma Multi-AI Agentic System

Author: Aaron Masuba
License: MIT
"""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageOps
from loguru import logger

from .base_agent import BaseAgent, AgentNotInitializedError, AgentTimeoutError


class VisionAgent(BaseAgent):
    """
    Analyzes chest X-ray images with the MedGemma multimodal model.

    Payload contract
    ----------------
    • image            : PIL.Image | bytes | str (path/URL)
    • analysis_type    : str   — one of {"detailed","findings","comparison","quality"}
    • context          : str   — optional clinical context
    • reference_report : str   — optional text for comparison mode
    """

    _VALID_ANALYSES = {"detailed", "findings", "comparison", "quality"}

    def __init__(self, cfg: Dict[str, Any], model_wrapper: "MedGemmaWrapper"):
        super().__init__(cfg, global_cfg=cfg.get("global", {}))
        self.model = model_wrapper

        # Agent-level configuration
        self.analysis_types = set(cfg["config"].get("analysis_types", []))
        if not self.analysis_types:
            self.analysis_types = self._VALID_ANALYSES

        self.conf_threshold: float = float(
            cfg["config"].get("confidence_threshold", 0.6)
        )
        self.max_size: Tuple[int, int] = tuple(
            cfg["config"]
            .get("max_image_size", [512, 512])
        )  # (w, h)

        self.preprocessing_cfg = cfg["config"].get("preprocessing", {})
        self._preprocess_enabled = self.preprocessing_cfg.get("resize", True)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _initialize(self) -> None:
        if not self.model or not self.model.initialized:
            raise AgentNotInitializedError(
                "MedGemmaWrapper must be initialized before VisionAgent"
            )
        self.log.info("VisionAgent ready ✔")

    async def _shutdown(self) -> None:
        # Nothing to clean up; model wrapper is owned by orchestrator.
        self.log.debug("VisionAgent shutdown complete")

    # ------------------------------------------------------------------ #
    # Core logic
    # ------------------------------------------------------------------ #

    async def _process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        img = self._coerce_image(payload.get("image"))
        analysis_type = payload.get("analysis_type", "detailed").lower().strip()
        if analysis_type not in self._VALID_ANALYSES:
            raise ValueError(
                f"Invalid analysis_type '{analysis_type}'. "
                f"Must be one of {sorted(self._VALID_ANALYSES)}"
            )

        context = payload.get("context") or ""
        reference_report = payload.get("reference_report") or ""

        t0 = time.perf_counter()
        if self._preprocess_enabled:
            img = self._preprocess(img)

        # Construct the prompt for MedGemma
        prompt = self._build_prompt(
            analysis_type=analysis_type,
            context=context,
            reference_report=reference_report,
        )

        self.log.debug(f"Prompt length: {len(prompt.split())} tokens")

        # Call the multimodal model (async safe)
        try:
            response = await asyncio.wait_for(
                self.model.generate(
                    image=img,
                    prompt=prompt,
                    max_length=self.model.cfg.generation.max_length,
                    temperature=self.model.cfg.generation.temperature,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            self.log.error("VisionAgent timed-out during model inference")
            raise AgentTimeoutError from exc

        elapsed = time.perf_counter() - t0
        self.log.info(f"Vision analysis completed in {elapsed:.2f}s")

        parsed = self._postprocess(response)

        return {
            "analysis_type": analysis_type,
            "vision_analysis": parsed["text"],
            "confidence_score": parsed.get("confidence", 0.0),
            "structured_findings": parsed.get("structured", {}),
            "timestamp": time.time(),
            "processing_time": elapsed,
        }

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _coerce_image(self, img_in: Any) -> Image.Image:
        """
        Accept PIL.Image, raw bytes, or path/URL and return a PIL.Image.
        """
        if img_in is None:
            raise ValueError("Payload is missing required 'image' field")

        if isinstance(img_in, Image.Image):
            img = img_in
        elif isinstance(img_in, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img_in))
        elif isinstance(img_in, (str, Path)):
            img = Image.open(img_in)
        else:
            raise ValueError("Unsupported image input type")

        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _preprocess(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_w, max_h = self.max_size
        if w > max_w or h > max_h:
            img = ImageOps.contain(img, self.max_size, Image.LANCZOS)
            self.log.debug(f"Image resized to {img.size}")
        # Additional preprocessing hooks (contrast, normalisation, etc.)
        if self.preprocessing_cfg.get("normalize"):
            img = ImageOps.autocontrast(img)
        if self.preprocessing_cfg.get("enhance_contrast"):
            img = ImageOps.equalize(img)
        return img

    @staticmethod
    def _build_prompt(
        analysis_type: str,
        context: str = "",
        reference_report: str = "",
    ) -> str:
        """
        Compose an instruction prompt for MedGemma based on the requested analysis.
        """
        if analysis_type == "detailed":
            instruction = (
                "Provide a comprehensive, radiology-style description of the chest "
                "X-ray, covering lungs, pleura, heart, mediastinum and bones."
            )
        elif analysis_type == "findings":
            instruction = (
                "List all abnormal findings in bullet form. "
                "Each finding should include location, severity and uncertainty."
            )
        elif analysis_type == "comparison":
            instruction = (
                "Compare the image with the reference report and highlight any new, "
                "resolved or unchanged findings."
            )
        else:  # quality
            instruction = (
                "Evaluate technical image quality (positioning, exposure, rotation, "
                "artifacts) and comment on diagnostic adequacy."
            )

        prompt_parts = [instruction]
        if context:
            prompt_parts.append(f"Patient context: {context}")
        if reference_report and analysis_type == "comparison":
            prompt_parts.append(f"Reference report:\n{reference_report}")
        return "\n".join(prompt_parts)

    def _postprocess(self, raw_text: str) -> Dict[str, Any]:
        """
        Simple post-processing to extract confidence and (optionally) structured
        findings from the model output. Extend this as needed.
        """
        confidence = 0.0
        structured: Dict[str, Any] = {}

        # Extract confidence if the model appended something like "Confidence: 0.82"
        for line in reversed(raw_text.splitlines()):
            if line.lower().startswith("confidence"):
                try:
                    confidence = float(line.split(":")[1].strip())
                    raw_text = "\n".join(raw_text.splitlines()[:-1])
                except Exception:  # noqa: BLE001
                    pass
                break

        # Basic heuristic: return findings as list if bullet points present
        if "-" in raw_text or "•" in raw_text:
            bullets = [
                l.lstrip("•- ").strip()
                for l in raw_text.splitlines()
                if l.lstrip().startswith(("-", "•"))
            ]
            structured["findings"] = bullets

        # Filter by confidence threshold
        if confidence and confidence < self.conf_threshold:
            self.log.warning(
                f"Analysis confidence {confidence:.2f} below threshold "
                f"{self.conf_threshold:.2f}"
            )

        return {"text": raw_text.strip(), "confidence": confidence, "structured": structured}


# --------------------------------------------------------------------------- #
# Stand-alone quick test (optional)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json
    from medgemma_multiagent.models.medgemma_wrapper import MedGemmaWrapper

    parser = argparse.ArgumentParser(description="VisionAgent standalone demo")
    parser.add_argument("image_path", help="Path to chest X-ray image")
    parser.add_argument(
        "--analysis_type",
        default="detailed",
        choices=VisionAgent._VALID_ANALYSES,
    )
    parser.add_argument("--cfg", default="config.yaml")
    args = parser.parse_args()

    async def _main() -> None:
        model = MedGemmaWrapper(config_path=args.cfg)
        await model.initialize()

        agent_cfg: Dict[str, Any] = {
            "name": "VisionAgent",
            "description": "Standalone Vision agent demo",
            "config": {},  # use defaults
        }
        agent = VisionAgent(agent_cfg, model)
        await agent.initialize()

        result = await agent.process(
            {"image": args.image_path, "analysis_type": args.analysis_type}
        )
        print(json.dumps(result, indent=2))

        await agent.shutdown()
        await model.cleanup()

    asyncio.run(_main())
