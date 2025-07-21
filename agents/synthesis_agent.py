"""
synthesis_agent.py
MedGemma Multi-AI Agentic System

Author: Aaron Masuba
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from loguru import logger

from .base_agent import BaseAgent, AgentNotInitializedError, AgentTimeoutError


class SynthesisAgent(BaseAgent):
    """
    Integrates outputs from RetrievalAgent, VisionAgent, DraftAgent, and RefinerAgent
    to produce the final, clinically-structured chest X-ray report.

    Payload contract
    ----------------
    • draft_report      : str    — text from DraftAgent
    • refined_findings  : Dict   — output of RefinerAgent
    • vision_analysis   : Dict   — output of VisionAgent
    • retrieved_context : Dict   — output of RetrievalAgent
    • patient_context   : str    — optional clinical context
    """

    def __init__(self, cfg: Dict[str, Any], model_wrapper: "MedGemmaWrapper"):
        super().__init__(cfg, global_cfg=cfg.get("global", {}))
        self.model = model_wrapper

        # Agent-level configuration
        self.multi_agent_integration = cfg["config"].get("multi_agent_integration", True)
        self.quality_assurance = cfg["config"].get("quality_assurance", True)
        self.report_formatting = cfg["config"].get("report_formatting", True)
        self.clinical_structure = cfg["config"].get("clinical_structure", True)
        self.final_validation = cfg["config"].get("final_validation", True)

    async def _initialize(self) -> None:
        if not self.model or not self.model.initialized:
            raise AgentNotInitializedError(
                "MedGemmaWrapper must be initialized before SynthesisAgent"
            )
        self.log.info("SynthesisAgent ready ✔")

    async def _shutdown(self) -> None:
        # Nothing to clean up; model wrapper is owned by orchestrator.
        self.log.debug("SynthesisAgent shutdown complete")

    async def _process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        draft = payload.get("draft_report", "")
        refined = payload.get("refined_findings", {})
        vision = payload.get("vision_analysis", {})
        retrieval = payload.get("retrieved_context", {})
        context = payload.get("patient_context", "")

        if not draft:
            raise ValueError("Payload must include 'draft_report'")

        t0 = time.perf_counter()

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(
            draft, refined, vision, retrieval, context
        )

        # Call model to synthesize final report
        try:
            synthesis_out = await asyncio.wait_for(
                self.model.generate(
                    prompt=prompt,
                    max_length=self.model.cfg.generation.max_length,
                    temperature=self.model.cfg.generation.temperature,
                    do_sample=False,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            self.log.error("SynthesisAgent timed-out during model inference")
            raise AgentTimeoutError from exc

        elapsed = time.perf_counter() - t0
        self.log.info(f"Synthesis completed in {elapsed:.2f}s")

        # Post-process final report
        final_text, confidence = self._postprocess_synthesis(synthesis_out)

        return {
            "final_report": final_text.strip(),
            "confidence_score": confidence,
            "timestamp": time.time(),
            "processing_time": elapsed,
        }

    def _build_synthesis_prompt(
        self,
        draft: str,
        refined: Dict[str, Any],
        vision: Dict[str, Any],
        retrieval: Dict[str, Any],
        context: str,
    ) -> str:
        parts = ["You are an expert radiologist synthesizing a final chest X-ray report."]
        if context:
            parts.append(f"Patient context: {context}")
        parts.append("### Draft Report ###")
        parts.append(draft)
        parts.append("### Extracted Findings ###")
        parts.append(refined.get("findings_summary", ""))
        parts.append("### Visual Observations ###")
        parts.append(vision.get("vision_analysis", ""))
        if retrieval.get("matched_reports"):
            parts.append("### Reference Cases ###")
            for r in retrieval["matched_reports"][:3]:
                snippet = r.get("content", "")[:200] + "..."
                parts.append(f"- (sim {r.get('similarity', 0.0):.2f}) {snippet}")
        parts.append(
            "Integrate all information into a single, structured radiology report. "
            "Ensure clarity, clinical accuracy, and standard report formatting."
        )
        return "\n\n".join(parts)

    def _postprocess_synthesis(self, raw_text: str) -> tuple[str, float]:
        """
        Extract optional confidence and clean the synthesized report.
        """
        confidence = 0.0
        lines = raw_text.splitlines()
        cleaned_lines = []
        for line in lines:
            if line.lower().startswith("confidence"):
                try:
                    confidence = float(line.split(":")[1].strip())
                    continue
                except Exception:
                    pass
            cleaned_lines.append(line)
        final_text = "\n".join(cleaned_lines)
        return final_text, confidence
