"""
draft_agent.py
MedGemma Multi-AI Agentic System

Author: Aaron Masuba
License: MIT
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from .base_agent import BaseAgent, AgentNotInitializedError, AgentTimeoutError


class DraftAgent(BaseAgent):
    """
    Generates initial chest X-ray report drafts using MedGemma.

    Payload contract
    ----------------
    • image              : PIL.Image | bytes | str (path/URL)
    • vision_analysis    : str   — visual observations from VisionAgent
    • retrieved_reports  : List[Dict] — similar reports from RetrievalAgent
    • patient_context    : str   — optional clinical context
    • draft_type         : str   — "structured" or "narrative" (default: "structured")
    """

    def __init__(self, cfg: Dict[str, Any], model_wrapper: "MedGemmaWrapper"):
        super().__init__(cfg, global_cfg=cfg.get("global", {}))
        self.model = model_wrapper

        # Agent-level configuration
        self.template_based = cfg["config"].get("template_based", True)
        self.use_retrieved_context = cfg["config"].get("use_retrieved_context", True)
        self.max_draft_length = cfg["config"].get("max_draft_length", 1000)
        self.include_confidence = cfg["config"].get("include_confidence", True)
        self.structured_output = cfg["config"].get("structured_output", True)

        # Draft templates
        self.templates = {
            "structured": self._get_structured_template(),
            "narrative": self._get_narrative_template()
        }

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def _initialize(self) -> None:
        if not self.model or not self.model.initialized:
            raise AgentNotInitializedError(
                "MedGemmaWrapper must be initialized before DraftAgent"
            )
        self.log.info("DraftAgent ready ✔")

    async def _shutdown(self) -> None:
        # Nothing to clean up; model wrapper is owned by orchestrator.
        self.log.debug("DraftAgent shutdown complete")

    # ------------------------------------------------------------------ #
    # Core logic
    # ------------------------------------------------------------------ #

    async def _process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        vision_analysis = payload.get("vision_analysis", "")
        retrieved_reports = payload.get("retrieved_reports", {}).get("matched_reports", [])
        patient_context = payload.get("patient_context", "")
        draft_type = payload.get("draft_type", "structured")

        if not vision_analysis:
            raise ValueError("Payload must include 'vision_analysis'")

        t0 = time.perf_counter()

        # Build context from retrieved reports
        context = self._build_context(retrieved_reports, patient_context)

        # Construct prompt for draft generation
        prompt = self._build_draft_prompt(
            vision_analysis=vision_analysis,
            context=context,
            draft_type=draft_type
        )

        self.log.debug(f"Draft prompt length: {len(prompt.split())} tokens")

        # Generate draft using MedGemma
        try:
            response = await asyncio.wait_for(
                self.model.generate(
                    prompt=prompt,
                    max_length=min(self.max_draft_length, self.model.cfg.generation.max_length),
                    temperature=self.model.cfg.generation.temperature,
                    do_sample=True,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            self.log.error("DraftAgent timed-out during model inference")
            raise AgentTimeoutError from exc

        elapsed = time.perf_counter() - t0
        self.log.info(f"Draft generation completed in {elapsed:.2f}s")

        # Post-process the draft
        processed_draft = self._postprocess_draft(response, draft_type)

        return {
            "draft_type": draft_type,
            "draft_report": processed_draft["text"],
            "confidence_score": processed_draft.get("confidence", 0.0),
            "structured_sections": processed_draft.get("sections", {}),
            "context_sources": len(retrieved_reports),
            "timestamp": time.time(),
            "processing_time": elapsed,
        }

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _build_context(self, retrieved_reports: List[Dict], patient_context: str) -> str:
        """Build context from retrieved reports and patient information."""
        context_parts = []

        if patient_context:
            context_parts.append(f"Patient Context: {patient_context}")

        if self.use_retrieved_context and retrieved_reports:
            context_parts.append("Similar Cases for Reference:")
            for i, report in enumerate(retrieved_reports[:3], 1):  # Top 3 only
                similarity = report.get("similarity", 0.0)
                content = report.get("content", "")[:200] + "..." if len(report.get("content", "")) > 200 else report.get("content", "")
                context_parts.append(f"{i}. (Similarity: {similarity:.2f}) {content}")

        return "\n\n".join(context_parts)

    def _build_draft_prompt(self, vision_analysis: str, context: str, draft_type: str) -> str:
        """Construct the prompt for draft generation."""
        template = self.templates.get(draft_type, self.templates["structured"])

        prompt_parts = [
            "You are an expert radiologist generating a chest X-ray report.",
            f"Template to follow:\n{template}",
            f"Visual Analysis:\n{vision_analysis}",
        ]

        if context:
            prompt_parts.append(f"Context:\n{context}")

        prompt_parts.append(
            "Generate a complete radiology report following the template structure. "
            "Be precise, concise, and clinically relevant."
        )

        return "\n\n".join(prompt_parts)

    def _get_structured_template(self) -> str:
        """Get the structured report template."""
        return """CHEST X-RAY REPORT

TECHNIQUE:
Single frontal chest radiograph

COMPARISON:
[Previous studies if available]

FINDINGS:
Lungs: [Describe lung fields, airways, pleura]
Heart: [Describe cardiac size, contour, mediastinum]
Bones: [Describe visible skeletal structures]
Soft Tissues: [Describe chest wall, diaphragm]

IMPRESSION:
[Summary of key findings and clinical significance]

RECOMMENDATIONS:
[Any follow-up recommendations if indicated]"""

    def _get_narrative_template(self) -> str:
        """Get the narrative report template."""
        return """CHEST X-RAY REPORT

The chest X-ray demonstrates [overall assessment]. The lungs show [lung findings]. 
The heart appears [cardiac findings]. The mediastinum is [mediastinal findings]. 
The visible skeletal structures show [bone findings]. The soft tissues appear [soft tissue findings].

IMPRESSION: [Clinical summary and recommendations]"""

    def _postprocess_draft(self, raw_text: str, draft_type: str) -> Dict[str, Any]:
        """Post-process the generated draft."""
        confidence = 0.0
        sections = {}

        # Extract confidence if present
        lines = raw_text.splitlines()
        for i, line in enumerate(lines):
            if line.lower().startswith("confidence"):
                try:
                    confidence = float(line.split(":")[1].strip())
                    raw_text = "\n".join(lines[:i] + lines[i+1:])
                except Exception:
                    pass
                break

        # Parse structured sections if structured format
        if draft_type == "structured" and self.structured_output:
            sections = self._parse_structured_sections(raw_text)

        # Clean up the text
        cleaned_text = self._clean_draft_text(raw_text)

        return {
            "text": cleaned_text,
            "confidence": confidence,
            "sections": sections
        }

    def _parse_structured_sections(self, text: str) -> Dict[str, str]:
        """Parse structured sections from the draft text."""
        sections = {}
        current_section = None
        current_content = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header
            if line.upper().endswith(":") and len(line.split()) <= 3:
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()

                # Start new section
                current_section = line.upper().replace(":", "")
                current_content = []
            else:
                # Add to current section
                if current_section:
                    current_content.append(line)

        # Save final section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _clean_draft_text(self, text: str) -> str:
        """Clean and format the draft text."""
        # Remove excessive whitespace
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)

        # Rejoin with appropriate spacing
        cleaned = "\n\n".join(lines)

        # Ensure length limits
        if len(cleaned) > self.max_draft_length:
            cleaned = cleaned[:self.max_draft_length] + "..."
            self.log.warning(f"Draft truncated to {self.max_draft_length} characters")

        return cleaned

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def get_template(self, template_type: str) -> str:
        """Get a specific template."""
        return self.templates.get(template_type, self.templates["structured"])

    def validate_draft(self, draft: str) -> Dict[str, Any]:
        """Validate the generated draft."""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "word_count": len(draft.split()),
            "section_count": len(self._parse_structured_sections(draft))
        }

        # Check length
        if len(draft) < 100:
            validation_results["issues"].append("Draft too short")
            validation_results["is_valid"] = False

        # Check for required sections in structured format
        if self.structured_output:
            required_sections = ["FINDINGS", "IMPRESSION"]
            sections = self._parse_structured_sections(draft)
            for section in required_sections:
                if section not in sections:
                    validation_results["issues"].append(f"Missing section: {section}")
                    validation_results["is_valid"] = False

        return validation_results


# --------------------------------------------------------------------------- #
# Stand-alone quick test (optional)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json
    from medgemma_multiagent.models.medgemma_wrapper import MedGemmaWrapper

    parser = argparse.ArgumentParser(description="DraftAgent standalone demo")
    parser.add_argument("--vision_analysis", required=True, help="Vision analysis text")
    parser.add_argument("--draft_type", default="structured", choices=["structured", "narrative"])
    parser.add_argument("--cfg", default="config.yaml")
    args = parser.parse_args()

    async def _main() -> None:
        model = MedGemmaWrapper(config_path=args.cfg)
        await model.initialize()

        agent_cfg: Dict[str, Any] = {
            "name": "DraftAgent",
            "description": "Standalone Draft agent demo",
            "config": {},  # use defaults
        }
        agent = DraftAgent(agent_cfg, model)
        await agent.initialize()

        result = await agent.process({
            "vision_analysis": args.vision_analysis,
            "draft_type": args.draft_type
        })
        print(json.dumps(result, indent=2))

        await agent.shutdown()
        await model.cleanup()

    asyncio.run(_main())
