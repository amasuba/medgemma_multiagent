"""
test_integration.py
MedGemma Multi-AI Agentic System

Integration/end-to-end tests for the orchestrated multi-agent workflow,
from input image to final synthesized chest X-ray report.
"""

import pytest
import asyncio
from pathlib import Path

from medgemma_multiagent.main import MedGemmaMultiAgent

@pytest.mark.asyncio
async def test_full_generation_pipeline(tmp_path):
    """
    End-to-end: Given a sample chest X-ray image and simulated report data,
    verify that the system:
        - Initializes all agents and core modules
        - Orchestrates the agent workflow
        - Produces a final report with structured findings
        - Contains expected sections and response fields
    """
    # Fixture: path to a sample image (can use a test asset or synthetic if none exist)
    image_path = "./data/images/example_xray.jpg"
    if not Path(image_path).exists():
        # Create a blank grayscale image for offline test
        from PIL import Image
        img = Image.new("L", (512, 512), 128)
        img.save(image_path)

    patient_context = "Integration test: 60-year-old male, cough."
    report_type = "detailed"  # Use 'detailed' for maximal agent path

    # Initialize the orchestrator and run entire pipeline
    system = MedGemmaMultiAgent(config_path="config.yaml")
    await system.initialize()

    try:
        result = await system.generate_report(
            image_path=image_path,
            patient_context=patient_context,
            report_type=report_type,
        )

        # --- Assertions ---
        assert "final_report" in result
        assert isinstance(result["final_report"], str)
        assert "findings" in result
        assert isinstance(result["findings"], list)
        assert "confidence_score" in result
        assert 0.0 <= result["confidence_score"] <= 1.0 or result["confidence_score"] == 0

        # Environment check: ensure all agent outputs available
        for agent_key in ["retrieval", "vision", "draft", "refiner", "synthesis"]:
            assert agent_key in result["agent_outputs"]

        # Check for structured key sections in synthesized report (at least one)
        final_report = result["final_report"].lower()
        assert any(
            section in final_report for section in ("findings", "impression", "technique", "recommendations")
        )

    finally:
        await system.shutdown()


@pytest.mark.asyncio
async def test_batch_processing():
    """
    Integration: Verify that batch processing produces valid results for multiple images.
    """
    system = MedGemmaMultiAgent(config_path="config.yaml")
    await system.initialize()

    image_dir = Path("./data/images")
    image_dir.mkdir(parents=True, exist_ok=True)
    test_images = []
    for i in range(2):
        img_path = image_dir / f"test_img_{i}.jpg"
        if not img_path.exists():
            from PIL import Image
            img = Image.new("L", (256, 256), 128 + i * 10)
            img.save(img_path)
        test_images.append(str(img_path))

    try:
        results = await system.process_batch(test_images, patient_context="Test batch", report_type="findings")
        assert isinstance(results, list)
        assert len(results) == len(test_images)
        for res in results:
            assert "final_report" in res
    finally:
        await system.shutdown()


@pytest.mark.asyncio
async def test_knowledge_base_ingestion():
    """
    Integration: Test adding reports to the knowledge base and retrieval.
    """
    import json
    from medgemma_multiagent.data.sample_reports import sample_reports  # or load via DataLoader

    system = MedGemmaMultiAgent(config_path="config.yaml")
    await system.initialize()
    try:
        await system.add_reports(sample_reports[:3])
        # Now, issue a retrieval request
        result = await system.agents["retrieval"].process({"query_text": "pulmonary edema", "top_k": 2})
        assert "matched_reports" in result
        assert isinstance(result["matched_reports"], list)
    finally:
        await system.shutdown()