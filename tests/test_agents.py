"""
test_agents.py
MedGemma Multi-AI Agentic System

Comprehensive test suite for all agent implementations.
Author: Your Name
License: MIT
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from PIL import Image
import numpy as np

# Import all agents
from medgemma_multiagent.agents.base_agent import BaseAgent, AgentNotInitializedError, AgentTimeoutError
from medgemma_multiagent.agents.retrieval_agent import RetrievalAgent
from medgemma_multiagent.agents.vision_agent import VisionAgent
from medgemma_multiagent.agents.draft_agent import DraftAgent
from medgemma_multiagent.agents.refiner_agent import RefinerAgent
from medgemma_multiagent.agents.synthesis_agent import SynthesisAgent


class TestConcreteAgent(BaseAgent):
    """Test implementation of BaseAgent for testing abstract functionality."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_called = False
        self._process_called = False
        self._shutdown_called = False

    async def _initialize(self):
        self._init_called = True
        await asyncio.sleep(0.01)  # Simulate initialization delay

    async def _process(self, payload):
        self._process_called = True
        return {"result": "test_output", "payload": payload}

    async def _shutdown(self):
        self._shutdown_called = True


@pytest.fixture
def base_agent_config():
    """Basic configuration for agent testing."""
    return {
        "name": "TestAgent",
        "description": "Test agent for unit testing",
        "global": {
            "timeout": 30,
            "retry_attempts": 2,
            "log_level": "DEBUG"
        }
    }


@pytest.fixture
def mock_model_wrapper():
    """Mock MedGemmaWrapper for testing."""
    wrapper = Mock()
    wrapper.initialized = True
    wrapper.generate = AsyncMock(return_value="Mock generated text")
    wrapper.cfg = Mock()
    wrapper.cfg.generation = Mock()
    wrapper.cfg.generation.max_length = 2048
    wrapper.cfg.generation.temperature = 0.7
    return wrapper


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    # Create a simple 100x100 RGB image
    image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(image_array, mode='RGB')


@pytest.fixture
def sample_reports():
    """Sample reports for retrieval testing."""
    return [
        {
            "report_id": "R001",
            "content": "Normal chest X-ray. Lungs are clear bilaterally.",
            "metadata": {"age": 45, "gender": "male"}
        },
        {
            "report_id": "R002", 
            "content": "Right lower lobe pneumonia with pleural effusion.",
            "metadata": {"age": 67, "gender": "female"}
        },
        {
            "report_id": "R003",
            "content": "Mild cardiomegaly. No acute pulmonary findings.",
            "metadata": {"age": 52, "gender": "male"}
        }
    ]


class TestBaseAgent:
    """Test cases for BaseAgent abstract class."""

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, base_agent_config):
        """Test complete agent lifecycle: init -> process -> shutdown."""
        agent = TestConcreteAgent(base_agent_config)

        # Test initialization
        assert not agent._initialized
        await agent.initialize()
        assert agent._initialized
        assert agent._init_called

        # Test processing
        payload = {"test_data": "value"}
        result = await agent.process(payload)
        assert agent._process_called
        assert result["result"] == "test_output"
        assert result["payload"] == payload

        # Test shutdown
        await agent.shutdown()
        assert agent._shutdown_called
        assert agent.is_shutdown()

    @pytest.mark.asyncio
    async def test_agent_not_initialized_error(self, base_agent_config):
        """Test that using agent before initialization raises error."""
        agent = TestConcreteAgent(base_agent_config)

        with pytest.raises(AgentNotInitializedError):
            await agent.process({"test": "data"})

    @pytest.mark.asyncio
    async def test_double_initialization(self, base_agent_config):
        """Test that double initialization is handled gracefully."""
        agent = TestConcreteAgent(base_agent_config)

        await agent.initialize()
        first_init_time = agent._init_called

        # Second initialization should be skipped
        await agent.initialize()
        assert agent._init_called == first_init_time

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, base_agent_config):
        """Test retry mechanism on failures."""
        class FailingAgent(BaseAgent):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.attempt_count = 0

            async def _initialize(self):
                pass

            async def _process(self, payload):
                self.attempt_count += 1
                if self.attempt_count < 2:
                    raise ValueError("Simulated failure")
                return {"success": True, "attempts": self.attempt_count}

            async def _shutdown(self):
                pass

        agent = FailingAgent(base_agent_config)
        await agent.initialize()

        result = await agent.process({"test": "retry"})
        assert result["success"] is True
        assert result["attempts"] == 2


class TestRetrievalAgent:
    """Test cases for RetrievalAgent."""

    @pytest.fixture
    def retrieval_config(self):
        return {
            "name": "RetrievalAgent",
            "description": "Test retrieval agent",
            "config": {},
            "global": {"timeout": 30, "retry_attempts": 2}
        }

    @pytest.fixture
    def retrieval_settings(self):
        return {
            "top_k": 3,
            "similarity_threshold": 0.5,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "search_methods": ["chromadb"],
            "vector_database": {
                "persist_directory": "./test_vector_db",
                "collection_name": "test_collection"
            }
        }

    @pytest.mark.asyncio
    async def test_retrieval_agent_initialization(self, retrieval_config, retrieval_settings):
        """Test RetrievalAgent initialization."""
        agent = RetrievalAgent(retrieval_config, retrieval_settings)

        # Mock the embedding model and database to avoid actual downloads
        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('chromadb.PersistentClient') as mock_chroma:

            mock_embedder = Mock()
            mock_st.return_value = mock_embedder

            mock_client = Mock()
            mock_collection = Mock()
            mock_client.list_collections.return_value = []
            mock_client.create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            await agent.initialize()
            assert agent._initialized
            assert agent._embedder == mock_embedder
            assert agent._chroma_client == mock_client

    @pytest.mark.asyncio
    async def test_retrieval_process(self, retrieval_config, retrieval_settings, sample_reports):
        """Test retrieval processing with mock data."""
        agent = RetrievalAgent(retrieval_config, retrieval_settings)

        # Mock dependencies
        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('chromadb.PersistentClient') as mock_chroma:

            # Setup mocks
            mock_embedder = Mock()
            mock_embedder.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_st.return_value = mock_embedder

            mock_collection = Mock()
            mock_collection.query.return_value = {
                "ids": [["0", "1"]],
                "distances": [[0.1, 0.2]]
            }

            mock_client = Mock()
            mock_client.list_collections.return_value = []
            mock_client.create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            await agent.initialize()

            # Add sample reports
            agent._report_store = sample_reports

            # Test retrieval
            result = await agent.process({
                "query_text": "pneumonia findings",
                "top_k": 2
            })

            assert "matched_reports" in result
            assert "num_matches" in result
            assert result["top_k"] == 2


class TestVisionAgent:
    """Test cases for VisionAgent."""

    @pytest.fixture
    def vision_config(self):
        return {
            "name": "VisionAgent",
            "description": "Test vision agent",
            "config": {
                "analysis_types": ["detailed", "findings"],
                "confidence_threshold": 0.6,
                "max_image_size": [512, 512],
                "preprocessing": {"resize": True, "normalize": False}
            },
            "global": {"timeout": 30}
        }

    @pytest.mark.asyncio
    async def test_vision_agent_initialization(self, vision_config, mock_model_wrapper):
        """Test VisionAgent initialization."""
        agent = VisionAgent(vision_config, mock_model_wrapper)
        await agent.initialize()

        assert agent._initialized
        assert agent.model == mock_model_wrapper

    @pytest.mark.asyncio
    async def test_vision_analysis_detailed(self, vision_config, mock_model_wrapper, sample_image):
        """Test detailed vision analysis."""
        agent = VisionAgent(vision_config, mock_model_wrapper)
        await agent.initialize()

        # Mock model response
        mock_model_wrapper.generate.return_value = "Bilateral lung fields are clear. Heart size normal."

        result = await agent.process({
            "image": sample_image,
            "analysis_type": "detailed",
            "context": "45-year-old male"
        })

        assert result["analysis_type"] == "detailed"
        assert "vision_analysis" in result
        assert "confidence_score" in result
        assert "processing_time" in result
        mock_model_wrapper.generate.assert_called_once()


class TestDraftAgent:
    """Test cases for DraftAgent."""

    @pytest.fixture
    def draft_config(self):
        return {
            "name": "DraftAgent",
            "description": "Test draft agent",
            "config": {
                "template_based": True,
                "use_retrieved_context": True,
                "max_draft_length": 1000,
                "structured_output": True
            },
            "global": {"timeout": 30}
        }

    @pytest.mark.asyncio
    async def test_draft_generation(self, draft_config, mock_model_wrapper):
        """Test draft report generation."""
        agent = DraftAgent(draft_config, mock_model_wrapper)
        await agent.initialize()

        # Mock model response
        mock_model_wrapper.generate.return_value = """CHEST X-RAY REPORT

        FINDINGS:
        The lungs are clear bilaterally.

        IMPRESSION:
        Normal chest radiograph."""

        vision_analysis = "Clear lung fields, normal heart size"
        retrieved_reports = {
            "matched_reports": [
                {"content": "Normal chest X-ray", "similarity": 0.8}
            ]
        }

        result = await agent.process({
            "vision_analysis": vision_analysis,
            "retrieved_reports": retrieved_reports,
            "patient_context": "45-year-old male",
            "draft_type": "structured"
        })

        assert result["draft_type"] == "structured"
        assert "draft_report" in result
        assert "confidence_score" in result
        assert "structured_sections" in result


class TestRefinerAgent:
    """Test cases for RefinerAgent."""

    @pytest.fixture
    def refiner_config(self):
        return {
            "name": "RefinerAgent",
            "description": "Test refiner agent",
            "config": {
                "entity_extraction": True,
                "finding_prioritization": True,
                "consistency_checking": True,
                "output_format": "structured"
            },
            "global": {"timeout": 30}
        }

    @pytest.mark.asyncio
    async def test_findings_extraction(self, refiner_config, mock_model_wrapper):
        """Test extraction of structured findings."""
        agent = RefinerAgent(refiner_config, mock_model_wrapper)
        await agent.initialize()

        # Mock LLM response with JSON findings
        json_response = '[{"disease": "pneumonia", "location": "right lower lobe", "severity": "moderate", "uncertainty": "false"}]'
        mock_model_wrapper.generate.return_value = json_response

        draft_report = "Right lower lobe pneumonia with possible bilateral small effusions."

        result = await agent.process({
            "draft_report": draft_report,
            "return_mode": "both"
        })

        assert "structured_findings" in result
        assert len(result["structured_findings"]) >= 1
        assert "findings_summary" in result


class TestSynthesisAgent:
    """Test cases for SynthesisAgent."""

    @pytest.fixture
    def synthesis_config(self):
        return {
            "name": "SynthesisAgent",
            "description": "Test synthesis agent",
            "config": {
                "multi_agent_integration": True,
                "quality_assurance": True,
                "report_formatting": True,
                "clinical_structure": True
            },
            "global": {"timeout": 30}
        }

    @pytest.mark.asyncio
    async def test_report_synthesis(self, synthesis_config, mock_model_wrapper):
        """Test synthesis of final report from all agent outputs."""
        agent = SynthesisAgent(synthesis_config, mock_model_wrapper)
        await agent.initialize()

        # Mock synthesized report
        synthesized_report = """CHEST X-RAY REPORT

        CLINICAL HISTORY: 45-year-old male with cough

        FINDINGS:
        The lungs demonstrate clear bilateral lung fields. Heart size is normal.
        No pleural effusions or pneumothorax identified.

        IMPRESSION:
        Normal chest radiograph.

        Confidence: 0.92"""

        mock_model_wrapper.generate.return_value = synthesized_report

        # Input from all agents
        draft_report = "Initial draft of normal chest X-ray"
        refined_findings = {
            "structured_findings": [],
            "findings_summary": "No abnormal findings"
        }
        vision_analysis = {
            "vision_analysis": "Clear lung fields, normal heart"
        }
        retrieved_context = {
            "matched_reports": [{"content": "Normal chest", "similarity": 0.8}]
        }

        result = await agent.process({
            "draft_report": draft_report,
            "refined_findings": refined_findings,
            "vision_analysis": vision_analysis,
            "retrieved_context": retrieved_context,
            "patient_context": "45-year-old male with cough"
        })

        assert "final_report" in result
        assert "confidence_score" in result
        assert result["confidence_score"] == 0.92
        assert "CHEST X-RAY REPORT" in result["final_report"]


# Integration tests for agent interactions
class TestAgentIntegration:
    """Integration tests for agent interactions."""

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, sample_image, sample_reports):
        """Test complete multi-agent workflow integration."""
        # This would test the full pipeline from image to final report
        # Implementation would involve mocking all agents and testing their interaction
        pass

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test how errors propagate through the agent pipeline."""
        # Test error handling across the multi-agent system
        pass

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in multi-agent scenarios."""
        # Test that timeouts are properly handled across agents
        pass


# Performance tests
class TestAgentPerformance:
    """Performance tests for agents."""

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        # Test multiple agents processing simultaneously
        pass

    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage patterns."""
        # Monitor memory usage during agent operations
        pass

    @pytest.mark.asyncio 
    async def test_processing_speed(self):
        """Test processing speed benchmarks."""
        # Benchmark agent processing speeds
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
