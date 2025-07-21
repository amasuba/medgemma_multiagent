import os
import io
import pytest
import asyncio
from PIL import Image
from pathlib import Path
from unittest.mock import AsyncMock, patch

from medgemma_multiagent.models.medgemma_wrapper import MedGemmaWrapper
from medgemma_multiagent.utils.model_config import ModelConfig, QuantizationConfig, GenerationConfig

@pytest.fixture(autouse=True)
def env_token(monkeypatch):
    # Ensure no real HF API token is used
    monkeypatch.delenv("HUGGINGFACE_API_TOKEN", raising=False)
    yield

def make_dummy_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "dummy.jpg"
    img = Image.new("RGB", (64, 64), color="gray")
    img.save(img_path)
    return img_path

class DummyModel:
    def __init__(self):
        self.generated = ["Hello world"]
    def generate(self, **kwargs):
        # Simulate generate API
        class Out:
            def __init__(self, text):
                self.text = text
        return ["Hello world"]

@pytest.mark.asyncio
async def test_local_initialization_and_generate(tmp_path, caplog):
    # Create minimal config
    cfg = {
        "model_name": "test-model",
        "device": "cpu",
        "cache_dir": str(tmp_path / "cache"),
        "quantization": {"enabled": False},
        "generation": {"max_length": 10, "temperature": 0.5},
        "use_hf_api": False,
    }
    wrapper = MedGemmaWrapper(cfg)
    # Patch AutoProcessor and AutoModelForCausalLM
    with patch("medgemma_multiagent.models.medgemma_wrapper.AutoProcessor.from_pretrained") as mock_proc, \
         patch("medgemma_multiagent.models.medgemma_wrapper.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_proc.return_value = AsyncMock(return_value=None)
        mock_model.return_value = AsyncMock(return_value=None)
        await wrapper.initialize()
    assert wrapper.initialized is True

    # Prepare dummy image and prompt
    img_path = make_dummy_image(tmp_path)
    # Mock processor and model generate
    wrapper.processor = AsyncMock()
    wrapper.processor.decode = lambda x, skip_special_tokens: "hi"
    wrapper.model = AsyncMock()
    wrapper.model.generate = AsyncMock(return_value=[[0,1,2]])
    # Monkeypatch processor decode
    with patch.object(wrapper.processor, "decode", return_value="generated text"):
        result = await wrapper.generate(image=img_path, prompt="Test", max_length=5)
    assert isinstance(result, str)
    assert "generated text" in result

@pytest.mark.asyncio
async def test_api_mode_generate(monkeypatch, tmp_path):
    # Simulate environment HF token and API mode
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "fake-token")
    cfg = {
        "model_name": "test-model",
        "device": "cpu",
        "cache_dir": str(tmp_path / "cache"),
        "quantization": {"enabled": False},
        "generation": {"max_length": 5, "temperature": 0.5},
        "use_hf_api": True,
    }
    wrapper = MedGemmaWrapper(cfg)
    # Patch login and InferenceClient
    with patch("medgemma_multiagent.models.medgemma_wrapper.login") as mock_login, \
         patch("medgemma_multiagent.models.medgemma_wrapper.InferenceClient") as mock_client_cls:
        mock_login.return_value = None
        client = mock_client_cls.return_value
        client.text_generation.return_value = type("R", (), {"generated_text": "api output"})
        await wrapper.initialize()
        out = await wrapper.generate(image=make_dummy_image(tmp_path), prompt="Hello")
    assert out == "api output"

def test_model_config_validation(tmp_path):
    # Valid config
    mc = ModelConfig(
        model_name="a-model",
        device="cpu",
        cache_dir=str(tmp_path / "cache"),
        use_hf_api=False,
        api_token=None
    )
    assert mc.model_name == "a-model"
    # Invalid device should raise
    with pytest.raises(ValueError):
        ModelConfig(
            model_name="m",
            device="gpu",
            cache_dir=str(tmp_path / "cache"),
            use_hf_api=False,
            api_token=None
        )

def test_quant_and_gen_config_defaults():
    qc = QuantizationConfig()
    gc = GenerationConfig()
    # Defaults
    assert qc.enabled is False and qc.bits in {4,8} or isinstance(qc.bits, int)
    assert gc.max_length == 2048
    assert 0.0 <= gc.temperature <= 1.0

@pytest.mark.asyncio
async def test_generate_with_bytes_input(tmp_path):
    # Test passing raw bytes
    cfg = {
        "model_name": "test-model",
        "device": "cpu",
        "cache_dir": str(tmp_path / "cache"),
        "quantization": {},
        "generation": {"max_length": 5},
        "use_hf_api": False,
    }
    wrapper = MedGemmaWrapper(cfg)
    # Initialize with minimal mocks
    wrapper.initialized = True
    wrapper.processor = AsyncMock()
    wrapper.processor.decode = lambda x, skip_special_tokens: "decoded"
    wrapper.model = AsyncMock()
    wrapper.model.generate = AsyncMock(return_value=[[0]])
    # Create raw bytes image
    img = Image.new("RGB", (32,32), "white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()
    out = await wrapper.generate(image=data, prompt="X")
    assert out == "decoded"
