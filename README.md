# MedGemma Multi-AI Agentic System

**MedGemma Multi-AI Agentic System** is a modular, multi-agent framework for automated chest X-ray report generation using Google’s MedGemma multimodal model. It mirrors clinical reasoning by coordinating specialized agents for retrieval, vision analysis, drafting, refinement, and synthesis.

## Key Features

- **Multi-Agent Architecture**  
  Five cooperating agents—Retrieval, Vision, Draft, Refiner, Synthesis—each focused on a distinct task for structured, interpretable reports.

- **MedGemma Integration**  
  Supports both local and Hugging Face Inference API modes with optional 4-bit quantization and GPU acceleration.

- **Knowledge Base Retrieval**  
  Semantic search over similar prior reports via ChromaDB or FAISS back-ends.

- **Configurable Drafting & Refinement**  
  Template-based draft generation and hybrid LLM/regex extraction of findings for clinical accuracy.

- **Comprehensive Evaluation**  
  Automatic metrics: RadGraph F1, BLEU, ROUGE, BERTScore, and GEMA-Score for explainable multi-agent assessment.

- **Extensible & Production-Ready**  
  FastAPI CLI & REST server, Docker support, structured logging, performance monitoring, and YAML/ENV configuration.

## Project Structure

```
medgemma_multiagent/
├── agents/
│   ├── base_agent.py
│   ├── retrieval_agent.py
│   ├── vision_agent.py
│   ├── draft_agent.py
│   ├── refiner_agent.py
│   └── synthesis_agent.py
├── models/
│   ├── medgemma_wrapper.py
│   └── model_config.py
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── message_queue.py
│   ├── data_loader.py
│   └── evaluation.py
├── notebooks/
│   ├── Quick_Start.ipynb
│   └── Advanced_Usage.ipynb
├── tests/
│   ├── test_agents.py
│   ├── test_model_wrapper.py
│   └── test_integration.py
├── data/
│   ├── images/
│   ├── reports/
│   └── processed/
├── config.yaml
├── requirements.txt
├── setup.py
├── main.py
└── README.md
```

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/medgemma-multiagent.git
   cd medgemma-multiagent
   ```

2. Create and activate a Python 3.10+ virtual environment  
   ```bash
   python3 -m venv menv
   source menv/bin/activate        # macOS/Linux
   menv\Scripts\activate.bat       # Windows
   ```

3. Install dependencies  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Configure environment  
   ```bash
   cp .env.example .env
   # Edit .env to set:
   # HUGGINGFACE_API_TOKEN=your_token_here
   ```

## Usage

### CLI Commands

- **Generate report**  
  ```bash
  python main.py generate \
    path/to/chest_xray.jpg \
    --patient-context "45-year-old male with cough" \
    --report-type detailed \
    --output-file result.json
  ```

- **Batch processing**  
  ```bash
  python main.py batch \
    ./data/images \
    --pattern "*.png" \
    --output-dir ./results
  ```

- **Evaluate system**  
  ```bash
  python main.py evaluate \
    ./data/test_data.json \
    --output-file evaluation_results.json
  ```

- **Serve API**  
  ```bash
  python main.py serve \
    --host 0.0.0.0 \
    --port 8000
  ```

### Python API

```python
from medgemma_multiagent.main import MedGemmaMultiAgent

# Initialize system
system = MedGemmaMultiAgent(config_path="config.yaml")
await system.initialize()

# Generate a report
result = await system.generate_report(
    image_path="chest_xray.jpg",
    patient_context="50-year-old female, chest pain",
    report_type="findings"
)
print(result["final_report"])

# Shutdown
await system.shutdown()
```

## Configuration

All settings reside in `config.yaml` (with overrides via `.env`):

- **models.medgemma**: model name, quantization, generation params  
- **agents.\***: timeouts, retry policies, agent-specific configs  
- **retrieval**: vector DB, top_k, thresholds  
- **evaluation**: metrics, batch sizes, output directory  
- **api**: host, port, CORS, documentation  
- **security**: JWT auth, rate limiting  

## Jupyter Notebooks

- **Quick_Start.ipynb**: Step-by-step demo from image upload to report.  
- **Advanced_Usage.ipynb**: Customizing agents, evaluation metrics, and batch workflows.

Launch notebooks:

```bash
jupyter notebook notebooks/Quick_Start.ipynb
```

## Testing

Run unit and integration tests:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Docker

Build and run:

```bash
docker build -t medgemma-multiagent .
docker run --gpus all -p 8000:8000 \
  -e HUGGINGFACE_API_TOKEN="${HUGGINGFACE_API_TOKEN}" \
  medgemma-multiagent
```

## Contributing

1. Fork and create a feature branch  
2. Run tests locally  
3. Submit a pull request  

## License

MIT License — see [LICENSE](LICENSE) for details.

