# Installation Guide

This document provides detailed installation instructions for setting up the MedGemma Multi-AI Agentic System in various environments: local development, GPU-enabled workstation, Docker container, and cloud deployment.

## Table of Contents

1. Prerequisites  
2. Local Development (CPU)  
3. GPU Workstation Setup  
4. Docker Deployment  
5. Cloud VM Deployment (e.g., AWS EC2, Azure VM)  
6. Post-Installation Steps  

## 1. Prerequisites

- Operating System: Linux (Ubuntu 20.04+), macOS (11+), or Windows 10+  
- Python 3.10 or higher  
- Git  
- 16 GB RAM minimum (32 GB+ recommended)  
- GPU with CUDA support (for GPU setup)  
- Hugging Face API token (export as `HUGGINGFACE_API_TOKEN`)  

## 2. Local Development (CPU)

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/medgemma-multiagent.git
   cd medgemma-multiagent
   ```

2. Create and activate Python virtual environment  
   ```bash
   python3 -m venv menv
   source menv/bin/activate       # macOS/Linux
   menv\Scripts\activate.bat      # Windows (PowerShell)
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
   # DEVICE=cpu
   ```

5. Verify installation  
   ```bash
   python main.py --help
   ```

## 3. GPU Workstation Setup

1. Ensure CUDA and NVIDIA drivers are installed (CUDA 11.8+).  

2. Clone repository and create virtual environment (as above).  

3. Install GPU-compatible packages  
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

4. Enable quantization support (optional)  
   ```bash
   # In .env or config.yaml:
   USE_QUANTIZATION=true
   ```

5. Configure environment  
   ```bash
   cp .env.example .env
   # Set:
   # HUGGINGFACE_API_TOKEN=your_token_here
   # DEVICE=cuda
   ```

6. Test GPU inference  
   ```bash
   python main.py generate path/to/chest_xray.jpg --report-type detailed
   ```

## 4. Docker Deployment

1. Build Docker image  
   ```bash
   docker build -t medgemma-multiagent:latest .
   ```

2. Run container (CPU)  
   ```bash
   docker run --rm \
     -e HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN} \
     -p 8000:8000 \
     medgemma-multiagent:latest \
     main.py serve
   ```

3. Run container (GPU)  
   ```bash
   docker run --rm --gpus all \
     -e HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN} \
     -p 8000:8000 \
     medgemma-multiagent:latest \
     main.py serve
   ```

4. Access API at `http://localhost:8000`

## 5. Cloud VM Deployment

### AWS EC2 (Ubuntu 22.04, GPU instance)

1. Launch EC2 instance (e.g., p3.2xlarge).  
2. SSH into instance and install Docker (optional) or follow GPU workstation setup.  
3. (With Docker) Pull and run image as above.  
4. Open port 8000 in security group.  

### Azure VM (Ubuntu + NVIDIA GPU)

1. Create VM with GPU SKU.  
2. Install NVIDIA drivers and CUDA.  
3. Follow GPU workstation or Docker steps.  

## 6. Post-Installation Steps

- Populate the vector database:  
  ```bash
  python main.py add-reports --reports-path ./data/reports/*.json
  ```
- Run sample notebook:  
  ```bash
  jupyter notebook notebooks/Quick_Start.ipynb
  ```
- Evaluate system on test set:  
  ```bash
  python main.py evaluate ./data/test_data.json
  ```
- Monitor logs in `./logs/` and performance metrics endpoint (if enabled).

*End of INSTALL.md*
