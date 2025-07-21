#!/usr/bin/env python3
"""
Setup script for MedGemma Multi-Agent System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="medgemma-multiagent",
    version="1.0.0",
    description="Multi-AI Agentic System for MedGemma Chest X-Ray Report Generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/medgemma-multiagent",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="medgemma, multiagent, chest-xray, radiology, ai, healthcare, medical-ai",
    entry_points={
        "console_scripts": [
            "medgemma-multiagent=medgemma_multiagent.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "medgemma_multiagent": [
            "data/*.json",
            "data/*.yaml",
            "config/*.yaml",
            "notebooks/*.ipynb",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/medgemma-multiagent/issues",
        "Source": "https://github.com/yourusername/medgemma-multiagent",
        "Documentation": "https://medgemma-multiagent.readthedocs.io/",
    },
)
