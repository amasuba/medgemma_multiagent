"""
data_loader.py
MedGemma Multi-AI Agentic System

Efficient data processing and loading for chest X-ray images and reports.
Author: Aaron Masuba
License: MIT
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Union
import json

from PIL import Image
import pandas as pd
from loguru import logger

class DataLoader:
    """
    DataLoader handles loading images and report data for the MedGemma multi-agent system.
    Supports:
      - Image loading (JPEG, PNG, DICOM via PIL)
      - Report loading (JSON, CSV, plain text)
      - Batch loading and validation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        config keys:
          directories:
            base: str
            images: str
            reports: str
            processed: str
          image_processing:
            supported_formats: List[str]
        """
        self.config = config
        dirs = config.get("directories", {})
        self.base_dir = Path(dirs.get("base", "./data"))
        self.images_dir = Path(dirs.get("images", self.base_dir / "images"))
        self.reports_dir = Path(dirs.get("reports", self.base_dir / "reports"))
        self.processed_dir = Path(dirs.get("processed", self.base_dir / "processed"))

        # Supported image extensions (lowercase, without dot)
        self.supported_images = {
            ext.lower().lstrip(".")
            for ext in config.get("image_processing", {}).get("supported_formats", [])
        }

        # Ensure directories exist
        for d in (self.images_dir, self.reports_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.log = logger.bind(component="DataLoader")

    def load_image(self, path: Union[str, Path]) -> Image.Image:
        """
        Load a single image and return as PIL.Image in RGB mode.
        Raises FileNotFoundError or OSError on failure.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {p}")
        ext = p.suffix.lower().lstrip(".")
        if ext not in self.supported_images:
            raise ValueError(f"Unsupported image format: .{ext}")
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def load_report(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a single report. Supports JSON (.json), CSV (.csv), and plain text (.txt).
        Returns a dict with keys:
          - report_id: str
          - content: str
          - metadata: dict (if CSV or JSON include extra columns)
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Report file not found: {p}")
        suffix = p.suffix.lower()
        if suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            # Expect data to have 'report_id', 'content', and optional 'metadata'
            report = {
                "report_id": data.get("report_id", p.stem),
                "content": data.get("content", ""),
                "metadata": data.get("metadata", {}),
            }
        elif suffix == ".csv":
            df = pd.read_csv(p)
            reports: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                report = {
                    "report_id": str(row.get("report_id", "")) or p.stem,
                    "content": str(row.get("content", "")),
                    "metadata": {
                        k: v for k, v in row.items()
                        if k not in ("report_id", "content")
                    },
                }
                reports.append(report)
            # If CSV contains multiple, return list; else return single dict
            return {"batch": reports} if len(reports) > 1 else reports[0]
        elif suffix in (".txt", ".md"):
            text = p.read_text(encoding="utf-8")
            report = {"report_id": p.stem, "content": text, "metadata": {}}
        else:
            raise ValueError(f"Unsupported report format: {suffix}")
        return report

    def load_images_batch(self, pattern: str = "*") -> List[Path]:
        """
        List all image files in images_dir matching the glob pattern.
        Returns a list of file Paths.
        """
        paths = list(self.images_dir.rglob(pattern))
        valid = [p for p in paths if p.suffix.lower().lstrip(".") in self.supported_images]
        self.log.info(f"Found {len(valid)} images matching '{pattern}'")
        return valid

    def load_reports_batch(self, pattern: str = "*") -> List[Path]:
        """
        List all report files in reports_dir matching the glob pattern.
        Returns a list of file Paths.
        """
        paths = list(self.reports_dir.rglob(pattern))
        # Filter known extensions
        valid_exts = {".json", ".csv", ".txt", ".md"}
        valid = [p for p in paths if p.suffix.lower() in valid_exts]
        self.log.info(f"Found {len(valid)} reports matching '{pattern}'")
        return valid

    def load_test_data(self, test_data_path: str) -> List[Dict[str, Any]]:
        """
        Load structured test dataset for evaluation.
        Expects a JSON lines file where each line is:
          { "image_path": "...", "report": {report dict} }
        Returns a list of dicts.
        """
        p = Path(test_data_path)
        if not p.is_file():
            raise FileNotFoundError(f"Test data file not found: {p}")
        records: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    self.log.warning(f"Skipping invalid JSON line: {line[:50]}...")
        self.log.info(f"Loaded {len(records)} test records from {p.name}")
        return records

    def save_processed(self, item: Dict[str, Any], filename: str) -> None:
        """
        Save processed item (e.g., generated report or metrics) as JSON to processed_dir.
        """
        out_path = self.processed_dir / f"{filename}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)
        self.log.debug(f"Saved processed file: {out_path}")

