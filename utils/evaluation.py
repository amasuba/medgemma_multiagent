"""
evaluation.py
MedGemma Multi-AI Agentic System

Comprehensive evaluation framework supporting RadGraph F1, BLEU, ROUGE,
BERTScore and the explainable multi-agent GEMA-Score.
Author: Aaron Masuba
License: MIT
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from evaluate import load as load_metric
from radgraph import RadGraphEvaluator  # hypothetical RadGraph evaluator package
from loguru import logger

class EvaluationManager:
    """
    Manages evaluation of MedGemma report generation against reference reports.
    Supports:
      - RadGraph F1 (entities & relations)
      - BLEU, ROUGE, BERTScore
      - GEMA-Score: granular, explainable multi-agent evaluation
      - Aggregation of metrics with mean & std
    """

    def __init__(self, cfg: Dict[str, Any]):
        """
        cfg keys (from config.evaluation):
          metrics: List[str]                      # e.g., ["radgraph_f1","bleu","rouge","bertscore"]
          batch_size: int
          output_dir: str
          radgraph.model_path: str
          gema_score.weights: Dict[str,float]
        """
        self.cfg = cfg
        self.logger = logger.bind(component="EvaluationManager")
        self.metrics_cfg = cfg.get("metrics", [])
        self.batch_size = cfg.get("batch_size", 32)
        self.output_dir = Path(cfg.get("output_dir", "./evaluation_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metrics
        self._load_text_metrics()
        if "radgraph_f1" in self.metrics_cfg:
            self.radgraph = RadGraphEvaluator(model_path=cfg["radgraph"]["model_path"])
        if "gema_score" in cfg and cfg["gema_score"].get("enabled", False):
            self.gema_weights = cfg["gema_score"].get("weights", {})
            # GEMA uses a custom evaluator (you would implement explainable sub-metrics)
            from .gema import GemaEvaluator  # local module to implement GEMA-Score
            self.gema = GemaEvaluator(self.gema_weights)

    def _load_text_metrics(self):
        self.text_metrics = {}
        if "bleu" in self.metrics_cfg:
            self.text_metrics["bleu"] = load_metric("bleu")
        if "rouge" in self.metrics_cfg:
            self.text_metrics["rouge"] = load_metric("rouge")
        if "bertscore" in self.metrics_cfg:
            self.text_metrics["bertscore"] = load_metric("bertscore")

    def evaluate(self, system: Any, test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs evaluation over test_records:
          each record: {"image_path":..., "report": {"content":..., "metadata":...}}
        Returns a dict of metric -> {"mean":..., "std":...} or raw scores.
        """
        all_scores: Dict[str, List[float]] = {m: [] for m in self.metrics_cfg}
        start_time = time.time()

        for idx in range(0, len(test_records), self.batch_size):
            batch = test_records[idx : idx + self.batch_size]
            hyps = []
            refs = []
            for rec in batch:
                # generate report
                out = asyncio.get_event_loop().run_until_complete(
                    system.generate_report(rec["image_path"], patient_context=rec.get("metadata", {}).get("context", ""), report_type="detailed")
                )
                hyp = out["final_report"].split()
                ref = rec["report"]["content"].split()
                hyps.append(hyp)
                refs.append([ref])

                # RadGraph
                if "radgraph_f1" in self.metrics_cfg:
                    rg_score = self.radgraph.score_texts(out["final_report"], rec["report"]["content"])
                    all_scores["radgraph_f1"].append(rg_score)

                # GEMA-Score
                if hasattr(self, "gema"):
                    gema_score = self.gema.score(out, rec["report"]["content"])
                    all_scores["gema_score"].append(gema_score)

            # Text metrics
            for m, metric in self.text_metrics.items():
                res = metric.compute(predictions=hyps, references=refs)
                # BLEU returns 'bleu'; ROUGE returns dict of scores
                if m == "bleu":
                    all_scores["bleu"].append(res["bleu"])
                elif m == "rouge":
                    # average over rouge1, rouge2, rougeL F1
                    avg = np.mean([res[k].mid.fmeasure for k in ["rouge1", "rouge2", "rougeL"]])
                    all_scores["rouge"].append(avg)
                elif m == "bertscore":
                    all_scores["bertscore"].extend(res["f1"])

        # Aggregate
        aggregated: Dict[str, Any] = {}
        for m, scores in all_scores.items():
            if scores:
                arr = np.array(scores)
                aggregated[m] = {"mean": float(arr.mean()), "std": float(arr.std())}
            else:
                aggregated[m] = {"mean": None, "std": None}

        # Save results
        out_path = self.output_dir / f"evaluation_{int(start_time)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        self.logger.info(f"Saved evaluation results to {out_path}")

        return aggregated
