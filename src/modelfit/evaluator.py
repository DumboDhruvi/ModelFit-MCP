"""Empirical Model Benchmarking and Accuracy Evaluator."""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from modelfit.adapter import ModelGateway, Prediction


@dataclass
class BenchmarkResult:
    model_id: str
    avg_latency_ms: float
    avg_confidence: float
    composite_score: float
    samples_evaluated: int
    predictions: List[Dict[str, Any]]
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    def __init__(self, gateway: Optional[ModelGateway] = None):
        self.gateway = gateway or ModelGateway.get_instance()

    def benchmark_models(
        self,
        model_ids: List[str],
        test_samples: List[Any],
        pipeline_tag: str = "image-classification",
        target_device: str = "auto"
    ) -> List[BenchmarkResult]:
        """
        Runs a live empirical benchmark across multiple candidate models that fit hardware.
        Evaluates average latency (ms), confidence score, and composite score.
        """
        if not model_ids:
            return []

        if not test_samples:
            test_samples = ["test_sample_default"]

        results: List[BenchmarkResult] = []

        for model_id in model_ids:
            try:
                self.gateway.load_model(model_id, pipeline_tag=pipeline_tag, target_device=target_device)
            except Exception as err:
                results.append(BenchmarkResult(
                    model_id=model_id,
                    avg_latency_ms=99999.0,
                    avg_confidence=0.0,
                    composite_score=0.0,
                    samples_evaluated=0,
                    predictions=[{"error": str(err)}]
                ))
                continue

            latencies: List[float] = []
            confidences: List[float] = []
            sample_preds: List[Dict[str, Any]] = []

            for sample in test_samples:
                t0 = time.perf_counter()
                try:
                    preds: List[Prediction] = self.gateway.predict(sample)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(dt_ms)

                    top_score = preds[0].score if preds else 0.0
                    confidences.append(top_score)
                    sample_preds.append({
                        "sample": str(sample),
                        "top_label": preds[0].label if preds else "unknown",
                        "top_score": top_score,
                        "latency_ms": round(dt_ms, 2)
                    })
                except Exception as e:
                    sample_preds.append({"sample": str(sample), "error": str(e)})

            avg_lat = sum(latencies) / len(latencies) if latencies else 99999.0
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            # Composite score: confidence penalized gracefully by excessive latency (>1000ms)
            latency_penalty = min(0.5, (avg_lat / 2000.0))
            composite = round(avg_conf * (1.0 - latency_penalty), 4)

            results.append(BenchmarkResult(
                model_id=model_id,
                avg_latency_ms=round(avg_lat, 2),
                avg_confidence=round(avg_conf, 4),
                composite_score=composite,
                samples_evaluated=len(latencies),
                predictions=sample_preds
            ))

        # Sort descending by composite score
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for i, res in enumerate(results, start=1):
            res.rank = i

        return results
