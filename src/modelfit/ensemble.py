"""High-Speed Ensemble Model Gateway and Aggregator."""

from typing import Any, Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
from modelfit.adapter import ModelGateway, Prediction
from modelfit.hardware import SystemSpecs, detect_system_specs
from modelfit.hf_client import HFHardwareClient


@dataclass
class EnsemblePrediction:
    label: str
    score: float
    votes: int
    participating_models: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnsembleGateway:
    def __init__(
        self,
        gateway: Optional[ModelGateway] = None,
        hf_client: Optional[HFHardwareClient] = None
    ):
        self.gateway = gateway or ModelGateway.get_instance()
        self.hf_client = hf_client or HFHardwareClient()

    def find_ensemble_candidates(
        self,
        query: str,
        pipeline_tag: str = "image-classification",
        max_models: int = 3,
        specs: Optional[SystemSpecs] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds lightweight, high-speed candidate models suitable for ensembling on host hardware.
        Filters for models that consume <= 35% of available RAM/VRAM so multiple models
        can be hot-swapped or executed sequentially with minimal overhead.
        """
        host_specs = specs or detect_system_specs()
        raw_models = self.hf_client.search_models(query=query, pipeline_tag=pipeline_tag)
        compatible = self.hf_client.filter_and_rank_models(raw_models, host_specs, preferred_precision="fp16")

        limit_gb = (host_specs.vram_available_gb or host_specs.ram_available_gb) * 0.35
        ensemble_pool = [
            m for m in compatible
            if m.get("memory_required_gb", 0.0) <= limit_gb
        ]

        return ensemble_pool[:max_models]

    def predict_ensemble(
        self,
        input_data: Any,
        model_ids: List[str],
        pipeline_tag: str = "image-classification",
        strategy: str = "weighted_average",
        target_device: str = "auto"
    ) -> List[EnsemblePrediction]:
        """
        Runs inference across multiple models and aggregates predictions using the selected strategy:
        - 'weighted_average': Combines normalized confidence scores across all models.
        - 'majority_vote': Highest frequency label wins.
        - 'top_confidence': Selects the single highest confidence prediction.
        """
        if not model_ids:
            raise ValueError("At least one model_id must be provided for ensemble.")

        all_model_preds: Dict[str, List[Prediction]] = {}

        for model_id in model_ids:
            try:
                self.gateway.load_model(model_id, pipeline_tag=pipeline_tag, target_device=target_device)
                preds = self.gateway.predict(input_data)
                all_model_preds[model_id] = preds
            except Exception:
                continue

        if not all_model_preds:
            return []

        if strategy == "top_confidence":
            best_label = ""
            best_score = -1.0
            best_model = ""
            for model_id, preds in all_model_preds.items():
                if preds and preds[0].score > best_score:
                    best_score = preds[0].score
                    best_label = preds[0].label
                    best_model = model_id
            return [EnsemblePrediction(
                label=best_label,
                score=round(best_score, 4),
                votes=1,
                participating_models=[best_model]
            )]

        elif strategy == "majority_vote":
            vote_counts: Dict[str, int] = defaultdict(int)
            vote_sources: Dict[str, List[str]] = defaultdict(list)
            total_scores: Dict[str, float] = defaultdict(float)

            for model_id, preds in all_model_preds.items():
                if preds:
                    top = preds[0]
                    vote_counts[top.label] += 1
                    vote_sources[top.label].append(model_id)
                    total_scores[top.label] += top.score

            ranked = sorted(vote_counts.items(), key=lambda item: item[1], reverse=True)
            return [
                EnsemblePrediction(
                    label=lbl,
                    score=round(total_scores[lbl] / count, 4),
                    votes=count,
                    participating_models=vote_sources[lbl]
                )
                for lbl, count in ranked
            ]

        else:  # "weighted_average" (default)
            label_scores: Dict[str, float] = defaultdict(float)
            label_counts: Dict[str, int] = defaultdict(int)
            label_sources: Dict[str, List[str]] = defaultdict(list)

            total_weight = float(len(all_model_preds))
            for model_id, preds in all_model_preds.items():
                for p in preds:
                    label_scores[p.label] += p.score
                    label_counts[p.label] += 1
                    if model_id not in label_sources[p.label]:
                        label_sources[p.label].append(model_id)

            ranked = sorted(label_scores.items(), key=lambda item: item[1] / total_weight, reverse=True)
            return [
                EnsemblePrediction(
                    label=lbl,
                    score=round(tot / total_weight, 4),
                    votes=label_counts[lbl],
                    participating_models=label_sources[lbl]
                )
                for lbl, tot in ranked
            ]
