"""Unified Local Model Adapter and Gateway with VRAM-safe hot-swapping."""

import gc
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Prediction:
    label: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": self.score}


class ModelGateway:
    _instance: Optional["ModelGateway"] = None

    def __init__(self):
        self.active_model_id: Optional[str] = None
        self.pipeline_tag: Optional[str] = None
        self.pipeline: Any = None
        self.target_device: str = "cpu"

    @classmethod
    def get_instance(cls) -> "ModelGateway":
        if cls._instance is None:
            cls._instance = ModelGateway()
        return cls._instance

    def load_model(
        self,
        model_id: str,
        pipeline_tag: str = "image-classification",
        target_device: str = "auto"
    ) -> Dict[str, Any]:
        """Loads a model into memory or hot-swaps an existing one, purging previous VRAM."""
        # 1. Unload old model and free GPU/system memory
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # 2. Determine target device
        resolved_device = target_device
        if resolved_device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    resolved_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    resolved_device = "mps"
                else:
                    resolved_device = "cpu"
            except ImportError:
                resolved_device = "cpu"

        # 3. Load HuggingFace pipeline
        from transformers import pipeline
        self.pipeline = pipeline(
            task=pipeline_tag,
            model=model_id,
            device=resolved_device
        )
        self.active_model_id = model_id
        self.pipeline_tag = pipeline_tag
        self.target_device = resolved_device

        return {
            "status": "loaded",
            "model_id": self.active_model_id,
            "device": self.target_device,
            "task": self.pipeline_tag
        }

    def predict(self, input_data: Any) -> List[Prediction]:
        """Abstract prediction method providing normalized output across models."""
        if self.pipeline is None:
            raise RuntimeError("No model is currently loaded. Call load_model() first.")

        raw_results = self.pipeline(input_data)

        # Normalize outputs across different vision and text models
        normalized: List[Prediction] = []
        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict) and "label" in item and "score" in item:
                    normalized.append(Prediction(label=item["label"], score=round(item["score"], 4)))
                elif isinstance(item, dict) and "generated_text" in item:
                    normalized.append(Prediction(label=item["generated_text"], score=1.0))
        elif isinstance(raw_results, dict) and "label" in raw_results:
            normalized.append(Prediction(label=raw_results["label"], score=round(raw_results.get("score", 1.0), 4)))

        return normalized


# Singleton instance for in-process usage
gateway = ModelGateway.get_instance()
