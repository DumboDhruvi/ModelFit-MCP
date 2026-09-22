"""Unified Local Model Adapter and Gateway with Multi-Modal Support and VRAM-Safe Hot-Swapping."""

import gc
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Prediction:
    label: str
    score: float
    box: Optional[Dict[str, float]] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"label": self.label, "score": self.score}
        if self.box:
            d["box"] = self.box
        if self.extra:
            d["extra"] = self.extra
        return d


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

        # 3. Load HuggingFace pipeline with CUDA OOM protection fallback
        from transformers import pipeline
        try:
            self.pipeline = pipeline(
                task=pipeline_tag,
                model=model_id,
                device=resolved_device
            )
            self.target_device = resolved_device
        except Exception as err:
            # Fallback to CPU if GPU encountered an OOM during load
            err_msg = str(err).lower()
            if ("out of memory" in err_msg or "cuda" in err_msg) and resolved_device != "cpu":
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                self.pipeline = pipeline(
                    task=pipeline_tag,
                    model=model_id,
                    device="cpu"
                )
                self.target_device = "cpu"
            else:
                raise err

        self.active_model_id = model_id
        self.pipeline_tag = pipeline_tag

        return {
            "status": "loaded",
            "model_id": self.active_model_id,
            "device": self.target_device,
            "task": self.pipeline_tag
        }

    def predict(self, input_data: Any, **kwargs) -> List[Prediction]:
        """
        Abstract prediction method providing normalized outputs across:
        - image-classification
        - object-detection
        - zero-shot-image-classification
        - text-generation / summarization
        """
        if self.pipeline is None:
            raise RuntimeError("No model is currently loaded. Call load_model() first.")

        raw_results = self.pipeline(input_data, **kwargs)

        normalized: List[Prediction] = []

        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    # Classification format: {"label": "...", "score": 0.95}
                    if "label" in item and "score" in item:
                        box = item.get("box")
                        normalized.append(Prediction(
                            label=item["label"],
                            score=round(float(item["score"]), 4),
                            box=box
                        ))
                    # Text generation format: {"generated_text": "..."}
                    elif "generated_text" in item:
                        normalized.append(Prediction(
                            label=item["generated_text"],
                            score=1.0
                        ))
                    # Summarization: {"summary_text": "..."}
                    elif "summary_text" in item:
                        normalized.append(Prediction(
                            label=item["summary_text"],
                            score=1.0
                        ))
        elif isinstance(raw_results, dict):
            if "label" in raw_results:
                normalized.append(Prediction(
                    label=raw_results["label"],
                    score=round(float(raw_results.get("score", 1.0)), 4),
                    box=raw_results.get("box")
                ))
            elif "generated_text" in raw_results:
                normalized.append(Prediction(label=raw_results["generated_text"], score=1.0))

        return normalized


# Singleton instance for in-process usage
gateway = ModelGateway.get_instance()
