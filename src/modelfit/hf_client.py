"""Hugging Face Hub querying and hardware-aware filtering."""

import json
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Optional
from modelfit.hardware import SystemSpecs, can_model_run


class HFHardwareClient:
    BASE_URL = "https://huggingface.co/api/models"

    def __init__(self, token: Optional[str] = None):
        self.token = token

    def search_models(
        self,
        query: str,
        pipeline_tag: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Queries Hugging Face Hub REST API for candidate models."""
        params = {
            "search": query,
            "sort": "downloads",
            "direction": "-1",
            "limit": limit,
        }
        if pipeline_tag:
            params["pipeline_tag"] = pipeline_tag

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "ModelFit-MCP/0.1.1"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except Exception:
            return []

    def filter_and_rank_models(
        self,
        models: List[Dict[str, Any]],
        specs: SystemSpecs,
        preferred_precision: str = "fp16",
    ) -> List[Dict[str, Any]]:
        """Filters models that fit the host hardware and ranks by downloads and likes."""
        compatible_models = []

        for m in models:
            model_id = m.get("id")
            downloads = m.get("downloads", 0)
            likes = m.get("likes", 0)
            pipeline_tag = m.get("pipeline_tag", "unknown")

            # Extract parameter count from tags (e.g., 'params:1.5B' or safetensors metadata)
            # Default to 0.5B for vision/classification tasks if not specified
            params_b = 0.5
            for tag in m.get("tags", []):
                if tag.startswith("params:"):
                    try:
                        val = tag.replace("params:", "").lower()
                        if "b" in val:
                            params_b = float(val.replace("b", ""))
                        elif "m" in val:
                            params_b = float(val.replace("m", "")) / 1000.0
                    except ValueError:
                        pass

            fit_analysis = can_model_run(params_b, specs, precision=preferred_precision)
            if fit_analysis["can_run"]:
                score = (downloads * 0.7) + (likes * 100 * 0.3)
                compatible_models.append({
                    "model_id": model_id,
                    "pipeline_tag": pipeline_tag,
                    "downloads": downloads,
                    "likes": likes,
                    "params_billions": params_b,
                    "target_device": fit_analysis["target"],
                    "memory_required_gb": fit_analysis["required_gb"],
                    "ranking_score": round(score, 1),
                })

        # Rank descending by score
        compatible_models.sort(key=lambda x: x["ranking_score"], reverse=True)
        return compatible_models
