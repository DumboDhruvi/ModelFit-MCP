"""ModelFit-MCP: Hardware-Aware Hugging Face Model Gateway for AI Agents."""

from modelfit.hardware import SystemSpecs, detect_system_specs, estimate_required_memory_gb, can_model_run
from modelfit.hf_client import HFHardwareClient
from modelfit.adapter import ModelGateway, Prediction, gateway
from modelfit.code_gen import generate_transformers_snippet, generate_ensemble_snippet, generate_client_snippet
from modelfit.evaluator import ModelEvaluator, BenchmarkResult
from modelfit.ensemble import EnsembleGateway, EnsemblePrediction

__all__ = [
    "SystemSpecs",
    "detect_system_specs",
    "estimate_required_memory_gb",
    "can_model_run",
    "HFHardwareClient",
    "ModelGateway",
    "Prediction",
    "gateway",
    "generate_transformers_snippet",
    "generate_ensemble_snippet",
    "generate_client_snippet",
    "ModelEvaluator",
    "BenchmarkResult",
    "EnsembleGateway",
    "EnsemblePrediction",
]
