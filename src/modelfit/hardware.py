"""Hardware detection and memory footprint estimation."""

import os
import platform
import shutil
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class SystemSpecs:
    os_name: str
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_available_gb: Optional[float] = None
    has_cuda: bool = False
    has_mps: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_system_specs() -> SystemSpecs:
    """Detects current machine specs (CPU, RAM, GPU, VRAM) safely and quickly."""
    os_name = platform.system()
    cpu_count = os.cpu_count() or 1

    # RAM detection
    if psutil:
        mem = psutil.virtual_memory()
        ram_total = round(mem.total / (1024**3), 2)
        ram_avail = round(mem.available / (1024**3), 2)
    else:
        ram_total = 8.0
        ram_avail = 4.0

    gpu_name = None
    vram_total = None
    vram_avail = None
    has_cuda = False
    has_mps = False

    # Check PyTorch / GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            vram_avail = round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_mps = True
            gpu_name = "Apple Silicon Unified Memory"
            vram_total = ram_total
            vram_avail = ram_avail
    except ImportError:
        # Fallback to nvidia-smi if torch is not installed
        if shutil.which("nvidia-smi"):
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                    encoding="utf-8",
                    timeout=2
                ).strip().split(",")
                if len(out) >= 3:
                    gpu_name = out[0].strip()
                    vram_total = round(float(out[1].strip()) / 1024, 2)
                    vram_avail = round(float(out[2].strip()) / 1024, 2)
                    has_cuda = True
            except Exception:
                pass

    return SystemSpecs(
        os_name=os_name,
        cpu_count=cpu_count,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        gpu_name=gpu_name,
        vram_total_gb=vram_total,
        vram_available_gb=vram_avail,
        has_cuda=has_cuda,
        has_mps=has_mps,
    )


def estimate_required_memory_gb(params_in_billions: float, precision: str = "fp16") -> float:
    """
    Estimates required memory in GB with a 25% safety margin for runtime activations and context.
    """
    bytes_per_param = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }.get(precision.lower(), 2.0)

    # Base weight memory in GB
    weight_gb = (params_in_billions * 1e9 * bytes_per_param) / (1024**3)
    # 25% overhead for KV cache, activations, and framework buffers
    return round(weight_gb * 1.25, 2)


def can_model_run(
    params_in_billions: float,
    specs: SystemSpecs,
    precision: str = "fp16"
) -> Dict[str, Any]:
    """Determines if a model can run on GPU, MPS, or CPU without OOM."""
    req_mem = estimate_required_memory_gb(params_in_billions, precision)

    # 1. Prefer GPU / VRAM if available
    if specs.has_cuda and specs.vram_available_gb is not None:
        if req_mem <= specs.vram_available_gb:
            return {
                "can_run": True,
                "target": "gpu",
                "required_gb": req_mem,
                "available_gb": specs.vram_available_gb
            }

    # 2. Apple Silicon unified memory
    if specs.has_mps and specs.ram_available_gb is not None:
        if req_mem <= specs.ram_available_gb:
            return {
                "can_run": True,
                "target": "mps",
                "required_gb": req_mem,
                "available_gb": specs.ram_available_gb
            }

    # 3. Fallback to CPU RAM
    if req_mem <= specs.ram_available_gb:
        return {
            "can_run": True,
            "target": "cpu",
            "required_gb": req_mem,
            "available_gb": specs.ram_available_gb
        }

    # 4. Out of memory rejection
    available = specs.vram_available_gb if specs.has_cuda else specs.ram_available_gb
    return {
        "can_run": False,
        "target": "none",
        "required_gb": req_mem,
        "available_gb": available,
        "reason": f"Model requires {req_mem} GB, but only {available} GB is available."
    }
