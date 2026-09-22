# ModelFit-MCP

[![CI](https://github.com/DumboDhruvi/ModelFit-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/DumboDhruvi/ModelFit-MCP/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Compliant](https://img.shields.io/badge/MCP-Compliant-green.svg)](https://modelcontextprotocol.io/)

> **Hardware-Aware Hugging Face Discovery, Memory Fitting, & Swappable Local Model Gateway for AI Agents.**

ModelFit-MCP connects AI agents (Claude Desktop, Cursor, custom autonomous agents) to Hugging Face with built-in hardware awareness. It profiles host specs (CPU, RAM, GPU, VRAM), calculates exact model memory footprints, filters out models that would trigger CUDA Out of Memory (OOM) errors, and spins up a **swappable local gateway** so models can be changed dynamically without modifying application code.

---

## Key Features
- **Zero OOM Crashes**: Automatically evaluates parameter count, precision (`fp32`, `fp16`, `int8`, `int4`), and 25% activation headroom before suggesting or loading models.
- **Swappable Architecture**: Your client code interacts with an abstract gateway (`gateway.predict()` or `POST /predict`). Models can be upgraded or hot-swapped without touching your application code.
- **Memory Purging**: Unloads old weights and triggers `torch.cuda.empty_cache()` on every swap to prevent VRAM memory leaks.
- **Multi-Modal**: Normalizes outputs across `image-classification`, `object-detection`, `text-generation`, and `zero-shot-image-classification`.
- **Cross-Language Ready**: First-class support for Python, Flutter/Dart, Node.js, and cURL.

---

## System Architecture

```mermaid
flowchart TD
    User["Your App\n(Plant App, Flutter, Web, CLI)"] -->|predict input| Gateway["Local Model Gateway\n(In-Process or http://127.0.0.1:7860)"]
    
    subgraph Engine["ModelFit-MCP Engine"]
        Gateway --> ActiveModel["Active Model\n(e.g., MobileNet / ViT)"]
        MCP["MCP Server / CLI\n(modelfit)"] -.->|Hot Swap + VRAM Cleanup| ActiveModel
        Profiler["hardware.py\n(VRAM & RAM Profiler)"] --> HFFilter["hf_client.py\n(Sizing & Ranking)"]
        HFFilter --> MCP
    end
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/DumboDhruvi/ModelFit-MCP.git
cd ModelFit-MCP
pip install -e .
```

### 2. CLI Usage
Inspect your hardware headroom:
```bash
modelfit specs
```

Search Hugging Face models guaranteed to fit your machine:
```bash
modelfit search "plant disease" --task image-classification
```

Start the local micro-API daemon:
```bash
modelfit serve --port 7860
```

Hot-swap models on the fly:
```bash
modelfit swap "google/vit-base-patch16-224" --task image-classification
```

---

## MCP Server Setup (Claude Desktop & Cursor)

Add ModelFit to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "modelfit": {
      "command": "modelfit-server"
    }
  }
}
```

---

## Integration Modes

### Mode A: In-Process Python Adapter (Zero Latency)
```python
from modelfit.adapter import gateway

# 1. Load initial model
gateway.load_model("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

# 2. Abstract prediction
results = gateway.predict("leaf.jpg")
print(results[0].label, results[0].score)

# 3. Hot-swap later with ZERO code changes below
gateway.load_model("google/vit-base-patch16-224")
results = gateway.predict("leaf.jpg")
```

### Mode B: Local Micro-API (Flutter, Node.js, Web, cURL)
Start the background daemon:
```bash
modelfit serve --port 7860
```

Query or swap over HTTP:
```bash
# Predict
curl -X POST http://127.0.0.1:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "leaf_sample.jpg"}'

# Hot-Swap
curl -X POST http://127.0.0.1:7860/swap \
  -H "Content-Type: application/json" \
  -d '{"model_id": "google/vit-base-patch16-224", "task": "image-classification"}'
```

*See [`examples/flutter_integration_example.dart`](examples/flutter_integration_example.dart) for a complete Flutter service.*

---

## Testing

Run the full test suite:
```bash
python3 -m unittest discover tests -v
```
*All tests complete in under 2 seconds.*

---

## License
MIT License. See `LICENSE` for details.
