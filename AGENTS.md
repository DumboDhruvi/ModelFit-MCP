# AGENTS.md - ModelFit MCP Project Guidelines

## Project Purpose
ModelFit-MCP is a Model Context Protocol (MCP) server that enables AI agents to autonomously find, rank, hardware-verify, and integrate Hugging Face models into user applications. It provides both an in-process Python adapter and a local micro-API for seamless model hot-swapping without modifying application code.

---

## Agent Roles and Responsibilities
When an AI agent (such as an LLM in Claude Desktop, Cursor, or an autonomous workflow) is asked to build or enhance a project with ML models:
1. **Never recommend models blindly**: Always verify local system hardware (RAM, GPU, VRAM) via `get_hardware_specs`.
2. **Filter before suggesting**: Use `search_compatible_models` with the detected hardware constraints to guarantee the model will not trigger a CUDA Out of Memory (OOM) error.
3. **Prefer swappable architectures**: Instead of hardcoding model-specific weights and pipelines into the user's project, scaffold using `adapter.py` (`ModelGateway`) or the local HTTP API (`local_api.py`).
4. **Hot-swap when upgrading**: If the user requests higher accuracy or a smaller footprint, call `swap_active_model`. Do not rewrite the consumer application's inference code.

---

## Memory Sizing Rules & Formulas
The hardware estimator in `hardware.py` uses the following formula to determine whether a model fits in memory:

$$\text{Weight Memory (GB)} = \frac{\text{Parameters} \times \text{Bytes per Parameter}}{1024^3}$$

$$\text{Required Memory (GB)} = \text{Weight Memory} \times 1.25 \quad (\text{Includes 25% overhead for KV cache & activations})$$

### Precision Constants
- **FP32**: 4.0 bytes / param
- **FP16 / BF16**: 2.0 bytes / param
- **INT8**: 1.0 byte / param
- **INT4 (AWQ, GPTQ, GGUF Q4)**: 0.5 bytes / param

### Target Device Hierarchy
1. **GPU (CUDA)**: Preferred if `has_cuda=True` and `required_gb <= vram_available_gb`.
2. **Apple Silicon (MPS)**: Preferred if `has_mps=True` and `required_gb <= ram_available_gb`.
3. **CPU**: Fallback if `required_gb <= ram_available_gb`.
4. **Reject (OOM Prevention)**: Reject if `required_gb > available_memory`.

---

## MCP Server Tools Reference

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `get_hardware_specs` | None | Returns detected CPU cores, total/available RAM, GPU model, and total/available VRAM. |
| `search_compatible_models` | `query` (str), `pipeline_tag` (str, opt), `precision` (str, default "fp16") | Searches Hugging Face Hub, filters by hardware fit, and ranks by quality/popularity. |
| `swap_active_model` | `model_id` (str), `pipeline_tag` (str, opt), `params_billions` (float, opt) | Checks hardware fit and hot-swaps the model in the local gateway, freeing old VRAM. |
| `get_active_model_status` | None | Returns currently loaded model, task, device, and API status. |
| `get_integration_code` | `model_id` (str), `pipeline_tag` (str), `target_device` (str, opt) | Emits drop-in Python inference code or local API client snippet. |
| `recommend_and_scaffold` | `query` (str), `pipeline_tag` (str) | One-shot discovery: detects hardware, picks the optimal model, and generates integration code. |

---

## Testing & Verification Guidelines
- All unit and integration tests are placed in `tests/`.
- Tests must use mocks for external network calls (`urllib.request.urlopen`) and heavy model weights (`transformers.pipeline`) so the entire test suite executes in **under 2 seconds**.
- Run tests via: `python3 -m unittest discover tests`.
