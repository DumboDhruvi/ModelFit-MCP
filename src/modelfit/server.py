"""ModelFit MCP Server."""

import sys
import json
from typing import Dict, Any
from modelfit.hardware import detect_system_specs, can_model_run
from modelfit.hf_client import HFHardwareClient
from modelfit.code_gen import generate_transformers_snippet
from modelfit.adapter import gateway

hf_client = HFHardwareClient()


def handle_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatches MCP tool calls with robust error handling."""
    if name == "get_hardware_specs":
        specs = detect_system_specs()
        return {"status": "success", "specs": specs.to_dict()}

    elif name == "search_compatible_models":
        query = args.get("query", "")
        pipeline_tag = args.get("pipeline_tag")
        precision = args.get("precision", "fp16")

        specs = detect_system_specs()
        raw_models = hf_client.search_models(query=query, pipeline_tag=pipeline_tag)
        compatible = hf_client.filter_and_rank_models(raw_models, specs, preferred_precision=precision)

        return {
            "status": "success",
            "detected_hardware": specs.to_dict(),
            "count": len(compatible),
            "models": compatible[:5]
        }

    elif name == "swap_active_model":
        model_id = args.get("model_id")
        if not model_id:
            return {"status": "error", "message": "Missing 'model_id' argument"}
        pipeline_tag = args.get("pipeline_tag", "image-classification")
        params_b = args.get("params_billions", 0.5)

        specs = detect_system_specs()
        fit = can_model_run(params_b, specs)
        if not fit["can_run"]:
            return {
                "status": "error",
                "message": f"Cannot swap to {model_id}: {fit['reason']}"
            }

        try:
            res = gateway.load_model(model_id, pipeline_tag=pipeline_tag)
            return {"status": "success", "active_model": res}
        except Exception as e:
            return {"status": "error", "message": f"Failed to load model: {str(e)}"}

    elif name == "get_active_model_status":
        return {
            "status": "success",
            "active_model": gateway.active_model_id,
            "task": gateway.pipeline_tag,
            "device": gateway.target_device
        }

    elif name == "get_integration_code":
        model_id = args.get("model_id")
        if not model_id:
            return {"status": "error", "message": "Missing 'model_id'"}
        pipeline_tag = args.get("pipeline_tag", "image-classification")
        target_device = args.get("target_device", "cpu")
        code = generate_transformers_snippet(model_id, pipeline_tag, target_device)
        return {"status": "success", "code": code}

    elif name == "recommend_and_scaffold":
        query = args.get("query", "")
        pipeline_tag = args.get("pipeline_tag", "image-classification")
        specs = detect_system_specs()

        raw_models = hf_client.search_models(query=query, pipeline_tag=pipeline_tag)
        compatible = hf_client.filter_and_rank_models(raw_models, specs)

        if not compatible:
            return {
                "status": "error",
                "message": f"No models found for '{query}' that fit current hardware ({specs.ram_available_gb} GB RAM available)."
            }

        top_model = compatible[0]
        code = generate_transformers_snippet(
            model_id=top_model["model_id"],
            pipeline_tag=top_model["pipeline_tag"],
            target_device=top_model["target_device"]
        )

        return {
            "status": "success",
            "selected_model": top_model,
            "integration_code": code
        }

    else:
        return {"status": "error", "message": f"Unknown tool '{name}'"}


def run_stdio_server():
    """STDIO JSON-RPC 2.0 loop for Model Context Protocol compliance."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/call":
                result = handle_tool_call(params.get("name"), params.get("arguments", {}))
                response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {"name": "get_hardware_specs", "description": "Detect host CPU, RAM, and GPU/VRAM"},
                            {"name": "search_compatible_models", "description": "Search HuggingFace models filtered by hardware fit"},
                            {"name": "swap_active_model", "description": "Hot-swap the active model with VRAM-safe memory purging"},
                            {"name": "get_active_model_status", "description": "Inspect currently loaded model and target device"},
                            {"name": "get_integration_code", "description": "Get drop-in Python inference code"},
                            {"name": "recommend_and_scaffold", "description": "1-shot model search, hardware check, and code scaffolding"}
                        ]
                    }
                }
            else:
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"status": "ok"}}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
