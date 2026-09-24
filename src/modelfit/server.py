"""ModelFit MCP Server."""

import sys
import json
from typing import Dict, Any
from modelfit.hardware import detect_system_specs, can_model_run
from modelfit.hf_client import HFHardwareClient
from modelfit.code_gen import generate_transformers_snippet, generate_ensemble_snippet
from modelfit.adapter import gateway
from modelfit.evaluator import ModelEvaluator
from modelfit.ensemble import EnsembleGateway

hf_client = HFHardwareClient()
evaluator = ModelEvaluator()
ensemble_gateway = EnsembleGateway()


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

    elif name == "benchmark_models":
        model_ids = args.get("model_ids", [])
        test_samples = args.get("test_samples", ["test_sample"])
        pipeline_tag = args.get("pipeline_tag", "image-classification")
        results = evaluator.benchmark_models(
            model_ids=model_ids,
            test_samples=test_samples,
            pipeline_tag=pipeline_tag
        )
        return {
            "status": "success",
            "evaluated_count": len(results),
            "leaderboard": [r.to_dict() for r in results],
            "recommended_model": results[0].model_id if results else None
        }

    elif name == "find_ensemble_models":
        query = args.get("query", "")
        pipeline_tag = args.get("pipeline_tag", "image-classification")
        max_models = args.get("max_models", 3)
        candidates = ensemble_gateway.find_ensemble_candidates(
            query=query,
            pipeline_tag=pipeline_tag,
            max_models=max_models
        )
        return {
            "status": "success",
            "count": len(candidates),
            "models": candidates,
            "recommendation": f"Ensemble of {len(candidates)} fast models fitting hardware headroom"
        }

    elif name == "ensemble_predict":
        model_ids = args.get("model_ids", [])
        input_data = args.get("input_data")
        if not input_data:
            return {"status": "error", "message": "Missing 'input_data' argument"}
        strategy = args.get("strategy", "weighted_average")
        pipeline_tag = args.get("pipeline_tag", "image-classification")

        preds = ensemble_gateway.predict_ensemble(
            input_data=input_data,
            model_ids=model_ids,
            pipeline_tag=pipeline_tag,
            strategy=strategy
        )
        return {
            "status": "success",
            "strategy": strategy,
            "predictions": [p.to_dict() for p in preds]
        }

    else:
        return {"status": "error", "message": f"Unknown tool '{name}'"}


import contextlib

TOOLS = [
    {
        "name": "get_hardware_specs",
        "description": "Detect host CPU, RAM, and GPU/VRAM hardware headroom",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "search_compatible_models",
        "description": "Search HuggingFace models filtered by hardware fit",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword e.g. plant disease"},
                "pipeline_tag": {"type": "string", "description": "Hugging Face pipeline task tag (e.g. image-classification, text-generation)"},
                "precision": {"type": "string", "description": "Target precision (fp32, fp16, int8, int4)", "default": "fp16"},
            },
        },
    },
    {
        "name": "swap_active_model",
        "description": "Hot-swap the active model with VRAM-safe memory purging",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Hugging Face model identifier"},
                "pipeline_tag": {"type": "string", "description": "Task pipeline tag (default: image-classification)"},
                "params_billions": {"type": "number", "description": "Parameter count in billions (default: 0.5)"},
            },
            "required": ["model_id"],
        },
    },
    {
        "name": "get_active_model_status",
        "description": "Inspect currently loaded model and target device",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_integration_code",
        "description": "Get drop-in Python inference code",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Target model ID"},
                "pipeline_tag": {"type": "string", "description": "Pipeline task tag"},
                "target_device": {"type": "string", "description": "Device (cpu or cuda)"},
            },
            "required": ["model_id"],
        },
    },
    {
        "name": "recommend_and_scaffold",
        "description": "1-shot model search, hardware check, and code scaffolding",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for model type or task"},
                "pipeline_tag": {"type": "string", "description": "Task pipeline tag"},
            },
        },
    },
    {
        "name": "benchmark_models",
        "description": "Run live accuracy and latency benchmarking on candidate models",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_ids": {"type": "array", "items": {"type": "string"}, "description": "List of model IDs to benchmark"},
                "test_samples": {"type": "array", "items": {"type": "string"}, "description": "Test sample inputs"},
                "pipeline_tag": {"type": "string", "description": "Pipeline tag"},
            },
            "required": ["model_ids"],
        },
    },
    {
        "name": "find_ensemble_models",
        "description": "Find ultra-fast models suitable for low-latency ensembling",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "pipeline_tag": {"type": "string", "description": "Pipeline tag"},
                "max_models": {"type": "integer", "description": "Maximum candidate models to return"},
            },
        },
    },
    {
        "name": "ensemble_predict",
        "description": "Execute ensemble prediction combining multiple models",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_data": {"type": "string", "description": "Input data string or file path"},
                "model_ids": {"type": "array", "items": {"type": "string"}, "description": "Model IDs for the ensemble"},
                "strategy": {"type": "string", "description": "Aggregation strategy: weighted_average or majority_vote"},
                "pipeline_tag": {"type": "string", "description": "Pipeline tag"},
            },
            "required": ["input_data"],
        },
    },
]


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

            if method == "initialize":
                protocol_version = params.get("protocolVersion", "2024-11-05")
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": protocol_version,
                        "capabilities": {
                            "tools": {},
                        },
                        "serverInfo": {
                            "name": "modelfit-mcp",
                            "version": "0.1.0",
                        },
                    },
                }
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": TOOLS,
                    },
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                with contextlib.redirect_stdout(sys.stderr):
                    result = handle_tool_call(tool_name, tool_args)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2),
                            }
                        ],
                        "isError": result.get("status") == "error",
                        **result,
                    },
                }
            elif method == "ping":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
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
