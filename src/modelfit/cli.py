"""Command Line Interface for ModelFit-MCP."""

import argparse
import json
import sys
import urllib.request
import urllib.parse
from modelfit.hardware import detect_system_specs
from modelfit.hf_client import HFHardwareClient
from modelfit.local_api import run_api_server
from modelfit.server import run_stdio_server


def cmd_specs(args):
    """Displays formatted host hardware specs."""
    specs = detect_system_specs()
    print("\n" + "=" * 50)
    print("        ModelFit Hardware Profile")
    print("=" * 50)
    print(f"  OS:              {specs.os_name}")
    print(f"  CPU Cores:       {specs.cpu_count}")
    print(f"  RAM Total:       {specs.ram_total_gb:.2f} GB")
    print(f"  RAM Available:   {specs.ram_available_gb:.2f} GB")
    if specs.has_cuda:
        print(f"  GPU:             {specs.gpu_name} (CUDA)")
        print(f"  VRAM Total:      {specs.vram_total_gb:.2f} GB")
        print(f"  VRAM Available:  {specs.vram_available_gb:.2f} GB")
    elif specs.has_mps:
        print(f"  GPU:             Apple Silicon (MPS Unified Memory)")
    else:
        print("  GPU:             None (CPU Fallback)")
    print("=" * 50 + "\n")


def cmd_search(args):
    """Searches Hugging Face for compatible models fitting host hardware."""
    specs = detect_system_specs()
    client = HFHardwareClient()
    print(f"\n[ModelFit] Searching Hugging Face for '{args.query}' (Task: {args.task or 'all'})...")

    raw_models = client.search_models(query=args.query, pipeline_tag=args.task, limit=30)
    compatible = client.filter_and_rank_models(raw_models, specs, preferred_precision=args.precision)

    if not compatible:
        print(f"[ModelFit] No models found fitting current hardware ({specs.ram_available_gb:.2f} GB available).")
        return

    print(f"\nFound {len(compatible)} hardware-compatible models (Top {min(args.limit, len(compatible))}):\n")
    print(f"{'#':<3} {'Model ID':<50} {'Device':<8} {'VRAM/RAM':<10} {'Downloads':<10} {'Score':<8}")
    print("-" * 92)
    for i, m in enumerate(compatible[:args.limit], 1):
        print(f"{i:<3} {m['model_id']:<50} {m['target_device'].upper():<8} {m['memory_required_gb']:<6.2f} GB {m['downloads']:<10} {m['ranking_score']:<8.1f}")
    print()


def cmd_serve(args):
    """Starts the local ModelFit micro-API daemon."""
    run_api_server(host=args.host, port=args.port)


def cmd_mcp(args):
    """Starts the STDIO Model Context Protocol server."""
    run_stdio_server()


def cmd_swap(args):
    """Hot-swaps the model in the running local API gateway."""
    url = f"http://{args.host}:{args.port}/swap"
    payload = json.dumps({"model_id": args.model_id, "task": args.task}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[ModelFit] Swapped successfully: {data}")
    except Exception as e:
        print(f"[ModelFit] Error swapping model: {e}", file=sys.stderr)


def cmd_predict(args):
    """Sends a prediction request to the running local API gateway."""
    url = f"http://{args.host}:{args.port}/predict"
    payload = json.dumps({"input": args.input}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[ModelFit] Error sending prediction: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        prog="modelfit",
        description="ModelFit-MCP: Hardware-Aware Hugging Face Discovery & Local Model Gateway"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # specs
    subparsers.add_parser("specs", help="Show host hardware specs (CPU, RAM, GPU, VRAM)")

    # search
    p_search = subparsers.add_parser("search", help="Search HF models compatible with local hardware")
    p_search.add_argument("query", help="Search keyword (e.g. 'plant disease')")
    p_search.add_argument("--task", default=None, help="Hugging Face pipeline tag (e.g. 'image-classification')")
    p_search.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16", "int8", "int4"], help="Model precision")
    p_search.add_argument("--limit", type=int, default=5, help="Number of results to show")

    # serve
    p_serve = subparsers.add_parser("serve", help="Run local HTTP micro-API daemon")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host binding")
    p_serve.add_argument("--port", type=int, default=7860, help="Port binding")

    # mcp
    subparsers.add_parser("mcp", help="Run MCP STDIO server for Claude / Cursor / agents")

    # swap
    p_swap = subparsers.add_parser("swap", help="Hot-swap model in running local API")
    p_swap.add_argument("model_id", help="Hugging Face model ID")
    p_swap.add_argument("--task", default="image-classification", help="Pipeline task")
    p_swap.add_argument("--host", default="127.0.0.1", help="Gateway host")
    p_swap.add_argument("--port", type=int, default=7860, help="Gateway port")

    # predict
    p_pred = subparsers.add_parser("predict", help="Send prediction to running local API")
    p_pred.add_argument("input", help="Input text, file path, or image path")
    p_pred.add_argument("--host", default="127.0.0.1", help="Gateway host")
    p_pred.add_argument("--port", type=int, default=7860, help="Gateway port")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "specs": cmd_specs,
        "search": cmd_search,
        "serve": cmd_serve,
        "mcp": cmd_mcp,
        "swap": cmd_swap,
        "predict": cmd_predict,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
