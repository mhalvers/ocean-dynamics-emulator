from __future__ import annotations

import argparse
from pathlib import Path

from ode.experiments import evaluate_checkpoint_against_benchmark, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an ode checkpoint on a fixed benchmark window set.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--benchmark", required=True, help="Path to the benchmark JSON spec.")
    parser.add_argument("--output", default=None, help="Optional path for the benchmark metrics JSON output.")
    parser.add_argument("--device", default="auto", help="Evaluation device: auto, cpu, cuda, or mps.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_checkpoint_against_benchmark(args.checkpoint, args.benchmark, device=args.device)
    summary = metrics["summary"]
    print(f"Benchmark: {metrics['benchmark_name']}")
    print(f"Checkpoint: {metrics['checkpoint_path']}")
    print(f"Data store: {metrics['evaluated_data_store']}")
    print(f"Windows: {summary['window_count']}")
    print(f"Overall MSE: {summary['overall_mse']:.6f}")
    print(f"Overall RMSE: {summary['overall_rmse']:.6f}")
    print(f"Persistence MSE: {summary['persistence_mse']:.6f}")
    if args.output:
        output_path = save_json(Path(args.output), metrics)
        print(f"Saved benchmark metrics to {output_path}")


if __name__ == "__main__":
    main()