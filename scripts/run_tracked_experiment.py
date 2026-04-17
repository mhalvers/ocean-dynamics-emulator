from __future__ import annotations

import argparse

from ode.experiments import record_existing_checkpoint, run_tracked_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ode experiment and record its benchmark results.")
    parser.add_argument("--config", default=None, help="Path to the training config JSON file.")
    parser.add_argument("--checkpoint", default=None, help="Optional path to an existing checkpoint to register instead of retraining.")
    parser.add_argument("--benchmark", default="benchmarks/ssh_standard_windows_v1.json", help="Path to the benchmark JSON spec.")
    parser.add_argument("--runs-dir", default="experiments/runs", help="Directory where per-run artifacts are stored.")
    parser.add_argument("--registry", default="experiments/registry.jsonl", help="Path to the experiment registry JSONL file.")
    parser.add_argument("--device", default="auto", help="Evaluation device for the benchmark step.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id. Defaults to a timestamped id from the config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint:
        manifest = record_existing_checkpoint(
            args.checkpoint,
            config_path=args.config,
            benchmark_path=args.benchmark,
            runs_dir=args.runs_dir,
            registry_path=args.registry,
            device=args.device,
            run_id=args.run_id,
        )
    else:
        if not args.config:
            raise SystemExit("--config is required unless --checkpoint is provided.")
        manifest = run_tracked_experiment(
            args.config,
            benchmark_path=args.benchmark,
            runs_dir=args.runs_dir,
            registry_path=args.registry,
            device=args.device,
            run_id=args.run_id,
        )
    print(f"Run id: {manifest['run_id']}")
    print(f"Run directory: {manifest['run_dir']}")
    print(f"Checkpoint: {manifest['checkpoint_path']}")
    if manifest["benchmark_overall_mse"] is not None:
        print(f"Benchmark overall MSE: {manifest['benchmark_overall_mse']:.6f}")
    if manifest["benchmark_ssh_mse"] is not None:
        print(f"Benchmark SSH MSE: {manifest['benchmark_ssh_mse']:.6f}")


if __name__ == "__main__":
    main()