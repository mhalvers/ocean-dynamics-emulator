from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def format_run_label(run_name: str) -> str:
    tokens = run_name.split("_")
    # Most run directories end with YYYY-MM-DD_to_YYYY-MM-DD; drop that suffix for cleaner labels.
    if len(tokens) >= 3 and tokens[-2] == "to":
        return "_".join(tokens[:-3])
    return run_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a bar plot of benchmark RMSE for each run in experiments/runs."
    )
    parser.add_argument(
        "--runs-root",
        default="experiments/runs",
        help="Directory containing run subdirectories with benchmark_metrics.json files.",
    )
    parser.add_argument(
        "--output",
        default="experiments/plots/run_rmse_bar.png",
        help="Output path for the generated RMSE bar plot.",
    )
    parser.add_argument(
        "--title",
        default="Forecast RMSE by Run",
        help="Plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    return parser


def collect_run_rmse(runs_root: Path) -> list[tuple[str, float]]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_root}")

    rmse_values: list[tuple[str, float]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "benchmark_metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        summary = metrics.get("summary", {})
        overall_rmse = summary.get("overall_rmse")
        if overall_rmse is None:
            continue
        rmse_values.append((run_dir.name, float(overall_rmse)))

    if not rmse_values:
        raise ValueError(f"No benchmark RMSE values found under {runs_root}")
    return rmse_values


def plot_rmse(values: list[tuple[str, float]], output_path: Path, title: str, show: bool) -> None:
    run_names = [name for name, _ in values]
    label_names = [format_run_label(name) for name in run_names]
    rmses = [rmse for _, rmse in values]

    figure, axis = plt.subplots(figsize=(max(8, 1.2 * len(run_names)), 5.5))
    bars = axis.bar(label_names, rmses, color="#2a9d8f")

    axis.set_title(title)
    axis.set_xlabel("Run")
    axis.set_ylabel("Overall RMSE")
    axis.tick_params(axis="x", rotation=30, labelsize=9)
    plt.setp(axis.get_xticklabels(), ha="right", rotation_mode="anchor")
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    for bar, rmse in zip(bars, rmses):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{rmse:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    output_path = Path(args.output)

    values = collect_run_rmse(runs_root)
    plot_rmse(values, output_path=output_path, title=args.title, show=args.show)

    print(f"Found {len(values)} runs with benchmark RMSE")
    for run_name, rmse in values:
        print(f"- {run_name}: {rmse:.6f}")
    print(f"Saved RMSE bar plot to {output_path}")


if __name__ == "__main__":
    main()
