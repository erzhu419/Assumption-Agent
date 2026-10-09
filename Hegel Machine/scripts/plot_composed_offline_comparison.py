#!/usr/bin/env python3
"""Plot the completed paper cohort without mixing research and evaluation costs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts/paper_composed_comparison_20261005/summary.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/composed_offline_comparison"
ARMS = ("full", "fixed_library", "passive_experiments", "task_local_memory")
LABELS = ("Full", "Fixed basis", "Fixed experiments", "Task-local")
COLORS = ("#4C7899", "#ADB3BA", "#C58B60", "#626D76")
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def plot_comparison(summary_path, output_stem):
    summary = json.loads(summary_path.read_text())
    if (summary["completed_cases"] != summary["planned_cases"]
            or summary["execution_failures"]):
        raise ValueError("The paper figure requires the complete registered cohort.")
    totals = summary["totals"]
    baseline = totals["task_local_memory"]["costs"]
    if any(baseline[resource] <= 0 for resource in RESOURCES):
        raise ValueError("Task-local research totals must be positive for normalization.")

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.linewidth": .7,
        "axes.edgecolor": "#73777B",
        "text.color": "#20262C",
        "axes.labelcolor": "#30363C",
        "xtick.color": "#50565C",
        "ytick.color": "#30363C",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(7, 6.2))
        grid = figure.add_gridspec(2, 2)
        axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]),
                figure.add_subplot(grid[1, :])]
        figure.subplots_adjust(left=.20, right=.98, bottom=.17, top=.9, wspace=.72, hspace=.85)
        for letter, title, ax in zip("ABC", (
                "Task accuracy", "Research resources", "Evaluation measurements"), axes):
            ax.set_title(f"{letter}  {title}", loc="left", pad=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", length=3, width=.6)
            ax.set_axisbelow(True)

        ax = axes[0]
        solved = [totals[arm]["solved"] for arm in ARMS]
        bars = ax.barh(range(len(ARMS)), solved, color=COLORS, height=.6)
        ax.set_yticks(range(len(ARMS)), LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, max(totals[arm]["tasks"] for arm in ARMS) * 1.28)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.grid(axis="x", color="#E5E8EB", linewidth=.6)
        ax.set_xlabel("Tasks passing terminal threshold")
        ax.bar_label(bars, labels=[f"{totals[arm]['solved']}/{totals[arm]['tasks']}"
            for arm in ARMS], padding=4, fontsize=9.5)

        ax = axes[1]
        width = .19
        maximum = 1.
        for index, (arm, color) in enumerate(zip(ARMS, COLORS)):
            ratios = [totals[arm]["costs"][resource] / baseline[resource]
                      for resource in RESOURCES]
            maximum = max(maximum, *ratios)
            positions = [group + (index - 1.5) * width for group in range(3)]
            bars = ax.bar(positions, ratios, color=color, width=width * .88)
            ax.bar_label(bars, labels=[f"{value:.2f}" for value in ratios],
                padding=3, fontsize=9.5, rotation=90)
        ax.axhline(1., color="#50565C", linestyle=(0, (3, 3)), linewidth=.8, zorder=1)
        ax.set_xticks(range(3), ("Response\nrows", "Meter\nunits", "Charged\ncompute"))
        ax.tick_params(axis="x", labelsize=9.5)
        ax.set_ylim(0, maximum * 1.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", color="#E5E8EB", linewidth=.6)
        ax.set_ylabel("Research cost / task-local cost")

        ax = axes[2]
        measurements = [totals[arm]["evaluation_costs"]["measurement_cost"] for arm in ARMS]
        bars = ax.barh(range(len(ARMS)), measurements, color=COLORS, height=.6)
        ax.set_yticks(range(len(ARMS)), LABELS)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1, max(measurements)) * 1.38)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.grid(axis="x", color="#E5E8EB", linewidth=.6)
        ax.set_xlabel("Instrument measurement units")
        ax.bar_label(bars, labels=[f"{value:,}" for value in measurements],
            padding=4, fontsize=9.5)

        figure.legend([Patch(facecolor=color) for color in COLORS], LABELS,
            loc="lower center", bbox_to_anchor=(.5, .025), ncol=4,
            frameon=False, handlelength=1.2, columnspacing=1.2, fontsize=9.5)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in ("pdf", "svg", "png"):
            path = output_stem.with_suffix(f".{extension}")
            figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.12)
            paths.append(path)
        plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
        help="Completed cohort summary JSON (default: registered paper cohort).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output stem for PDF, SVG and PNG (default: paper/figures/composed_offline_comparison).")
    args = parser.parse_args()
    for path in plot_comparison(args.summary, args.output):
        print(path)


if __name__ == "__main__":
    main()
