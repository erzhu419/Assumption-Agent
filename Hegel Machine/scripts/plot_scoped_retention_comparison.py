#!/usr/bin/env python3
"""Plot successor research costs from the completed retention experiment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts/paper_scoped_retention_20261005/summary.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/scoped_retention_comparison"
ARMS = ("conditional_retention", "unmarked_retention", "global_discard")
LABELS = ("Conditional retention", "Unmarked retention", "Global discard")
COLORS = ("#4C7899", "#ADB3BA", "#C58B60")
PHASES = ("first_changed", "original_return", "changed_return")
PHASE_LABELS = ("First\nchange", "Original\nreturn", "Changed\nreturn")
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")


def plot_comparison(summary_path, output_stem):
    summary = json.loads(summary_path.read_text())
    if (summary["completed_cases"] != summary["planned_cases"]
            or summary["execution_failures"]):
        raise ValueError("The paper figure requires the complete registered cohort.")
    totals = summary["totals"]
    phases = {arm: {item["phase"]: item["costs"] for item in totals[arm]["successors"]}
              for arm in ARMS}
    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.linewidth": .7,
        "axes.edgecolor": "#73777B", "text.color": "#20262C",
        "axes.labelcolor": "#30363C", "xtick.color": "#50565C",
        "ytick.color": "#30363C", "xtick.labelsize": 10, "ytick.labelsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(7, 6.4))
        grid = figure.add_gridspec(2, 2)
        axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]),
                figure.add_subplot(grid[1, :])]
        figure.subplots_adjust(left=.13, right=.97, bottom=.17, top=.86, wspace=.65, hspace=.72)
        width = .23
        for letter, title, resource, ax in zip("ABC", (
                "Observations", "Measurement units", "Charged computation"), RESOURCES, axes):
            maximum = 0
            for index, (arm, color) in enumerate(zip(ARMS, COLORS)):
                values = [phases[arm][phase][resource] for phase in PHASES]
                maximum = max(maximum, *values)
                positions = [group + (index - 1) * width for group in range(len(PHASES))]
                ax.bar(positions, values, color=color, width=width * .9, zorder=3)
            ax.set_title(f"{letter}  {title}", loc="left", pad=12)
            ax.set_xticks(range(len(PHASES)), PHASE_LABELS)
            ax.set_ylim(0, max(1, maximum) * 1.12)
            ax.set_ylabel(f"Total ({summary['completed_cases']} instances)")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", length=3, width=.6)
            ax.grid(axis="y", color="#E5E8EB", linewidth=.6)
            ax.set_axisbelow(True)
        figure.legend([Patch(facecolor=color) for color in COLORS], LABELS,
            loc="upper center", bbox_to_anchor=(.5, .99), ncol=3,
            frameon=False, handlelength=1.2, columnspacing=1.2, fontsize=9.5)
        figure.text(.13, .035,
            "Successor research only; shared prefix and challenge excluded.\n"
            "Sequence totals include both. Terminal evaluation is reported separately.",
            fontsize=9.5, color="#50565C", va="bottom", linespacing=1.5)
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
        help="Completed retention cohort summary JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output stem for PDF, SVG and PNG.")
    args = parser.parse_args()
    for path in plot_comparison(args.summary, args.output):
        print(path)


if __name__ == "__main__":
    main()
