#!/usr/bin/env python3
"""Plot independent mechanism contrasts and their resource trade-offs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(summary_path, output):
    summary = json.loads(summary_path.read_text())
    if summary["execution_failures"] or summary["completed_cases"] != summary["planned_cases"]:
        raise ValueError("plot the complete registered cohort, including unsuccessful research")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.subplots_adjust(left=.06, right=.985, bottom=.26, top=.83, wspace=.5)
    labels = ["Source", "Units", "Scope", "Hidden", "New instrument", "Noise", "Simplify", "Return"]
    positions = np.arange(8)
    colors = {"shared": "#24678d", "task_local": "#b07740"}
    for offset, arm in ((-.17, "shared"), (.17, "task_local")):
        stages = summary["totals"][arm]["stages"]
        rates = [item["solved"] / item["tasks"] for item in stages]
        axes[0].bar(positions + offset, rates, width=.32, color=colors[arm],
                    label="Persistent" if arm == "shared" else "Task-local")
    axes[0].set_xticks(positions, labels, rotation=45, ha="right")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Fraction passing independent evaluation")
    axes[0].set_title("a  Joint research sequence", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")

    keys = ["expression", "units", "methods", "measurement", "sequence"]
    names = ["New expression", "New units", "Learned method", "New measurement", "Whole sequence"]
    for index, key in enumerate(keys):
        result = summary["paired_uncertainty"][key]["observation_cost"]
        value = result["mean_retained_minus_control_per_seed"]
        low, high = result["paired_95_percent_interval"]
        axes[1].errorbar(value, index, xerr=[[value-low], [high-value]], fmt="o",
                         color="#24678d", capsize=3, markersize=5)
    axes[1].axvline(0, color=".6", linewidth=.8)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Paired observation difference per seed\nPersistent − intervention control")
    axes[1].set_title("b  Mechanism interventions", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color=".9", linewidth=.5)

    full, local = (summary["totals"][arm] for arm in ("shared", "task_local"))
    ratios = [full["costs"][key] / local["costs"][key] for key in
              ("observation_cost", "measurement_cost", "compute_cost")]
    ratios.append(full["training_cpu_seconds"] / local["training_cpu_seconds"])
    bars = axes[2].bar(range(4), ratios, color=["#24678d"]*3 + ["#b07740"], width=.65)
    for bar, value in zip(bars, ratios):
        axes[2].annotate(f"{value:.2f}×", (bar.get_x() + bar.get_width()/2, value),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    axes[2].axhline(1, color=".5", linewidth=.8, linestyle="--")
    axes[2].set_xticks(range(4), ["Observations", "Measurements", "Charged compute", "CPU time"],
                       rotation=40, ha="right")
    axes[2].set_ylim(0, max(1.2, max(ratios)*1.18))
    axes[2].set_ylabel("Persistent / task-local total")
    axes[2].set_title("c  Whole-sequence cost", loc="left", fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix("."+suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper/figures/joint_research_growth"))
    args = parser.parse_args()
    plot(args.summary, args.output)
