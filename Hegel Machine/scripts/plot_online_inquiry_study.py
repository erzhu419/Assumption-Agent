#!/usr/bin/env python3
"""Plot the first registered online-rewrite trace and its actual alternatives."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.online_inquiry import predict_action

def main():
    study = ROOT / "artifacts/online_inquiry_20261005_main"
    case = json.loads((study / "case_79101013/case.json").read_text())
    record = next(r for r in case["results"] if r["arm"] == "recursive"
                  and r["condition"] == "same_structure_changed_surface")
    report = record["research"]
    event = next(e for e in report["events"] if e["event"] == "method_rewritten")
    nodes = report["nodes"]
    spec = record["public_spec"]
    binding = next(n["answer"] for n in nodes if n["goal"] == "coordinate_mapping")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(7, 6.5),
                                      gridspec_kw={"height_ratios": [1.05, 1]})
    colors = {"response_law": "#315B82", "coordinate_mapping": "#30846F",
              "sensor_calibration": "#C77D36"}
    positions = {n["question_id"]: 2 - i for i, n in enumerate(nodes)}
    for node in nodes:
        y = positions[node["question_id"]]
        ax.plot([node["opened_at"], node["resolved_at"]], [y, y],
                lw=9, alpha=.22, color=colors[node["goal"]], solid_capstyle="round")
        ax.scatter([node["opened_at"]], [y], s=45, facecolor="white",
                   edgecolor=colors[node["goal"]], zorder=4)
        ax.scatter([node["resolved_at"]], [y], s=45, color=colors[node["goal"]], zorder=4)
    for e in report["events"]:
        if e["event"] == "experiment_selected" and e["reason"] != "confirm_frozen_hypothesis":
            y = positions[e["question_id"]]
            ax.scatter([e["event_id"]], [y], marker="s", s=27, color="#333333", zorder=5)
        elif e["event"] == "parent_resumed":
            child = next(n for n in nodes if n["question_id"] == e["consumes_answer"])
            ax.add_patch(FancyArrowPatch((child["resolved_at"], positions[child["question_id"]]+.08),
                (e["event_id"], positions[e["question_id"]]-.09), arrowstyle="-|>",
                mutation_scale=12, color="#555555", lw=1.3))
    ax.scatter([event["event_id"]], [2], marker="*", s=150, color="#AD3A3A", zorder=6)
    ax.annotate("Rewrite before response probe", (event["event_id"], 2),
                xytext=(9, 2.62), arrowprops={"arrowstyle": "-", "color": "#AD3A3A"},
                color="#AD3A3A", fontsize=9.5)
    ax.set(yticks=[0, 1, 2], yticklabels=["Sensor calibration", "Coordinate mapping", "Response law"],
           xlabel="Recorded execution event", xlim=(-1, 28), ylim=(-.45, 2.95))
    ax.set_title("a  Supporting questions precede their evidence", loc="left", fontsize=11, pad=12)
    ax.text(0, -.34, "○ Open   ■ Select experiment   ● Resolve   ↑ Return answer",
            transform=ax.transAxes, fontsize=9.5, color="#444444")

    rows = []
    for mechanism in spec["finite_language"]["mechanisms"]:
        candidate = {**binding, "mechanism": mechanism, "calibration_sign": 1,
                     "calibration_offset": 0}
        rows.append([mechanism.title(),
                     f'{predict_action(candidate, event["previous_action"], spec):.2f}',
                     f'{predict_action(candidate, event["selected_action"], spec):.2f}'])
    table_ax.axis("off")
    table_ax.set_title("b  Returned binding changes a useful test", loc="left", fontsize=11, pad=16)
    table = table_ax.table(cellText=rows, colLabels=["Candidate", "Old probe", "Revised probe"],
                          cellLoc="center", colWidths=[.4, .3, .3], bbox=[0, .27, 1, .57])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDE3E8")
        if r == 0:
            cell.set_facecolor("#EAF0F5")
            cell.set_text_props(weight="bold")
        elif c == 1 and r in (2, 3):
            cell.set_facecolor("#F9DFD8")
        elif c == 2:
            cell.set_facecolor("#E2F0E8")
    table_ax.text(0, .94, "Child answer: the second control is the driver", fontsize=10)
    table_ax.text(0, .19, "Old: (−0.6, −1.0)       Revised: (−1.0, −0.6)", fontsize=9.5)
    table_ax.text(0, .04, "Old probe aliases quadratic and absolute responses.\n"
                  "Revised probe separates all four remaining laws.", fontsize=9.5, va="top")
    fig.subplots_adjust(left=.25, right=.98, bottom=.07, top=.92, hspace=.66)
    target = ROOT / "paper/figures/online_support_inquiry"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(target.with_suffix("."+ext), dpi=300, bbox_inches="tight")
    print(target.with_suffix(".pdf"))

if __name__ == "__main__":
    main()
