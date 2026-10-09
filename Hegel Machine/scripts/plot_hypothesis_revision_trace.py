#!/usr/bin/env python3
"""Plot a recorded hypothesis revision and later reuse, without rerunning it."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts/paper_revision_choice_20261005/case_812676_confirmation.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/hypothesis_revision_trace"
INK = "#26323C"
MUTED = "#606F79"
BLUE = "#416F92"
TEAL = "#397D77"
ORANGE = "#A8683B"


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def read_trace(path):
    tasks = json.loads(path.read_text())["results"]["change_b"]["tasks"]
    changed, returned = tasks[:2]
    refinements = [a for a in changed["actions"]
                   if a["action"] == "refine_relation_parameter_space"]
    failure = next(a for a in changed["actions"]
                   if a["action"] == "test_learned_acquisition_method")
    confirm = next(a for a in changed["actions"]
                   if a["action"] == "validate_frozen_program" and a["accepted"])
    return_confirm = next(a for a in returned["actions"]
                          if a["action"] == "validate_frozen_program" and a["accepted"])
    # The figure describes this actual C -> D -> C trajectory. A different
    # selected program requires revising its labels, not silently reusing them.
    assert changed["concepts_added"] == ["concept_0004"]
    assert returned["selected_kind"] == "retained_calibrated_call"
    assert returned["used_sources"] == ["concept_0003"]
    assert not returned["concepts_added"]
    assert all(len(a["after_intervals"]) == 1 for a in refinements)
    rows = [a["evidence_rows"] for a in refinements]
    widths = [a["after_intervals"][0][1] - a["after_intervals"][0][0]
              for a in refinements]
    return changed, returned, failure, confirm, return_confirm, rows, widths


def draw_trace(source, output_stem):
    changed, returned, failure, confirm, return_confirm, rows, widths = read_trace(source)
    style = {"font.family": "DejaVu Sans", "font.size": 8.5,
             "text.color": INK, "axes.labelcolor": INK,
             "xtick.color": MUTED, "ytick.color": MUTED,
             "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
             "ps.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(6.3, 4.65), facecolor="white")
        flow = fig.add_axes((.035, .63, .94, .34))
        flow.set(xlim=(-.006, 1.006), ylim=(0, 1))
        flow.axis("off")
        flow.text(0, .99, "a  A recorded revision and return", va="top",
                  fontsize=10, fontweight="semibold")
        stages = [
            ("Old C fails", "Changed relation\nValidation MSE\n" +
             scientific(failure["validation_loss"]), "#FCF5EF", ORANGE),
            ("Propose D", "Revise the nested\nrelation; retain\nold computations", "#F5F8FB", BLUE),
            ("Confirm D", f"{confirm['observation_rows']} fresh rows\n"
             f"{changed['evaluation_rows']} held-out rows\nMSE " + scientific(changed["mse"]),
             "#F2F8F6", TEAL),
            ("Old regime returns", "Call stored C\nNo new concept\nMSE " +
             scientific(returned["mse"]), "#F2F8F6", TEAL),
        ]
        for i, (title, body, fill, edge) in enumerate(stages):
            x = i * .253
            flow.add_patch(FancyBboxPatch((x, .28), .232, .53,
                           boxstyle="round,pad=0.003,rounding_size=0.02",
                           facecolor=fill, edgecolor=edge, linewidth=.85))
            flow.text(x + .116, .70, title, ha="center", va="center",
                      fontsize=8.1, fontweight="semibold")
            flow.text(x + .116, .475, body, ha="center", va="center",
                      fontsize=8, linespacing=1.35)
            if i < 3:
                flow.add_patch(FancyArrowPatch((x + .234, .55), (x + .251, .55),
                               arrowstyle="-|>", mutation_scale=7,
                               color=MUTED, linewidth=.9))
        flow.text(.5, .15,
                  f"Return: {returned['costs']['observation_cost']} rows total "
                  f"(including {return_confirm['observation_rows']} fresh confirmation rows)",
                  ha="center", va="center", fontsize=8, color=MUTED)
        flow.text(.5, .025, "Recorded return mode: retained_calibrated_call",
                  ha="center", va="center", fontsize=8, color=MUTED)

        ax = fig.add_axes((.125, .14, .845, .40))
        ax.set_title("b  Experiments narrow the unresolved relation parameter",
                     loc="left", fontsize=10, fontweight="semibold", pad=12)
        ax.plot(rows, widths, "o-", color=BLUE, linewidth=1.4, markersize=3.5,
                markeredgewidth=0, zorder=3)
        total = rows[-1] + confirm["observation_rows"]
        ax.axvspan(rows[-1], total, facecolor="#EAF3EF", alpha=.85, zorder=0)
        ax.axvline(rows[-1], color=TEAL, linewidth=.8, linestyle="--")
        ax.text((rows[-1] + total) / 2, 1.8e-4,
                f"+{confirm['observation_rows']} fresh rows\nConfirm frozen D",
                ha="center", va="center", color=TEAL, fontsize=8.3,
                linespacing=1.4)
        ax.annotate(scientific(widths[-1]), (rows[-1], widths[-1]),
                    xytext=(-10, -14), textcoords="offset points", ha="right",
                    color=BLUE, fontsize=8)
        ax.set(xlim=(38, total + 1), ylim=(2e-7, 6e-3), yscale="log",
               xlabel="Acquired rows in the changed task",
               ylabel="Radius interval width")
        ax.set_xticks([40, 60, 80, 100, total])
        ax.set_yticks([1e-3, 1e-4, 1e-5, 1e-6])
        ax.tick_params(axis="both", labelsize=8, length=3)
        ax.tick_params(axis="y", which="minor", left=False)
        ax.grid(axis="y", which="major", color="#E1E7EB", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#ABB7BF")
        fig.text(.125, .035, "Each point: updated candidate interval after acquired evidence.",
                 fontsize=8, color=MUTED)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(output_stem.with_suffix("." + extension), dpi=200)
        plt.close(fig)
    print(json.dumps({"refinement_points": len(rows), "research_rows": rows[-1],
                      "initial_plotted_width": widths[0], "final_width": widths[-1],
                      "fresh_confirmation_rows": confirm["observation_rows"],
                      "heldout_rows": changed["evaluation_rows"],
                      "return_fresh_confirmation_rows": return_confirm["observation_rows"],
                      "output_stem": str(output_stem)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    draw_trace(args.input, args.output_stem)
