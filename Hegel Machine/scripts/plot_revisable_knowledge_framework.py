#!/usr/bin/env python3
"""Draw experimental distinguishability and recursive revision of research."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper/figures/revisable_knowledge_framework"
INK = "#26323C"
MUTED = "#606F79"
LINE = "#AEBBC4"
BLUE = "#416F92"
TEAL = "#397D77"
ORANGE = "#A8683B"


def draw_framework(output_stem):
    # Draw at manuscript width so the smallest labels remain readable.
    style = {
        "font.family": "DejaVu Sans", "font.size": 9.2,
        "text.color": INK, "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure, ax = plt.subplots(figsize=(7.0, 6.6))
        figure.subplots_adjust(left=.02, right=.98, bottom=.02, top=.985)
        ax.set(xlim=(0, 100), ylim=(0, 100))
        ax.axis("off")

        def text(x, y, label, *, size=9.2, color=INK, weight="normal", align="center"):
            ax.text(x, y, label, fontsize=size, color=color, fontweight=weight,
                    ha=align, va="center", linespacing=1.35)

        def box(x, y, width, height, *, fill="white", edge=LINE):
            ax.add_patch(FancyBboxPatch((x, y), width, height,
                boxstyle="round,pad=0.02,rounding_size=1.0",
                facecolor=fill, edgecolor=edge, linewidth=.8))

        def arrow(start, end, *, color=MUTED):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                mutation_scale=8, linewidth=.85, color=color, shrinkA=2, shrinkB=2))

        text(50, 98, "Hypotheses and observations evolve through recursive inquiry",
             size=11.0, weight="semibold")
        text(3, 92.2, "a  What can the current experiments distinguish?", size=10.3,
             weight="semibold", align="left")

        box(3, 77.8, 40, 10.5, fill="#F5F8FB", edge=BLUE)
        text(23, 85.0, "Candidate explanations", weight="semibold", color=BLUE)
        text(23, 80.9, "Representations, laws and scopes")
        box(57, 77.8, 40, 10.5, fill="#F2F8F6", edge=TEAL)
        text(77, 85.0, "Available experiments", weight="semibold", color=TEAL)
        text(77, 80.9, "Inputs, measurements and readouts")

        arrow((23, 77.8), (39, 72.5), color=BLUE)
        arrow((77, 77.8), (61, 72.5), color=TEAL)
        box(24, 62.0, 52, 10.5, fill="#F6F8FA")
        text(50, 69.2, "Experimental signatures", weight="semibold")
        text(50, 65.0, r"$h\;\mapsto\;\left(F_{e,\psi}(h)\right)_{e,\psi}$", size=10.8)
        text(50, 57.7, "Equal signatures: indistinguishable under these tests.", color=MUTED)
        text(50, 53.4, "An informative added measurement or probe can separate them.", color=TEAL)

        ax.plot([3, 97], [49.5, 49.5], color="#DCE3E8", linewidth=.8)
        text(3, 45.7, "b  A blocked claim opens a supporting investigation", size=10.3,
             weight="semibold", align="left")

        steps = (
            (3, 30.0, "Parent question", "Proposed explanation;\nunresolved premise", "#F5F8FB", BLUE),
            (36, 30.0, "Child hypothesis", "Design and conduct\na supporting investigation", "white", LINE),
            (69, 30.0, "Returned evidence", "Supported answer,\ncounterexample or unknown", "#F2F8F6", TEAL),
            (69, 10.5, "Revise the parent", "Change its representation,\nexperiment or explanation", "#FCF5EF", ORANGE),
            (36, 10.5, "Test consequences", "Earlier valid scope\nand new predictions", "white", LINE),
            (3, 10.5, "Accumulate knowledge", "Explanations and\nresearch procedures", "#F5F8FB", BLUE),
        )
        for left, bottom, title, body, fill, edge in steps:
            box(left, bottom, 28, 11.8, fill=fill, edge=edge)
            text(left + 14, bottom + 8.9, title, size=9.1, weight="semibold")
            text(left + 14, bottom + 4.0, body, size=9.0)
        arrow((31, 35.9), (36, 35.9))
        arrow((64, 35.9), (69, 35.9))
        arrow((83, 30), (83, 22.3), color=TEAL)
        arrow((69, 16.4), (64, 16.4), color=ORANGE)
        arrow((36, 16.4), (31, 16.4), color=BLUE)
        arrow((17, 22.3), (17, 30), color=BLUE)
        text(50, 4.3, "Revise explanations and tests; expand or simplify when evidence warrants.",
             size=9.2, color=BLUE)

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in ("svg", "pdf", "png"):
            path = output_stem.with_suffix(f".{extension}")
            figure.savefig(path, dpi=260, bbox_inches="tight", pad_inches=.07)
            paths.append(path)
        plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output stem for SVG, PDF and PNG.")
    args = parser.parse_args()
    for path in draw_framework(args.output):
        print(path)


if __name__ == "__main__":
    main()
