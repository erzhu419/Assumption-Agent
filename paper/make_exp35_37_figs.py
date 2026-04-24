"""Matplotlib figures for Exp 35 (5-judge scale-up) + Exp 37 (kappa).

Two figures:
  fig:five-judge-forest  — forest plot of wr_ext with 95% Wilson CI
                           for W076/W077/W078 x {gemini, haiku, mini,
                           opus, gpt-5.4}. KEEP threshold at 0.60.
  fig:kappa-partition    — strip+box plot of pairwise kappa by
                           partition (within-cheap / within-exp /
                           cross-tier). Shows the 'cross-tier not
                           lower' finding.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

AUTO = Path(__file__).parent.parent / "phase four" / "autonomous"
FIGS = Path(__file__).parent / "figs"

# Palette: cheap cool, expensive warm, KEEP threshold deep teal
C_CHEAP = {"gemini": "#4A90E2", "claude_haiku": "#5DADE2", "gpt_mini": "#3498DB"}
C_EXP = {"claude_opus": "#E74C3C", "gpt5": "#C0392B"}
KEEP_COLOR = "#16A085"


def wilson_ci(k, n):
    if n == 0:
        return 0.5, 0.5
    r = binomtest(k, n).proportion_ci(method="wilson")
    return r.low, r.high


def counts_from_verdicts(v):
    ne = sum(1 for x in v.values() if x == "ext")
    nb = sum(1 for x in v.values() if x == "base")
    return ne, nb


# ==================== Fig A: 5-judge forest plot ====================

def fig_forest():
    exp35 = json.loads((AUTO / "exp35_expensive_extended_log.json").read_text())
    exp36 = json.loads((AUTO / "exp36_cheap_verdicts_log.json").read_text())

    # Build per-(cand, judge) -> (wr, lo, hi, n)
    merged = {}  # cand -> {judge: (wr, lo, hi, n)}
    for r in exp35["results"] + exp36["results"]:
        cid = r["cand_id"]
        merged.setdefault(cid, {})
        for fam, v in r["verdicts"].items():
            ne, nb = counts_from_verdicts(v)
            n = ne + nb
            wr = ne / n if n else 0.5
            lo, hi = wilson_ci(ne, n)
            merged[cid][fam] = (wr, lo, hi, n)

    keeps = ["W076", "W077", "W078"]
    judge_order = ["gemini", "claude_haiku", "gpt_mini", "claude_opus", "gpt5"]
    judge_label = {
        "gemini": "gemini-3-flash", "claude_haiku": "claude-haiku-4.5",
        "gpt_mini": "gpt-5.4-mini",
        "claude_opus": "claude-opus-4.6", "gpt5": "gpt-5.4",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, cid in zip(axes, keeps):
        ys = np.arange(len(judge_order))[::-1]
        for y, j in zip(ys, judge_order):
            if j not in merged.get(cid, {}):
                continue
            wr, lo, hi, n = merged[cid][j]
            color = C_CHEAP.get(j, C_EXP.get(j, "grey"))
            tier = "cheap" if j in C_CHEAP else "expensive"
            ax.errorbar([wr], [y], xerr=[[wr - lo], [hi - wr]],
                         fmt="o", color=color, ms=8, capsize=4,
                         elinewidth=1.8, markeredgewidth=0)
            ax.text(hi + 0.01, y, f"wr={wr:.2f} (n={n})",
                     va="center", fontsize=8, color="#555")
        # Reference lines
        ax.axvline(0.50, color="#888", linestyle="--", linewidth=1, alpha=0.6)
        ax.axvline(0.60, color=KEEP_COLOR, linestyle="-", linewidth=1.5,
                     alpha=0.8, label="KEEP (+10pp)")
        # cosmetics
        ax.set_xlim(0.20, 0.80)
        ax.set_yticks(ys)
        ax.set_yticklabels([judge_label[j] for j in judge_order], fontsize=9)
        ax.set_title(f"{cid}", fontsize=11, fontweight="bold")
        ax.set_xlabel("wr\\_ext")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.tick_params(axis="both", labelsize=8)
        # tier band shading (judge_order: 3 cheap on top rows y=4,3,2;
        # 2 expensive on bottom rows y=1,0)
        ax.axhspan(1.5, 4.5, facecolor="#E8F4FD", alpha=0.3, zorder=0)  # cheap
        ax.axhspan(-0.5, 1.5, facecolor="#FDEDEC", alpha=0.3, zorder=0) # exp

    # Add tier labels only on the leftmost subplot
    axes[0].text(0.22, 3.0, "CHEAP", fontsize=8, rotation=90,
                   va="center", color="#2980B9", fontweight="bold")
    axes[0].text(0.22, 0.5, "EXPENSIVE", fontsize=8, rotation=90,
                   va="center", color="#C0392B", fontweight="bold")

    axes[-1].legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("wr\\_ext under 5 judges across 3 KEEP candidates, "
                  "95\\% Wilson CI", fontsize=11, y=1.00)
    fig.tight_layout()
    out = FIGS / "exp35_five_judge_forest.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved → {out.name}")
    plt.close(fig)


# ==================== Fig B: kappa partition strip plot ====================

def fig_kappa():
    data = json.loads((AUTO / "exp37_kappa_cross_tier_log.json").read_text())

    partitions = [
        ("within_cheap", "within-cheap\n(3 judges, 3 pairs $\\times$ 3 candidates)", "#4A90E2"),
        ("within_exp", "within-expensive\n(2 judges, 1 pair $\\times$ 3 candidates)", "#E74C3C"),
        ("cross_tier", "cross-tier\n(cheap $\\times$ expensive, 6 pairs $\\times$ 3 candidates)", "#16A085"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_positions = np.arange(len(partitions))
    for i, (key, label, color) in enumerate(partitions):
        vals = data["partition_means"][key]["values"]
        if not vals:
            continue
        # Jitter
        rng = np.random.RandomState(42 + i)
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        xs = np.full(len(vals), i) + jitter
        ax.scatter(xs, vals, s=60, color=color, alpha=0.65,
                    edgecolor="white", linewidth=0.8, zorder=3)
        mean_k = np.mean(vals)
        # Mean bar
        ax.hlines(mean_k, i - 0.22, i + 0.22, color=color,
                    linewidth=3, zorder=4)
        # Annotate mean
        ax.text(i + 0.3, mean_k, f"mean $\\kappa$ = {mean_k:+.2f}",
                 va="center", fontsize=9, color=color, fontweight="bold")

    # Reference 0
    ax.axhline(0, color="#888", linestyle=":", linewidth=1)
    # Reference Landis-Koch bands
    for y, label, c in [(0.2, "'fair'", "#999"),
                          (0.4, "'moderate'", "#999"),
                          (0.6, "'substantial'", "#999")]:
        ax.axhline(y, color=c, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.text(len(partitions) - 0.5, y + 0.02, label,
                 fontsize=7, color="#777", ha="right")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([p[1] for p in partitions], fontsize=9)
    ax.set_ylabel("Cohen's $\\kappa$", fontsize=10)
    ax.set_ylim(-0.3, 0.8)
    ax.set_title("Pairwise judge agreement by partition\n"
                  "Cross-tier $\\kappa$ comparable to within-tier "
                  "$\\to$ stability assumption holds",
                  fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = FIGS / "exp37_kappa_partitions.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved → {out.name}")
    plt.close(fig)


if __name__ == "__main__":
    fig_forest()
    fig_kappa()
