"""Remake fig4_candidate_distribution.png properly.

The previous version claimed '2/11 passed' in the title but only plotted
ONE bar. It also didn't include the cross-LLM candidate (W078) that
also committed. Fix:
  - Show all 12 candidates (11 success-distilled + 1 cross-LLM)
  - Color green for KEEP (wr >= 0.60), red for REVERT
  - Both reference lines: 0.50 parity + 0.60 KEEP
  - Exact wr label above each bar
  - 95% Wilson CI error bars (n=50 each)
"""

import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import binomtest

# CJK-capable fonts available in matplotlib's font cache on this box
matplotlib.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

AUTO = Path(__file__).parent.parent / "phase four" / "autonomous"
FIGS = Path(__file__).parent / "figs"


def wilson(k, n):
    r = binomtest(k, n).proportion_ci(method="wilson")
    return r.low, r.high


def main():
    # Load 11 success-distilled
    val = json.loads((AUTO / "validation_log_parallel.json").read_text())
    rows = []
    for entry in val:
        for r in entry["results"]:
            ab = r["ab"]
            k = ab["wins_a"]  # wr_a is wr_ext (A = ext)
            n = ab["wins_a"] + ab["wins_b"]
            if n == 0:
                continue
            wr = k / n
            lo, hi = wilson(k, n)
            rows.append({
                "tid": r["tid"],
                "label": r["candidate"][:14],
                "wr": wr,
                "lo": lo, "hi": hi, "n": n,
                "decision": r["decision"],
            })

    # Note: val log already contains WCROSSL01 in the second entry (the
    # cross-LLM validation pass), so no manual append needed.

    # Sort by wr ascending so the KEEPs sit on the right
    rows.sort(key=lambda r: r["wr"])

    # Short English glosses for context
    gloss = {
        "上工治未病": "prevent disease",
        "别高效解决一个被看错": "don't solve misread",
        "凡事预则立": "plan ahead",
        "急则治其标": "urgent vs root",
        "覆水难收": "look forward",
        "亲兄弟，明算账": "clean accounts",
        "想理解行为": "incentives",
        "不谋全局者": "whole vs region",
        "若不是品牌": "brand vs commodity",
        "凡益之道": "change with times",
        "没有调查": "no investigation,",
        "是骡子是马": "mule or horse",
    }
    def short(r):
        raw = r["label"]
        for k in gloss:
            if raw.startswith(k):
                return f"{k}\n({gloss[k]})"
        return raw

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#27AE60" if r["decision"] == "KEEP" else "#C0392B"
              for r in rows]
    xs = list(range(len(rows)))
    wrs = [r["wr"] for r in rows]
    err_lo = [r["wr"] - r["lo"] for r in rows]
    err_hi = [r["hi"] - r["wr"] for r in rows]

    bars = ax.bar(xs, wrs, color=colors, edgecolor="#333", linewidth=0.8,
                    alpha=0.85, zorder=3)
    ax.errorbar(xs, wrs, yerr=[err_lo, err_hi], fmt="none",
                 ecolor="#333", capsize=3, elinewidth=1, zorder=4)

    # reference lines
    ax.axhline(0.50, color="#888", linestyle="--", linewidth=1, zorder=2,
                 label="parity (0.50)")
    ax.axhline(0.60, color="#16A085", linestyle="--", linewidth=1.5,
                 zorder=2, label="KEEP threshold (+10pp)")

    # annotations
    for i, r in enumerate(rows):
        ax.text(i, r["hi"] + 0.008, f"{r['wr']:.2f}",
                 ha="center", va="bottom", fontsize=8.5,
                 color="#333", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([short(r) for r in rows], fontsize=7.5,
                         rotation=40, ha="right")
    ax.set_ylabel("wr\\_ext vs base library (held-out 50)", fontsize=10)
    ax.set_ylim(0.30, 0.78)
    ax.set_title("All 12 candidates from inner-loop orchestrator, "
                  "single-family A/B at $n=50$\n"
                  "Green: KEEP (wr $\\geq$ 0.60). Red: REVERT. "
                  "Error bars: 95\\% Wilson CI.", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=1)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    # right margin note
    ax.text(1.01, 0.62, "KEEP", transform=ax.get_yaxis_transform(),
             color="#16A085", fontsize=9, fontweight="bold",
             va="center")
    ax.text(1.01, 0.50, "parity", transform=ax.get_yaxis_transform(),
             color="#888", fontsize=8, va="center")

    plt.tight_layout()
    out = FIGS / "fig4_candidate_distribution.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved → {out.name}  (n_rows = {len(rows)}, "
          f"KEEPs = {sum(1 for r in rows if r['decision']=='KEEP')})")


if __name__ == "__main__":
    main()
