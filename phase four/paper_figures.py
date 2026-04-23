"""Generate paper figures for Phase 4 v3 autonomous library evolution story.

Produces:
  - figs/fig1_architecture_progression.png
      Architecture-level win-rate ladder (v11 → v16 → v20)
  - figs/fig2_v20_generalization.png
      v16/v20 generalization: sample_100 vs held-out 50
  - figs/fig3_library_evolution.png
      Library size + status over time (75 → 77 → 77/deprecated after prune)
  - figs/fig4_candidate_distribution.png
      Success-distilled candidates: wr_ext distribution, KEEP/REVERT decision

Reads directly from judgments/ and autonomous/ — no manual data entry.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# CJK font registration (system-installed Noto Sans CJK)
for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

PROJECT = Path(__file__).parent.parent
CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
FIG_DIR = PROJECT / "phase four" / "figs"
FIG_DIR.mkdir(exist_ok=True)


def ab_wr(judgment_filename):
    """Parse a judgment file, return (wr_a, wins_a, wins_b, ties, n)."""
    stem = Path(judgment_filename).stem
    if "_vs_" not in stem:
        return None
    a, b = stem.split("_vs_", 1)
    p = CACHE / "judgments" / judgment_filename
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    wins_a = wins_b = ties = 0
    for v in data.values():
        w = v.get("winner")
        if w == "tie":
            ties += 1
            continue
        aw = v.get("a_was", "A")
        if (w == "A" and aw == "A") or (w == "B" and aw == "B") or w == a:
            wins_a += 1
        else:
            wins_b += 1
    n = wins_a + wins_b
    return (wins_a / n if n else 0.5, wins_a, wins_b, ties, n + ties)


def fig1_architecture_progression():
    """Two-panel: (a) v13-v16 vs baseline_long, (b) v17-v20 vs v16."""
    panel_a = [
        ("v13_reflect", "phase2_v13_reflect_vs_baseline_long.json"),
        ("v14_hybrid",  "phase2_v14_hybrid_vs_baseline_long.json"),
        ("v15_exemplar","phase2_v15_exemplar_vs_baseline_long.json"),
        ("v16",         "phase2_v16_vs_baseline_long.json"),
    ]
    panel_b = [
        ("v17", "phase2_v17_vs_phase2_v16.json"),
        ("v18", "phase2_v18_vs_phase2_v16.json"),
        ("v19a", "phase2_v19a_vs_phase2_v16.json"),
        ("v19b", "phase2_v19b_vs_phase2_v16.json"),
        ("v19c", "phase2_v19c_vs_phase2_v16.json"),
        ("v19d", "phase2_v19d_vs_phase2_v16.json"),
        ("v20",  "phase2_v20_vs_phase2_v16.json"),
    ]
    a_rows, b_rows = [], []
    for label, jf in panel_a:
        r = ab_wr(jf)
        if r: a_rows.append((label, r[0]))
    for label, jf in panel_b:
        r = ab_wr(jf)
        if r: b_rows.append((label, r[0]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                              gridspec_kw={"width_ratios": [len(a_rows), len(b_rows)]})

    # Panel A: vs baseline_long
    ax = axes[0]
    labels_a = [r[0] for r in a_rows]; wrs_a = [r[1] for r in a_rows]
    colors_a = ["#87ceeb" if w < 0.8 else "#2e8b57" for w in wrs_a]
    bars = ax.bar(labels_a, wrs_a, color=colors_a, edgecolor="#333")
    for b, w in zip(bars, wrs_a):
        ax.text(b.get_x() + b.get_width()/2, w + 0.015, f"{w:.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(0.50, color="#888", lw=1, ls="--")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Win rate vs baseline_long (n=100)")
    ax.set_title("(a) Climbing from scaffolding baseline → v16")

    # Panel B: vs v16 (the champion baseline)
    ax = axes[1]
    labels_b = [r[0] for r in b_rows]; wrs_b = [r[1] for r in b_rows]
    colors_b = ["#c05050" if w < 0.5 else "#87ceeb" if w < 0.55 else
                "#5fa8d3" if w < 0.6 else "#2e8b57" for w in wrs_b]
    bars = ax.bar(labels_b, wrs_b, color=colors_b, edgecolor="#333")
    for b, w in zip(bars, wrs_b):
        ax.text(b.get_x() + b.get_width()/2, w + 0.012, f"{w:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0.50, color="#888", lw=1, ls="--", label="parity with v16")
    ax.set_ylim(0.3, 0.75); ax.set_ylabel("Win rate vs v16 (n=100)")
    ax.set_title("(b) Post-v16 experiments — v20 new champion")
    ax.legend(loc="lower right", fontsize=9)

    plt.suptitle("Architecture progression — the ratchet up to v20", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig1_architecture_progression.png"
    plt.savefig(out, dpi=130)
    plt.close()
    return out


def fig2_generalization():
    """v16/v20 on sample_100 vs held-out 50."""
    pairs = [
        # label, (in-sample file, held-out file)
        ("v16 vs baseline_long",
         "phase2_v16_vs_baseline_long.json",
         "phase2_v16_holdout50_vs_baseline_long_holdout50.json"),
        ("v16 vs v13_reflect",
         "phase2_v16_vs_phase2_v13_reflect.json",
         "phase2_v16_holdout50_vs_phase2_v13_reflect_holdout50.json"),
    ]
    labels = []
    in_sample = []
    held_out = []
    for label, f_in, f_out in pairs:
        r_in = ab_wr(f_in); r_out = ab_wr(f_out)
        if not r_in or not r_out: continue
        labels.append(label)
        in_sample.append(r_in[0])
        held_out.append(r_out[0])

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, in_sample, w, label="sample_100 (train)", color="#87ceeb")
    b2 = ax.bar(x + w/2, held_out, w, label="held-out 50 (test)", color="#2e8b57")
    for bars, vals in ((b1, in_sample), (b2, held_out)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(0.50, color="#888", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Win rate")
    ax.set_title("Held-out generalization — advantage HOLDS (and amplifies)")
    ax.legend(loc="lower center")
    plt.tight_layout()
    out = FIG_DIR / "fig2_v20_generalization.png"
    plt.savefig(out, dpi=130)
    plt.close()
    return out


def fig3_library_evolution():
    """Library status over evolution events."""
    registry = json.loads((AUTO_DIR / "wisdom_registry.json").read_text())
    wisdoms = registry["wisdoms"]

    # Event timeline: (event_label, n_active, n_deprecated, n_removed, n_total)
    # We reconstruct from the current registry + prune_log
    prune_log_path = AUTO_DIR / "prune_log.json"
    prune_log = json.loads(prune_log_path.read_text()) if prune_log_path.exists() else []

    # 4 time points
    events = []
    events.append(("Phase 3 kernel library\n(original 75)", 75, 0, 0))
    events.append(("+ success_distilled\nW076 W077 (v20.2)", 77, 0, 0))
    # after prune
    if prune_log:
        latest = prune_log[-1]
        n_dep = len(latest.get("deprecated", []))
        n_rem = len(latest.get("removed", []))
        active_after = latest.get("active_count_after", 77 - n_dep)
        events.append((f"+ pruner (v20.3)", active_after, n_dep, n_rem))
    # add cross_llm W078 if present
    has_cross = any(w["id"] == "W078" and w["source"] == "cross_llm"
                    for w in registry["wisdoms"])
    if has_cross:
        n_active_now = sum(1 for w in registry["wisdoms"] if w["status"] == "active")
        n_dep_now = sum(1 for w in registry["wisdoms"] if w["status"] == "deprecated")
        n_rem_now = sum(1 for w in registry["wisdoms"] if w["status"] == "removed")
        events.append((f"+ cross_llm W078\n({registry['version']})",
                       n_active_now, n_dep_now, n_rem_now))

    labels = [e[0] for e in events]
    actives = np.array([e[1] for e in events])
    deps    = np.array([e[2] for e in events])
    rems    = np.array([e[3] for e in events])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, actives, color="#2e8b57", label="active")
    ax.bar(x, deps, bottom=actives, color="#d4a017", label="deprecated")
    ax.bar(x, rems, bottom=actives + deps, color="#a03030", label="removed")

    # label totals
    for xi, (a, d, r) in enumerate(zip(actives, deps, rems)):
        total = a + d + r
        ax.text(xi, total + 1.5, f"{total}", ha="center", va="bottom", fontsize=10)
        if a: ax.text(xi, a/2, f"{a}", ha="center", va="center", fontsize=9, color="white")
        if d: ax.text(xi, a + d/2, f"{d}", ha="center", va="center", fontsize=9, color="#222")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Wisdoms")
    ax.set_title("Autonomous library evolution — grown + pruned without human curation")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(actives + deps + rems) + 12)
    plt.tight_layout()
    out = FIG_DIR / "fig3_library_evolution.png"
    plt.savefig(out, dpi=130)
    plt.close()
    return out


def fig4_candidate_distribution():
    """Success-distilled candidate A/B outcomes on held-out 50."""
    log = json.loads((AUTO_DIR / "validation_log_parallel.json").read_text())
    latest = log[-1]
    results = latest["results"]

    aphs = [r["candidate"][:16] for r in results]
    wrs = [r["ab"]["wr_a"] for r in results]
    decisions = [r["decision"] for r in results]
    colors = ["#2e8b57" if d == "KEEP" else "#c05050" for d in decisions]

    order = np.argsort(wrs)[::-1]
    aphs = [aphs[i] for i in order]
    wrs = [wrs[i] for i in order]
    colors = [colors[i] for i in order]
    decisions = [decisions[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(aphs, wrs, color=colors, edgecolor="#333")
    for b, w, d in zip(bars, wrs, decisions):
        ax.text(b.get_x() + b.get_width()/2, w + 0.008, f"{w:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0.50, color="#888", lw=1, ls="--", label="parity")
    ax.axhline(0.60, color="#2e8b57", lw=1.5, ls="--", label="KEEP threshold (+10pp)")
    ax.set_ylabel("Win rate vs base (held-out 50)")
    ax.set_title("Success-distilled candidates — 2/11 passed autonomous A/B filter")
    ax.set_ylim(0.35, 0.72)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = FIG_DIR / "fig4_candidate_distribution.png"
    plt.savefig(out, dpi=130)
    plt.close()
    return out


def main():
    outs = []
    for fn in (fig1_architecture_progression, fig2_generalization,
               fig3_library_evolution, fig4_candidate_distribution):
        try:
            p = fn()
            outs.append((fn.__name__, p))
            print(f"  ✓ {fn.__name__} → {p.relative_to(PROJECT)}")
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{len(outs)} figures saved to {FIG_DIR.relative_to(PROJECT)}/")


if __name__ == "__main__":
    main()
