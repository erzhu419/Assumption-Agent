"""Build the local Assumption Graph from existing project artifacts.

Usage:
  python -m assumption_os.build_graph
  python -m assumption_os.build_graph --out "phase four/assumption_graph"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import (
    ingest_artifacts,
    load_exp82_hypotheses,
    load_residual_nodes,
    load_strategy_nodes,
    load_wisdom_nodes,
)
from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .selector import MetaproductivitySelector


def build(root: Path, out: Path, *, include_exp82: bool = True, fresh: bool = False) -> dict:
    if fresh:
        out.mkdir(parents=True, exist_ok=True)
        for name in ("nodes.jsonl", "edges.jsonl", "evidence.jsonl", "trials.jsonl", "build_summary.json"):
            path = out / name
            if path.exists():
                path.unlink()
    store = JsonlGraphStore(out)
    artifact_groups = []

    strategies = root / "phase zero" / "kb" / "strategies"
    if strategies.exists():
        artifact_groups.append(load_strategy_nodes(strategies))

    wisdom = root / "phase two" / "analysis" / "cache" / "wisdom_library.json"
    if wisdom.exists():
        artifact_groups.append(load_wisdom_nodes(wisdom))

    residuals = root / "phase four" / "residuals" / "v16_residuals.json"
    if residuals.exists():
        artifact_groups.append(load_residual_nodes(residuals))

    exp82 = root / "phase six" / "exp82" / "hypotheses.jsonl"
    if include_exp82 and exp82.exists():
        artifact_groups.append(load_exp82_hypotheses(exp82))

    ingest_artifacts(store, artifact_groups)
    store.flush()

    graph = SimpleAssumptionGraph(store)
    selector = MetaproductivitySelector(graph)
    probe = selector.rank(
        "世界模型外推失败：一次性构建所有代码 vs 先在最简单场景替换一个核心模块",
        seeds=["incremental", "控制变量", "S15", "S01"],
        top_k=5,
    )
    summary = {
        "out": str(out),
        "nodes": len(store.nodes),
        "edges": len(store.edges),
        "evidence": len(store.evidence),
        "trials": len(store.trials),
        "probe_top": [s.to_dict() for s in probe],
    }
    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--out", default="phase four/assumption_graph", help="graph output dir")
    ap.add_argument("--no-exp82", action="store_true", help="skip phase six/exp82 hypotheses")
    ap.add_argument("--fresh", action="store_true", help="rebuild graph JSONL files from source artifacts")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    summary = build(root, out, include_exp82=not args.no_exp82, fresh=args.fresh)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
