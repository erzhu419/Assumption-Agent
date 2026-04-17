"""
Strategy Execution Graph (SEG): directed graph over operational steps.

Captures the TIME-ORDERED structure that a Markov kernel flattens away.
Two strategies with identical step compositions but different orderings will
have identical kernels but very different SEG distances.

Node: one step, annotated with its abstract action category (see ACTION_SPACE).
Edge: sequential step transition, or a recursive call to another strategy
      derived from the step's `on_difficulty` hint.

Distance: normalized graph edit distance (networkx), with substitution costs
based on action-category mismatch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import _config as cfg
from formalization.core.kernel_builder import classify_step_text


@dataclass
class SEGNode:
    step_id: int
    action_cat: str
    description: str
    recursive_target: Optional[str] = None


@dataclass
class SEG:
    strategy_id: str
    nodes: List[SEGNode] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)  # (from_id, to_id, kind)


def build_seg(strategy: Dict) -> SEG:
    sid = strategy["id"]
    nodes: List[SEGNode] = []
    edges: List[tuple] = []
    prev_id: Optional[int] = None
    for step in strategy.get("operational_steps", []):
        step_id = int(step.get("step", len(nodes) + 1))
        action_text = step.get("action", "") or ""
        on_diff = step.get("on_difficulty") or ""
        cat = classify_step_text(action_text)
        recursive_target = None
        if on_diff:
            # Try to extract a strategy reference like "S07" from on_difficulty text.
            import re
            m = re.search(r"S\d{2}", on_diff)
            if m:
                recursive_target = m.group(0)
        nodes.append(SEGNode(
            step_id=step_id,
            action_cat=cat,
            description=action_text[:80],
            recursive_target=recursive_target,
        ))
        if prev_id is not None:
            edges.append((prev_id, step_id, "sequential"))
        if recursive_target:
            edges.append((step_id, -1, f"recursive:{recursive_target}"))
        prev_id = step_id
    return SEG(strategy_id=sid, nodes=nodes, edges=edges)


def seg_distance(seg_a: SEG, seg_b: SEG) -> float:
    """Normalized graph edit distance between two SEGs.

    Uses networkx's greedy GED (exact is NP-hard but our graphs are tiny).
    """
    import networkx as nx
    G_a = nx.DiGraph()
    G_b = nx.DiGraph()
    for n in seg_a.nodes:
        G_a.add_node(n.step_id, action=n.action_cat)
    for f, t, k in seg_a.edges:
        G_a.add_edge(f, t, kind=k)
    for n in seg_b.nodes:
        G_b.add_node(n.step_id, action=n.action_cat)
    for f, t, k in seg_b.edges:
        G_b.add_edge(f, t, kind=k)

    def node_sub(n1, n2):
        return 0.0 if n1.get("action") == n2.get("action") else 1.0

    def edge_sub(e1, e2):
        return 0.0 if e1.get("kind") == e2.get("kind") else 0.3

    try:
        ged = nx.graph_edit_distance(
            G_a, G_b,
            node_subst_cost=node_sub,
            edge_subst_cost=edge_sub,
            timeout=3.0,
        )
    except Exception:
        return 1.0

    max_size = max(len(seg_a.nodes), len(seg_b.nodes), 1)
    if ged is None:
        return 1.0
    return float(ged) / max_size
