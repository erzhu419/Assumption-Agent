"""Reproduce two actual M3 source models that are misclassified by AST checking."""
import json
from pathlib import Path

from hegel_machine.cognition.m3_scale_universe import make_m3_large_scale_suite, solve_m3_workspace_direct
from hegel_machine.cognition.open_workspace_wormhole import canonicalize_python_model_ast
import hegel_machine.cognition.m3_scale_universe as m3

ROOT = Path(__file__).resolve().parent
REPO = Path(m3.__file__).resolve().parents[3]


def main():
    graph = json.loads((REPO / "docs/evidence/v17_0_m3_hipporag_wormhole_scale_20261119/wormhole_forest_graph.json").read_text())
    mismatch_nodes = []
    for nid, node in graph["nodes"].items():
        conclusion = node["claim_contract"]["conclusion_or_action"]
        if str(conclusion.get("ast_skeleton_digest", "")).startswith("mismatch:"):
            mismatch_nodes.append({"node_id": nid, **conclusion, "evidence_status": node["claim_contract"]["evidence_status"]})
    suite = make_m3_large_scale_suite(7027404)
    source_results = []
    for name, (factory, evaluate) in suite["sources"][:2]:
        lab = factory()
        ws = ROOT / "m3_ast_source_workspaces" / name
        sub = solve_m3_workspace_direct(lab, ws)
        ev = evaluate(ws, sub, lab.observations)
        code = (ws / sub["model_path"]).read_text()
        meta = canonicalize_python_model_ast(code)
        source_results.append({"episode": name, "submission": sub, "evaluation": ev,
                               "model_source": code, "canonicalization": meta})
    result = {"seed": 7027404, "graph_node_count": len(graph["nodes"]),
              "mismatch_nodes": mismatch_nodes, "actual_source_reproductions": source_results}
    (ROOT / "m3_ast_rejection_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"node_count": len(graph["nodes"]), "mismatch_count": len(mismatch_nodes),
                     "reproductions": [{"episode": r["episode"], "pass": r["evaluation"]["strict_pass"],
                                        "digest": r["canonicalization"]["ast_skeleton_digest"],
                                        "status": r["canonicalization"]["skeleton_status"],
                                        "mismatch_detail": r["canonicalization"]["canonical_ast_dump"]}
                                       for r in source_results]}, indent=2))


if __name__ == "__main__":
    main()
