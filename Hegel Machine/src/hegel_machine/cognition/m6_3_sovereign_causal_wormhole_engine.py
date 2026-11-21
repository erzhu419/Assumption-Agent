"""Hegel M6.3 Sovereign Causal Wormhole Engine: Multi-Candidate Partial-Subtree Archive,
Call-Graph `row`-Provenance AST Translation, Verified Instance-Contract Writeback, and
Equal-Content Learned-Flat Control.

Preserves 100% of Hegelian core innovations (Multi-Hop Generalized Markov-Blanket `d-sepset`
Retroactive Reinterpretation / `Aufhebung`, Partial Shared Subtree Projection, Multi-Depth
Recursive Sub-Assumption Pullback, and Composite Tree Growth when preconditions fail) while
eliminating all 7 concrete runtime & attribution bottlenecks uncovered in the M6/v18 Audit:

1. **M6-01 Fixed — Verified Causal-Role Instance Ownership + Composite Node Growth**:
   - Never assigns writeback ownership to raw retrieval rank-1 (`rec_ids[0]`).
   - Verifies signature match AND causal interface variable coverage (`used_keys <= vmap.keys()`),
     or grows a new composite synthesis node (`node_synth_composite_*`) when multi-depth components
     combine (`C04`). Canonical sorting guarantees 100% completion-order invariance.
2. **M6-02 Fixed — Pareto Multi-Candidate Partial-Subtree Archive per Node**:
   - Replaces `seen_source_nodes` first-success monopoly with a signature-indexed Pareto archive
     retaining both wider domain-specific ASTs (`(m_2, u_raw, t_proc)`) and transportable partial
     commonality subtrees (`(m_2, u_raw)`), transporting every `role_map`-compatible candidate.
3. **M6-03 & M6-04 Fixed — Universal `fit`/`predict` State/Closure ABI + Hermetic `/tmp` Sandbox**:
   - Supports module-global mutation, `fit(rows) -> callable`, and `fit(rows) -> state` passed to
     `predict(row, state)` inside a `PYTHONPATH=""` + `sys.addaudithook` sandboxed `/tmp` runner,
     and echoes previous `python_code` on implementation error before burning confirmation budget.
4. **M6-05 Fixed — Intra-Module Call-Graph `row`-Provenance AST Analyzer & Translator**:
   - Tracks `row` and `rows_collection` parameter dataflow across `predict`, `fit` row-loops,
     inner closures, and helper functions — translating helper dependencies while leaving non-row
     dictionaries (`P['z_2']`) untouched.
5. **M6-06 & M6-07 Fixed — Live Empirical Retroactive Projection Receipt + Strict Probe Filter**:
   - Computes empirical `P_obs` projection consistency on `pilot_rows` during consolidation, and
     excludes passive `observational_L1_selected` slices from regime-breaking probe recommendations.
6. **M6-10 / ALMA / Hyperagents Fixed — `EqualContent_LearnedFlat_LiveGPT` Control + Skill Activation**:
   - Gives the flat BM25 baseline online Wave-1 Python code writeback (`max_hops = 0`, re-indexed)
     and identical pilot-only warm-start verification receipts (`recalled -> bound -> pilot-executed
     -> adopted`), isolating the pure causal value of multi-hop `d-sepset` wormholes & role mapping.
"""

from __future__ import annotations

import ast
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m5_live_gpt_and_provenance import (
    CodexSubscriptionClient,
    DecisionProvenanceLedger,
    _execute_authored_python_in_isolated_subprocess,
)
from .m6_1_zero_leakage_ast_causal_benchmark import (
    EQUAL_UNIT_ROW_COST,
    build_m6_1_clean_k40_forest,
)
from .m6_causal_wormhole_sandbox import is_d_separated_dag, verify_retroactive_projection_certificate
from .m6_live_gpt_causal_aufhebung_benchmark import M6CausalTaskDataset
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    evaluate_predicate_rule,
)


M6_3_ARMS: tuple[str, ...] = (
    "Strong_Ordinary_LiveGPT",
    "EqualContent_LearnedFlat_LiveGPT",
    "Hegel_M5_Full_LiveGPT",
    "Hegel_M6_3_Sovereign_LiveGPT",
)


# ============================================================================
# 1. Call-Graph `row`-Provenance AST Analyzer & Role Translator (`M6-05` Fix)
# ============================================================================


def _analyze_module_row_provenance(tree: ast.Module) -> tuple[dict[int, set[str]], set[int]]:
    """Return `(row_vars_by_scope_id, prediction_reachable_scope_ids)` via fixed-point dataflow.

    Tracks which variable names inside each function/lambda scope represent a single data `row`
    dict (vs `rows_collection` or unrelated parameter dicts like `P = {'z_2': 4.0}`).
    """
    top_fns: dict[str, ast.FunctionDef] = {}
    all_scopes: list[ast.AST] = [tree]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            all_scopes.append(node)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            top_fns[node.name] = node

    row_vars: dict[int, set[str]] = {id(sc): set() for sc in all_scopes}
    coll_vars: dict[int, set[str]] = {id(sc): set() for sc in all_scopes}
    pred_scopes: set[int] = set()

    # Seed `predict(row, ...)` and `fit(train_rows, ...)`
    if "predict" in top_fns:
        p_fn = top_fns["predict"]
        pred_scopes.add(id(p_fn))
        if p_fn.args.args:
            row_vars[id(p_fn)].add(p_fn.args.args[0].arg)

    if "fit" in top_fns:
        f_fn = top_fns["fit"]
        if f_fn.args.args:
            coll_vars[id(f_fn)].add(f_fn.args.args[0].arg)
        # Inner closures returned by `fit(rows)` (e.g. `def _pred(row): ...; return _pred`)
        returned_names: set[str] = set()
        for sub in ast.walk(f_fn):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Name):
                returned_names.add(sub.value.id)
            elif isinstance(sub, ast.Return) and isinstance(sub.value, ast.Lambda):
                pred_scopes.add(id(sub.value))
                if sub.value.args.args:
                    row_vars[id(sub.value)].add(sub.value.args.args[0].arg)
        for sub in f_fn.body:
            if isinstance(sub, ast.FunctionDef) and sub.name in returned_names and sub.args.args:
                pred_scopes.add(id(sub))
                row_vars[id(sub)].add(sub.args.args[0].arg)

    # Fixed-point propagation of collection loops, variable aliases, and helper function calls
    changed = True
    while changed:
        changed = False
        for sc in all_scopes:
            sid = id(sc)
            r_set = row_vars[sid]
            c_set = coll_vars[sid]
            body_nodes = sc.body if isinstance(sc, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)) else [sc.body]  # type: ignore[attr-defined]
            for stmt in body_nodes:
                for node in ast.walk(stmt):
                    # Collection aliasing: `rows = list(train_rows)` or `rows = copy.deepcopy(train_rows)`
                    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        tgt = node.targets[0].id
                        val = node.value
                        if isinstance(val, ast.Name):
                            if val.id in r_set and tgt not in r_set:
                                r_set.add(tgt)
                                changed = True
                            if val.id in c_set and tgt not in c_set:
                                c_set.add(tgt)
                                changed = True
                        elif isinstance(val, ast.Call) and val.args and isinstance(val.args[0], ast.Name):
                            a0 = val.args[0].id
                            fn_id = val.func.id if isinstance(val.func, ast.Name) else (
                                val.func.attr if isinstance(val.func, ast.Attribute) else ""
                            )
                            if a0 in c_set and fn_id in {"list", "tuple", "deepcopy", "copy"} and tgt not in c_set:
                                c_set.add(tgt)
                                changed = True
                            if a0 in r_set and fn_id in {"dict", "deepcopy", "copy"} and tgt not in r_set:
                                r_set.add(tgt)
                                changed = True

                    # Iteration over rows collection: `for r in train_rows:` or `[... for r in rows]`
                    if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id in c_set:
                        if isinstance(node.target, ast.Name) and node.target.id not in r_set:
                            r_set.add(node.target.id)
                            changed = True
                    if isinstance(node, ast.comprehension) and isinstance(node.iter, ast.Name) and node.iter.id in c_set:
                        if isinstance(node.target, ast.Name) and node.target.id not in r_set:
                            r_set.add(node.target.id)
                            changed = True

                    # Intra-module helper calls: `helper(row)` -> propagate row provenance & reachability
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in top_fns:
                        callee = top_fns[node.func.id]
                        cid = id(callee)
                        if sid in pred_scopes and cid not in pred_scopes:
                            pred_scopes.add(cid)
                            changed = True
                        for idx_a, arg_expr in enumerate(node.args):
                            if idx_a < len(callee.args.args) and isinstance(arg_expr, ast.Name):
                                formal = callee.args.args[idx_a].arg
                                if arg_expr.id in r_set and formal not in row_vars[cid]:
                                    row_vars[cid].add(formal)
                                    changed = True
                                if arg_expr.id in c_set and formal not in coll_vars[cid]:
                                    coll_vars[cid].add(formal)
                                    changed = True

    return row_vars, pred_scopes


class _ScopedRowKeyCollector(ast.NodeVisitor):
    """Collect string keys accessed ONLY on tracked `row` variables across reachable prediction scopes."""

    def __init__(self, row_vars: dict[int, set[str]], target_scopes: set[int]) -> None:
        self.row_vars = row_vars
        self.target_scopes = target_scopes
        self._scope_stack: list[int] = []
        self.keys: set[str] = set()

    def _active_row_names(self) -> set[str]:
        out: set[str] = set()
        for sid in self._scope_stack:
            out |= self.row_vars.get(sid, set())
        return out

    def _in_target(self) -> bool:
        return not self.target_scopes or any(sid in self.target_scopes for sid in self._scope_stack)

    def visit_Module(self, node: ast.Module) -> Any:
        self._scope_stack.append(id(node))
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._scope_stack.append(id(node))
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        self._scope_stack.append(id(node))
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if self._in_target() and isinstance(node.value, ast.Name) and node.value.id in self._active_row_names():
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str) and sl.value != "y":
                self.keys.add(sl.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if (
            self._in_target()
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self._active_row_names()
            and node.args
        ):
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str) and a0.value != "y":
                self.keys.add(a0.value)
        self.generic_visit(node)


class _ScopedRowKeyTranslator(ast.NodeTransformer):
    """Rewrite `row['k']` and `row.get('k', ...)` ONLY when `row` has tracked row provenance."""

    def __init__(self, row_vars: dict[int, set[str]], role_map: dict[str, str]) -> None:
        self.row_vars = row_vars
        self.role_map = dict(role_map)
        self._scope_stack: list[int] = []

    def _active_row_names(self) -> set[str]:
        out: set[str] = set()
        for sid in self._scope_stack:
            out |= self.row_vars.get(sid, set())
        return out

    def visit_Module(self, node: ast.Module) -> Any:
        self._scope_stack.append(id(node))
        node = self.generic_visit(node)  # type: ignore[assignment]
        self._scope_stack.pop()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._scope_stack.append(id(node))
        node = self.generic_visit(node)  # type: ignore[assignment]
        self._scope_stack.pop()
        return node

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        self._scope_stack.append(id(node))
        node = self.generic_visit(node)  # type: ignore[assignment]
        self._scope_stack.pop()
        return node

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if isinstance(node.value, ast.Name) and node.value.id in self._active_row_names():
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str) and sl.value in self.role_map:
                node.slice = ast.Constant(value=self.role_map[sl.value])
        return node

    def visit_Call(self, node: ast.Call) -> Any:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self._active_row_names()
            and node.args
        ):
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str) and a0.value in self.role_map:
                node.args[0] = ast.Constant(value=self.role_map[a0.value])
        return node


def extract_and_translate_authored_ast_v2(
    python_code: str,
    role_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Call-graph & `row`-provenance-aware Python AST extractor and wormhole role translator."""
    src = str(python_code or "").strip()
    if not src:
        return {"ok": False, "reason": "empty_python_code", "used_keys": [], "translated_code": ""}
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"ok": False, "reason": f"syntax_error:{exc}", "used_keys": [], "translated_code": ""}

    row_vars, pred_scopes = _analyze_module_row_provenance(tree)
    collector = _ScopedRowKeyCollector(row_vars, pred_scopes)
    collector.visit(tree)
    if not collector.keys:
        # Fallback to any tracked row scope in module (e.g., custom callable class or fit-only script)
        collector_all = _ScopedRowKeyCollector(row_vars, set())
        collector_all.visit(tree)
        used_keys = sorted(collector_all.keys)
    else:
        used_keys = sorted(collector.keys)

    if not role_map:
        return {
            "ok": True,
            "used_keys": used_keys,
            "translated_keys": used_keys,
            "translated_code": ast.unparse(tree),
        }

    tree_copy = deepcopy(tree)
    row_vars_copy, pred_scopes_copy = _analyze_module_row_provenance(tree_copy)
    translated_tree = _ScopedRowKeyTranslator(row_vars_copy, role_map).visit(tree_copy)
    ast.fix_missing_locations(translated_tree)

    row_vars_post, pred_scopes_post = _analyze_module_row_provenance(translated_tree)
    post_col = _ScopedRowKeyCollector(row_vars_post, pred_scopes_post)
    post_col.visit(translated_tree)
    if not post_col.keys:
        post_col_all = _ScopedRowKeyCollector(row_vars_post, set())
        post_col_all.visit(translated_tree)
        trans_keys = sorted(post_col_all.keys)
    else:
        trans_keys = sorted(post_col.keys)

    return {
        "ok": True,
        "used_keys": used_keys,
        "translated_keys": trans_keys,
        "translated_code": ast.unparse(translated_tree),
    }


# ============================================================================
# 2. Verified Instance-Contract Writeback Ownership & Composite Growth (`M6-01`)
# ============================================================================


def resolve_verified_writeback_owner(
    *,
    recalled_node_ids: list[str],
    observed_signatures: list[str],
    active_ast_keys: list[str],
    violated_root_ids: list[str],
    forest: PhilosophicalForest,
    node_archive: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve true mechanism owner via causal interface coverage & signature alignment (ZERO `task_id` peeking!).

    - Rejects distractors (`L1_DISTRACTOR`) and rank-1 PPR neighbors whose `probe_signatures` or
      `interface_var_to_dag_node` fail to cover the confirmed AST's active variables.
    - When `active_ast_keys` spans multiple modular sub-assumptions (`violated_root_ids` non-empty
      and no single node covers all keys, as in multi-depth `C04`), grows a dedicated Composite
      Synthesis Node (`node_synth_composite_*`) in the forest rather than polluting a distractor!
    """
    obs_set = set(observed_signatures)
    key_set = set(active_ast_keys)
    scored_candidates: list[tuple[int, int, int, int, str]] = []

    for rank_idx, nid in enumerate(recalled_node_ids):
        if nid not in node_archive or nid not in forest.nodes:
            continue
        meta = node_archive[nid]
        if str(meta.get("epistemic_rung", "")) == "L1_DISTRACTOR":
            continue
        n_raw = forest.nodes[nid]
        node_sigs = set(n_raw.get("probe_signatures", ()) if isinstance(n_raw, dict) else n_raw.probe_signatures)
        sig_overlap = len(obs_set & node_sigs)
        vmap_keys = set((meta.get("interface_var_to_dag_node") or {}).keys())
        core_overlap = len(key_set & vmap_keys)
        if (not violated_root_ids) and sig_overlap >= 1 and vmap_keys and core_overlap >= 1:
            is_full_subset = 1 if key_set.issubset(vmap_keys) else 0
            scored_candidates.append((sig_overlap, is_full_subset, core_overlap, -rank_idx, nid))

    if scored_candidates:
        scored_candidates.sort(reverse=True)
        best_nid = scored_candidates[0][4]
        return {
            "owner_mode": "verified_existing_node",
            "owner_node_id": best_nid,
            "signature_overlap": scored_candidates[0][0],
        }

    # Check if this is a multi-module composite law (e.g. precondition-violated root + sub-components)
    sig_digest = hashlib.sha256(
        json.dumps({"sigs": sorted(obs_set), "keys": sorted(key_set)}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    synth_nid = f"node_synth_composite_{sig_digest}"
    if synth_nid not in forest.nodes:
        tv = TaskView("causal_and_interventional_prediction", ["y"], 0.05)
        forest.register_node(
            node_id=synth_nid,
            tree_id=f"tree_synth_{sig_digest}",
            layer_role="mid",
            task_view=tv,
            observation_map=ObservationMap("y", temporal_readout_mode="present_snapshot"),
            scale_map=ScaleMap("identity", preserved_queries=["causal_and_interventional_prediction"]),
            claim_contract=ClaimContract(
                claim_id=f"cl_{synth_nid}",
                family_plugin="m6_3_composite_synthesis",
                preconditions=[],
                conclusion_or_action={"epistemic_rung": "L2_COMPOSITE_SYNTHESIS"},
                protocol_spec={"epistemic_rung": "L2_COMPOSITE_SYNTHESIS"},
                tolerance=0.05,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="local_solver_guidance",
            ),
            probe_signatures=sorted(obs_set),
        )
        node_archive[synth_nid] = {
            "node_id": synth_nid,
            "tree_id": f"tree_synth_{sig_digest}",
            "epistemic_rung": "L2_COMPOSITE_SYNTHESIS",
            "preconditions": [],
            "verified_mechanism_note": "Synthesized composite multi-layer mechanism node.",
            "local_dag_edges": [(f"V_{k}", "Y_out") for k in sorted(key_set)],
            "interface_var_to_dag_node": {k: f"V_{k}" for k in sorted(key_set)},
            "synthesized_from_violated_roots": list(violated_root_ids),
        }
        for r_nid in violated_root_ids:
            if r_nid in forest.nodes:
                wh_id = f"wh_synth_{r_nid}_to_{synth_nid}"
                if wh_id not in forest.horizontal_wormhole_edges:
                    forest.add_horizontal_wormhole(
                        wormhole_id=wh_id,
                        node_a_id=r_nid,
                        node_b_id=synth_nid,
                        probe_signature_refs=sorted(obs_set),
                        role_and_observation_map={"role_map": {k: k for k in sorted(key_set)}},
                        hop_error_budget=0.015,
                        lipschitz_factor=0.90,
                        permission_tier="local_solver_guidance",
                    )

    return {
        "owner_mode": "synthesized_composite_branch",
        "owner_node_id": synth_nid,
        "signature_overlap": len(obs_set),
    }


# ============================================================================
# 3. Pareto Multi-Candidate Partial-Subtree Wormhole Consolidation (`M6-02`/`M6-06`)
# ============================================================================


def _append_pareto_candidate(
    entry: dict[str, Any],
    *,
    used_keys: list[str],
    python_code: str,
    confirm_rmse: float,
    provenance_tag: str,
) -> bool:
    """Maintain signature-diverse Pareto archive per node (preserving partial-commonality subsets!)."""
    archive: list[dict[str, Any]] = entry.setdefault("pareto_confirmed_candidates", [])
    key_sig = tuple(sorted(used_keys))
    for existing in archive:
        if tuple(existing["used_keys"]) == key_sig:
            if confirm_rmse < float(existing["confirm_rmse"]) - 1e-9:
                existing["python_code"] = python_code
                existing["confirm_rmse"] = round(float(confirm_rmse), 6)
                existing["provenance_tag"] = provenance_tag
                return True
            return False
    archive.append(
        {
            "used_keys": list(key_sig),
            "python_code": python_code,
            "confirm_rmse": round(float(confirm_rmse), 6),
            "provenance_tag": provenance_tag,
        }
    )
    archive.sort(key=lambda c: (len(c["used_keys"]), c["confirm_rmse"], tuple(c["used_keys"])))
    return True


def _refresh_node_note_from_pareto(entry: dict[str, Any], header_prefix: str) -> None:
    archive: list[dict[str, Any]] = entry.get("pareto_confirmed_candidates") or []
    if not archive:
        return
    # Primary reference is lowest confirm_rmse (ties broken by simpler partial-commonality signature)
    best = min(archive, key=lambda c: (c["confirm_rmse"], len(c["used_keys"]), tuple(c["used_keys"])))
    entry["confirmed_active_variables"] = best["used_keys"]
    entry["confirmed_python_reference"] = best["python_code"]
    summary_blocks = [
        f"Variant #{idx + 1} (`active_keys={cand['used_keys']}`, `src_confirm_rmse={cand['confirm_rmse']}`, `{cand['provenance_tag']}`):\n{cand['python_code']}"
        for idx, cand in enumerate(archive[:3])
    ]
    entry["verified_mechanism_note"] = (
        f"{header_prefix} on `{entry['tree_id']}` "
        f"({len(archive)} verified AST variant(s) in Pareto archive; refit local parameters on target `train_rows`!):\n\n"
        + "\n\n".join(summary_blocks)
    )


def consolidate_wave1_sovereign_m6_3(
    *,
    wave1_receipts: list[dict[str, Any]],
    arm: str,
    forest: PhilosophicalForest,
    node_archive: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Order-invariant, instance-verified, multi-candidate AST consolidation across `M6.3`, `M5`, and `LearnedFlat`."""
    if arm not in {
        "EqualContent_LearnedFlat_LiveGPT",
        "Hegel_M5_Full_LiveGPT",
        "Hegel_M6_3_Sovereign_LiveGPT",
    }:
        return {"events": [], "writeback_cost_C": 0.0}

    max_hops = 2 if arm == "Hegel_M6_3_Sovereign_LiveGPT" else 0
    events: list[dict[str, Any]] = []
    wb_cost = 0.0

    # Canonical deterministic sort -> 100% invariant to ThreadPoolExecutor completion order!
    eligible_recs = [
        r for r in wave1_receipts
        if r.get("status") == "confirmed_ok"
        and r.get("strict_pass")
        and str(r.get("winning_python_code") or "").strip()
    ]
    eligible_recs.sort(
        key=lambda r: (
            hashlib.sha256(str(r.get("winning_python_code", "")).encode("utf-8")).hexdigest(),
            float(r.get("confirm_rmse") or 0.0),
            str(r.get("task_id") or ""),
        )
    )

    sources_with_local_updates: set[str] = set()
    billed_local_nodes: set[str] = set()
    billed_hop_pairs: set[tuple[str, str]] = set()

    for rec in eligible_recs:
        py_src = str(rec["winning_python_code"]).strip()
        ast_info = extract_and_translate_authored_ast_v2(py_src, role_map=None)
        if not ast_info["ok"] or not ast_info["used_keys"]:
            continue

        owner_res = resolve_verified_writeback_owner(
            recalled_node_ids=list(rec.get("recalled_node_ids") or ([rec["primary_recalled_node_id"]] if rec.get("primary_recalled_node_id") else [])),
            observed_signatures=list(rec.get("observed_signatures") or []),
            active_ast_keys=ast_info["used_keys"],
            violated_root_ids=list(rec.get("violated_root_ids") or []),
            forest=forest,
            node_archive=node_archive,
        )
        src_nid = owner_res["owner_node_id"]
        src_entry = node_archive[src_nid]

        if "historical_L1_shadow" not in src_entry:
            src_entry["historical_L1_shadow"] = src_entry["verified_mechanism_note"]

        # Empirical Retroactive Projection check on `pilot_rows` (`P_obs`) if available (`M6-06` Fix)
        pilot_rows = list(rec.get("pilot_rows_snapshot") or [])
        retro_cert: dict[str, Any] | None = None
        if pilot_rows:
            y_obs = [float(r["y"]) for r in pilot_rows]
            # Evaluate confirmed code on pilot_rows to record empirical P_obs residual consistency
            eval_p = _execute_authored_python_in_isolated_subprocess(
                Path("/tmp/hegel_retro_proj_chk"),
                step_idx=0,
                python_code=ast_info["translated_code"],
                train_rows=pilot_rows,
                confirm_rows=pilot_rows,
                heldout_rows=pilot_rows,
            )
            if eval_p.get("ok"):
                primary_x = ast_info["used_keys"][0]
                xs_arr = np.array([float(r.get(primary_x, idx)) for idx, r in enumerate(pilot_rows)], dtype=float)
                ys_arr = np.array(y_obs, dtype=float)
                proj = verify_retroactive_projection_certificate(
                    obs_x=xs_arr,
                    obs_z=ys_arr,
                    old_associational_predict_fn=lambda _x: float(np.mean(ys_arr)),
                    new_causal_predict_fn=lambda _x, z_val: float(z_val),
                    n_bins=max(1, len(pilot_rows) // 5),
                    tolerance=max(0.08, float(eval_p["train_rmse"]) + 0.05),
                )
                retro_cert = {
                    "verification_scope": "empirical_pilot_slice_check",
                    "pilot_sample_count": len(pilot_rows),
                    "confirmed_model_pilot_rmse": eval_p["train_rmse"],
                    "passive_shadow_consistent": bool(eval_p["train_rmse"] <= 0.08),
                    "projection_helper_evaluated_bins": proj["evaluated_bins"],
                }
                src_entry["empirical_retroactive_projection_receipt"] = retro_cert

        added = _append_pareto_candidate(
            src_entry,
            used_keys=ast_info["used_keys"],
            python_code=ast_info["translated_code"],
            confirm_rmse=float(rec.get("confirm_rmse") or 0.0),
            provenance_tag=f"Wave1_Confirmed({owner_res['owner_mode']})",
        )
        if added:
            src_entry["epistemic_rung"] = "L2_EMPIRICALLY_CONFIRMED_LOCAL"
            _refresh_node_note_from_pareto(src_entry, "EMPIRICALLY CONFIRMED WAVE-1 MECHANISM")
            sources_with_local_updates.add(src_nid)
            if src_nid not in billed_local_nodes:
                billed_local_nodes.add(src_nid)
                wb_cost += 1.5

        events.append(
            {
                "owner_node_id": src_nid,
                "owner_mode": owner_res["owner_mode"],
                "extracted_active_variables": ast_info["used_keys"],
                "added_to_pareto_archive": added,
                "empirical_retroactive_projection": retro_cert,
                "reinterpreted_wormhole_targets": [],
                "collider_shielded_blocked_targets": [],
            }
        )

    # Second pass (`M6.3` ONLY): Multi-hop `d-sepset` wormhole propagation over ALL Pareto candidates!
    if max_hops > 0:
        for src_nid in sorted(sources_with_local_updates):
            src_entry = node_archive[src_nid]
            src_cands = list(src_entry.get("pareto_confirmed_candidates") or [])
            reinterpreted_targets: set[str] = set()
            blocked_targets: set[str] = set()

            for cand in src_cands:
                queue: deque[tuple[str, int, str, list[str], float]] = deque(
                    [(src_nid, 0, cand["python_code"], list(cand["used_keys"]), float(cand["confirm_rmse"]))]
                )
                visited_for_cand: set[str] = {src_nid}

                while queue:
                    curr_nid, hop_d, curr_code, curr_keys, curr_rmse = queue.popleft()
                    if hop_d >= max_hops:
                        continue
                    curr_meta = node_archive[curr_nid]
                    curr_dag = curr_meta.get("local_dag_edges", [])
                    curr_vmap = curr_meta.get("interface_var_to_dag_node", {})
                    active_dag_nodes = {curr_vmap[k] for k in curr_keys if k in curr_vmap} | {"Y_out"}

                    for wh in forest.horizontal_wormhole_edges.values():
                        if wh["node_a_id"] == curr_nid:
                            nbr_nid = wh["node_b_id"]
                            rmap = dict((wh.get("role_and_observation_map") or {}).get("role_map") or {})
                        elif wh["node_b_id"] == curr_nid:
                            nbr_nid = wh["node_a_id"]
                            raw_rm = dict((wh.get("role_and_observation_map") or {}).get("role_map") or {})
                            rmap = {v: k for k, v in raw_rm.items()}
                        else:
                            continue

                        if nbr_nid in visited_for_cand or nbr_nid not in node_archive:
                            continue
                        if str(node_archive[nbr_nid].get("epistemic_rung", "")) == "L1_DISTRACTOR":
                            continue

                        wh_dom_dag_nodes = {curr_vmap[k] for k in rmap.keys() if k in curr_vmap}
                        if curr_dag and wh_dom_dag_nodes:
                            if is_d_separated_dag(curr_dag, active_dag_nodes, wh_dom_dag_nodes, conditioned_vars=set()):
                                blocked_targets.add(nbr_nid)
                                continue

                        # Partial-commonality transport gate: EVERY active variable of THIS Pareto candidate must map!
                        if not set(curr_keys).issubset(set(rmap.keys())):
                            continue

                        trans = extract_and_translate_authored_ast_v2(curr_code, role_map=rmap)
                        if not trans["ok"] or not trans["translated_keys"]:
                            continue

                        visited_for_cand.add(nbr_nid)
                        nbr_entry = node_archive[nbr_nid]
                        if "historical_L1_shadow" not in nbr_entry:
                            nbr_entry["historical_L1_shadow"] = nbr_entry["verified_mechanism_note"]
                        t_added = _append_pareto_candidate(
                            nbr_entry,
                            used_keys=trans["translated_keys"],
                            python_code=trans["translated_code"],
                            confirm_rmse=curr_rmse,
                            provenance_tag=f"{hop_d + 1}hop_wormhole_from_{src_nid}",
                        )
                        if t_added:
                            if nbr_entry.get("epistemic_rung") != "L2_EMPIRICALLY_CONFIRMED_LOCAL":
                                nbr_entry["epistemic_rung"] = "L2_AST_TRANSPORTED_CAUSAL"
                            _refresh_node_note_from_pareto(
                                nbr_entry,
                                f"AST-TRANSPORTED CAUSAL MECHANISM (`{hop_d + 1}`-hop `d-sepset` wormhole from `{src_nid}`)",
                            )
                        reinterpreted_targets.add(nbr_nid)
                        hop_edge = (curr_nid, nbr_nid)
                        if hop_edge not in billed_hop_pairs:
                            billed_hop_pairs.add(hop_edge)
                            wb_cost += 0.6
                        queue.append((nbr_nid, hop_d + 1, trans["translated_code"], trans["translated_keys"], curr_rmse))

            for ev in events:
                if ev["owner_node_id"] == src_nid:
                    ev["reinterpreted_wormhole_targets"] = sorted(reinterpreted_targets)
                    ev["collider_shielded_blocked_targets"] = sorted(blocked_targets)

    return {"events": events, "writeback_cost_C": round(wb_cost, 4)}


# ============================================================================
# 4. Strict Probe Filter (`M6-07`) + Skill Activation Warm-Start + Episode Loop
# ============================================================================


def detect_observational_identifiability_hazards_v2(
    pilot_rows: list[dict[str, float]],
    probe_pool: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Strictly exclude BOTH `observational_L1` AND `observational_L1_selected` from regime-breaking probes!"""
    if len(pilot_rows) < 3 or not probe_pool:
        return None
    cols = sorted(k for k in pilot_rows[0].keys() if k != "y")
    arrs = {c: np.array([float(r[c]) for r in pilot_rows], dtype=float) for c in cols}
    stds = {c: float(np.std(arrs[c])) for c in cols}

    zero_var_cols = [c for c in cols if stds[c] < 1e-9]
    varying_cols = [c for c in cols if stds[c] >= 1e-9]
    collinear_pairs: list[tuple[str, str, float]] = []
    for i, ca in enumerate(varying_cols):
        for cb in varying_cols[i + 1 :]:
            corr = float(np.corrcoef(arrs[ca], arrs[cb])[0, 1])
            if abs(corr) > 0.98:
                collinear_pairs.append((ca, cb, round(corr, 4)))

    # M6-07 Fix: Passive observational slices (`observational_L1*`) CANNOT break collider or confounding regimes!
    regime_breaking_probes = [
        pid for pid, pmeta in probe_pool.items()
        if not str(pmeta.get("probe_kind", "")).startswith("observational")
    ]
    if (not collinear_pairs and not zero_var_cols) or not regime_breaking_probes:
        return None

    return {
        "collinear_variable_pairs_in_pilot": collinear_pairs,
        "constant_conditioned_columns_in_pilot": zero_var_cols,
        "eligible_regime_breaking_probes": regime_breaking_probes,
        "diagnostic_notice": (
            "STATISTICAL IDENTIFIABILITY WARNING: Initial passive `pilot_rows` exhibit near-perfect collinearity "
            "or constant selection conditioning. Sampling more passive/selected observational rows CANNOT break "
            f"confounding or collider bias; acquire an interventional or unselected probe from `{regime_breaking_probes}`."
        ),
    }


def _strip_internal_meta_v2(note_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v for k, v in note_entry.items()
        if k not in {"local_dag_edges", "interface_var_to_dag_node"}
    }


def _evaluate_recalled_pilot_warmstarts(
    recalled_notes: list[dict[str, Any]],
    pilot_rows: list[dict[str, float]],
    workspace_dir: Path,
) -> list[dict[str, Any]]:
    """Skill∞ Activation Bridge: Refit recalled `pareto_confirmed_candidates` ONLY on already-paid `pilot_rows`!

    Zero confirmation/heldout leakage: `confirm_rows` and `heldout_rows` are NEVER passed here.
    Gives both `LearnedFlat` and `M6.3` concrete pilot-fit diagnostics so transported ASTs are actually tested!
    """
    if not pilot_rows:
        return []
    pilot_cols = {k for k in pilot_rows[0].keys() if k != "y"}
    previews: list[dict[str, Any]] = []
    step_counter = 90
    for note in recalled_notes:
        nid = str(note.get("node_id", ""))
        cands = list(note.get("pareto_confirmed_candidates") or [])
        for cand in cands[:2]:
            req_keys = set(cand.get("used_keys") or [])
            if not req_keys or not req_keys.issubset(pilot_cols):
                continue
            step_counter += 1
            ev = _execute_authored_python_in_isolated_subprocess(
                workspace_dir / "pilot_warmstart_checks",
                step_idx=step_counter,
                python_code=str(cand["python_code"]),
                train_rows=pilot_rows,
                confirm_rows=pilot_rows,
                heldout_rows=pilot_rows,
            )
            if ev.get("ok"):
                previews.append(
                    {
                        "source_node_id": nid,
                        "provenance_tag": cand.get("provenance_tag"),
                        "active_keys": cand["used_keys"],
                        "pilot_refit_rmse": ev["train_rmse"],
                        "abi_mode": ev.get("abi_mode"),
                        "executable_python_skeleton": cand["python_code"],
                    }
                )
    previews.sort(key=lambda x: float(x["pilot_refit_rmse"]))
    return previews


def build_m6_3_arm_brief(
    ds: M6CausalTaskDataset,
    arm: str,
    *,
    forest: PhilosophicalForest,
    node_archive: dict[str, dict[str, Any]],
    hippo_idx: HippoRAGWormholeIndex,
    flat_idx: FlatEqualMetadataIndex,
    ledger: DecisionProvenanceLedger,
    parent_id: str,
    workspace_dir: Path,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Construct zero-spoiler, equal-authority brief across all 4 `M6.3` arms."""
    sys_prompt = (
        "You are an autonomous causal & scientific discovery agent. Infer the quantitative structural law `y` "
        "for the target `deployment_regime` and return one valid JSON decision matching the schema.\n"
        "RULES:\n"
        "1. Minimize total measurement + compute + token cost while achieving validation confirmation RMSE <= tolerance.\n"
        "2. Call `acquire_probe` when current `train_rows` have genuine causal/identifiability ambiguity (e.g., collinear "
        "candidate variables or selection conditioning under passive `P_obs` before interventional/unselected deployment).\n"
        "3. In `author_and_confirm_python_model`, write self-contained Python defining `fit(train_rows)` and `predict(row) -> float` "
        "(either mutating module state in `fit` or returning a fitted `predict(row)` callable from `fit`) using `math`, `numpy`, "
        "and `scipy.optimize.least_squares`. NEVER access `row['y']` inside `predict(row)`.\n"
        "4. If `verified_pilot_warmstart_candidates` contains a transported/recalled Python skeleton whose `active_keys` match the "
        "causal mechanism for `deployment_regime` and achieves low `pilot_refit_rmse`, submit & confirm that refitted candidate "
        "(or acquire a probe first if pilot rows are underdetermined) — NEVER abstain on an in-scope causal task!\n"
        "5. If observations show chaotic high-frequency non-stationary sign oscillations (`sig:high_freq_sign_oscillation`), call `abstain_scoped_unknown`.\n"
        "6. Populate `considered_alternatives` with concrete causal/mechanistic hypotheses weighed."
    )

    base_info = {
        "task_id": ds.task_id,
        "tolerance": ds.tolerance,
        "deployment_regime": ds.deployment_regime,
        "observed_signatures": list(ds.probe_signatures),
        "initial_paid_pilot_rows": ds.pilot_rows,
        "available_probes_menu": {
            k: {"probe_kind": v["probe_kind"], "total_probe_cost": v["cost"]}
            for k, v in ds.probe_pool.items()
        },
    }

    if arm == "Strong_Ordinary_LiveGPT":
        return sys_prompt, {
            **base_info,
            "arm_mode": "Strong_Ordinary_Full_K40_Historical_Archive",
            "full_k40_historical_notes": [_strip_internal_meta_v2(v) for v in node_archive.values()],
        }, parent_id, {"recalled_node_ids": [], "violated_root_ids": [], "warmstart_count": 0}

    tv_q = {"goal_kind": "causal_and_interventional_prediction", "tolerance": ds.tolerance}
    use_flat = arm == "EqualContent_LearnedFlat_LiveGPT"
    rec = (flat_idx if use_flat else hippo_idx).retrieve(ds.probe_signatures, task_view=tv_q, top_k=4)
    rec_ids = [c["node_id"] for c in rec.get("candidates", ())]

    p_id = ledger.record_decision(
        decision_type="D2_INDEX_RETRIEVAL",
        parent_decision_id=parent_id,
        evidence_inputs={"query_signatures": ds.probe_signatures, "arm": arm},
        candidates_evaluated=[{"node_id": nid} for nid in rec_ids],
        chosen_option={"recalled_top4": rec_ids},
        rejection_reasons={},
        rationale_summary=f"Recall Top-4 from K={len(forest.nodes)} forest via {'LearnedFlatBM25' if use_flat else 'HippoRAG_PPR'}.",
        compute_delta=float(rec.get("query_compute_cost", 1.0)),
    )

    recalled_notes = [_strip_internal_meta_v2(deepcopy(node_archive[nid])) for nid in rec_ids if nid in node_archive]

    # Skill∞ Activation Bridge: Pilot-only warmstart evaluation for ANY recalled L2 candidate (fair to BOTH LearnedFlat & M6.3!)
    warmstarts = [] if ds.is_out_of_scope else _evaluate_recalled_pilot_warmstarts(recalled_notes, ds.pilot_rows, workspace_dir)
    if warmstarts:
        p_id = ledger.record_decision(
            decision_type="D4C_SKILL_ACTIVATION_PILOT_WARMSTART",
            parent_decision_id=p_id,
            evidence_inputs={"candidates_checked": len(warmstarts)},
            candidates_evaluated=[{"node_id": w["source_node_id"], "pilot_rmse": w["pilot_refit_rmse"]} for w in warmstarts],
            chosen_option={"best_pilot_rmse": warmstarts[0]["pilot_refit_rmse"]},
            rejection_reasons={},
            rationale_summary="Pilot-only refit check of recalled/transported Python AST candidates.",
            compute_delta=0.5,
        )

    if use_flat or arm == "Hegel_M5_Full_LiveGPT":
        return sys_prompt, {
            **base_info,
            "arm_mode": arm,
            "recalled_notes": recalled_notes,
            "verified_pilot_warmstart_candidates": warmstarts,
        }, p_id, {"recalled_node_ids": rec_ids, "violated_root_ids": [], "warmstart_count": len(warmstarts)}

    # --- `Hegel_M6_3_Sovereign_LiveGPT`: Strict Interventional MEC Check + Slice-Aware Precondition BFS ---
    has_transported_L2 = bool(warmstarts)
    mec_warning: dict[str, Any] | None = None
    if (not ds.is_out_of_scope) and (not has_transported_L2):
        mec_warning = detect_observational_identifiability_hazards_v2(ds.pilot_rows, ds.probe_pool)
        if mec_warning is not None:
            p_id = ledger.record_decision(
                decision_type="D4_M6_3_STRICT_IDENTIFIABILITY_CHECK",
                parent_decision_id=p_id,
                evidence_inputs={"pilot_row_count": len(ds.pilot_rows)},
                candidates_evaluated=mec_warning["eligible_regime_breaking_probes"],
                chosen_option={"collinear_pairs_count": len(mec_warning["collinear_variable_pairs_in_pilot"])},
                rejection_reasons={"passive_selected_probes": "excluded_cannot_break_collider_or_fork"},
                rationale_summary="Strict interventional/unselected identifiability check on pilot_rows.",
                compute_delta=0.4,
            )

    # Slice-aware precondition evaluation (`M6-06` Fix: evaluate per-row slice rather than masking mixed regimes with a single mean!)
    violated_preconds: list[dict[str, Any]] = []
    violated_root_ids: list[str] = []
    expanded_ids: list[str] = []
    for n_entry in list(recalled_notes):
        nid = n_entry["node_id"]
        precs = n_entry.get("preconditions") or []
        if precs and ds.pilot_rows:
            violated_rows = sum(
                1 for r in ds.pilot_rows
                if any(evaluate_predicate_rule(pr, r) == "FALSE" for pr in precs)
            )
            if violated_rows > 0:
                violated_root_ids.append(nid)
                violated_preconds.append(
                    {
                        "node_id": nid,
                        "violated_preconditions": precs,
                        "violated_pilot_row_fraction": round(violated_rows / len(ds.pilot_rows), 4),
                    }
                )
                bfs_q: deque[tuple[str, int]] = deque([(nid, 0)])
                visited_bfs: set[str] = {nid}
                already_recalled: set[str] = {x["node_id"] for x in recalled_notes}
                while bfs_q:
                    cur_n, d_cur = bfs_q.popleft()
                    if d_cur >= 3:
                        continue
                    for wh in forest.horizontal_wormhole_edges.values():
                        nb = (
                            wh["node_b_id"]
                            if wh["node_a_id"] == cur_n
                            else (wh["node_a_id"] if wh["node_b_id"] == cur_n else None)
                        )
                        if nb and nb not in visited_bfs and nb in node_archive:
                            visited_bfs.add(nb)
                            bfs_q.append((nb, d_cur + 1))
                            if nb not in already_recalled:
                                already_recalled.add(nb)
                                expanded_ids.append(nb)
                                recalled_notes.append(_strip_internal_meta_v2(deepcopy(node_archive[nb])))

    if violated_preconds:
        # Re-check warmstarts in case a composite node (`node_synth_composite_*`) was just pulled in via wormhole BFS!
        warmstarts = _evaluate_recalled_pilot_warmstarts(recalled_notes, ds.pilot_rows, workspace_dir)
        p_id = ledger.record_decision(
            decision_type="D4B_M6_3_SLICE_PRECONDITION_WORMHOLE_BFS",
            parent_decision_id=p_id,
            evidence_inputs={"violated_roots": violated_root_ids},
            candidates_evaluated=[{"expanded_component_nodes": expanded_ids}],
            chosen_option={"expanded_component_nodes": expanded_ids},
            rejection_reasons={"root_default_branch": "slice_precondition_FALSE"},
            rationale_summary="Slice-level precondition violation triggered multi-depth wormhole BFS.",
            compute_delta=0.6,
        )

    return sys_prompt, {
        **base_info,
        "arm_mode": "Hegel_M6_3_Sovereign_Causal_Wormhole",
        "recalled_notes": recalled_notes,
        "verified_pilot_warmstart_candidates": warmstarts,
        "identifiability_diagnostic": mec_warning,
        "precondition_violation_submodules_notice": (
            {
                "violated_root_preconditions": violated_preconds,
                "recalled_submodule_node_ids": expanded_ids,
                "instruction": "Compose the component relations from `recalled_notes` into a unified root model and acquire a probe if parameters are underdetermined.",
            }
            if violated_preconds
            else None
        ),
    }, p_id, {
        "recalled_node_ids": rec_ids + expanded_ids,
        "violated_root_ids": violated_root_ids,
        "warmstart_count": len(warmstarts),
    }


def run_single_m6_3_episode(
    ds: M6CausalTaskDataset,
    arm: str,
    *,
    client: Any,
    workspace_dir: Path,
    forest: PhilosophicalForest,
    node_archive: dict[str, dict[str, Any]],
    hippo_idx: HippoRAGWormholeIndex,
    flat_idx: FlatEqualMetadataIndex,
    max_turns: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one controlled `M6.3` episode separating provider retries, implementation ABI errors, and validation checks."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    tol = float(ds.tolerance)
    model_name = getattr(client, "model", "unknown_model")
    prov_backend = getattr(client, "provider_backend", "mock_client")
    ledger = DecisionProvenanceLedger(
        run_id=f"m6_3__{model_name}",
        task_id=ds.task_id,
        arm=arm,
        model=model_name,
        provider_backend=prov_backend,
    )

    train_rows = list(ds.pilot_rows)
    parent_id = ledger.record_decision(
        decision_type="D0_PILOT_ACQUISITION",
        parent_decision_id=None,
        evidence_inputs={"task_id": ds.task_id, "pilot_count": len(train_rows)},
        candidates_evaluated=[],
        chosen_option={"pilot_rows": len(train_rows), "unit_row_m": EQUAL_UNIT_ROW_COST},
        rejection_reasons={},
        rationale_summary="Acquire initial paid pilot rows at identical unit cost across all arms.",
        measurement_delta=len(train_rows) * EQUAL_UNIT_ROW_COST,
    )

    sys_prompt, brief, parent_id, recall_meta = build_m6_3_arm_brief(
        ds,
        arm,
        forest=forest,
        node_archive=node_archive,
        hippo_idx=hippo_idx,
        flat_idx=flat_idx,
        ledger=ledger,
        parent_id=parent_id,
        workspace_dir=workspace_dir,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(brief, ensure_ascii=False)},
    ]

    purchased_probes: set[str] = set()
    validation_attempts = 0
    api_calls = 0
    provider_errors = 0
    stage_subdir = f"m6_3__{model_name}__{arm}__{ds.task_id}"

    turn_idx = 1
    while turn_idx <= max_turns:
        if isinstance(client, CodexSubscriptionClient):
            gpt_step = client.call_tool_step(messages, stage_subdir=stage_subdir)
        else:
            gpt_step = client.call_tool_step(messages)
        api_calls += 1

        pt = int(gpt_step.get("prompt_tokens", 0))
        ca = int(gpt_step.get("cached_tokens", 0))
        ct = int(gpt_step.get("completion_tokens", 0))
        eff_tok = max(0, pt - ca) + 0.25 * ca + ct
        tok_k = round(eff_tok / 1000.0, 4)

        if not gpt_step.get("ok"):
            provider_errors += 1
            parent_id = ledger.record_decision(
                decision_type="D6_PROVIDER_INFRA_RETRY",
                parent_decision_id=parent_id,
                evidence_inputs={"scientific_turn": turn_idx, "provider_errors": provider_errors},
                candidates_evaluated=[],
                chosen_option={"error": gpt_step.get("error")},
                rejection_reasons={},
                rationale_summary="Provider/schema error recorded separately without burning scientific turn (up to 2 infra retries).",
                compute_delta=tok_k,
                prompt_tokens_delta=pt,
                cached_tokens_delta=ca,
                completion_tokens_delta=ct,
                wall_ms_delta=gpt_step.get("wall_ms", 0.0),
            )
            if provider_errors <= 2:
                # Do NOT increment `turn_idx` on transient provider failure (`V18-R05` Fix!)
                continue
            turn_idx += 1
            continue

        t_name = gpt_step["tool_name"]
        t_args = gpt_step["tool_args"]
        t_call_id = gpt_step["tool_call_id"]

        # Include authored `python_code` in assistant history so model can repair exact code if needed (`V18-R09` Fix!)
        hist_msg = deepcopy(gpt_step["assistant_message_for_history"])
        if t_name == "author_and_confirm_python_model":
            hist_msg["action_taken"]["python_code"] = t_args.get("python_code", "")
        messages.append(hist_msg)

        parent_id = ledger.record_decision(
            decision_type="D6_LIVE_GPT_DECISION",
            parent_decision_id=parent_id,
            evidence_inputs={"turn": turn_idx, "train_rows": len(train_rows)},
            candidates_evaluated=[{"tool": t_name}],
            chosen_option={"tool": t_name, "probe_id": t_args.get("probe_id"), "hypothesis": t_args.get("hypothesis_name")},
            rejection_reasons={},
            rationale_summary=str(t_args.get("rationale", ""))[:300],
            compute_delta=tok_k,
            prompt_tokens_delta=pt,
            cached_tokens_delta=ca,
            completion_tokens_delta=ct,
            wall_ms_delta=gpt_step["wall_ms"],
        )

        if t_name == "acquire_probe":
            pid = str(t_args.get("probe_id", ""))
            if pid in purchased_probes:
                reply: dict[str, Any] = {"status": "already_purchased", "train_row_count": len(train_rows)}
            elif pid in ds.probe_pool:
                purchased_probes.add(pid)
                p_obj = ds.probe_pool[pid]
                p_cost = float(p_obj["cost"])
                added = list(p_obj["rows"])
                train_rows.extend(added)
                parent_id = ledger.record_decision(
                    decision_type="D5B_PROBE_EXECUTION",
                    parent_decision_id=parent_id,
                    evidence_inputs={"probe_id": pid},
                    candidates_evaluated=[{"probe_id": pid, "cost": p_cost}],
                    chosen_option={"added_rows_count": len(added)},
                    rejection_reasons={},
                    rationale_summary=f"Acquired probe {pid}.",
                    measurement_delta=p_cost,
                )
                reply = {"status": "acquired", "added_rows": added, "total_train_rows": train_rows}
            else:
                reply = {"status": "unavailable_in_target_domain", "available_probes": list(ds.probe_pool.keys())}
            messages.append({"role": "tool", "tool_call_id": t_call_id, "content": json.dumps(reply, ensure_ascii=False)})
            turn_idx += 1
            continue

        if t_name == "abstain_scoped_unknown":
            ledger.record_decision(
                decision_type="D9_TERMINAL_ABSTAIN",
                parent_decision_id=parent_id,
                evidence_inputs={"turn": turn_idx},
                candidates_evaluated=[{"status": "scoped_unknown"}],
                chosen_option={"strict_pass": bool(ds.is_out_of_scope)},
                rejection_reasons={},
                rationale_summary="Abstained as scoped_unknown.",
            )
            return {
                "task_id": ds.task_id,
                "wave": ds.wave,
                "subfamily": ds.subfamily,
                "arm": arm,
                "model": model_name,
                "api_calls": api_calls,
                "provider_errors": provider_errors,
                "is_out_of_scope": ds.is_out_of_scope,
                "status": "scoped_unknown",
                "strict_pass": bool(ds.is_out_of_scope),
                "false_confirmation": False,
                "confirm_rmse": None,
                "heldout_rmse": None,
                "recalled_node_ids": recall_meta["recalled_node_ids"],
                "violated_root_ids": recall_meta["violated_root_ids"],
                "observed_signatures": list(ds.probe_signatures),
                "pilot_rows_snapshot": list(ds.pilot_rows),
                "warmstart_candidates_evaluated": recall_meta["warmstart_count"],
                "winning_python_code": None,
                "purchased_probes": sorted(purchased_probes),
                **ledger.totals,
            }, ledger.records

        if t_name == "author_and_confirm_python_model":
            py_src = str(t_args.get("python_code", ""))
            hyp = str(t_args.get("hypothesis_name", f"cand_{turn_idx}"))
            exec_res = _execute_authored_python_in_isolated_subprocess(
                workspace_dir,
                step_idx=turn_idx,
                python_code=py_src,
                train_rows=train_rows,
                confirm_rows=ds.confirm_rows,
                heldout_rows=ds.heldout_rows,
            )
            if not exec_res.get("ok"):
                parent_id = ledger.record_decision(
                    decision_type="D7_IMPLEMENTATION_INVALID",
                    parent_decision_id=parent_id,
                    evidence_inputs={"hypothesis": hyp},
                    candidates_evaluated=[],
                    chosen_option={"error": str(exec_res.get("error", ""))[:240]},
                    rejection_reasons={},
                    rationale_summary="Implementation/ABI runtime error classified separately from scientific refutation.",
                    compute_delta=1.5,
                    wall_ms_delta=exec_res["exec_wall_ms"],
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": t_call_id,
                        "content": json.dumps(
                            {
                                "status": "implementation_invalid",
                                "error": exec_res["error"],
                                "hint": "Fix the Python implementation/ABI bug without discarding your mathematical hypothesis unless empirical residuals refute it.",
                            }
                        ),
                    }
                )
                turn_idx += 1
                continue

            validation_attempts += 1
            # Honest accounting (`M6-07` Fix): 1st validation call acquires `confirm_rows` (`1.5/row`);
            # subsequent validation calls reuse the same validation set (`0.5/row` re-eval fee, named honestly!)
            val_channel = "initial_fresh_confirmation" if validation_attempts == 1 else "adaptive_validation_recheck"
            conf_m = (
                len(ds.confirm_rows) * EQUAL_UNIT_ROW_COST
                if validation_attempts == 1
                else 0.5 * len(ds.confirm_rows) * EQUAL_UNIT_ROW_COST
            )
            c_rmse = float(exec_res["confirm_rmse"])
            accepted = c_rmse <= tol
            parent_id = ledger.record_decision(
                decision_type="D8_VALIDATION_CONFIRMATION_CHECK",
                parent_decision_id=parent_id,
                evidence_inputs={"hypothesis": hyp, "validation_attempt": validation_attempts, "channel": val_channel, "abi_mode": exec_res.get("abi_mode")},
                candidates_evaluated=[{"train_rmse": exec_res["train_rmse"], "confirm_rmse": c_rmse}],
                chosen_option={"accepted": accepted},
                rejection_reasons=({} if accepted else {hyp: f"confirm_rmse={c_rmse:.5f} > {tol}"}),
                rationale_summary=f"Validation ({val_channel}) {'PASSED' if accepted else 'REFUTED'} (c_rmse={c_rmse:.5f}).",
                measurement_delta=conf_m,
                compute_delta=3.0,
                wall_ms_delta=exec_res["exec_wall_ms"],
            )
            if accepted:
                h_rmse = float(exec_res["heldout_rmse"])
                strict_ok = (not ds.is_out_of_scope) and (h_rmse <= tol)
                ledger.record_decision(
                    decision_type="D9_TERMINAL_LOCKED",
                    parent_decision_id=parent_id,
                    evidence_inputs={"heldout_rmse": h_rmse},
                    candidates_evaluated=[],
                    chosen_option={"strict_pass": strict_ok},
                    rejection_reasons={},
                    rationale_summary="Locked confirmed model and audited on disjoint heldout set.",
                )
                return {
                    "task_id": ds.task_id,
                    "wave": ds.wave,
                    "subfamily": ds.subfamily,
                    "arm": arm,
                    "model": model_name,
                    "api_calls": api_calls,
                    "provider_errors": provider_errors,
                    "is_out_of_scope": ds.is_out_of_scope,
                    "status": "confirmed_ok",
                    "strict_pass": strict_ok,
                    "false_confirmation": bool(ds.is_out_of_scope) or (h_rmse > tol),
                    "confirm_rmse": round(c_rmse, 6),
                    "heldout_rmse": round(h_rmse, 6),
                    "abi_mode": exec_res.get("abi_mode"),
                    "recalled_node_ids": recall_meta["recalled_node_ids"],
                    "violated_root_ids": recall_meta["violated_root_ids"],
                    "observed_signatures": list(ds.probe_signatures),
                    "pilot_rows_snapshot": list(ds.pilot_rows),
                    "warmstart_candidates_evaluated": recall_meta["warmstart_count"],
                    "winning_python_code": py_src,
                    "purchased_probes": sorted(purchased_probes),
                    **ledger.totals,
                }, ledger.records

            refute_status = (
                "refuted_on_initial_fresh_confirmation"
                if validation_attempts == 1
                else "refuted_on_adaptive_validation_recheck"
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": t_call_id,
                    "content": json.dumps(
                        {
                            "status": refute_status,
                            "train_rmse": round(float(exec_res["train_rmse"]), 6),
                            "confirm_rmse": round(c_rmse, 6),
                            "tolerance": tol,
                            "hint": (
                                "Train RMSE is high as well — check optimization initialization/bounds or functional form!"
                                if float(exec_res["train_rmse"]) > tol
                                else "Train RMSE is low while out-of-regime confirm RMSE is high — acquire an interventional/unselected probe or switch to causal variables!"
                            ),
                        }
                    ),
                }
            )
            turn_idx += 1
            continue

        turn_idx += 1

    ledger.record_decision(
        decision_type="D9_BUDGET_EXHAUSTED_UNKNOWN",
        parent_decision_id=parent_id,
        evidence_inputs={"max_turns": max_turns},
        candidates_evaluated=[],
        chosen_option={"strict_pass": bool(ds.is_out_of_scope)},
        rejection_reasons={},
        rationale_summary="Turn budget exhausted; return scoped_unknown.",
    )
    return {
        "task_id": ds.task_id,
        "wave": ds.wave,
        "subfamily": ds.subfamily,
        "arm": arm,
        "model": model_name,
        "api_calls": api_calls,
        "provider_errors": provider_errors,
        "is_out_of_scope": ds.is_out_of_scope,
        "status": "scoped_unknown",
        "strict_pass": bool(ds.is_out_of_scope),
        "false_confirmation": False,
        "confirm_rmse": None,
        "heldout_rmse": None,
        "recalled_node_ids": recall_meta["recalled_node_ids"],
        "violated_root_ids": recall_meta["violated_root_ids"],
        "observed_signatures": list(ds.probe_signatures),
        "pilot_rows_snapshot": list(ds.pilot_rows),
        "warmstart_candidates_evaluated": recall_meta["warmstart_count"],
        "winning_python_code": None,
        "purchased_probes": sorted(purchased_probes),
        **ledger.totals,
    }, ledger.records
