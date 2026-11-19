#!/usr/bin/env python3
"""Focused offline controls of exact V16.6 modules; no model/provider calls.

Use --repo '/path/to/checkout/Hegel Machine' and --output result.json.
Namespace package shims bypass unrelated eager package __init__ imports.
Core functions are unchanged. Workflow acquisitions/fits are explicit toy
boundary fixtures. step_question is AST-isolated with a supplied typed query
result and simple mission-accounting stubs; this does not replay its full query
engine or demonstrate that the stored experiment produced that result shape.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

COMMIT = "675f52612a9d0f1fb208da8bdb0552de3c571758"


def selected_functions(path, names, env):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if len(selected) != len(names):
        raise ValueError("Missing selected function")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("m0_semantics_results.json"))
    args = p.parse_args()
    root = args.repo.resolve()
    src = root / "src/hegel_machine/cognition"
    sys.dont_write_bytecode = True
    for name, folder in [("hegel_machine", root / "src/hegel_machine"), ("hegel_machine.cognition", src)]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(folder)]
        sys.modules[name] = mod
    modules = {n: importlib.import_module("hegel_machine.cognition." + n) for n in (
        "question_semantics", "wormhole_forest", "growth_methods", "workflow_execution",
        "models", "language", "family_plugins", "cross_domain_clusters", "cross_domain_arms")}
    qs, wf, gm, ex = [modules[n] for n in ("question_semantics", "wormhole_forest", "growth_methods", "workflow_execution")]
    md, lang, cl, arms = [modules[n] for n in ("models", "language", "cross_domain_clusters", "cross_domain_arms")]
    results = {}

    # Existing live-question update path, without constructing the LLM agent.
    env = selected_functions(src / "online_inquiry_llm.py", ["open_assigned_question"], {
        "deepcopy": deepcopy, "adopt_question_semantics": qs.adopt_question_semantics})
    q = qs.adopt_question_semantics(None, {
        "id": "q", "question": "Scoped query", "kind": "mechanism", "answer_type": "scoped_outcome",
        "answer_resolution_ref": {"kind": "fixed"}, "query_semantics": {"mode": "model_query", "query_expression": {"op": "HypothesisTest"}},
        "protocol_spec": {"reset": "cold"}, "task_view": {"goal_kind": "predict", "display_label": "Old"}})
    q["semantic_answer"] = {"status": "answered", "value": {"outcome_status": "SUPPORTED_IN_SCOPE"}}
    q["query_execution"] = {"basis": "old", "answer": deepcopy(q["semantic_answer"])}
    agent = types.SimpleNamespace(questions=[q], research_input_bindings={"consumer": {"slot": "q"}},
                                  research_candidates={"consumer": {"status": "evaluated", "current_evaluation": {"mse": 0}}})
    env["open_assigned_question"](agent, {"id": "q", "task_view": {"goal_kind": "predict", "display_label": "New"}})
    cosmetic_ok = q["meaning_version"] == 1 and "semantic_answer" in q
    env["open_assigned_question"](agent, {"id": "q", "protocol_spec": {"reset": "warm"}})
    results["semantic_change_corrected"] = {
        "cosmetic_edit_retained_answer": cosmetic_ok, "meaning_version": q["meaning_version"],
        "answer_cleared": "semantic_answer" not in q, "query_cache_cleared": "query_execution" not in q,
        "history_kept": len(q["meaning_history"]), "consumer": agent.research_candidates["consumer"]}

    old_cc = {"preconditions": [{"field": "payload", "op": "==", "value": {"notes": "mode_A"}}]}
    new_cc = {"preconditions": [{"field": "payload", "op": "==", "value": {"notes": "mode_B"}}]}
    oq = qs.adopt_question_semantics(None, {"id": "opaque", "claim_contract": old_cc,
        "semantic_answer": {"status": "answered", "value": True}})
    nq = qs.adopt_question_semantics(oq, {"id": "opaque", "claim_contract": new_cc})
    results["cosmetic_key_overstrips_semantic_literal"] = {
        "old_predicate": wf.evaluate_predicate_rule(old_cc["preconditions"][0], {"payload": {"notes": "mode_A"}}),
        "new_predicate": wf.evaluate_predicate_rule(new_cc["preconditions"][0], {"payload": {"notes": "mode_A"}}),
        "meaning_unchanged": nq["meaning_version"] == oq["meaning_version"],
        "answer_retained": "semantic_answer" in nq}

    tv = {"goal_kind": "some_adjacent_flip", "tolerance": 0.0}
    cc = {"claim_id": "c", "preconditions": [], "conclusion_or_action": {"guarantee": "some_adjacent_flip"},
          "protocol_spec": {}, "evidence_status": "SUPPORTED_IN_SCOPE", "permission_tier": "retrieve_candidate"}
    def register(forest, ident, tree, *, contract=None, paths=None, question=None):
        return forest.register_node(node_id=ident, tree_id=tree, layer_role="mid", task_view=tv,
            observation_map={"raw_channel": "label"}, claim_contract=deepcopy(contract or cc),
            probe_signatures=["sig:ordered_opposite_endpoints"], support_paths=paths, question_spec=question)
    forest = wf.PhilosophicalForest()
    register(forest, "a", "tree_a", question={"id": "fq", "kind": "mechanism", "answer_type": "scoped_outcome"})
    register(forest, "b", "tree_b")
    register(forest, "weak", "tree_a")
    forest.add_local_implication_edge(stronger_node_id="a", weaker_node_id="weak", preserved_contract={"claim": "old"}, scope={})
    forest.add_horizontal_wormhole(wormhole_id="wh", node_a_id="a", node_b_id="b", probe_signature_refs=["sig:ordered_opposite_endpoints"],
        role_and_observation_map={}, hop_error_budget=0)
    forest.questions["fq"]["semantic_answer"] = {"status": "answered", "value": {"outcome_status": "SUPPORTED_IN_SCOPE"}}
    new_contract = deepcopy(cc)
    new_contract["conclusion_or_action"] = {"guarantee": "changed_contract"}
    revised = forest.empirical_revise_node("a", witness_episode_id="controlled_witness", cause="claim_refuted_or_unprofitable", updated_claim_contract=new_contract)
    wh_recalled = [r for r in forest.recall_mid_layer_wormholes(["sig:ordered_opposite_endpoints"], task_view=tv) if r["entry_kind"] == "horizontal_mid_wormhole"]
    results["forest_revision_identity_gap"] = {
        "current_a_version": revised["version"], "edge_a_version": forest.horizontal_wormhole_edges["wh"]["node_a_version"],
        "stale_edge_recalled_count": len(wh_recalled), "question_link_after_revision": revised["question_id"],
        "old_question_meaning": forest.questions["fq"]["meaning_version"],
        "old_question_answer_retained": "semantic_answer" in forest.questions["fq"],
        "old_implication_still_returns_parent": bool(forest.weaker_parents_of("a"))}

    missing_exclusion = {"permission_tier": "retrieve_candidate", "hop_error_budget": 0,
        "align_conditions": {"exclusion_predicates": [{"field": "sensor_clipped", "op": "==", "value": True}]}}
    results["unknown_exclusion_and_declared_tier"] = wf.authorize_abstraction_use(requested_tier="validated_pruning",
        task_view=tv, target_features={}, claim_contract=cc, wormhole_edge=missing_exclusion,
        heldout_confirmed=True, implication_verified=True)

    # No mutation of the gate: create a true failed target precondition.
    h = arms.HegelV166Researcher()
    blocked_cc = deepcopy(cc)
    blocked_cc["preconditions"] = [{"field": "reset_protocol_verified", "op": "==", "value": True}]
    register(h.forest, "blocked_a", "ta", contract=blocked_cc)
    register(h.forest, "blocked_b", "tb", contract=blocked_cc)
    h.forest.add_horizontal_wormhole(wormhole_id="blocked_wh", node_a_id="blocked_a", node_b_id="blocked_b",
        probe_signature_refs=["sig:ordered_opposite_endpoints"], role_and_observation_map={}, hop_error_budget=0)
    task = cl.PublicOrderedBoundaryTask(task_id="controlled_boundary", surface_domain="test", length=16,
        cost_per_query=1, task_view=tv, meter=cl.CostMeter(obs_budget=100, meas_budget=100, compute_budget=1000),
        private_oracle_fn=lambda i: int(i >= 7))
    answer = h.solve_target("controlled_boundary", task)
    results["m1_ordered_gate_denial_not_enforced"] = {
        "authorization": h.alignment_checks[-1]["authorization"], "answer": answer,
        "phase_costs": task.meter.snapshot()["phase_breakdown"]}

    rf = wf.PhilosophicalForest()
    rf.record_raw_episode({"episode_id": "raw", "rows": [1, 2]})
    register(rf, "a", "t1"); register(rf, "b", "t2")
    register(rf, "and_child", "t3", paths=[[{"node_id": "a", "version": 1}, {"node_id": "b", "version": 1}]])
    register(rf, "or_child", "t4", paths=[[{"node_id": "a", "version": 1}], [{"node_id": "b", "version": 1}]])
    pruned = rf.without_node_version("a", 1)
    results["and_or_removal_corrected"] = {"nodes": sorted(pruned.nodes), "raw_archive_preserved": pruned.raw_episode_archive == rf.raw_episode_archive}
    na = {"node_id": "rejecting", "claim_contract": {"preconditions": [], "prediction": False}}
    nb = {"node_id": "accepting", "claim_contract": {"preconditions": [], "prediction": True}}
    comp = wf.fidelity_compress_nodes(na, nb, [{"observed": True}],
        lambda node, q: "ACC" if node["claim_contract"]["prediction"] == q["observed"] else "REJ")
    results["fidelity_accepts_conflicting_verdicts"] = {k: v for k, v in comp.items() if k != "merged_node"}
    results["fidelity_unknown_promotion_unchecked"] = {k: v for k, v in wf.fidelity_compress_nodes(na, nb, [{}],
        lambda node, q: "ACC" if node["node_id"].startswith("merged:") else "UNK").items() if k != "merged_node"}

    # Actual compiler, binder and replay, with explicitly synthetic acquisition/fit.
    bounds = {"u": (0.0, 10.0)}
    spec = md.TaskSpec("branch", "Controlled branch", ("u",), (), (), feature_bounds=bounds, context={"control_roles": {"u": "focus"}})
    channels = (types.SimpleNamespace(name="ch", cost_per_row=1),)
    diag = {"feature_bounds": bounds, "control_roles": {"u": "focus"}, "channel_names": ["ch"], "feature_count": 1}
    def workflow(predicate=True, child=False, failure=False):
        trace = [
            {"event_id": "m", "action": "measure", "produces": ["rows"], "recipe": {"sequence": [{"u": u} for u in (1,3,5,9)], "channels": ["ch"]}},
            {"event_id": "c", "action": "scientific_refutation_witness", "consumes": ["rows"], "outcome_status": "REFUTED_IN_SCOPE", "recipe": ({"predicate_spec": {"kind": "diagnostic_predicate", "field": "monotone_flag", "op": "==", "value": True}} if predicate else {})},
            {"event_id": "branch", "action": "measure", "produces": ["extra"], "when_outcome": {"from_event": "c", "outcomes": ["REFUTED_IN_SCOPE"]}, "recipe": {"sequence": [{"u": 2}, {"u": 7}], "channels": ["ch"]}},
        ]
        if child:
            trace.append({"event_id": "child", "action": "measure", "consumes": ["extra"], "recipe": {"sequence": [{"u": 4}, {"u": 8}], "channels": ["ch"]}})
        trace.extend([
            {"event_id": "f", "action": "fit", "consumes": ["rows"], "produces": ["fit"], "recipe": {"estimator": "residual_expression_search", "channels": ["ch"]}},
            {"event_id": "confirm", "action": "confirm", "consumes": ["fit"], "recipe": {"sequence": [{"u": 4}, {"u": 8}], "channels": ["ch"]}},
        ])
        book = gm.ResearchMethodBook(); book.learn(trace, diag, success=True, task_id="source")
        ep = ex.WorkflowEpisode(book, spec, channels, context={"control_roles": {"u": "focus"}})
        ep.diagnostics["monotone_flag"] = True
        acquisitions, actions = [], []
        def acquire(ch, purpose, probe):
            acquisitions.append(probe.probe_id)
            return {"execution_error": True, "reason": "fixture_failure"} if failure else [md.Observation(r, r["u"]) for r in probe.sequence]
        cand = md.Candidate("controlled", lang.Program("feature", name="u"), "fit")
        out = ep.replay(acquire=acquire, project=lambda obs, a, b: obs,
            fit=lambda ch, tr, va: [{"candidate": cand, "training_loss": 0., "validation_loss": 0.}],
            legal_views=[("ch",)], remaining=lambda: {"observation_remaining": 100, "measurement_remaining": 100, "compute_remaining": 1000},
            confirmation_length=2, threshold=1e-6, diagnostics={}, actions=actions)
        return {"adequate": out["adequate"], "execution_error": out.get("execution_error", False),
                "node_outcomes": out["node_outcomes"], "skipped_steps": out["skipped_steps"],
                "acquired_probe_ids": acquisitions, "method_example_count": len(book.methods[0]["examples"]),
                "compiled_node_count": len(book.methods[0]["nodes"])}
    results["workflow_current_predicate_corrected"] = workflow()
    results["workflow_historical_outcome_without_predicate"] = workflow(predicate=False)
    results["workflow_execution_error_isolated"] = workflow(failure=True)

    # Exact result-acceptance path, query evaluator supplied at its explicit API.
    qc_env = {"deepcopy": deepcopy, "compile_question": qs.compile_question, "validate_answer_type": qs.validate_answer_type,
        "_CONTRACT_MEANING_FIELDS": qs._CONTRACT_MEANING_FIELDS, "_strip_cosmetic": qs._strip_cosmetic,
        "effective_question_mission": lambda q, m: {"purpose": "theory_learning"},
        "mission_compute_remaining": lambda a, m: 1000, "debit_mission_compute": lambda *a: None,
        "refresh_dependencies": lambda a: None,
        "evaluate_on_evidence": lambda *a, **k: {"status": "answered", "value": {"outcome_status": "EXECUTION_ERROR"}, "compute_cost": 0}}
    selected_functions(src / "question_continuation.py", ["executed_plan_records", "_versioned_support", "_author_work", "_discard_answer", "step_question"], qc_env)
    eq = qs.adopt_question_semantics(None, {"id": "error_q", "kind": "mechanism", "answer_type": "scoped_outcome", "answer_resolution_ref": {"kind": "fixed"}, "query_semantics": {"mode": "model_query", "query_expression": {"op": "HypothesisTest"}}})
    ea = types.SimpleNamespace(questions=[eq], observations=[], rows=[], costs={"compute_cost": 0}, research_mission={})
    er = qc_env["step_question"](ea, eq, None)
    results["nested_execution_error_result_accepted"] = {"return_status": er["status"], "semantic_answer": eq.get("semantic_answer")}

    provenance = []
    for file in sorted(src.glob("*.py")):
        raw = file.read_bytes()
        provenance.append({"path": str(file.relative_to(root)), "git_blob_sha1": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()})
    out = {"commit": COMMIT, "scope": __doc__, "source_files": provenance, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "controls": len(results), "result_keys": list(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
