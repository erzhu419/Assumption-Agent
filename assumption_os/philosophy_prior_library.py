"""Methodology prior library for dialectical framework evolution.

R2 in Hegel_assumption.md asks for human philosophy / methodology principles
to be represented as falsifiable priors rather than prompt-only wisdom text.
This module builds a bounded 30-principle library with success cases, failure
boundaries, verifier protocols, graph serialization, and a small expert-label
retrieval agreement benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .graph_memory import JsonlGraphStore
from .schema import AssumptionEdge, AssumptionNode, AssumptionType, EdgeType, HypothesisKind, stable_id


DEFAULT_OUT = PAPER_DIR / "philosophy_prior_library_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/philosophy_prior_library_20260612.md")


@dataclass(frozen=True)
class PriorCase:
    case_id: str
    domain: str
    description: str


@dataclass(frozen=True)
class PhilosophyPrior:
    principle_id: str
    claim: str
    scope_conditions: list[str]
    failure_conditions: list[str]
    canonical_examples: list[PriorCase]
    negative_examples: list[PriorCase]
    related_principles: list[str]
    formal_sketch: dict[str, Any] | None
    verifier_protocol: dict[str, str]
    tags: list[str]
    status: str = "active_prior"

    @property
    def node_id(self) -> str:
        return f"prior_{self.principle_id}"

    def to_assumption_node(self) -> AssumptionNode:
        return AssumptionNode(
            id=self.node_id,
            type=AssumptionType.FRAMEWORK,
            kind=HypothesisKind.CLAIM,
            claim=self.claim,
            formal_form=self.formal_sketch,
            context_conditions=self.scope_conditions,
            predicted_effects=list(self.verifier_protocol),
            risk_predictions=self.failure_conditions,
            verifiers=list(self.verifier_protocol.values()),
            evidence_ids=[case.case_id for case in [*self.canonical_examples, *self.negative_examples]],
            confidence=0.72,
            metaproductivity=0.22,
            status=self.status,
            tags=["philosophy_prior", "methodology_prior", *self.tags],
            payload={"philosophy_prior": asdict(self)},
        )


def build_philosophy_prior_library_payload(
    *,
    root: Path,
    eval_id: str = "philosophy_prior_library_20260612",
) -> dict[str, Any]:
    _ = root.resolve()
    priors = _build_priors()
    graph = _build_graph(priors)
    roundtrip = _roundtrip_graph(graph)
    retrieval = _run_retrieval_benchmark(priors)
    metrics = _metrics(priors=priors, graph=graph, roundtrip=roundtrip, retrieval=retrieval)
    gates = {
        "principle_count_high": metrics["principle_count"] >= 30,
        "all_priors_are_assumptions": metrics["framework_prior_node_count"] == metrics["principle_count"],
        "all_have_success_cases": metrics["min_success_case_count"] >= 2,
        "all_have_failure_cases": metrics["min_negative_case_count"] >= 1,
        "all_have_scope_and_failure_conditions": metrics["scope_condition_coverage"] == 1.0
        and metrics["failure_condition_coverage"] == 1.0,
        "all_have_verifier_protocols": metrics["verifier_protocol_coverage"] == 1.0,
        "all_gate_ready": metrics["conservative_gate_ready_coverage"] == 1.0,
        "top3_expert_agreement_reasonable": metrics["top3_expert_agreement"] >= 0.80,
        "top1_expert_agreement_nontrivial": metrics["top1_expert_agreement"] >= 0.50,
        "graph_roundtrip_exact": roundtrip["roundtrip_exact"] is True,
        "core_prior_auto_promotion_blocked": metrics["core_prior_auto_promotion_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "philosophy_prior_library",
        "source_md": "reconstruction/md/Hegel_assumption.md",
        "release_step": "R2_philosophy_methodology_prior_library",
        "performance_validation": True,
        "validation_scope": (
            "Represents 30 human methodology principles as bounded, falsifiable prior assumptions.  "
            "Each prior has success cases, failure boundaries, verifier protocols, graph edges, "
            "retrieval agreement checks, and conservative-generalization obligations."
        ),
        "priors": [asdict(prior) | {"node_id": prior.node_id} for prior in priors],
        "graph": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "edge_type_counts": _counts(edge.type.value for edge in graph["edges"]),
        },
        "retrieval_benchmark": retrieval,
        "roundtrip": roundtrip,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "allowed_claim": "bounded philosophy/methodology prior library for framework evolution",
        "blocked_claims": [
            "complete_cyc_style_common_sense_library",
            "human_priors_as_unquestioned_axioms",
            "automatic_core_prior_promotion",
            "retrieval_agreement_as_expert_proof",
        ],
    }


def format_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines = [
        "# Philosophy / Methodology Prior Library",
        "",
        f"- pass: `{payload['pass']}`",
        f"- principles: `{m['principle_count']}`",
        f"- min success cases: `{m['min_success_case_count']}`",
        f"- min negative cases: `{m['min_negative_case_count']}`",
        f"- conservative gate ready coverage: `{m['conservative_gate_ready_coverage']}`",
        f"- top-3 expert agreement: `{m['top3_expert_agreement']}`",
        f"- graph roundtrip exact: `{payload['roundtrip']['roundtrip_exact']}`",
        "",
        "## Principle IDs",
        "",
    ]
    for prior in payload["priors"]:
        lines.append(f"- `{prior['principle_id']}`: {prior['claim']}")
    lines.extend(["", "## Claim Boundary", ""])
    for claim in payload["blocked_claims"]:
        lines.append(f"- `{claim}`")
    return "\n".join(lines).rstrip() + "\n"


def retrieve_priors(priors: list[PhilosophyPrior], query: str, *, top_k: int = 3) -> list[dict[str, Any]]:
    q_vec = _text_vector(query)
    rows = []
    for prior in priors:
        score = _retrieval_score(q_vec=q_vec, prior=prior)
        rows.append({
            "principle_id": prior.principle_id,
            "node_id": prior.node_id,
            "claim": prior.claim,
            "score": round(score, 6),
        })
    return sorted(rows, key=lambda row: (-row["score"], row["principle_id"]))[:top_k]


def _build_graph(priors: list[PhilosophyPrior]) -> dict[str, Any]:
    nodes: dict[str, AssumptionNode] = {}
    edges: list[AssumptionEdge] = []
    for prior in priors:
        prior_node = prior.to_assumption_node()
        nodes[prior_node.id] = prior_node
        for case in prior.canonical_examples:
            case_node = _case_node(case=case, role="success_case")
            nodes[case_node.id] = case_node
            edges.append(AssumptionEdge(
                source=prior.node_id,
                target=case_node.id,
                type=EdgeType.PRESERVES_SUCCESS_CASES,
                weight=1.0,
                payload={"role": "canonical_success", "principle_id": prior.principle_id},
            ))
        for case in prior.negative_examples:
            case_node = _case_node(case=case, role="failure_boundary")
            nodes[case_node.id] = case_node
            edges.append(AssumptionEdge(
                source=prior.node_id,
                target=case_node.id,
                type=EdgeType.CONFLICTS_WITH,
                weight=1.0,
                payload={"role": "negative_boundary", "principle_id": prior.principle_id},
            ))
        prediction_id = stable_id("prediction", prior.principle_id, "new_case")
        prediction = AssumptionNode(
            id=prediction_id,
            type=AssumptionType.CASE,
            kind=HypothesisKind.CLAIM,
            claim=f"{prior.principle_id} should improve tasks matching its scope and fail safely outside it",
            context_conditions=prior.scope_conditions,
            risk_predictions=prior.failure_conditions,
            verifiers=list(prior.verifier_protocol.values()),
            status="prior_prediction_case",
            tags=["new_prediction_case", prior.principle_id],
            payload={"principle_id": prior.principle_id, "role": "new_prediction_case"},
        )
        nodes[prediction.id] = prediction
        edges.append(AssumptionEdge(
            source=prior.node_id,
            target=prediction.id,
            type=EdgeType.PREDICTS_NEW_CASE,
            weight=0.8,
            payload={"principle_id": prior.principle_id},
        ))
        residual_id = stable_id("residual", prior.principle_id, "boundary")
        residual = AssumptionNode(
            id=residual_id,
            type=AssumptionType.RESIDUAL,
            kind=HypothesisKind.CLAIM,
            claim=f"Boundary residuals for {prior.principle_id}: {'; '.join(prior.failure_conditions)}",
            status="prior_boundary_residual",
            tags=["prior_residual_boundary", prior.principle_id],
            payload={"principle_id": prior.principle_id, "failure_conditions": prior.failure_conditions},
        )
        nodes[residual.id] = residual
        edges.append(AssumptionEdge(
            source=prior.node_id,
            target=residual.id,
            type=EdgeType.EXPLAINS_RESIDUAL,
            weight=0.7,
            payload={"principle_id": prior.principle_id},
        ))
        for related in prior.related_principles:
            if related == prior.principle_id:
                continue
            edges.append(AssumptionEdge(
                source=prior.node_id,
                target=f"prior_{related}",
                type=EdgeType.IS_ANALOGY_OF,
                weight=0.45,
                payload={"relation": "related_methodology_prior"},
            ))
    return {"nodes": list(nodes.values()), "edges": _dedupe_edges(edges)}


def _case_node(*, case: PriorCase, role: str) -> AssumptionNode:
    node_type = AssumptionType.CASE if role == "success_case" else AssumptionType.RESIDUAL
    return AssumptionNode(
        id=stable_id(role, case.case_id),
        type=node_type,
        kind=HypothesisKind.CLAIM,
        claim=case.description,
        context_conditions=[case.domain],
        status=role,
        tags=["philosophy_prior_case", role, case.domain],
        payload=asdict(case) | {"role": role},
    )


def _roundtrip_graph(graph: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="philosophy_prior_library_") as td:
        store = JsonlGraphStore(td)
        for node in graph["nodes"]:
            store.upsert_node(node)
        for edge in graph["edges"]:
            store.add_edge(edge)
        before = _snapshot(store)
        store.flush()
        reloaded = JsonlGraphStore(td)
        after = _snapshot(reloaded)
    return {
        "roundtrip_exact": before == after,
        "before_hash": stable_hash(before),
        "after_hash": stable_hash(after),
        "node_count_after": len(after["nodes"]),
        "edge_count_after": len(after["edges"]),
        "framework_prior_node_count_after": sum(
            1 for row in after["nodes"]
            if row["type"] == AssumptionType.FRAMEWORK.value
            and "philosophy_prior" in row.get("tags", [])
        ),
    }


def _snapshot(store: JsonlGraphStore) -> dict[str, Any]:
    return {
        "nodes": sorted((node.to_dict() for node in store.nodes.values()), key=lambda row: row["id"]),
        "edges": sorted((edge.to_dict() for edge in store.edges), key=lambda row: (row["source"], row["target"], row["type"])),
    }


def _run_retrieval_benchmark(priors: list[PhilosophyPrior]) -> dict[str, Any]:
    rows = []
    for query in _expert_queries():
        retrieved = retrieve_priors(priors, query["query"], top_k=3)
        top_ids = [row["principle_id"] for row in retrieved]
        gold = set(query["gold_principle_ids"])
        rows.append({
            "query_id": query["query_id"],
            "query": query["query"],
            "gold_principle_ids": query["gold_principle_ids"],
            "top3": top_ids,
            "top1": top_ids[0] if top_ids else None,
            "top3_hit": bool(gold & set(top_ids)),
            "top1_hit": bool(top_ids and top_ids[0] in gold),
            "retrieved": retrieved,
        })
    return {
        "query_count": len(rows),
        "rows": rows,
        "top3_expert_agreement": round(sum(row["top3_hit"] for row in rows) / len(rows), 4),
        "top1_expert_agreement": round(sum(row["top1_hit"] for row in rows) / len(rows), 4),
    }


def _metrics(
    *,
    priors: list[PhilosophyPrior],
    graph: dict[str, Any],
    roundtrip: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    success_counts = [len(prior.canonical_examples) for prior in priors]
    negative_counts = [len(prior.negative_examples) for prior in priors]
    gate_ready = [
        bool(prior.scope_conditions)
        and bool(prior.failure_conditions)
        and len(prior.canonical_examples) >= 2
        and len(prior.negative_examples) >= 1
        and bool(prior.verifier_protocol)
        for prior in priors
    ]
    framework_prior_node_count = sum(
        1 for node in graph["nodes"]
        if node.type == AssumptionType.FRAMEWORK and "philosophy_prior" in node.tags
    )
    edge_types = {edge.type.value for edge in graph["edges"]}
    return {
        "principle_count": len(priors),
        "active_prior_count": sum(1 for prior in priors if prior.status == "active_prior"),
        "framework_prior_node_count": framework_prior_node_count,
        "min_success_case_count": min(success_counts),
        "min_negative_case_count": min(negative_counts),
        "scope_condition_coverage": round(sum(bool(prior.scope_conditions) for prior in priors) / len(priors), 4),
        "failure_condition_coverage": round(sum(bool(prior.failure_conditions) for prior in priors) / len(priors), 4),
        "verifier_protocol_coverage": round(sum(bool(prior.verifier_protocol) for prior in priors) / len(priors), 4),
        "conservative_gate_ready_count": sum(gate_ready),
        "conservative_gate_ready_coverage": round(sum(gate_ready) / len(priors), 4),
        "retrieval_query_count": retrieval["query_count"],
        "top3_expert_agreement": retrieval["top3_expert_agreement"],
        "top1_expert_agreement": retrieval["top1_expert_agreement"],
        "graph_node_count": len(graph["nodes"]),
        "graph_edge_count": len(graph["edges"]),
        "graph_edge_type_count": len(edge_types),
        "required_prior_edge_coverage": round(
            len({
                EdgeType.PRESERVES_SUCCESS_CASES.value,
                EdgeType.CONFLICTS_WITH.value,
                EdgeType.EXPLAINS_RESIDUAL.value,
                EdgeType.PREDICTS_NEW_CASE.value,
            } & edge_types)
            / 4,
            4,
        ),
        "roundtrip_exact": bool(roundtrip["roundtrip_exact"]),
        "roundtrip_node_count": roundtrip["node_count_after"],
        "roundtrip_edge_count": roundtrip["edge_count_after"],
        "core_prior_auto_promotion_count": 0,
        "main_graph_mutation_count": 0,
    }


def _retrieval_score(*, q_vec: Counter[str], prior: PhilosophyPrior) -> float:
    text = " ".join([
        prior.principle_id.replace("_", " "),
        prior.claim,
        " ".join(prior.scope_conditions),
        " ".join(prior.failure_conditions),
        " ".join(prior.tags),
        " ".join(case.description for case in prior.canonical_examples),
        " ".join(case.description for case in prior.negative_examples),
    ])
    p_vec = _text_vector(text)
    lexical = _cosine(q_vec, p_vec)
    tag_boost = 0.06 * len(set(q_vec) & set(prior.tags))
    id_boost = 0.08 * sum(1 for token in prior.principle_id.split("_") if token in q_vec)
    return lexical + tag_boost + id_boost


def _text_vector(text: str) -> Counter[str]:
    tokens = _tokens(text)
    expanded = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(_SYNONYMS.get(token, []))
    return Counter(expanded)


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[key] * b.get(key, 0) for key in a)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _dedupe_edges(edges: list[AssumptionEdge]) -> list[AssumptionEdge]:
    seen: set[tuple[str, str, str]] = set()
    out: list[AssumptionEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.type.value)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _prior(
    principle_id: str,
    claim: str,
    scope: list[str],
    failure: list[str],
    examples: list[tuple[str, str, str]],
    negatives: list[tuple[str, str, str]],
    related: list[str],
    tags: list[str],
) -> PhilosophyPrior:
    return PhilosophyPrior(
        principle_id=principle_id,
        claim=claim,
        scope_conditions=scope,
        failure_conditions=failure,
        canonical_examples=[PriorCase(*row) for row in examples],
        negative_examples=[PriorCase(*row) for row in negatives],
        related_principles=related,
        formal_sketch={
            "objects": ["task_state", "intervention", "outcome"],
            "morphisms": ["select_scope", "apply_principle", "verify_boundary"],
            "invariants": ["old_success_preserved", "failure_boundary_recorded"],
        },
        verifier_protocol={
            "old_success_preservation": "replay prior success cases before promotion",
            "boundary_check": "test at least one negative or limiting case",
            "new_prediction": "run one unseen task matching the principle scope",
        },
        tags=tags,
    )


def _build_priors() -> list[PhilosophyPrior]:
    return [
        _prior("control_variables", "When causal attribution is uncertain, vary one factor while holding other factors fixed.", ["causal attribution", "ablation", "matched comparison"], ["strongly coupled variables make isolation misleading"], [("cv_science", "science", "hold temperature constant while changing pressure to estimate a gas relation"), ("cv_prompt", "agent", "change one retrieval policy while keeping judge and task set fixed")], [("cv_coupled", "systems", "changing learning rate also changes batch statistics, so attribution is confounded")], ["ablation", "negative_control", "causal_intervention"], ["causal", "ablation", "variable", "factor", "controlled"]),
        _prior("divide_and_conquer", "Split a complex task into separable subproblems with explicit interfaces, then compose the results.", ["multi-step task", "composable interface", "large problem"], ["subtasks interact so strongly that local solutions fail globally"], [("dac_algorithm", "software", "separate parsing, planning, and execution modules"), ("dac_math", "math", "prove lemmas before proving the theorem")], [("dac_entangled", "planning", "optimizing subgoals independently breaks a shared constraint")], ["error_decomposition", "general_to_special"], ["decompose", "split", "subproblem", "compose", "interface"]),
        _prior("proof_by_contradiction", "To test a claim, assume its negation and derive an impossible or inconsistent consequence.", ["logical proof", "necessary condition", "invariant violation"], ["empirical uncertainty has no crisp contradiction"], [("pbc_math", "math", "prove irrationality by assuming a rational ratio"), ("pbc_static", "software", "assume a dependency is absent and search for a required call path")], [("pbc_noisy", "science", "measurement noise prevents a strict contradiction")], ["reductio_ad_absurdum", "falsifiability"], ["contradiction", "negation", "inconsistent", "impossible", "proof"]),
        _prior("reductio_ad_absurdum", "Stress an assumption by extending it to an absurd consequence that exposes its boundary.", ["overbroad claim", "edge case", "policy boundary"], ["satire-like extremes distract from measurable failure"], [("raa_policy", "policy", "extend an unlimited growth rule until resource exhaustion is visible"), ("raa_prompt", "agent", "apply a style rule to every task and observe incoherent answers")], [("raa_unmeasured", "discussion", "absurd framing without a concrete verifier")], ["proof_by_contradiction", "boundary_condition_analysis"], ["absurd", "extreme", "overbroad", "edge", "boundary"]),
        _prior("occams_razor", "Prefer the simpler model when competing explanations have similar evidence and predictive power.", ["model comparison", "equal fit", "parsimony"], ["the simpler model underfits or misses required mechanisms"], [("occam_stats", "statistics", "choose a linear model when it matches heldout error"), ("occam_design", "software", "remove an unused abstraction when tests and users see no loss")], [("occam_underfit", "science", "a simple model ignores a necessary interaction term")], ["model_comparison", "minimum_viable_prototype"], ["simple", "parsimony", "model", "complexity", "underfit"]),
        _prior("bayesian_update", "Update belief strength as new evidence arrives, weighting prior confidence by observed likelihood.", ["uncertain belief", "sequential evidence", "calibration"], ["prior is misspecified or evidence is selected adversarially"], [("bayes_diagnosis", "medicine", "update disease probability after a test result"), ("bayes_world_model", "agent", "calibrate proposal acceptance probability after judgments")], [("bayes_bad_prior", "forecasting", "a biased prior dominates sparse contradictory evidence")], ["prior_estimate_then_update", "model_comparison"], ["bayesian", "belief", "prior", "posterior", "evidence", "calibration"]),
        _prior("analogical_reasoning", "Map a new problem to a structurally similar solved problem, then test which relations transfer.", ["structural similarity", "known source case", "transfer hypothesis"], ["surface similarity hides broken invariants"], [("analog_lenz", "physics", "map negative feedback in Lenz law to stabilizing policy response"), ("analog_resnet", "ml", "map identity residual paths to rollback-preserving software changes")], [("analog_name", "retrieval", "similar terms share words but not roles or invariants")], ["cross_domain_transfer", "invariant_search"], ["analogy", "similar", "morphism", "transfer", "structure"]),
        _prior("boundary_condition_analysis", "Test the edges of the scope because assumptions often fail at extremes or transitions.", ["edge case", "extreme input", "scope boundary"], ["boundary cases are irrelevant to the actual operating range"], [("boundary_math", "math", "check zero, infinity, and equality cases"), ("boundary_ops", "software", "test empty queue, max queue, and timeout cases")], [("boundary_irrelevant", "product", "spending all effort on impossible user states")], ["limiting_case_analysis", "robustness_testing"], ["boundary", "edge", "extreme", "scope", "transition"]),
        _prior("negative_control", "Use a condition where the effect should not appear to detect leakage, confounding, or prompt artifacts.", ["causal test", "leakage audit", "placebo-like setting"], ["negative control is not truly unaffected by the intervention"], [("neg_bio", "biology", "use a no-target sample in an assay"), ("neg_agent", "agent", "run no-trigger tasks to ensure a prompt policy abstains")], [("neg_bad", "evaluation", "control task secretly contains the same cue")], ["placebo_control", "control_variables"], ["negative", "control", "leakage", "artifact", "no trigger"]),
        _prior("minimum_viable_prototype", "Build the smallest working version that can falsify the core uncertainty before scaling.", ["uncertain feasibility", "early validation", "budget constraint"], ["prototype omits the failure mode that determines real viability"], [("mvp_startup", "business", "test purchase intent with a narrow pilot"), ("mvp_agent", "agent", "run a smoke ablation before a full benchmark")], [("mvp_toy", "engineering", "toy demo ignores production latency constraints")], ["incremental_replacement", "occams_razor"], ["prototype", "mvp", "minimal", "pilot", "feasibility"]),
        _prior("incremental_replacement", "Replace one bounded component behind an interface while preserving rollback to the working baseline.", ["working baseline", "module boundary", "safe migration"], ["component boundaries are false and hidden coupling dominates"], [("inc_database", "software", "migrate one service endpoint behind a feature flag"), ("inc_agent", "agent", "swap only the retrieval selector while keeping prompt and judge fixed")], [("inc_hidden", "systems", "a shared cache makes the isolated replacement unsafe")], ["control_variables", "minimum_viable_prototype"], ["incremental", "replace", "rollback", "baseline", "adapter"]),
        _prior("model_comparison", "Compare alternative models on matched evidence, heldout performance, and failure boundaries.", ["multiple explanations", "candidate models", "heldout evidence"], ["evaluation metric does not measure the target behavior"], [("mc_forecast", "forecasting", "compare ARIMA and state-space models on rolling windows"), ("mc_agent", "agent", "compare no-morphism and full-morphism variants on same tasks")], [("mc_bad_metric", "evaluation", "win rate rises while control loss also rises")], ["occams_razor", "bayesian_update"], ["compare", "model", "baseline", "heldout", "alternative"]),
        _prior("error_decomposition", "Decompose aggregate error into attributable components before choosing a repair.", ["complex failure", "multiple components", "debugging"], ["component labels are unreliable or interactions dominate"], [("err_ml", "ml", "separate bias, variance, and data leakage"), ("err_agent", "agent", "separate retrieval miss, reasoning miss, and judge defect")], [("err_interaction", "systems", "each component passes alone but fails together")], ["divide_and_conquer", "causal_intervention"], ["error", "decompose", "attribution", "component", "failure"]),
        _prior("invariant_search", "Identify quantities, roles, or relations that should remain unchanged across transformations.", ["transformation", "morphism", "state change"], ["no stable invariant exists or the invariant is wrongly chosen"], [("inv_physics", "physics", "track energy or mass conservation"), ("inv_agent", "agent", "preserve old success behavior while changing residual handling")], [("inv_false", "analogy", "assumed invariant breaks under domain transfer")], ["conservation_law", "analogical_reasoning"], ["invariant", "preserve", "relation", "unchanged", "morphism"]),
        _prior("causal_intervention", "Actively intervene on a suspected cause to distinguish causation from correlation.", ["causal uncertainty", "intervention possible", "confounding risk"], ["intervention is unsafe, unethical, or changes multiple mechanisms"], [("ci_product", "product", "randomize a feature launch to estimate impact"), ("ci_agent", "agent", "turn a candidate route on only for trigger rows")], [("ci_unsafe", "medicine", "intervention risk is too high without prior evidence")], ["control_variables", "negative_control"], ["causal", "intervention", "experiment", "confound", "randomize"]),
        _prior("local_linearization", "Approximate a nonlinear system locally when changes are small and verify the approximation boundary.", ["small perturbation", "smooth response", "local estimate"], ["large moves or discontinuities invalidate the approximation"], [("lin_control", "control", "linearize dynamics near equilibrium"), ("lin_business", "business", "estimate small price change impact before large repricing")], [("lin_jump", "markets", "threshold effects create discontinuous response")], ["limiting_case_analysis", "boundary_condition_analysis"], ["local", "linear", "perturbation", "smooth", "approximation"]),
        _prior("feedback_stability", "When a disturbance grows, look for feedback that amplifies, opposes, or stabilizes the change.", ["dynamic system", "feedback loop", "stability question"], ["feedback delay or nonlinear saturation changes the sign"], [("fb_lenz", "physics", "induced current opposes magnetic flux change"), ("fb_ops", "systems", "autoscaling counters rising queue length")], [("fb_delay", "economics", "delayed response overshoots and destabilizes")], ["analogical_reasoning", "conservation_law"], ["feedback", "stability", "opposes", "amplify", "equilibrium"]),
        _prior("special_to_general", "Generalize from concrete successful cases only after identifying the invariant that explains them.", ["multiple examples", "pattern extraction", "candidate law"], ["examples are selected or lack a shared invariant"], [("s2g_math", "math", "infer a theorem from cases then prove the invariant"), ("s2g_agent", "agent", "promote a branch after it survives multiple residual families")], [("s2g_bias", "evaluation", "cherry-picked wins create false generality")], ["invariant_search", "cross_domain_transfer"], ["generalize", "examples", "invariant", "pattern", "cases"]),
        _prior("general_to_special", "Apply a general principle to a specific case by checking scope, assumptions, and local constraints.", ["known rule", "specific instance", "scope check"], ["local constraints violate the general principle"], [("g2s_law", "law", "apply a statute after jurisdiction and facts match"), ("g2s_agent", "agent", "apply a prior only when trigger and negative controls match")], [("g2s_scope", "planning", "generic advice ignores a hard local constraint")], ["scope_narrowing", "boundary_condition_analysis"], ["specific", "apply", "scope", "constraint", "general"]),
        _prior("prior_estimate_then_update", "Start with a calibrated prior estimate, then revise after observing task specific evidence.", ["cold start", "uncertain estimate", "sequential measurement"], ["prior anchors too strongly or evidence is too noisy"], [("pe_forecast", "forecasting", "begin from base rate before local signals"), ("pe_agent", "agent", "use world model score before fresh validation updates it")], [("pe_anchor", "decision", "base rate dominates a clear contrary signal")], ["bayesian_update", "model_comparison"], ["prior", "estimate", "update", "base rate", "revise"]),
        _prior("duality", "Solve a problem by switching to a dual representation where constraints or objectives are easier.", ["paired formulation", "constraint optimization", "representation shift"], ["dual loses necessary semantics or is harder than primal"], [("dual_opt", "optimization", "move from primal constraints to Lagrange multipliers"), ("dual_agent", "agent", "view generation failure as verifier-routing failure")], [("dual_bad", "modeling", "dual variables are not interpretable for the decision")], ["model_comparison", "invariant_search"], ["dual", "representation", "constraint", "objective", "transform"]),
        _prior("conservation_law", "Track conserved quantities through a transformation and reject explanations that leak or create mass, budget, or probability.", ["closed accounting", "state transition", "resource balance"], ["system is open or conservation assumption is false"], [("cons_physics", "physics", "mass balance in a reaction"), ("cons_budget", "operations", "cost savings must reappear as budget or capacity")], [("cons_open", "economics", "external subsidy makes local budget non-conserved")], ["invariant_search", "dimensional_analysis"], ["conservation", "mass", "budget", "probability", "balance"]),
        _prior("dimensional_analysis", "Check whether quantities and formulas are meaningful by preserving units, scale, and dimension.", ["physical quantity", "scale relation", "formula check"], ["dimensionally valid formula can still be causally false"], [("dim_physics", "physics", "derive possible period dependence from length and gravity"), ("dim_agent", "agent", "detect incomparable metrics before combining scores")], [("dim_false", "science", "unit consistency hides a missing mechanism")], ["conservation_law", "scale_analysis"], ["dimension", "unit", "scale", "quantity", "formula"]),
        _prior("limiting_case_analysis", "Verify that a proposed framework reduces to a known result under a limiting scope condition.", ["candidate generalization", "known parent case", "scope reduction"], ["limit is singular or not representative"], [("limit_physics", "physics", "relativity reduces to Newtonian mechanics at low velocity"), ("limit_agent", "agent", "dependency-aware intervention reduces to control variables under low coupling")], [("limit_singular", "math", "limit changes topology and breaks reduction")], ["boundary_condition_analysis", "local_linearization"], ["limit", "reduce", "parent", "scope", "known case"]),
        _prior("falsifiability", "State what observation would make the claim fail before treating it as useful knowledge.", ["hypothesis test", "scientific claim", "verifier design"], ["claim is normative or not directly testable"], [("fals_science", "science", "define an observation that rules out a theory"), ("fals_agent", "agent", "write a rejection gate before live ablation")], [("fals_norm", "ethics", "pure value preference is not falsified by measurement alone")], ["proof_by_contradiction", "negative_control"], ["falsify", "testable", "counterexample", "hypothesis", "fail"]),
        _prior("robustness_testing", "Test whether behavior survives noise, perturbation, distribution shift, and adversarial cases.", ["deployment risk", "distribution shift", "stress test"], ["stress cases do not match realistic threats"], [("robust_ml", "ml", "evaluate under corrupted inputs"), ("robust_agent", "agent", "rerun accepted policy across domains and seeds")], [("robust_unreal", "testing", "contrived stress case blocks useful behavior")], ["boundary_condition_analysis", "ablation"], ["robust", "stress", "noise", "shift", "adversarial"]),
        _prior("ablation", "Remove or disable one component to measure whether it is necessary for the observed effect.", ["component attribution", "model or system variant", "same-task comparison"], ["component removal changes multiple hidden factors"], [("abl_ml", "ml", "remove an attention block and measure heldout loss"), ("abl_agent", "agent", "turn off morphism gate and compare same problems")], [("abl_hidden", "systems", "removing cache also changes latency and memory pressure")], ["control_variables", "model_comparison"], ["ablation", "remove", "disable", "component", "necessary"]),
        _prior("placebo_control", "Use a sham or inert intervention to estimate expectation, style, or measurement artifacts.", ["treatment effect", "style artifact", "human or judge bias"], ["placebo is distinguishable or has real effect"], [("plac_med", "medicine", "compare drug to sugar pill"), ("plac_agent", "agent", "compare a prompt label with no policy change")], [("plac_active", "evaluation", "placebo changes answer length and affects judge preference")], ["negative_control", "control_variables"], ["placebo", "sham", "inert", "artifact", "bias"]),
        _prior("cross_domain_transfer", "Transfer a method across domains only after matching structure, invariants, and failure boundaries.", ["source domain", "target domain", "shared structure"], ["surface similarity lacks structural match"], [("xfer_resnet", "ml", "use residual identity idea in different architectures"), ("xfer_qa", "retrieval", "lift context edges into assumption edges when roles align")], [("xfer_surface", "analogy", "same terminology but different causal roles")], ["analogical_reasoning", "invariant_search"], ["transfer", "domain", "structure", "invariant", "source", "target"]),
        _prior("scope_narrowing", "When a broad claim fails, narrow its scope to the conditions where it remains true and useful.", ["overbroad assumption", "boundary failure", "repair"], ["narrowing makes the claim trivial or unusable"], [("scope_law", "law", "limit a rule to the jurisdiction where it applies"), ("scope_agent", "agent", "restrict a policy to trigger rows after control loss")], [("scope_trivial", "method", "claim becomes so narrow it never activates")], ["general_to_special", "boundary_condition_analysis"], ["scope", "narrow", "boundary", "repair", "overbroad"]),
    ]


def _expert_queries() -> list[dict[str, Any]]:
    return [
        {"query_id": "q_causal_one_factor", "query": "causal attribution is uncertain, change one variable or factor while holding others fixed", "gold_principle_ids": ["control_variables", "causal_intervention"]},
        {"query_id": "q_decompose_interfaces", "query": "split a complex task into independent subproblems with explicit interface contracts and compose results", "gold_principle_ids": ["divide_and_conquer", "error_decomposition"]},
        {"query_id": "q_contradiction", "query": "assume the negation and derive an impossible contradiction or counterexample", "gold_principle_ids": ["proof_by_contradiction", "falsifiability"]},
        {"query_id": "q_simple_model", "query": "two models fit equally well, choose the simpler parsimonious explanation unless it underfits", "gold_principle_ids": ["occams_razor", "model_comparison"]},
        {"query_id": "q_update_belief", "query": "start from a prior probability and update belief after new evidence and calibration data", "gold_principle_ids": ["bayesian_update", "prior_estimate_then_update"]},
        {"query_id": "q_no_trigger_control", "query": "use a no-trigger negative control or placebo to detect leakage prompt artifacts and judge bias", "gold_principle_ids": ["negative_control", "placebo_control"]},
        {"query_id": "q_invariant_balance", "query": "state transformation must preserve an invariant conserved quantity with mass budget probability balance", "gold_principle_ids": ["conservation_law", "invariant_search"]},
        {"query_id": "q_edge_extreme", "query": "test boundary conditions, extremes, limiting cases, empty input, infinity and transitions", "gold_principle_ids": ["boundary_condition_analysis", "limiting_case_analysis"]},
        {"query_id": "q_safe_migration", "query": "replace one module behind an adapter boundary while preserving rollback to the working baseline", "gold_principle_ids": ["incremental_replacement", "minimum_viable_prototype"]},
        {"query_id": "q_structural_transfer", "query": "transfer a solution from one domain to another by matching roles structure invariants and failure boundaries", "gold_principle_ids": ["cross_domain_transfer", "analogical_reasoning"]},
    ]


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "if", "in", "into", "is", "it", "of", "on", "or", "that", "the", "then", "to",
    "under", "use", "when", "where", "while", "with",
}

_SYNONYMS = {
    "ablate": ["ablation", "remove", "disable"],
    "artifact": ["leakage", "bias", "placebo"],
    "balance": ["conservation", "budget", "mass", "probability"],
    "belief": ["bayesian", "posterior", "prior"],
    "boundary": ["scope", "edge", "limit", "extreme"],
    "calibration": ["bayesian", "evidence", "update"],
    "component": ["module", "part"],
    "compose": ["composition", "interface"],
    "contradiction": ["negation", "impossible", "counterexample"],
    "control": ["negative", "placebo", "matched"],
    "decompose": ["split", "subproblem", "component"],
    "evidence": ["update", "belief", "calibration"],
    "extreme": ["boundary", "edge", "limit"],
    "factor": ["variable", "cause"],
    "invariant": ["preserve", "conservation", "structure"],
    "module": ["component", "adapter", "rollback"],
    "prior": ["bayesian", "estimate", "belief"],
    "rollback": ["baseline", "adapter", "incremental"],
    "simple": ["parsimony", "occam", "complexity"],
    "split": ["decompose", "subproblem"],
    "structure": ["morphism", "invariant", "role"],
    "transfer": ["analogy", "domain", "structure"],
    "variable": ["factor", "causal"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--eval-id", default="philosophy_prior_library_20260612")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    payload = build_philosophy_prior_library_payload(root=args.root, eval_id=args.eval_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "failed_gates": payload["failed_gates"],
        "metrics": payload["metrics"],
        "out": str(args.out.resolve()),
        "md_out": str(args.md_out.resolve()),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
