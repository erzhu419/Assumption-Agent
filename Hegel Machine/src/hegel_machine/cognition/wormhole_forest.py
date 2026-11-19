"""Local Philosophical Implication Trees + Horizontal Mid-Layer Wormholes (Hegel V16.6).

Core architectural guarantees:
1. Universal 4-contract separation: ``TaskView``, ``ObservationMap`` (preserving
   ``LensTransform``), ``ScaleMap`` (optional coarse-graining + explicit closure
   scope), and ``ClaimContract`` — wired directly into ``question_semantics``.
2. Forest of local conditional implication trees (vertical edges = verified local
   implication or CEGAR state/probe refinement) interconnected across domains by
   first-class ``HorizontalWormholeEdge`` bridges between intermediate-layer nodes.
3. Non-transitive error budget composition on wormhole chains (Girard & Pappas 2007;
   ISS Prop. 4): pairwise epsilon-bridges never collapse into an equivalence class.
4. 4-tier permission gate (``retrieve_candidate`` -> ``bounded_trial`` ->
   ``local_solver_guidance`` -> ``validated_pruning``) replacing self-certified
   ``rg_irrelevant_slots``, plus bounded exploration budget for uncovered clusters.
5. 5-cause competing failure diagnosis + ALIGN near-miss condition split + CEGAR
   refinement + vertical backoff + clean wormhole exit when unprofitable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from statistics import mean, pstdev
from typing import Any

from .question_semantics import _strip_cosmetic, adopt_question_semantics, evaluate_decision_region


PERMISSION_ORDER = {
    "retrieve_candidate": 1,
    "bounded_trial": 2,
    "local_solver_guidance": 3,
    "validated_pruning": 4,
}

DIAGNOSIS_CAUSES = (
    "execution_error",
    "observation_map_mismatch",
    "protocol_or_scope_mismatch",
    "abstraction_closure_loss",
    "claim_refuted_or_unprofitable",
)


def canonical_digest(payload: Any) -> str:
    cleaned = _strip_cosmetic(payload)
    encoded = json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class TaskView:
    goal_kind: str
    target_metric: str
    tolerance: float
    input_domain: dict[str, Any] = field(default_factory=dict)
    budget_spec: dict[str, Any] = field(default_factory=dict)
    display_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if not out["display_label"]:
            out.pop("display_label")
        return out


@dataclass
class ObservationMap:
    raw_channel: str
    readout_transform: str = "identity"
    unit_scale: float = 1.0
    sign_convention: int = 1
    parameter_semantics: dict[str, str] = field(default_factory=dict)
    noise_std_estimate: float = 0.0
    clipping_ceiling: float | None = None
    display_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if not out["display_label"]:
            out.pop("display_label")
        return out


@dataclass
class ScaleMap:
    coarse_graining_op: str
    window_or_partition: dict[str, Any] = field(default_factory=dict)
    preserved_queries: list[str] = field(default_factory=list)
    closure_scope: str = "steady_state_only"  # exact | steady_state_only | transient_approx | not_closed
    proposed_discarded_variables: list[str] = field(default_factory=list)
    verified_discarded_in_scope: list[dict[str, Any]] = field(default_factory=list)
    error_budget: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimContract:
    claim_id: str
    family_plugin: str
    preconditions: list[dict[str, Any]]
    conclusion_or_action: dict[str, Any]
    protocol_spec: dict[str, Any]
    tolerance: float
    evidence_status: str = "candidate"  # candidate | SUPPORTED_IN_SCOPE | NOT_DETECTED_UNDER_PROTOCOL | REFUTED_IN_SCOPE | UNKNOWN
    permission_tier: str = "retrieve_candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_predicate_rule(rule: dict[str, Any], features: dict[str, Any]) -> str:
    """Evaluate one scope/inclusion/exclusion predicate; missing keys return UNKNOWN."""
    key = rule.get("field")
    if not key or key in {"task_id", "family_id", "hidden_family", "oracle", "ground_truth"}:
        return "UNKNOWN"
    if key not in features or features[key] is None:
        return "UNKNOWN"
    actual = features[key]
    op = rule.get("op", "==")
    expected = rule.get("value")
    if op == "==":
        return "TRUE" if actual == expected else "FALSE"
    if op == "!=":
        return "TRUE" if actual != expected else "FALSE"
    if op == "in":
        return "TRUE" if actual in expected else "FALSE"
    if op == "not_in":
        return "TRUE" if actual not in expected else "FALSE"
    if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(float(actual)):
        return "UNKNOWN"
    val = float(actual)
    exp = float(expected)
    if op == "<":
        return "TRUE" if val < exp else "FALSE"
    if op == "<=":
        return "TRUE" if val <= exp else "FALSE"
    if op == ">":
        return "TRUE" if val > exp else "FALSE"
    if op == ">=":
        return "TRUE" if val >= exp else "FALSE"
    return "UNKNOWN"


def _tier_name_from_rank(rank: int) -> str:
    clamped = max(1, min(4, int(rank)))
    return next(name for name, r in PERMISSION_ORDER.items() if r == clamped)


def _validate_evidence_receipt(
    evidence_receipt: dict[str, Any] | None,
    claim_contract: dict[str, Any],
    task_view: dict[str, Any],
) -> bool:
    if not isinstance(evidence_receipt, dict):
        return False
    if evidence_receipt.get("passed") is not True:
        return False
    expected_claim = claim_contract.get("claim_id")
    if expected_claim:
        if evidence_receipt.get("claim_id") != expected_claim or evidence_receipt.get("claim_id") == "*":
            return False
    elif not evidence_receipt.get("claim_id") or evidence_receipt.get("claim_id") == "*":
        return False

    if "observed_error" not in evidence_receipt:
        return False
    raw_err = evidence_receipt["observed_error"]
    if not isinstance(raw_err, (int, float)) or isinstance(raw_err, bool):
        return False
    err_val = float(raw_err)
    if not math.isfinite(err_val) or err_val < 0.0:
        return False
    tol = task_view.get("tolerance", claim_contract.get("tolerance"))
    if tol is not None:
        if not math.isfinite(float(tol)) or err_val > float(tol) + 1e-12:
            return False

    expected_target = task_view.get("target_id")
    rec_target = evidence_receipt.get("target_id")
    if not expected_target or not rec_target or rec_target != expected_target:
        return False

    expected_version = task_view.get("node_version", claim_contract.get("node_version"))
    rec_version = evidence_receipt.get("node_version")
    if expected_version is None or not isinstance(rec_version, int) or isinstance(rec_version, bool) or rec_version != expected_version:
        return False

    requires_model_binding = (
        task_view.get("goal_kind") == "open_workspace_mechanism_prediction"
        or str(claim_contract.get("family_plugin", "")).endswith("_ast")
        or task_view.get("expected_model_sha256") is not None
    )
    expected_model_sha = task_view.get("expected_model_sha256")
    rec_model_sha = evidence_receipt.get("model_sha256")
    if requires_model_binding:
        if not expected_model_sha or not rec_model_sha or rec_model_sha != expected_model_sha:
            return False

    requires_conf_digest = (
        task_view.get("goal_kind") == "open_workspace_mechanism_prediction"
        or str(claim_contract.get("family_plugin", "")) in {"m3_open_workspace_ast", "m2_open_workspace_ast", "open_code_agent_ast"}
        or task_view.get("expected_confirmation_digest") is not None
    )
    expected_conf_digest = task_view.get("expected_confirmation_digest")
    rec_conf_digest = evidence_receipt.get("confirmation_digest")
    if requires_conf_digest:
        if not expected_conf_digest or not rec_conf_digest or rec_conf_digest != expected_conf_digest:
            return False

    return True


def authorize_abstraction_use(
    *,
    requested_tier: str,
    task_view: dict[str, Any],
    target_features: dict[str, Any],
    claim_contract: dict[str, Any],
    scale_map: dict[str, Any] | None = None,
    wormhole_edge: dict[str, Any] | None = None,
    heldout_confirmed: bool = False,
    evidence_receipt: dict[str, Any] | None = None,
    implication_verified: bool = False,
    trial_budget_ok: bool = True,
) -> dict[str, Any]:
    """Enforce the 4-tier permission gate; proposed discarded variables & unknown exclusions never self-certify."""
    if requested_tier not in PERMISSION_ORDER:
        raise ValueError(f"Unsupported permission tier: {requested_tier}")

    receipt_ok = _validate_evidence_receipt(evidence_receipt, claim_contract, task_view)
    if evidence_receipt is not None:
        confirmed = bool(receipt_ok)
    elif task_view.get("goal_kind") == "open_workspace_mechanism_prediction" and task_view.get("target_id") is not None:
        confirmed = False
    else:
        confirmed = bool(heldout_confirmed)

    # Compute declared permission ceiling from ClaimContract & HorizontalWormholeEdge
    declared_max_rank = 4
    if "permission_tier" in claim_contract and claim_contract["permission_tier"] in PERMISSION_ORDER:
        declared_max_rank = min(declared_max_rank, PERMISSION_ORDER[claim_contract["permission_tier"]])
    if wormhole_edge is not None and wormhole_edge.get("permission_tier") in PERMISSION_ORDER:
        declared_max_rank = min(declared_max_rank, PERMISSION_ORDER[wormhole_edge["permission_tier"]])
    if receipt_ok and evidence_receipt.get("upgrade_to_tier") in PERMISSION_ORDER:
        upgrade_cap = PERMISSION_ORDER["validated_pruning"] if implication_verified else PERMISSION_ORDER["local_solver_guidance"]
        declared_max_rank = max(declared_max_rank, min(upgrade_cap, PERMISSION_ORDER[evidence_receipt["upgrade_to_tier"]]))

    def cap_tier(rank: int) -> str:
        return _tier_name_from_rank(min(rank, declared_max_rank))

    reasons: list[str] = []
    # Check explicit preconditions & ALIGN near-miss exclusions
    preconditions = list(claim_contract.get("preconditions", ()))
    exclusions: list[dict[str, Any]] = []
    if wormhole_edge is not None:
        align_conds = wormhole_edge.get("align_conditions", {})
        preconditions.extend(align_conds.get("inclusion_predicates", ()))
        child_required = {
            (r.get("field"), r.get("op"), r.get("value"))
            for r in claim_contract.get("preconditions", ())
            if isinstance(r, dict)
        }
        for ex_r in align_conds.get("exclusion_predicates", ()):
            if isinstance(ex_r, dict) and (ex_r.get("field"), ex_r.get("op"), ex_r.get("value")) in child_required:
                continue
            exclusions.append(ex_r)

    precond_results = [evaluate_predicate_rule(r, target_features) for r in preconditions]
    excl_results = [evaluate_predicate_rule(r, target_features) for r in exclusions]

    if "FALSE" in precond_results:
        return {
            "granted": False,
            "granted_tier": None,
            "requested_tier": requested_tier,
            "reason": "precondition_violated",
            "precondition_results": precond_results,
        }
    if "TRUE" in excl_results:
        return {
            "granted": False,
            "granted_tier": None,
            "requested_tier": requested_tier,
            "reason": "align_near_miss_exclusion_triggered",
            "exclusion_results": excl_results,
        }

    # Check ScaleMap closure & discarded variable verification for tiers >= local_solver_guidance
    if scale_map is not None and PERMISSION_ORDER[requested_tier] >= PERMISSION_ORDER["local_solver_guidance"]:
        goal = task_view.get("goal_kind", "")
        preserved = set(scale_map.get("preserved_queries", ()))
        if preserved and goal not in preserved:
            return {
                "granted": False,
                "granted_tier": "retrieve_candidate",
                "requested_tier": requested_tier,
                "reason": "scale_map_does_not_preserve_task_view",
            }
        if scale_map.get("closure_scope") == "not_closed":
            return {
                "granted": False,
                "granted_tier": cap_tier(PERMISSION_ORDER["bounded_trial"]),
                "requested_tier": requested_tier,
                "reason": "coarse_graining_not_closed_for_target",
            }
        proposed_discarded = set(scale_map.get("proposed_discarded_variables", ()))
        verified_entries = scale_map.get("verified_discarded_in_scope", ())
        verified_vars = {
            item["variable"] for item in verified_entries
            if item.get("task_view_goal") in {goal, "*"}
            and item.get("error_bound") is not None
            and float(item["error_bound"]) <= float(task_view.get("tolerance", 0.0))
            and item.get("evidence_ref")
        }
        unverified_discards = proposed_discarded - verified_vars
        if unverified_discards:
            reasons.append(f"unverified_proposed_discards:{sorted(unverified_discards)}")
            return {
                "granted": False,
                "granted_tier": cap_tier(PERMISSION_ORDER["bounded_trial"] if trial_budget_ok else 1),
                "requested_tier": requested_tier,
                "reason": "proposed_discarded_variables_require_in_scope_verification",
                "details": reasons,
            }

    # Wormhole error budget check
    if wormhole_edge is not None and PERMISSION_ORDER[requested_tier] >= PERMISSION_ORDER["local_solver_guidance"]:
        hop_err = float(wormhole_edge.get("hop_error_budget", 0.0))
        tol = float(task_view.get("tolerance", 0.0))
        if hop_err > tol and not confirmed:
            return {
                "granted": False,
                "granted_tier": cap_tier(PERMISSION_ORDER["bounded_trial"] if trial_budget_ok else 1),
                "requested_tier": requested_tier,
                "reason": "wormhole_error_exceeds_tolerance_without_target_confirmation",
                "hop_error_budget": hop_err,
                "task_tolerance": tol,
            }

    if requested_tier == "retrieve_candidate":
        return {"granted": True, "granted_tier": "retrieve_candidate", "requested_tier": requested_tier, "reason": "signature_recall_allowed"}

    # For tiers >= bounded_trial, check declared permission ceiling unless upgraded by receipt
    if PERMISSION_ORDER[requested_tier] > declared_max_rank:
        # Also note if unknown predicates/exclusions exist
        has_unknown_excl = "UNKNOWN" in excl_results
        has_unknown_prec = "UNKNOWN" in precond_results
        reason = (
            "unknown_exclusion_predicate_and_declared_permission_cap"
            if has_unknown_excl
            else ("unknown_precondition_and_declared_permission_cap" if has_unknown_prec else "exceeds_declared_permission_tier")
        )
        return {
            "granted": False,
            "granted_tier": _tier_name_from_rank(declared_max_rank),
            "requested_tier": requested_tier,
            "reason": reason,
            "declared_max_tier": _tier_name_from_rank(declared_max_rank),
        }

    if requested_tier == "bounded_trial":
        if not trial_budget_ok:
            return {"granted": False, "granted_tier": "retrieve_candidate", "requested_tier": requested_tier, "reason": "trial_budget_exceeded"}
        return {"granted": True, "granted_tier": "bounded_trial", "requested_tier": requested_tier, "reason": "bounded_trial_authorized"}

    if "UNKNOWN" in precond_results:
        return {
            "granted": False,
            "granted_tier": cap_tier(PERMISSION_ORDER["bounded_trial"] if trial_budget_ok else 1),
            "requested_tier": requested_tier,
            "reason": "unknown_precondition_requires_probe_before_guidance",
        }
    if "UNKNOWN" in excl_results:
        return {
            "granted": False,
            "granted_tier": cap_tier(PERMISSION_ORDER["bounded_trial"] if trial_budget_ok else 1),
            "requested_tier": requested_tier,
            "reason": "unknown_exclusion_predicate_requires_probe_before_guidance",
            "exclusion_results": excl_results,
        }

    if requested_tier == "local_solver_guidance":
        if claim_contract.get("evidence_status") not in {"SUPPORTED_IN_SCOPE", "derived"}:
            return {
                "granted": False,
                "granted_tier": cap_tier(PERMISSION_ORDER["bounded_trial"] if trial_budget_ok else 1),
                "requested_tier": requested_tier,
                "reason": "claim_not_supported_in_scope",
            }
        if not confirmed:
            return {
                "granted": False,
                "granted_tier": cap_tier(PERMISSION_ORDER["bounded_trial"]),
                "requested_tier": requested_tier,
                "reason": "heldout_target_point_confirmation_required",
            }
        return {"granted": True, "granted_tier": "local_solver_guidance", "requested_tier": requested_tier, "reason": "in_scope_alignment_and_heldout_confirmed"}

    # validated_pruning
    if not (implication_verified and confirmed and claim_contract.get("evidence_status") == "SUPPORTED_IN_SCOPE"):
        return {
            "granted": False,
            "granted_tier": cap_tier(PERMISSION_ORDER["local_solver_guidance"] if confirmed else PERMISSION_ORDER["bounded_trial"]),
            "requested_tier": requested_tier,
            "reason": "validated_pruning_requires_verified_implication_and_in_scope_support",
        }
    return {"granted": True, "granted_tier": "validated_pruning", "requested_tier": requested_tier, "reason": "verified_implication_pruning_authorized"}


def compose_wormhole_chain(edges: list[dict[str, Any]], *, target_tolerance: float) -> dict[str, Any]:
    """Compose horizontal wormhole hops WITHOUT assuming epsilon-closeness is transitive.

    For hop errors e_1, ..., e_k and Lipschitz factors L_1, ..., L_k:
      e_total = sum_{j=1}^k (prod_{m=j+1}^k L_m) * e_j.
    If e_total > target_tolerance, multi-hop chaining cannot inherit solver/pruning
    guarantees and downgrades to candidate retrieval / bounded trial.
    """
    if not edges:
        raise ValueError("compose_wormhole_chain requires at least one edge")
    k = len(edges)
    eps = [float(edge.get("hop_error_budget", 0.0)) for edge in edges]
    lips = [max(1.0, float(edge.get("lipschitz_factor", 1.0))) for edge in edges]
    total_err = 0.0
    for j in range(k):
        amplification = 1.0
        for m in range(j + 1, k):
            amplification *= lips[m]
        total_err += amplification * eps[j]
    within_tol = total_err <= float(target_tolerance) + 1e-12
    max_tier = (
        min((PERMISSION_ORDER.get(e.get("permission_tier", "retrieve_candidate"), 1) for e in edges))
        if within_tol else PERMISSION_ORDER["bounded_trial"]
    )
    tier_name = next(name for name, rank in PERMISSION_ORDER.items() if rank == max_tier)
    return {
        "hops": k,
        "cumulative_error_budget": total_err,
        "target_tolerance": float(target_tolerance),
        "within_tolerance": within_tol,
        "transitive_equivalence_assumed": False,
        "effective_permission_tier": tier_name,
        "edge_ids": [e["wormhole_id"] for e in edges],
    }


def fidelity_compress_nodes(
    node_a: dict[str, Any],
    node_b: dict[str, Any],
    evaluation_queries: list[dict[str, Any]],
    evaluator_fn,
) -> dict[str, Any]:
    """Check aligned Acceptance, Rejection, Conflict, AND Unknown preservation before merging.

    Fidelity compression requires:
      1. Zero direct conflicts: {res_a, res_b} != {'ACC', 'REJ'} on any query;
      2. Zero unauthorized promotions from {'UNK', 'UNK'} to 'ACC' or 'REJ';
      3. Zero lost accepted cases: Acc(A) U Acc(B) <= Acc(merged) when non-conflicting;
      4. Zero lost true rejections: Rej(A) U Rej(B) <= Rej(merged) when non-conflicting;
      5. Zero newly fabricated false rejections on valid cases.
    """
    merged_preconditions = list({
        json.dumps(p, sort_keys=True): deepcopy(p)
        for p in [*node_a["claim_contract"]["preconditions"], *node_b["claim_contract"]["preconditions"]]
    }.values())
    merged_node = deepcopy(node_a)
    merged_node["node_id"] = f"merged:{node_a['node_id']}+{node_b['node_id']}"
    merged_node["claim_contract"]["preconditions"] = merged_preconditions
    merged_node["claim_contract"]["disjunctive_branches"] = [
        deepcopy(node_a["claim_contract"]),
        deepcopy(node_b["claim_contract"]),
    ]

    lost_accepts, lost_rejects, false_new_rejects = [], [], []
    conflicting_verdicts, unauthorized_promotions = [], []
    for idx, q in enumerate(evaluation_queries):
        res_a = evaluator_fn(node_a, q)  # 'ACC' | 'REJ' | 'UNK' (UNK = out of scope / undecided)
        res_b = evaluator_fn(node_b, q)
        res_m = evaluator_fn(merged_node, q)
        # 0a. Contradictory source nodes ({ACC, REJ}) CANNOT be fidelity-compressed!
        if {res_a, res_b} == {"ACC", "REJ"}:
            conflicting_verdicts.append(idx)
            continue
        # 0b. Merged node must NEVER promote {'UNK', 'UNK'} into 'ACC' or 'REJ'!
        if res_a == "UNK" and res_b == "UNK" and res_m != "UNK":
            unauthorized_promotions.append(idx)
        # 1. Any query accepted by an applicable branch (without a genuine empirical REJ) must stay ACC
        if ("ACC" in {res_a, res_b}) and ("REJ" not in {res_a, res_b}) and res_m != "ACC":
            lost_accepts.append(idx)
        # 2. Any genuine empirical rejection must be preserved
        if ("REJ" in {res_a, res_b}) and res_m != "REJ":
            lost_rejects.append(idx)
        # 3. Merged rule must NEVER fabricate a new REJ where neither branch rejected
        if res_m == "REJ" and ("REJ" not in {res_a, res_b}):
            false_new_rejects.append(idx)

    valid = not (
        lost_accepts
        or lost_rejects
        or false_new_rejects
        or conflicting_verdicts
        or unauthorized_promotions
    )
    return {
        "kind": "FidelityCompression",
        "accepted": valid,
        "merged_node": merged_node if valid else None,
        "lost_accept_indices": lost_accepts,
        "lost_reject_indices": lost_rejects,
        "false_new_reject_indices": false_new_rejects,
        "conflicting_verdict_indices": conflicting_verdicts,
        "unauthorized_unknown_promotion_indices": unauthorized_promotions,
    }


class PhilosophicalForest:
    """Forest of Local Conditional Implication Trees + Horizontal Mid-Layer Wormholes."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_history: dict[str, list[dict[str, Any]]] = {}
        self.local_implication_edges: list[dict[str, Any]] = []
        self.coarse_graining_edges: list[dict[str, Any]] = []
        self.horizontal_wormhole_edges: dict[str, dict[str, Any]] = {}
        self.wormhole_history: dict[str, list[dict[str, Any]]] = {}
        self.conflict_relations: list[dict[str, Any]] = []
        self.raw_episode_archive: list[dict[str, Any]] = []
        self.questions: dict[str, dict[str, Any]] = {}

    def record_raw_episode(self, episode: dict[str, Any]) -> None:
        self.raw_episode_archive.append(deepcopy(episode))

    def register_node(
        self,
        *,
        node_id: str,
        tree_id: str,
        layer_role: str,  # 'root' | 'mid' | 'leaf'
        task_view: TaskView | dict[str, Any],
        observation_map: ObservationMap | dict[str, Any],
        claim_contract: ClaimContract | dict[str, Any],
        scale_map: ScaleMap | dict[str, Any] | None = None,
        probe_signatures: list[str] | None = None,
        program_steps: list[dict[str, Any]] | None = None,
        question_spec: dict[str, Any] | None = None,
        support_paths: list[list[dict[str, Any]]] | None = None,
        source_cluster_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        tv = task_view.to_dict() if isinstance(task_view, TaskView) else deepcopy(task_view)
        om = observation_map.to_dict() if isinstance(observation_map, ObservationMap) else deepcopy(observation_map)
        sm = scale_map.to_dict() if isinstance(scale_map, ScaleMap) else deepcopy(scale_map)
        cc = claim_contract.to_dict() if isinstance(claim_contract, ClaimContract) else deepcopy(claim_contract)
        prog = deepcopy(program_steps) if program_steps is not None else []

        sem_digest = canonical_digest({
            "task_view": tv,
            "observation_map": om,
            "scale_map": sm,
            "claim_contract": cc,
            "program_steps": prog,
        })
        existing = self.nodes.get(node_id)
        version = 1 if existing is None else existing["version"] + 1
        if existing is not None:
            self.node_history.setdefault(node_id, []).append(deepcopy(existing))

        q_record = None
        if question_spec is not None:
            base_q = {
                k: deepcopy(v) for k, v in question_spec.items()
                if k not in {"meaning_version", "meaning_digest", "meaning_history"}
            }
            incoming_q = {
                **base_q,
                "task_view": tv,
                "observation_map": om,
                "protocol_spec": deepcopy(cc.get("protocol_spec", {})),
                "claim_contract": cc,
                **({"scale_map": sm} if sm is not None else {}),
            }
            q_record = adopt_question_semantics(self.questions.get(incoming_q["id"]), incoming_q)
            self.questions[incoming_q["id"]] = q_record

        node = {
            "node_id": node_id,
            "version": version,
            "semantic_digest": sem_digest,
            "tree_id": tree_id,
            "layer_role": layer_role,
            "task_view": tv,
            "observation_map": om,
            "scale_map": sm,
            "claim_contract": cc,
            "probe_signatures": list(probe_signatures or ()),
            "program_steps": prog,
            "question_id": q_record["id"] if q_record else None,
            "question_meaning_version": q_record["meaning_version"] if q_record else None,
            "support_paths": deepcopy(support_paths) if support_paths is not None else [],
            "source_cluster_ids": list(dict.fromkeys(source_cluster_ids or ())),
            "utility_ledger": (
                deepcopy(existing["utility_ledger"])
                if existing is not None
                else {"trials": 0, "successes": 0, "total_savings": 0.0, "total_overhead": 0.0, "archived_low_utility": False}
            ),
        }
        self.nodes[node_id] = node
        return deepcopy(node)

    def add_local_implication_edge(
        self,
        *,
        stronger_node_id: str,
        weaker_node_id: str,
        preserved_contract: dict[str, Any],
        scope: dict[str, Any],
        evidence_status: str = "derived",
    ) -> dict[str, Any]:
        stronger = self.nodes[stronger_node_id]
        weaker = self.nodes[weaker_node_id]
        if stronger["tree_id"] != weaker["tree_id"]:
            raise ValueError(
                "Local implication edges are intra-tree only; use HorizontalWormholeEdge across distinct trees!"
            )
        # Prevent cycles in local implication DAG
        if self._can_reach_implication(weaker_node_id, stronger_node_id):
            raise ValueError("Cycle detected in local_implication_edges")
        edge = {
            "stronger_node_id": stronger_node_id,
            "stronger_version": stronger["version"],
            "weaker_node_id": weaker_node_id,
            "weaker_version": weaker["version"],
            "preserved_contract": deepcopy(preserved_contract),
            "scope": deepcopy(scope),
            "evidence_status": evidence_status,
        }
        self.local_implication_edges.append(edge)
        return deepcopy(edge)

    def _can_reach_implication(self, start: str, goal: str) -> bool:
        stack, seen = [start], set()
        while stack:
            cur = stack.pop()
            if cur == goal:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            for edge in self.local_implication_edges:
                if edge["stronger_node_id"] == cur:
                    stack.append(edge["weaker_node_id"])
        return False

    def weaker_parents_of(self, node_id: str) -> list[dict[str, Any]]:
        """Return verified weaker parent nodes bound strictly to current live endpoint versions."""
        if node_id not in self.nodes:
            return []
        cur_stronger_ver = self.nodes[node_id]["version"]
        parents = []
        for edge in self.local_implication_edges:
            w_id = edge["weaker_node_id"]
            if (edge["stronger_node_id"] == node_id
                    and edge["stronger_version"] == cur_stronger_ver
                    and edge["evidence_status"] in {"derived", "SUPPORTED_IN_SCOPE"}
                    and w_id in self.nodes
                    and edge["weaker_version"] == self.nodes[w_id]["version"]):
                parents.append(deepcopy(self.nodes[w_id]))
        return parents

    def _can_reach_coarse_graining(self, start: str, goal: str) -> bool:
        stack, seen = [start], set()
        while stack:
            cur = stack.pop()
            if cur == goal:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            for edge in self.coarse_graining_edges:
                if edge["fine_node_id"] == cur:
                    stack.append(edge["coarse_node_id"])
        return False

    def add_coarse_graining_edge(
        self,
        *,
        fine_node_id: str,
        coarse_node_id: str,
        scale_map: ScaleMap | dict[str, Any],
        empirical_closure_certificate: dict[str, Any],
        evidence_status: str = "SUPPORTED_IN_SCOPE",
    ) -> dict[str, Any]:
        """Connect a fine-scale hypothesis node to its coarse-grained emergent node via operator Pi_s."""
        fine = self.nodes[fine_node_id]
        coarse = self.nodes[coarse_node_id]
        if fine["tree_id"] != coarse["tree_id"]:
            raise ValueError("Coarse-graining edges connect scales within the same local system tree")
        if self._can_reach_coarse_graining(coarse_node_id, fine_node_id):
            raise ValueError("Cycle detected in coarse_graining_edges")
        sm = scale_map.to_dict() if isinstance(scale_map, ScaleMap) else deepcopy(scale_map)
        edge = {
            "fine_node_id": fine_node_id,
            "fine_version": fine["version"],
            "coarse_node_id": coarse_node_id,
            "coarse_version": coarse["version"],
            "scale_map": sm,
            "empirical_closure_certificate": deepcopy(empirical_closure_certificate),
            "evidence_status": evidence_status,
        }
        self.coarse_graining_edges.append(edge)
        return deepcopy(edge)

    def fine_scale_children_of(self, coarse_node_id: str) -> list[dict[str, Any]]:
        """Return live fine-scale underlying nodes when coarse-graining closure breaks (dialectical escalation)."""
        if coarse_node_id not in self.nodes:
            return []
        cur_ver = self.nodes[coarse_node_id]["version"]
        out = []
        for edge in self.coarse_graining_edges:
            f_id = edge["fine_node_id"]
            if (
                edge["coarse_node_id"] == coarse_node_id
                and edge["coarse_version"] == cur_ver
                and f_id in self.nodes
                and edge["fine_version"] == self.nodes[f_id]["version"]
            ):
                out.append(deepcopy(self.nodes[f_id]))
        return out

    def coarse_scale_parents_of(self, fine_node_id: str) -> list[dict[str, Any]]:
        """Return live coarse-grained emergent nodes reachable via Pi_s from ``fine_node_id``."""
        if fine_node_id not in self.nodes:
            return []
        cur_ver = self.nodes[fine_node_id]["version"]
        out = []
        for edge in self.coarse_graining_edges:
            c_id = edge["coarse_node_id"]
            if (
                edge["fine_node_id"] == fine_node_id
                and edge["fine_version"] == cur_ver
                and c_id in self.nodes
                and edge["coarse_version"] == self.nodes[c_id]["version"]
            ):
                out.append(deepcopy(self.nodes[c_id]))
        return out

    def add_horizontal_wormhole(
        self,
        *,
        wormhole_id: str,
        node_a_id: str,
        node_b_id: str,
        probe_signature_refs: list[str],
        role_and_observation_map: dict[str, Any],
        hop_error_budget: float,
        lipschitz_factor: float = 1.0,
        align_conditions: dict[str, Any] | None = None,
        permission_tier: str = "bounded_trial",
        compatible_goal_kinds: list[str] | None = None,
        source_episode_refs: list[str] | None = None,
        scale_regime: str | None = None,
    ) -> dict[str, Any]:
        node_a = self.nodes[node_a_id]
        node_b = self.nodes[node_b_id]
        if node_a["tree_id"] == node_b["tree_id"]:
            raise ValueError("HorizontalWormholeEdge must connect nodes across distinct local trees")
        if node_a["layer_role"] != "mid" or node_b["layer_role"] != "mid":
            raise ValueError("HorizontalWormholeEdge connects intermediate-layer ('mid') nodes across trees")
        prev = self.horizontal_wormhole_edges.get(wormhole_id)
        version = 1 if prev is None else prev["version"] + 1
        if prev is not None:
            self.wormhole_history.setdefault(wormhole_id, []).append(deepcopy(prev))
        edge = {
            "wormhole_id": wormhole_id,
            "version": version,
            "node_a_id": node_a_id,
            "node_a_version": node_a["version"],
            "tree_a_id": node_a["tree_id"],
            "node_b_id": node_b_id,
            "node_b_version": node_b["version"],
            "tree_b_id": node_b["tree_id"],
            "probe_signature_refs": list(probe_signature_refs),
            "role_and_observation_map": deepcopy(role_and_observation_map),
            "hop_error_budget": float(hop_error_budget),
            "lipschitz_factor": float(lipschitz_factor),
            "align_conditions": deepcopy(align_conditions or {
                "characteristic_predicates": [],
                "inclusion_predicates": [],
                "exclusion_predicates": [],
                "revision_provenance": [],
            }),
            "permission_tier": permission_tier,
            "compatible_goal_kinds": list(compatible_goal_kinds) if compatible_goal_kinds is not None else [],
            "stale_endpoint_revision": False,
            "source_episode_refs": list(source_episode_refs or ()),
            **({"scale_regime": scale_regime} if scale_regime is not None else {}),
        }
        self.horizontal_wormhole_edges[wormhole_id] = edge
        return deepcopy(edge)

    def revalidate_wormhole_after_revision(
        self,
        wormhole_id: str,
        *,
        witness_episode_id: str,
        updated_role_and_observation_map: dict[str, Any] | None = None,
        updated_permission_tier: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Explicitly revalidate & rebind a HorizontalWormholeEdge to revised live endpoint versions."""
        edge = deepcopy(self.horizontal_wormhole_edges[wormhole_id])
        self.wormhole_history.setdefault(wormhole_id, []).append(deepcopy(edge))
        live_a = self.nodes[edge["node_a_id"]]
        live_b = self.nodes[edge["node_b_id"]]
        edge["node_a_version"] = live_a["version"]
        edge["node_b_version"] = live_b["version"]
        edge["stale_endpoint_revision"] = False
        if updated_role_and_observation_map is not None:
            edge["role_and_observation_map"] = deepcopy(updated_role_and_observation_map)
        if updated_permission_tier is not None:
            edge["permission_tier"] = updated_permission_tier
        edge["align_conditions"].setdefault("revision_provenance", []).append({
            "episode_id": witness_episode_id,
            "action": "revalidate_endpoint_versions",
            "node_a_version": live_a["version"],
            "node_b_version": live_b["version"],
            "reason": reason,
        })
        edge["version"] += 1
        self.horizontal_wormhole_edges[wormhole_id] = edge
        return deepcopy(edge)

    def refine_wormhole_with_near_miss(
        self,
        wormhole_id: str,
        *,
        near_miss_episode_id: str,
        exclusion_predicate: dict[str, Any] | None = None,
        inclusion_predicate: dict[str, Any] | None = None,
        retract_overstrict_inclusion: dict[str, Any] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """ALIGN-style near-miss condition refinement on a HorizontalWormholeEdge."""
        edge = deepcopy(self.horizontal_wormhole_edges[wormhole_id])
        self.wormhole_history.setdefault(wormhole_id, []).append(deepcopy(edge))
        conds = edge["align_conditions"]
        if exclusion_predicate and exclusion_predicate not in conds["exclusion_predicates"]:
            conds["exclusion_predicates"].append(deepcopy(exclusion_predicate))
        if inclusion_predicate and inclusion_predicate not in conds["inclusion_predicates"]:
            conds["inclusion_predicates"].append(deepcopy(inclusion_predicate))
        if retract_overstrict_inclusion and retract_overstrict_inclusion in conds["inclusion_predicates"]:
            conds["inclusion_predicates"].remove(retract_overstrict_inclusion)
        conds["revision_provenance"].append({
            "episode_id": near_miss_episode_id,
            "added_exclusion": deepcopy(exclusion_predicate),
            "added_inclusion": deepcopy(inclusion_predicate),
            "retracted_inclusion": deepcopy(retract_overstrict_inclusion),
            "reason": reason,
        })
        edge["version"] += 1
        self.horizontal_wormhole_edges[wormhole_id] = edge
        return deepcopy(edge)

    def empirical_revise_node(
        self,
        node_id: str,
        *,
        witness_episode_id: str,
        cause: str,
        updated_task_view: dict[str, Any] | None = None,
        updated_observation_map: dict[str, Any] | None = None,
        updated_scale_map: dict[str, Any] | None = None,
        updated_claim_contract: dict[str, Any] | None = None,
        updated_program_steps: list[dict[str, Any]] | None = None,
        updated_question_spec: dict[str, Any] | None = None,
        reinterpreted_counterexample_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Versioned empirical revision (propagates meaning version to QuestionSpec & marks stale edges)."""
        if cause not in DIAGNOSIS_CAUSES:
            raise ValueError(f"Unknown diagnosis cause: {cause}")
        old = self.nodes[node_id]
        q_spec_to_pass = updated_question_spec
        if q_spec_to_pass is None and old.get("question_id") and old["question_id"] in self.questions:
            q_spec_to_pass = deepcopy(self.questions[old["question_id"]])

        updated = self.register_node(
            node_id=node_id,
            tree_id=old["tree_id"],
            layer_role=old["layer_role"],
            task_view=updated_task_view if updated_task_view is not None else old["task_view"],
            observation_map=updated_observation_map if updated_observation_map is not None else old["observation_map"],
            scale_map=updated_scale_map if updated_scale_map is not None else old["scale_map"],
            claim_contract=updated_claim_contract if updated_claim_contract is not None else old["claim_contract"],
            probe_signatures=old["probe_signatures"],
            program_steps=updated_program_steps if updated_program_steps is not None else old.get("program_steps", []),
            question_spec=q_spec_to_pass,
            support_paths=old["support_paths"],
            source_cluster_ids=old["source_cluster_ids"],
        )
        updated["revision_record"] = {
            "kind": "EmpiricalRevision",
            "previous_version": old["version"],
            "previous_semantic_digest": old["semantic_digest"],
            "witness_episode_id": witness_episode_id,
            "diagnosis_cause": cause,
            "reinterpreted_counterexample_ids": list(reinterpreted_counterexample_ids or ()),
        }
        self.nodes[node_id] = updated
        # Mark incident horizontal wormholes stale until explicitly revalidated!
        for wh in self.horizontal_wormhole_edges.values():
            if ((wh["node_a_id"] == node_id and wh["node_a_version"] != updated["version"]) or
                    (wh["node_b_id"] == node_id and wh["node_b_version"] != updated["version"])):
                wh["stale_endpoint_revision"] = True
        return deepcopy(updated)

    def recall_mid_layer_wormholes(
        self,
        observed_signatures: list[str],
        *,
        task_view: dict[str, Any],
        allow_horizontal_wormholes: bool = True,
    ) -> list[dict[str, Any]]:
        """Recall candidate intermediate nodes & live-version horizontal wormholes by ProbeSignatures."""
        sig_set = set(observed_signatures)
        goal = task_view.get("goal_kind")
        candidates: list[dict[str, Any]] = []
        for node in self.nodes.values():
            if node["utility_ledger"].get("archived_low_utility"):
                continue
            if node["claim_contract"].get("evidence_status") == "REFUTED_IN_SCOPE":
                continue
            overlap = sorted(sig_set & set(node.get("probe_signatures", ())))
            if not overlap:
                continue
            # Direct node match
            candidates.append({
                "entry_kind": "local_tree_node",
                "target_node": deepcopy(node),
                "wormhole_edge": None,
                "matched_signatures": overlap,
                "score": len(overlap),
            })
            if not allow_horizontal_wormholes or node["layer_role"] != "mid":
                continue
            # Horizontal mid-layer wormhole bridge jumps across distinct trees!
            for wh in self.horizontal_wormhole_edges.values():
                if wh.get("stale_endpoint_revision"):
                    continue
                if wh["node_a_id"] not in self.nodes or wh["node_b_id"] not in self.nodes:
                    continue
                live_a = self.nodes[wh["node_a_id"]]
                live_b = self.nodes[wh["node_b_id"]]
                if wh["node_a_version"] != live_a["version"] or wh["node_b_version"] != live_b["version"]:
                    continue
                compat_goals = wh.get("compatible_goal_kinds") or ()
                if compat_goals and goal is not None and goal not in compat_goals:
                    continue
                peer_id = None
                if wh["node_a_id"] == node["node_id"]:
                    peer_id = wh["node_b_id"]
                elif wh["node_b_id"] == node["node_id"]:
                    peer_id = wh["node_a_id"]
                if peer_id is None:
                    continue
                peer = self.nodes[peer_id]
                if peer["utility_ledger"].get("archived_low_utility"):
                    continue
                if peer["claim_contract"].get("evidence_status") == "REFUTED_IN_SCOPE":
                    continue
                wh_overlap = sorted(sig_set & set(wh["probe_signature_refs"]))
                if wh_overlap:
                    candidates.append({
                        "entry_kind": "horizontal_mid_wormhole",
                        "source_mid_node_id": node["node_id"],
                        "target_node": deepcopy(peer),
                        "wormhole_edge": deepcopy(wh),
                        "matched_signatures": wh_overlap,
                        "score": len(wh_overlap) + 0.5,
                    })
        candidates.sort(key=lambda item: (-item["score"], item["target_node"]["node_id"]))
        return candidates

    def without_node_version(self, node_id: str, version: int) -> "PhilosophicalForest":
        """Clean versioned object removal preserving raw episodes and independent OR-supported objects."""
        clone = deepcopy(self)
        cur = clone.nodes.get(node_id)
        if cur is None or cur["version"] != version:
            return clone
        clone.nodes.pop(node_id)
        # Remove wormholes incident to this removed version
        clone.horizontal_wormhole_edges = {
            wid: edge for wid, edge in clone.horizontal_wormhole_edges.items()
            if not ((edge["node_a_id"] == node_id and edge["node_a_version"] == version) or
                    (edge["node_b_id"] == node_id and edge["node_b_version"] == version))
        }
        clone.local_implication_edges = [
            e for e in clone.local_implication_edges
            if not ((e["stronger_node_id"] == node_id and e["stronger_version"] == version) or
                    (e["weaker_node_id"] == node_id and e["weaker_version"] == version))
        ]
        clone.coarse_graining_edges = [
            e for e in clone.coarse_graining_edges
            if not ((e["fine_node_id"] == node_id and e["fine_version"] == version) or
                    (e["coarse_node_id"] == node_id and e["coarse_version"] == version))
        ]
        # Propagate AND/OR support invalidation: each path in support_paths is an AND-conjunction;
        # a node survives iff it has no support_paths dependency or >= 1 complete OR-path remains!
        changed = True
        while changed:
            changed = False
            for nid in list(clone.nodes.keys()):
                paths = clone.nodes[nid].get("support_paths")
                if not paths:
                    continue
                surviving_paths = []
                for and_premises in paths:
                    all_hold = all(
                        prem["node_id"] in clone.nodes
                        and clone.nodes[prem["node_id"]]["version"] == prem.get("version", clone.nodes[prem["node_id"]]["version"])
                        for prem in and_premises
                    )
                    if all_hold:
                        surviving_paths.append(and_premises)
                if not surviving_paths:
                    clone.nodes.pop(nid)
                    changed = True
                else:
                    clone.nodes[nid]["support_paths"] = surviving_paths
        # Raw episode archive stays 100% intact so ordinary/rebuild baselines are never robbed of history!
        return clone

    def record_utility_feedback(self, node_id: str, *, saved_cost: float, overhead_cost: float, succeeded: bool) -> None:
        if node_id not in self.nodes:
            return
        led = self.nodes[node_id]["utility_ledger"]
        led["trials"] += 1
        led["successes"] += int(bool(succeeded))
        led["total_savings"] += float(saved_cost)
        led["total_overhead"] += float(overhead_cost)
        # Minton utility problem guard: demote persistently net-negative nodes after >= 3 trials
        if led["trials"] >= 3 and (led["total_savings"] - led["total_overhead"]) < -2.0:
            led["archived_low_utility"] = True

    def compress_compatible_nodes(
        self,
        *,
        keeper_node_id: str,
        redundant_node_id: str,
        evaluation_queries: list[dict[str, Any]],
        evaluator_fn,
    ) -> dict[str, Any]:
        """Run zero-conflict, zero-false-rejection FidelityCompression and archive redundant duplicate."""
        if keeper_node_id not in self.nodes or redundant_node_id not in self.nodes:
            return {"accepted": False, "reason": "missing_node"}
        node_a = self.nodes[keeper_node_id]
        node_b = self.nodes[redundant_node_id]
        if node_a["utility_ledger"].get("archived_low_utility") or node_b["utility_ledger"].get("archived_low_utility"):
            return {"accepted": False, "reason": "already_archived"}
        report = fidelity_compress_nodes(node_a, node_b, evaluation_queries, evaluator_fn)
        if not report["accepted"]:
            return report
        # Union observable signatures & provenance on keeper while preserving versioned history
        merged_sigs = list(dict.fromkeys([*node_a.get("probe_signatures", ()), *node_b.get("probe_signatures", ())]))
        node_a["probe_signatures"] = merged_sigs
        node_a["source_cluster_ids"] = list(
            dict.fromkeys([*node_a.get("source_cluster_ids", ()), *node_b.get("source_cluster_ids", ())])
        )
        node_a.setdefault("fidelity_merged_members", []).append({
            "redundant_node_id": redundant_node_id,
            "redundant_version": node_b["version"],
            "query_count": len(evaluation_queries),
        })
        node_b["utility_ledger"]["archived_low_utility"] = True
        node_b["utility_ledger"]["archived_reason"] = "fidelity_compressed"
        node_b["utility_ledger"]["merged_into_node_id"] = keeper_node_id
        # Rebind live wormholes from redundant_node_id to keeper_node_id when across distinct trees
        for wid, wh in list(self.horizontal_wormhole_edges.items()):
            if wh.get("stale_endpoint_revision"):
                continue
            if wh["node_a_id"] == redundant_node_id and wh["tree_b_id"] != node_a["tree_id"]:
                wh["node_a_id"] = keeper_node_id
                wh["node_a_version"] = node_a["version"]
                wh["tree_a_id"] = node_a["tree_id"]
            elif wh["node_b_id"] == redundant_node_id and wh["tree_a_id"] != node_a["tree_id"]:
                wh["node_b_id"] = keeper_node_id
                wh["node_b_version"] = node_a["version"]
                wh["tree_b_id"] = node_a["tree_id"]
        return {
            **report,
            "keeper_node_id": keeper_node_id,
            "archived_redundant_node_id": redundant_node_id,
        }


class TwoStageWormholeGate:
    """SATzilla Stage-0 presolver + Stage-1 paired same-state cost gate + uncovered exploration arm."""

    def __init__(
        self,
        *,
        margin: float = 0.5,
        probe_budget_fraction_cap: float = 0.25,
        uncovered_exploration_budget_calls: int = 2,
    ) -> None:
        self.margin = float(margin)
        self.probe_budget_fraction_cap = float(probe_budget_fraction_cap)
        self.uncovered_exploration_budget_calls = int(uncovered_exploration_budget_calls)
        self.exploration_uses_by_signature: dict[str, int] = {}
        # Paired development observations keyed by cluster signature: list of (transfer_rem_cost, direct_rem_cost, transfer_ok, direct_ok)
        self.paired_dev_records: dict[str, list[tuple[float, float, bool, bool]]] = {}

    def observe_paired_dev_rollout(
        self,
        signature_key: str,
        *,
        transfer_remaining_cost: float,
        direct_remaining_cost: float,
        transfer_quality_ok: bool,
        direct_quality_ok: bool,
    ) -> None:
        self.paired_dev_records.setdefault(signature_key, []).append((
            float(transfer_remaining_cost),
            float(direct_remaining_cost),
            bool(transfer_quality_ok),
            bool(direct_quality_ok),
        ))

    def decide_stage1(
        self,
        signature_key: str,
        *,
        remaining_budget: float,
        estimated_probe_and_bind_cost: float,
        allow_exploration_on_uncovered: bool = True,
    ) -> dict[str, Any]:
        if estimated_probe_and_bind_cost > self.probe_budget_fraction_cap * remaining_budget:
            return {
                "admit_transfer": False,
                "mode": "rejected_probe_cap",
                "reason": "estimated_probe_exceeds_remaining_budget_fraction_cap",
            }
        records = self.paired_dev_records.get(signature_key, ())
        if len(records) < 2:
            used = self.exploration_uses_by_signature.get(signature_key, 0)
            if allow_exploration_on_uncovered and used < self.uncovered_exploration_budget_calls:
                self.exploration_uses_by_signature[signature_key] = used + 1
                return {
                    "admit_transfer": True,
                    "mode": "exploration_arm_no_prior_guarantee",
                    "reason": "uncovered_cluster_bounded_exploration_budget",
                    "exploration_index": used + 1,
                }
            return {
                "admit_transfer": False,
                "mode": "direct_fallback_uncovered",
                "reason": "insufficient_dev_coverage_and_exploration_budget_spent",
            }
        # Quality first: never admit transfer if dev quality degraded relative to direct!
        transfer_pass_rate = mean(1.0 if r[2] else 0.0 for r in records)
        direct_pass_rate = mean(1.0 if r[3] else 0.0 for r in records)
        if transfer_pass_rate + 1e-9 < direct_pass_rate:
            return {
                "admit_transfer": False,
                "mode": "rejected_quality_guard",
                "reason": "dev_transfer_pass_rate_below_direct",
                "transfer_pass_rate": transfer_pass_rate,
                "direct_pass_rate": direct_pass_rate,
            }
        paired_diffs = [r[0] - r[1] for r in records]
        n = len(records)
        mean_paired_diff = mean(paired_diffs)
        se_paired_diff = pstdev(paired_diffs) / math.sqrt(n)
        empirical_upper_diff = mean_paired_diff + 0.35 * se_paired_diff
        admit = (empirical_upper_diff < -self.margin) or (transfer_pass_rate > direct_pass_rate)
        return {
            "admit_transfer": bool(admit),
            "mode": "calibrated_paired_diff_empirical_gate",
            "mean_paired_diff": mean_paired_diff,
            "empirical_upper_diff": empirical_upper_diff,
            "margin": self.margin,
            "sample_count": n,
        }
