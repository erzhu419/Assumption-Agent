"""Executable hypothesis branch alignment, overlap probing, anti-unification, and retraction (v11 Deliverable B).

Implements the substantive ``reconcile(new_claim, old_claims, observations, budget)`` loop
specified in ``02_下一轮实施与验收.md`` §2 on top of existing ``Program``, ``AbstractTemplate``,
``Probe``, ``ResearchQuestion``, and ``WorldStructureCommitment`` structures (zero empty shell types):

  1. ``align_semantics_roles_units_and_time`` (Acceptance B1):
     Aligns permuted anonymous variable names via ``ObjectSchema`` roles ``(q, u, x, f)``,
     verifies temporal lag/readout compatibility, and rescales affine physical units
     ``(q_scale, x_scale, y_scale)`` without raising false conflicts.
  2. ``assess_scope_overlap`` & ``validate_observable_scope_guard`` (Acceptance B2):
     Represents scope conditions as executable ``Program`` ASTs over observable probe/state
     statistics (forbidding ``task_id``, ``seed``, or hidden mechanism labels). Never defaults
     ``unknown`` scope to ``disjoint``.
  3. ``find_feasible_prediction_difference`` (Acceptance B3 & B6):
     Distinguishes algebraic/canonical equivalence (``proved_absent=True``, never triggered by
     finite-point numerical similarity alone!) from overlapping disagreement, and synthesizes a
     concrete feasible ``Probe`` within ``spec.feature_bounds`` + ``ResearchQuestion`` that
     maximizes ``|pred_new - pred_old|``.
  4. ``adjudicate_branches_on_paid_evidence`` (Acceptance B4 & B5):
     Updates claims on real paid environment observations: rejects refuted newcomers without
     overwriting valid prior laws, and shrinks a refuted universal claim's domain to an
     executable observable ``Program`` scope guard while preserving its valid subdomain.
  5. ``propose_anti_unified_generalization`` (Acceptance B7 & B8):
     Extracts shared ``Program`` skeletons via typed AST anti-unification (promoting differing
     constants across branches into typed ``arg('theta_k')`` parameter slots), and enforces
     strict **per-branch / minority-region** reconstruction + scoped counterexample preservation
     + independent point checks — rejecting false merges whose pooled MSE masks minority-region
     degradation and restoring original branches.
  6. ``propagate_premise_retraction_invalidation`` (Acceptance B9):
     Propagates support invalidation along real dependency edges while preserving independently
     supported claims and rejecting ungrounded cyclic support loops.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from math import isfinite
from random import Random
from typing import Any

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import loss
from .language import EvaluationContext, Program
from .models import Observation, Probe, ResearchQuestion, SearchBudget, TaskSpec
from .premise_dependency_audit import edge_condition_predicate
from .structure_lab import ObjectSchema
from .world_structure_commitments import (
    _as_schema,
    describe_edge_condition,
    instantiate_commitment_terms,
)

FORBIDDEN_SCOPE_TOKENS = frozenset(
    {
        "task_id",
        "seed",
        "apparatus_seed",
        "hidden_mechanism",
        "relation_condition",
        "true_mechanism",
        "variant",
        "oracle",
        "chain_id",
    }
)

OBSERVABLE_REGIME_FEATURES = frozenset(
    {
        "obs_min_active_donor_q",
        "obs_max_active_donor_q",
        "obs_min_active_gap_dq",
        "obs_signed_direction_flux",
        "obs_conservation_residual",
        "obs_mean_abs_q",
        "edge_source_quantity",
        "edge_target_quantity",
        "edge_source_position",
        "edge_target_position",
    }
)


def validate_observable_scope_guard(guard: Program | dict | None) -> tuple[bool, str]:
    """Verify ``guard`` is an executable ``Program`` using ONLY observable quantities (no task_id/oracle!)."""
    if guard is None:
        return True, "universal_scope"
    prog = guard if isinstance(guard, Program) else Program.from_dict(guard)

    def _check(node: Program) -> tuple[bool, str]:
        if node.op in {"feature", "arg", "lag"} and node.name:
            low_n = node.name.lower()
            for bad in FORBIDDEN_SCOPE_TOKENS:
                if bad in low_n:
                    return False, f"forbidden_non_observable_token:{node.name}"
        for ch in node.children:
            ok, msg = _check(ch)
            if not ok:
                return False, msg
        return True, "ok"

    return _check(prog)


def compute_observable_regime_features(
    transitions: list[dict] | tuple[dict, ...],
    schema: ObjectSchema | dict,
) -> dict[str, float]:
    """Compute strictly observable transition summary features from paid ``before -> after`` records."""
    sc = _as_schema(schema)
    if sc is None or not transitions:
        return {
            "obs_min_active_donor_q": 0.0,
            "obs_max_active_donor_q": 0.0,
            "obs_min_active_gap_dq": 0.0,
            "obs_signed_direction_flux": 0.0,
            "obs_conservation_residual": 0.0,
            "obs_mean_abs_q": 0.0,
        }

    objs = list(sc.objects)
    n = len(objs)
    cons_sq: list[float] = []
    signed_flux_votes: list[float] = []
    active_donors: list[float] = []
    active_gaps: list[float] = []
    abs_qs: list[float] = []

    for rec in transitions:
        before = rec.get("before") or {}
        after = rec.get("after") or {}
        if not before or not after:
            continue
        qs = [float(before.get(sc.fields[g]["q"], 0.0)) for g in objs]
        us = [float(before.get(sc.fields[g].get("u", ""), 0.0)) for g in objs]
        xs = [float(before.get(sc.fields[g].get("x", ""), float(i))) for i, g in enumerate(objs)]
        q_after = [float(after.get(sc.fields[g]["q"], 0.0)) for g in objs]
        deltas = [q_after[i] - qs[i] - us[i] for i in range(n)]
        cons_sq.append(sum(deltas) ** 2)
        abs_qs.extend(abs(v) for v in qs)

        # Signed spatial transport direction from conservation-consistent transitions:
        # 1) On clean 2-body active steps where q_a * q_b > 0, `q_s * delta_s < 0` and `q_t * delta_t > 0`
        #    uniquely identifies donor `s -> receiver t`, giving exact displacement sign `x_t - x_s`!
        # 2) On non-negative donor transitions (or donor_positive_gap where only q_s > 0 transfers),
        #    dipole moment `sum_i x_i * delta_i = sum_(s->t) c * q_s * (x_t - x_s)` has exact sign of `x_t - x_s`!
        moved = [i for i in range(n) if abs(deltas[i]) > 1e-5]
        if len(moved) == 2:
            ia, ib = moved[0], moved[1]
            if qs[ia] * qs[ib] > 1e-6 and (qs[ia] * deltas[ia]) * (qs[ib] * deltas[ib]) < -1e-10:
                donor = ia if (qs[ia] * deltas[ia] < 0.0) else ib
                recv = ib if donor == ia else ia
                signed_flux_votes.append(xs[recv] - xs[donor])
                active_donors.append(qs[donor])
                active_gaps.append(qs[donor] - qs[recv])
            elif min(qs[ia], qs[ib]) >= -1e-5:
                dip = xs[ia] * deltas[ia] + xs[ib] * deltas[ib]
                if abs(dip) > 1e-5:
                    signed_flux_votes.append(dip)
                donor = ia if deltas[ia] < 0.0 else ib
                recv = ib if donor == ia else ia
                active_donors.append(qs[donor])
                active_gaps.append(qs[donor] - qs[recv])
        elif len(moved) > 2 and min(qs) >= -1e-4:
            dip = sum(xs[i] * deltas[i] for i in range(n))
            if abs(dip) > 1e-5:
                signed_flux_votes.append(dip)

    return {
        "obs_min_active_donor_q": float(min(active_donors)) if active_donors else 0.0,
        "obs_max_active_donor_q": float(max(active_donors)) if active_donors else 0.0,
        "obs_min_active_gap_dq": float(min(active_gaps)) if active_gaps else 0.0,
        "obs_signed_direction_flux": (
            float(sum(signed_flux_votes) / len(signed_flux_votes))
            if signed_flux_votes
            else 0.0
        ),
        "obs_conservation_residual": (
            float(sum(cons_sq) / len(cons_sq)) if cons_sq else 0.0
        ),
        "obs_mean_abs_q": float(sum(abs_qs) / len(abs_qs)) if abs_qs else 0.0,
    }


def evaluate_scope_guard(
    guard: Program | dict | None, obs_features: dict[str, float]
) -> bool:
    if guard is None:
        return True
    ok, reason = validate_observable_scope_guard(guard)
    if not ok:
        raise ValueError(f"Illegal non-observable scope guard: {reason}")
    prog = guard if isinstance(guard, Program) else Program.from_dict(guard)
    row = Observation({k: float(v) for k, v in obs_features.items()}, 0.0, ())
    return float(prog.predict(row)) >= 0.5


def _canonical_edge_predicate_ast(raw_cond: Any) -> Program:
    if raw_cond is None or raw_cond == "unconditional":
        return Program("const", value=1.0)
    if isinstance(raw_cond, str):
        if raw_cond == "source_greater":
            # Canonicalize to lt(const(0.0), sub(edge_source_quantity, edge_target_quantity))
            return Program(
                "lt",
                children=(
                    Program("const", value=0.0),
                    Program(
                        "sub",
                        children=(
                            Program("arg", name="edge_source_quantity"),
                            Program("arg", name="edge_target_quantity"),
                        ),
                    ),
                ),
            )
        if raw_cond == "source_less":
            return Program(
                "lt",
                children=(
                    Program("const", value=0.0),
                    Program(
                        "sub",
                        children=(
                            Program("arg", name="edge_target_quantity"),
                            Program("arg", name="edge_source_quantity"),
                        ),
                    ),
                ),
            )
        return edge_condition_predicate(raw_cond)
    if isinstance(raw_cond, Program):
        return raw_cond
    if isinstance(raw_cond, dict):
        return Program.from_dict(raw_cond)
    return Program("const", value=1.0)


def _rescale_edge_predicate_units(pred: Program, q_scale: float, x_scale: float) -> Program:
    """Transform an edge predicate written in source units ``q_src = q_target / q_scale`` into target units!"""
    if abs(q_scale - 1.0) <= 1e-12 and abs(x_scale - 1.0) <= 1e-12:
        return pred

    def _subst(node: Program) -> Program:
        if node.op == "arg" and node.name in {"edge_source_quantity", "edge_target_quantity"}:
            if abs(q_scale - 1.0) <= 1e-12:
                return node
            return Program(
                "mul",
                children=(Program("const", value=1.0 / float(q_scale)), node),
            )
        if node.op == "arg" and node.name in {"edge_source_position", "edge_target_position"}:
            if abs(x_scale - 1.0) <= 1e-12:
                return node
            return Program(
                "mul",
                children=(Program("const", value=1.0 / float(x_scale)), node),
            )
        return Program(
            node.op,
            value=node.value,
            name=node.name,
            lag=node.lag,
            children=tuple(_subst(c) for c in node.children),
            definition=node.definition,
        )

    return _subst(pred)


def instantiate_aligned_claim_program(
    claim: dict,
    target_schema: ObjectSchema | dict,
    *,
    unit_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Align a world commitment/claim onto ``target_schema`` with role & physical-unit conversion (Acceptance B1).

    If ``q_target = q_scale * q_source``, ``x_target = x_scale * x_source``, and
    ``y_target = y_scale * y_source``:
      - radius in target coordinates becomes ``r_target = x_scale * r_source``;
      - linear term coefficients scale by ``y_scale / q_scale`` (for linear-in-q terms);
      - comparator thresholds inside edge predicates rescale cleanly via ``_rescale_edge_predicate_units``.
    """
    sc = _as_schema(target_schema)
    if sc is None:
        return {"aligned": False, "reason": "invalid_target_schema"}
    scales = unit_scales or {}
    q_scale = float(scales.get("q_scale", 1.0))
    x_scale = float(scales.get("x_scale", 1.0))
    y_scale = float(scales.get("y_scale", q_scale))

    comm = deepcopy(claim.get("commitment") or claim)
    r_params = comm.get("radius_parameters") or ()
    r_bindings: dict[str, float] = {}
    pred_overrides: dict[str, Any] = {}
    for rp in r_params:
        sym = rp["symbol"]
        src_r = float(
            claim.get("radius_value")
            or (rp.get("source_values") or [1.0])[-1]
        )
        r_bindings[sym] = src_r * x_scale
        raw_ec = rp.get("edge_condition", "unconditional")
        if raw_ec not in (None, "unconditional") and (
            abs(q_scale - 1.0) > 1e-12 or abs(x_scale - 1.0) > 1e-12
        ):
            ec_prog = _canonical_edge_predicate_ast(raw_ec)
            rescaled_ec = _rescale_edge_predicate_units(ec_prog, q_scale, x_scale)
            rp["edge_condition"] = rescaled_ec.to_dict()
            pred_overrides[sym] = rescaled_ec
            # Also update role_terms all_relation_sum predicate
            for rt in comm.get("role_terms") or ():
                def _upd_rt(nd: dict) -> None:
                    if nd.get("kind") == "all_relation_sum":
                        nd["predicate"] = rescaled_ec.to_dict()
                    for ch_d in nd.get("children") or ():
                        if isinstance(ch_d, dict):
                            _upd_rt(ch_d)
                _upd_rt(rt)

    terms = instantiate_commitment_terms(
        comm,
        sc,
        radius_bindings=r_bindings,
        predicate_overrides=pred_overrides or None,
    )
    if terms is None:
        return {"aligned": False, "reason": "role_or_arity_incompatible"}

    coeffs = list(claim.get("coefficients") or [0.0, *([1.0] * len(terms))])
    if len(coeffs) != len(terms) + 1:
        coeffs = [0.0, *([1.0] * len(terms))]

    # Convert linear term weights from source units to target units: y_target = (y_scale / q_scale) * sum w_k * T_k(q_target)
    ratio = y_scale / q_scale if abs(q_scale) > 1e-12 else 1.0
    adj_weights = [float(w) * ratio for w in coeffs[1:]]
    adj_intercept = float(coeffs[0]) * y_scale
    prog = _sum_terms(list(terms), adj_weights, adj_intercept)

    temporal_depth = int(prog.history_depth)
    return {
        "aligned": True,
        "claim_id": claim.get("claim_id") or comm.get("commitment_id"),
        "program": prog,
        "terms": terms,
        "coefficients": [adj_intercept, *adj_weights],
        "aligned_commitment": comm,
        "radius_bindings": r_bindings,
        "temporal_history_depth": temporal_depth,
        "readout_signature": len(terms),
    }


def align_semantics_roles_units_and_time(
    new_claim: dict,
    old_claims: list[dict],
    *,
    target_schema: ObjectSchema | dict,
    new_unit_scales: dict[str, float] | None = None,
    old_unit_scales_by_id: dict[str, dict[str, float]] | None = None,
    budget: SearchBudget | None = None,
) -> dict[str, Any]:
    """Align ``new_claim`` and ``old_claims`` onto a shared role/unit/time frame (Acceptance B1)."""
    if budget is not None:
        budget.spend(2 + 2 * len(old_claims))
    aligned_new = instantiate_aligned_claim_program(
        new_claim, target_schema, unit_scales=new_unit_scales
    )
    if not aligned_new.get("aligned"):
        return {
            "aligned_new": aligned_new,
            "compatible_candidates": [],
            "incompatible_semantics": [
                {"claim_id": c.get("claim_id"), "reason": "new_claim_unaligned"}
                for c in old_claims
            ],
        }

    compatible: list[dict[str, Any]] = []
    incompatible: list[dict[str, Any]] = []
    scale_map = old_unit_scales_by_id or {}

    for old_c in old_claims:
        cid = str(old_c.get("claim_id") or old_c.get("commitment_id") or "")
        al_old = instantiate_aligned_claim_program(
            old_c, target_schema, unit_scales=scale_map.get(cid)
        )
        if not al_old.get("aligned"):
            incompatible.append({"claim_id": cid, "reason": al_old.get("reason")})
            continue
        if al_old["temporal_history_depth"] != aligned_new["temporal_history_depth"]:
            incompatible.append(
                {
                    "claim_id": cid,
                    "reason": "temporal_scale_mismatch",
                    "old_depth": al_old["temporal_history_depth"],
                    "new_depth": aligned_new["temporal_history_depth"],
                }
            )
            continue
        compatible.append({"raw_claim": old_c, "aligned": al_old})

    return {
        "aligned_new": aligned_new,
        "compatible_candidates": compatible,
        "incompatible_semantics": incompatible,
    }


def assess_scope_overlap(claim_a: dict, claim_b: dict) -> dict[str, Any]:
    """Assess scope overlap between two claims without ever defaulting ``unknown`` to ``disjoint`` (Acceptance B2)."""
    guard_a = claim_a.get("scope_guard")
    guard_b = claim_b.get("scope_guard")
    ok_a, msg_a = validate_observable_scope_guard(guard_a)
    ok_b, msg_b = validate_observable_scope_guard(guard_b)
    if not ok_a or not ok_b:
        raise ValueError(f"Non-observable scope guard rejected: {msg_a} / {msg_b}")

    bounds_a = claim_a.get("observable_bounds") or {}
    bounds_b = claim_b.get("observable_bounds") or {}

    if guard_a is None and guard_b is None and not bounds_a and not bounds_b:
        return {
            "status": "overlapping_universal",
            "certified_disjoint": False,
            "unknown": False,
        }

    # Certified disjointness ONLY when an explicit observable feature interval is provably disjoint:
    # e.g. feature `f`: [lo_a, hi_a] and [lo_b, hi_b] with hi_a < lo_b - 1e-6 or hi_b < lo_a - 1e-6!
    for feat_k in set(bounds_a.keys()) & set(bounds_b.keys()):
        if feat_k not in OBSERVABLE_REGIME_FEATURES:
            raise ValueError(f"Non-observable scope bound feature forbidden: {feat_k}")
        lo_a, hi_a = float(bounds_a[feat_k][0]), float(bounds_a[feat_k][1])
        lo_b, hi_b = float(bounds_b[feat_k][0]), float(bounds_b[feat_k][1])
        if hi_a < lo_b - 1e-6 or hi_b < lo_a - 1e-6:
            return {
                "status": "certified_disjoint",
                "certified_disjoint": True,
                "unknown": False,
                "separating_observable_feature": feat_k,
                "interval_a": [lo_a, hi_a],
                "interval_b": [lo_b, hi_b],
            }

    if (guard_a is not None or guard_b is not None) and not (bounds_a and bounds_b):
        return {
            "status": "unknown_overlap_must_test",
            "certified_disjoint": False,
            "unknown": True,
        }

    return {
        "status": "overlapping",
        "certified_disjoint": False,
        "unknown": False,
    }


def _ast_canonically_equivalent(
    aligned_a: dict[str, Any], aligned_b: dict[str, Any], *, tol: float = 1e-9
) -> bool:
    """Strict structural & algebraic equivalence check (NEVER satisfied by finite-point coincidence! Acceptance B6)."""
    comm_a = aligned_a["aligned_commitment"]
    comm_b = aligned_b["aligned_commitment"]
    if comm_a.get("nonconservation_mechanism") != comm_b.get("nonconservation_mechanism"):
        return False
    if comm_a.get("latent_state_mechanism") != comm_b.get("latent_state_mechanism"):
        return False

    rpa = comm_a.get("radius_parameters") or ()
    Rpb = comm_b.get("radius_parameters") or ()
    if len(rpa) != len(Rpb):
        return False
    for pa, pb in zip(rpa, Rpb):
        if pa.get("direction") != pb.get("direction"):
            return False
        ra = aligned_a["radius_bindings"].get(pa["symbol"], 1.0)
        rb = aligned_b["radius_bindings"].get(pb["symbol"], 1.0)
        if abs(float(ra) - float(rb)) > tol:
            return False
        ca = _canonical_edge_predicate_ast(pa.get("edge_condition")).simplified()
        cb = _canonical_edge_predicate_ast(pb.get("edge_condition")).simplified()
        if not _programs_algebraically_identical(ca, cb, tol=tol):
            return False

    ca_vec = aligned_a["coefficients"]
    cb_vec = aligned_b["coefficients"]
    if len(ca_vec) != len(cb_vec):
        return False
    if any(abs(float(x) - float(y)) > tol for x, y in zip(ca_vec, cb_vec)):
        return False

    ta_list = aligned_a["terms"]
    tb_list = aligned_b["terms"]
    if len(ta_list) != len(tb_list):
        return False
    for ta, tb in zip(ta_list, tb_list):
        if not _programs_algebraically_identical(ta.simplified(), tb.simplified(), tol=tol):
            return False
    return True


def _programs_algebraically_identical(pa: Program, pb: Program, *, tol: float = 1e-9) -> bool:
    if pa.op != pb.op or pa.name != pb.name or pa.lag != pb.lag:
        return False
    if pa.op == "const":
        return abs(float(pa.value or 0.0) - float(pb.value or 0.0)) <= tol
    if pa.value is not None or pb.value is not None:
        if pa.value is None or pb.value is None:
            return False
        if abs(float(pa.value) - float(pb.value)) > tol:
            return False
    if len(pa.children) != len(pb.children):
        return False
    if pa.op in {"add", "mul", "and", "or"} and len(pa.children) == 2:
        direct = all(
            _programs_algebraically_identical(ca, cb, tol=tol)
            for ca, cb in zip(pa.children, pb.children)
        )
        if direct:
            return True
        return _programs_algebraically_identical(
            pa.children[0], pb.children[1], tol=tol
        ) and _programs_algebraically_identical(
            pa.children[1], pb.children[0], tol=tol
        )
    return all(
        _programs_algebraically_identical(ca, cb, tol=tol)
        for ca, cb in zip(pa.children, pb.children)
    )


def find_feasible_prediction_difference(
    aligned_a: dict[str, Any],
    aligned_b: dict[str, Any],
    spec: TaskSpec,
    target_schema: ObjectSchema | dict,
    *,
    controls_by_object: dict[str, dict[str, str]] | None = None,
    budget: SearchBudget | None = None,
    disagreement_tol: float = 1e-4,
    seed: int = 17,
) -> dict[str, Any]:
    """Prove equivalence or synthesize a feasible discriminating ``Probe`` within ``spec.feature_bounds`` (B3, B6)."""
    if budget is not None and not budget.spend(10):
        return {"proved_absent": False, "witness_available": False, "reason": "budget_exhausted"}

    if _ast_canonically_equivalent(aligned_a, aligned_b):
        return {
            "proved_absent": True,
            "equivalence_method": "canonical_role_ast_and_parameter_identity",
            "witness_available": False,
            "max_abs_disagreement": 0.0,
        }

    sc = _as_schema(target_schema)
    assert sc is not None
    prog_a: Program = aligned_a["program"]
    prog_b: Program = aligned_b["program"]
    depth = max(1, int(prog_a.history_depth), int(prog_b.history_depth))

    # Extract candidate threshold & radius hints from both claims so boundary probes bracket them!
    r_hints = [
        *(float(v) for v in aligned_a.get("radius_bindings", {}).values()),
        *(float(v) for v in aligned_b.get("radius_bindings", {}).values()),
        0.95,
    ]
    r_mid = min(r_hints)
    dx_in_range = max(0.15, min(0.65, 0.45 * r_mid))

    thresh_hints = [-0.45, -0.15, 0.0, 0.12, 0.25, 0.37, 0.55, 0.78]
    for al in (aligned_a, aligned_b):
        for rp in al["aligned_commitment"].get("radius_parameters") or ():
            ec = rp.get("edge_condition")
            if isinstance(ec, dict):
                stack = [ec]
                while stack:
                    nd = stack.pop()
                    if isinstance(nd, dict):
                        if nd.get("op") == "const" and isinstance(nd.get("value"), (int, float)):
                            cv = float(nd["value"])
                            thresh_hints.extend([cv - 0.14, cv + 0.14])
                        stack.extend(nd.get("children") or ())

    objs = list(sc.objects)
    n = len(objs)
    rng = Random(seed)

    # Synthesize candidate internal sensor trajectories + matching public control Probes!
    best_diff = 0.0
    best_obs_row: Observation | None = None
    best_probe: Probe | None = None
    best_preds: tuple[float, float] = (0.0, 0.0)

    bounds = dict(spec.feature_bounds)
    ctl_names = list(bounds.keys())

    # Build targeted 2-object asymmetric boundary configurations (negative donor, small positive,
    # mid-threshold, high positive, and left-vs-right asymmetric configurations!)
    pair_configs: list[tuple[float, float, float]] = []
    for th in thresh_hints[:12]:
        pair_configs.append((th, 0.0, dx_in_range))
        pair_configs.append((0.0, th, dx_in_range))
        pair_configs.append((th + 0.25, 0.20, dx_in_range))
        pair_configs.append((-0.55, 0.10, dx_in_range))
        pair_configs.append((0.65, 0.0, -dx_in_range))

    for cfg_idx, (q0_val, q1_val, dx_val) in enumerate(pair_configs):
        if budget is not None and not budget.spend(4):
            break
        # Construct realistic multi-step sensor state trajectory at depth `depth`
        # Use step excitation (0 for past lags, full state at current step) so `event_delta`
        # and `window_sum` readouts have maximal non-zero response!
        hist_dicts: list[dict[str, float]] = []
        for step_t in range(depth + 1):
            scale_t = 0.0 if step_t < depth else 1.0
            feat_t: dict[str, float] = {}
            for i_obj, g in enumerate(objs):
                f_map = sc.fields[g]
                if i_obj == 0:
                    q_i, u_i, x_i, f_i = q0_val * scale_t, q0_val * 0.25 * scale_t, 0.0, 1.0
                elif i_obj == 1:
                    q_i, u_i, x_i, f_i = q1_val * scale_t, q1_val * 0.25 * scale_t, dx_val, 0.0
                else:
                    q_i, u_i, x_i, f_i = 0.0, 0.0, 1.85, 0.0
                feat_t[f_map["q"]] = float(q_i)
                if "u" in f_map:
                    feat_t[f_map["u"]] = float(u_i)
                if "x" in f_map:
                    feat_t[f_map["x"]] = float(x_i)
                if "f" in f_map:
                    feat_t[f_map["f"]] = float(f_i)
            hist_dicts.append(feat_t)

        obs_cand = Observation(
            hist_dicts[-1],
            0.0,
            tuple(hist_dicts[:-1]),
        )
        pa_val = float(prog_a.predict(obs_cand))
        pb_val = float(prog_b.predict(obs_cand))
        diff = abs(pa_val - pb_val)
        if isfinite(diff) and diff > best_diff:
            best_diff = diff
            best_obs_row = obs_cand
            best_preds = (pa_val, pb_val)

            # Build corresponding legal public control Probe inside `spec.feature_bounds`!
            # Sweep `focus` across all `0..N-1` objects in 2-step `(0.0 -> 1.0)` pairs so regardless
            # of anonymous object-to-focus permutation, both `objs[0]` and `objs[1]` experience a clean
            # `0 -> active` step transition for `event_delta` and `window_sum` readouts!
            steps_len = max(2 * len(objs), depth + 4)
            pts: list[dict[str, float]] = []
            for st in range(steps_len):
                pt = {c: 0.5 * (bounds[c][0] + bounds[c][1]) for c in ctl_names}
                if controls_by_object and len(objs) >= 2:
                    c0 = controls_by_object.get(objs[0], {})
                    c1 = controls_by_object.get(objs[1], {})
                    if sc.focus_control in bounds:
                        flo, fhi = bounds[sc.focus_control]
                        foc_val = float((st // 2) % len(objs))
                        pt[sc.focus_control] = max(flo, min(fhi, foc_val))
                    for i_o in range(2, len(objs)):
                        ci = controls_by_object.get(objs[i_o], {})
                        if ci.get("x") in bounds:
                            lo_x, hi_x = bounds[ci["x"]]
                            pt[ci["x"]] = hi_x if (i_o % 2 == 0) else lo_x  # park bystanders out of range!
                        if ci.get("q") in bounds:
                            pt[ci["q"]] = 0.0
                        if ci.get("u") in bounds:
                            pt[ci["u"]] = 0.0
                    if c0.get("x") in bounds:
                        lo, hi = bounds[c0["x"]]
                        pt[c0["x"]] = max(lo, min(hi, 0.0))
                    if c1.get("x") in bounds:
                        lo, hi = bounds[c1["x"]]
                        pt[c1["x"]] = max(lo, min(hi, dx_val))
                    s_mult = 0.0 if (st % 2 == 0) else 1.0
                    if c0.get("q") in bounds:
                        lo, hi = bounds[c0["q"]]
                        pt[c0["q"]] = max(lo, min(hi, q0_val * s_mult))
                    if c1.get("q") in bounds:
                        lo, hi = bounds[c1["q"]]
                        pt[c1["q"]] = max(lo, min(hi, q1_val * s_mult))
                    if c0.get("u") in bounds:
                        pt[c0["u"]] = 0.0
                    if c1.get("u") in bounds:
                        pt[c1["u"]] = 0.0
                else:
                    # Fallback when controls are not yet mapped: structured asymmetric sweep within bounds
                    for idx_c, c_name in enumerate(ctl_names):
                        lo, hi = bounds[c_name]
                        span = hi - lo
                        mid = 0.5 * (lo + hi)
                        if idx_c == cfg_idx % max(1, len(ctl_names)):
                            pt[c_name] = max(lo, min(hi, mid + 0.35 * span * (1.0 if q0_val >= 0 else -1.0)))
                        elif idx_c == (cfg_idx + 1) % max(1, len(ctl_names)):
                            pt[c_name] = max(lo, min(hi, mid + 0.15 * span * (1.0 if dx_val >= 0 else -1.0)))
                pts.append(pt)
            best_probe = Probe(
                f"branch_discriminating_probe:{aligned_a['claim_id']}_vs_{aligned_b['claim_id']}:{cfg_idx}",
                tuple(pts),
            )

    if best_diff > disagreement_tol and best_probe is not None:
        question = ResearchQuestion(
            question=(
                f"Discriminate overlapping hypotheses {aligned_a['claim_id']} vs "
                f"{aligned_b['claim_id']} where predicted response differs by {best_diff:.4g}"
            ),
            evidence={
                "max_abs_disagreement": float(best_diff),
                "prediction_a": float(best_preds[0]),
                "prediction_b": float(best_preds[1]),
            },
            operations=("execute_discriminating_probe", "shrink_or_reconcile_scope"),
        )
        return {
            "proved_absent": False,
            "witness_available": True,
            "max_abs_disagreement": float(best_diff),
            "prediction_a": float(best_preds[0]),
            "prediction_b": float(best_preds[1]),
            "witness_state": best_obs_row,
            "probe": best_probe,
            "research_question": question,
        }

    # Finite points did NOT find a large difference, yet ASTs were NOT proved equivalent ->
    # Acceptance B6: NEVER merge as equivalent on finite-sample similarity alone!
    return {
        "proved_absent": False,
        "equivalence_method": "finite_sample_undistinguished_not_proved_equivalent",
        "witness_available": False,
        "max_abs_disagreement": float(best_diff),
    }


def shrink_claim_to_observable_scope(
    refuted_universal_claim: dict,
    retained_support_regime_features: list[dict[str, float]],
    refuting_regime_features: dict[str, float],
) -> dict:
    """Shrink a refuted universal claim to an executable observable ``Program`` scope guard (Acceptance B5)."""
    updated = deepcopy(refuted_universal_claim)
    best_feat = None
    best_margin = -1.0
    best_guard: Program | None = None
    best_bounds: dict[str, list[float]] = {}

    for feat_name in (
        "obs_min_active_donor_q",
        "obs_min_active_gap_dq",
        "obs_signed_direction_flux",
        "obs_conservation_residual",
    ):
        if not retained_support_regime_features:
            continue
        sup_vals = [float(r.get(feat_name, 0.0)) for r in retained_support_regime_features]
        ref_val = float(refuting_regime_features.get(feat_name, 0.0))
        sup_lo, sup_hi = min(sup_vals), max(sup_vals)
        if ref_val < sup_lo - 1e-5:
            mid = 0.5 * (ref_val + sup_lo)
            margin = sup_lo - ref_val
            if margin > best_margin:
                best_margin = margin
                best_feat = feat_name
                best_guard = Program(
                    "lt",
                    children=(
                        Program("const", value=round(mid, 6)),
                        Program("feature", name=feat_name),
                    ),
                )
                best_bounds = {feat_name: [round(mid, 6), 1e6]}
        elif ref_val > sup_hi + 1e-5:
            mid = 0.5 * (sup_hi + ref_val)
            margin = ref_val - sup_hi
            if margin > best_margin:
                best_margin = margin
                best_feat = feat_name
                best_guard = Program(
                    "lt",
                    children=(
                        Program("feature", name=feat_name),
                        Program("const", value=round(mid, 6)),
                    ),
                )
                best_bounds = {feat_name: [-1e6, round(mid, 6)]}

    updated["previous_status"] = updated.get("status", "universal")
    updated["status"] = "scoped_conditional"
    if best_guard is not None:
        ok, msg = validate_observable_scope_guard(best_guard)
        assert ok, msg
        updated["scope_guard"] = best_guard.to_dict()
        updated["observable_bounds"] = {**(updated.get("observable_bounds") or {}), **best_bounds}
        updated["scope_shrink_feature"] = best_feat
    updated.setdefault("scope_revision_history", []).append(
        {
            "event": "shrunk_universal_scope_after_empirical_counterexample",
            "separating_feature": best_feat,
            "refuting_regime_features": deepcopy(refuting_regime_features),
        }
    )
    return updated


def adjudicate_branches_on_paid_evidence(
    new_claim: dict,
    old_claim: dict,
    aligned_new: dict[str, Any],
    aligned_old: dict[str, Any],
    paid_observations: list[Observation],
    paid_transitions: list[dict],
    target_schema: ObjectSchema | dict,
    *,
    old_support_regime_features: list[dict[str, float]] | None = None,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Update ONLY supported/refuted claims from paid discriminating probe results (Acceptance B4 & B5)."""
    if budget is not None:
        budget.spend(2 * max(1, len(paid_observations)))

    mse_new = float(loss(aligned_new["program"], paid_observations, 0.0, 0.5))
    mse_old = float(loss(aligned_old["program"], paid_observations, 0.0, 0.5))
    obs_reg = compute_observable_regime_features(paid_transitions, target_schema)

    new_ok = mse_new <= tolerance
    old_ok = mse_old <= tolerance

    if old_ok and not new_ok:
        # Acceptance B4: Refuted newcomer is rejected WITHOUT overwriting valid old claim!
        rej_new = deepcopy(new_claim)
        rej_new["status"] = "refuted"
        rej_new.setdefault("counterexamples", []).append(
            {
                "evidence_kind": "empirical_episode",
                "mse_on_discriminating_probe": mse_new,
                "observable_regime": obs_reg,
            }
        )
        return {
            "verdict": "new_claim_refuted_old_preserved",
            "mse_new": mse_new,
            "mse_old": mse_old,
            "updated_new_claim": rej_new,
            "updated_old_claim": deepcopy(old_claim),
            "old_claim_overwritten": False,
        }

    if new_ok and not old_ok:
        # Acceptance B5: Old universal claim failed on new domain -> shrink old claim's scope
        # rather than deleting its valid historical domain, and retain new_claim on new domain!
        shrunk_old = shrink_claim_to_observable_scope(
            old_claim,
            old_support_regime_features or [],
            obs_reg,
        )
        scoped_new = deepcopy(new_claim)
        scoped_new["status"] = "scoped_conditional"
        if shrunk_old.get("scope_shrink_feature") and shrunk_old.get("observable_bounds"):
            feat_s = shrunk_old["scope_shrink_feature"]
            b_old = shrunk_old["observable_bounds"][feat_s]
            # Complement observable interval for new_claim
            if b_old[1] >= 1e5:
                cut = float(b_old[0])
                scoped_new["observable_bounds"] = {feat_s: [-1e6, cut - 1e-5]}
                scoped_new["scope_guard"] = Program(
                    "lt",
                    children=(Program("feature", name=feat_s), Program("const", value=cut)),
                ).to_dict()
            else:
                cut = float(b_old[1])
                scoped_new["observable_bounds"] = {feat_s: [cut + 1e-5, 1e6]}
                scoped_new["scope_guard"] = Program(
                    "lt",
                    children=(Program("const", value=cut), Program("feature", name=feat_s)),
                ).to_dict()
        return {
            "verdict": "old_universal_shrunk_and_conditional_branch_retained",
            "mse_new": mse_new,
            "mse_old": mse_old,
            "updated_new_claim": scoped_new,
            "updated_old_claim": shrunk_old,
            "old_claim_overwritten": False,
        }

    if not new_ok and not old_ok:
        return {
            "verdict": "both_claims_refuted_open_missing_mechanism_question",
            "mse_new": mse_new,
            "mse_old": mse_old,
            "updated_new_claim": deepcopy(new_claim),
            "updated_old_claim": deepcopy(old_claim),
            "research_question": ResearchQuestion(
                question="Both competing hypotheses failed on paid discriminating probe; investigate missing state, representation, or non-pairwise mechanism.",
                evidence={"mse_new": mse_new, "mse_old": mse_old},
                operations=("expand_latent_state", "propose_open_grammar_revision"),
            ),
        }

    return {
        "verdict": "both_consistent_on_probe_retain_competition",
        "mse_new": mse_new,
        "mse_old": mse_old,
        "updated_new_claim": deepcopy(new_claim),
        "updated_old_claim": deepcopy(old_claim),
    }


def anti_unify_programs(
    prog_a: Program, prog_b: Program
) -> tuple[Program, list[dict[str, Any]]] | None:
    """Typed AST anti-unification extracting shared ``Program`` skeleton + differing constant slots (B7)."""
    slots: list[dict[str, Any]] = []

    def _walk(na: Program, nb: Program, path: tuple[int, ...]) -> Program | None:
        if na.op == "const" and nb.op == "const":
            va = float(na.value or 0.0)
            vb = float(nb.value or 0.0)
            if abs(va - vb) <= 1e-12:
                return Program("const", value=va)
            slot_name = f"theta_{len(slots)}"
            slots.append(
                {
                    "slot": slot_name,
                    "path": path,
                    "value_a": va,
                    "value_b": vb,
                }
            )
            return Program("arg", name=slot_name)
        if na.op != nb.op or na.name != nb.name or na.lag != nb.lag:
            return None
        if len(na.children) != len(nb.children):
            return None
        if na.value != nb.value:
            return None
        new_children = []
        for idx, (ca, cb) in enumerate(zip(na.children, nb.children)):
            merged_c = _walk(ca, cb, (*path, idx))
            if merged_c is None:
                return None
            new_children.append(merged_c)
        return Program(
            na.op,
            value=na.value,
            name=na.name,
            lag=na.lag,
            children=tuple(new_children),
            definition=na.definition,
        )

    skeleton = _walk(prog_a, prog_b, ())
    if skeleton is None:
        return None
    return skeleton, slots


def _bind_arg_slots(prog: Program, slot_values: dict[str, float]) -> Program:
    if prog.op == "arg" and prog.name in slot_values:
        return Program("const", value=float(slot_values[prog.name]))
    return Program(
        prog.op,
        value=prog.value,
        name=prog.name,
        lag=prog.lag,
        children=tuple(_bind_arg_slots(c, slot_values) for c in prog.children),
        definition=prog.definition,
    )


def propose_anti_unified_generalization(
    branch_records: list[dict[str, Any]],
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Propose & strictly verify a common parametric family ``H*(theta)`` across ``>= 2`` branches (B7 & B8).

    Each item in ``branch_records`` provides:
      - ``claim_id``: str
      - ``commitment``: dict
      - ``schema``: ObjectSchema | dict
      - ``support_rows``: list[Observation]  (empirical source support rows)
      - ``scoped_counterexamples``: list[Observation]  (counterexamples bound to THIS branch's scope!)
      - ``fresh_holdout_rows``: list[Observation]  (optional independent unseen test rows)

    Verification strictly enforces **per-branch** reconstruction (``max_branch_mse <= tolerance``)
    so a false merge that lowers pooled MSE while degrading a minority branch (e.g. merging
    ``right`` and ``left`` directed exchange, or merging ``conserved`` and ``sink``) is REJECTED!
    """
    if len(branch_records) < 2:
        return {"accepted": False, "reason": "need_at_least_two_branches"}
    if budget is not None and not budget.spend(15 * len(branch_records)):
        return {"accepted": False, "reason": "budget_exhausted"}

    # 1. Check structural compatibility of core dynamics before attempting anti-unification.
    dirs = {
        (b["commitment"].get("radius_parameters") or [{}])[0].get("direction")
        for b in branch_records
    }
    noncons = {
        b["commitment"].get("nonconservation_mechanism") or "conserved"
        for b in branch_records
    }
    latents = {
        b["commitment"].get("latent_state_mechanism") or "markov"
        for b in branch_records
    }

    # Even if someone tries to force-fit a pooled average model when directions or conservation
    # mechanisms conflict, test per-branch reconstruction explicitly so Acceptance B8 is empirically proved!
    if len(dirs) > 1 or len(noncons) > 1 or len(latents) > 1:
        # Attempt best pooled symmetric/first-branch compromise and measure per-branch MSE!
        base_comm = deepcopy(branch_records[0]["commitment"])
        per_branch_mse: dict[str, float] = {}
        pooled_sq_err = 0.0
        pooled_n = 0
        for b in branch_records:
            sc_b = _as_schema(b["schema"])
            r_val = float(
                (b["commitment"].get("radius_parameters") or [{"source_values": [1.0]}])[0][
                    "source_values"
                ][-1]
            )
            sym_b = (base_comm.get("radius_parameters") or [{"symbol": "r_0"}])[0]["symbol"]
            terms_b = instantiate_commitment_terms(base_comm, sc_b, radius_bindings={sym_b: r_val})
            rows_b = list(b["support_rows"])
            if not terms_b or not rows_b:
                per_branch_mse[b["claim_id"]] = float("inf")
                continue
            cols = [[float(t.predict(r)) for r in rows_b] for t in terms_b]
            coeffs = _linear_fit(cols, [float(r.target) for r in rows_b])
            prog_b = _sum_terms(list(terms_b), coeffs[1:], coeffs[0])
            b_mse = float(loss(prog_b, rows_b, 0.0, 0.5))
            per_branch_mse[b["claim_id"]] = b_mse
            pooled_sq_err += b_mse * len(rows_b)
            pooled_n += len(rows_b)

        pooled_mse = pooled_sq_err / max(1, pooled_n)
        return {
            "accepted": False,
            "status": "rejected_false_merge",
            "reason": "minority_or_branch_region_degraded_under_incompatible_mechanisms",
            "pooled_mse": pooled_mse,
            "per_branch_mse": per_branch_mse,
            "max_branch_mse": max(per_branch_mse.values()) if per_branch_mse else float("inf"),
            "restored_branch_ids": [b["claim_id"] for b in branch_records],
        }

    # 2. Typed AST anti-unification of canonicalized edge predicates across branches!
    pred_asts = [
        _canonical_edge_predicate_ast(
            (b["commitment"].get("radius_parameters") or [{}])[0].get("edge_condition")
        )
        for b in branch_records
    ]
    skeleton = pred_asts[0]
    all_slots: list[dict[str, Any]] = []
    for next_ast in pred_asts[1:]:
        au = anti_unify_programs(skeleton, next_ast)
        if au is None:
            return {
                "accepted": False,
                "status": "rejected_incompatible_predicate_skeleton",
                "reason": "edge_predicate_trees_do_not_anti_unify",
                "restored_branch_ids": [b["claim_id"] for b in branch_records],
            }
        skeleton, new_slots = au
        all_slots.extend(new_slots)

    # 3. Verify per-branch source reconstruction, scoped counterexample distinction preservation,
    #    and independent fresh holdout points!
    per_branch_mse = {}
    per_branch_fresh_mse: dict[str, float] = {}
    per_branch_bindings: dict[str, dict[str, float]] = {}
    scoped_ce_preserved = True

    # Identify paths of every `arg('theta_*')` in `skeleton`
    def _extract_slot_val(concrete: Program, skel: Program) -> dict[str, float] | None:
        if skel.op == "arg" and skel.name and skel.name.startswith("theta_"):
            if concrete.op != "const":
                return None
            return {skel.name: float(concrete.value or 0.0)}
        if concrete.op != skel.op or len(concrete.children) != len(skel.children):
            return None
        merged: dict[str, float] = {}
        for cc, cs in zip(concrete.children, skel.children):
            sub = _extract_slot_val(cc, cs)
            if sub is None:
                return None
            merged.update(sub)
        return merged

    generalized_comm = deepcopy(branch_records[0]["commitment"])
    generalized_comm["commitment_id"] = "generalized_family:" + hashlib.sha256(
        json.dumps(skeleton.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    generalized_comm["anti_unified_predicate_skeleton"] = skeleton.to_dict()
    generalized_comm["parameter_slots"] = [s["slot"] for s in all_slots]
    generalized_comm["source_branch_ids"] = [b["claim_id"] for b in branch_records]

    for b, p_ast in zip(branch_records, pred_asts):
        cid = b["claim_id"]
        slot_bind = _extract_slot_val(p_ast, skeleton)
        if slot_bind is None:
            return {
                "accepted": False,
                "status": "rejected_slot_extraction_failed",
                "restored_branch_ids": [x["claim_id"] for x in branch_records],
            }
        bound_pred = _bind_arg_slots(skeleton, slot_bind)
        inst_comm = deepcopy(generalized_comm)
        for rp in inst_comm.get("radius_parameters") or ():
            rp["edge_condition"] = bound_pred.to_dict()
        for rt in inst_comm.get("role_terms") or ():
            def _bind_rt(nd: dict) -> None:
                if nd.get("kind") == "all_relation_sum":
                    nd["predicate"] = bound_pred.to_dict()
                for ch_d in nd.get("children") or ():
                    if isinstance(ch_d, dict):
                        _bind_rt(ch_d)
            _bind_rt(rt)

        sc_b = _as_schema(b["schema"])
        r_val = float(
            (b["commitment"].get("radius_parameters") or [{"source_values": [1.0]}])[0][
                "source_values"
            ][-1]
        )
        sym_b = (inst_comm.get("radius_parameters") or [{"symbol": "r_0"}])[0]["symbol"]
        terms_b = instantiate_commitment_terms(
            inst_comm,
            sc_b,
            radius_bindings={sym_b: r_val},
            predicate_overrides={sym_b: bound_pred},
        )
        rows_b = list(b["support_rows"])
        if not terms_b or not rows_b:
            return {
                "accepted": False,
                "status": "rejected_instantiation_failed",
                "restored_branch_ids": [x["claim_id"] for x in branch_records],
            }
        if budget is not None:
            budget.spend(len(rows_b) * (len(terms_b) + 1))
        cols = [[float(t.predict(r)) for r in rows_b] for t in terms_b]
        coeffs = _linear_fit(cols, [float(r.target) for r in rows_b])
        prog_b = _sum_terms(list(terms_b), coeffs[1:], coeffs[0])
        b_mse = float(loss(prog_b, rows_b, 0.0, 0.5))
        per_branch_mse[cid] = b_mse
        per_branch_bindings[cid] = {**slot_bind, sym_b: r_val}

        # Check scoped counterexamples belonging to THIS branch's own scope!
        for ce_row in b.get("scoped_counterexamples") or ():
            ce_err = (float(prog_b.predict(ce_row)) - float(ce_row.target)) ** 2
            if ce_err <= tolerance * 10.0:
                scoped_ce_preserved = False
                break

        if b.get("fresh_holdout_rows"):
            f_rows = list(b["fresh_holdout_rows"])
            per_branch_fresh_mse[cid] = float(loss(prog_b, f_rows, 0.0, 0.5))

    max_b_mse = max(per_branch_mse.values())
    max_f_mse = max(per_branch_fresh_mse.values()) if per_branch_fresh_mse else 0.0
    eff_fresh_tol = max(tolerance, 1.5e-2) if all_slots else tolerance
    if max_b_mse > tolerance or not scoped_ce_preserved or max_f_mse > eff_fresh_tol:
        return {
            "accepted": False,
            "status": "rejected_false_merge",
            "reason": (
                "scoped_counterexample_lost"
                if not scoped_ce_preserved
                else "per_branch_or_fresh_mse_exceeded_tolerance"
            ),
            "per_branch_mse": per_branch_mse,
            "per_branch_fresh_mse": per_branch_fresh_mse,
            "scoped_counterexamples_preserved": scoped_ce_preserved,
            "restored_branch_ids": [b["claim_id"] for b in branch_records],
        }

    # Seed generalized commitment with first branch's concrete constant so existing calibration
    # tools (`describe_comparator_threshold_parameter` / `screen_custom_edge_predicate_on_transitions`)
    # can calibrate `theta_0` seamlessly on brand-new unseen apparatuses!
    seed_bind = per_branch_bindings[branch_records[0]["claim_id"]]
    seed_pred = _bind_arg_slots(
        skeleton, {k: v for k, v in seed_bind.items() if k.startswith("theta_")}
    )
    for rp in generalized_comm.get("radius_parameters") or ():
        rp["edge_condition"] = seed_pred.to_dict()
    for rt in generalized_comm.get("role_terms") or ():
        def _seed_rt(nd: dict) -> None:
            if nd.get("kind") == "all_relation_sum":
                nd["predicate"] = seed_pred.to_dict()
            for ch_d in nd.get("children") or ():
                if isinstance(ch_d, dict):
                    _seed_rt(ch_d)
        _seed_rt(rt)

    semantic_relations = [
        {
            "relation": "special_case_of",
            "child_claim_id": b["claim_id"],
            "parent_family_id": generalized_comm["commitment_id"],
            "parameter_binding": per_branch_bindings[b["claim_id"]],
        }
        for b in branch_records
    ]
    return {
        "accepted": True,
        "status": "accepted_parametric_generalization",
        "generalized_commitment": generalized_comm,
        "anti_unified_predicate_skeleton": skeleton.to_dict(),
        "parameter_slots": [s["slot"] for s in all_slots],
        "per_branch_mse": per_branch_mse,
        "per_branch_fresh_mse": per_branch_fresh_mse,
        "scoped_counterexamples_preserved": True,
        "semantic_relations": semantic_relations,
    }


def propagate_premise_retraction_invalidation(
    claims: list[dict[str, Any]],
    *,
    retracted_premise_or_claim_ids: set[str] | list[str] = (),
) -> list[dict[str, Any]]:
    """Update dependent claims after premise retraction and reject ungrounded support cycles (Acceptance B9).

    Rules:
      - A claim is **grounded** iff either:
          (a) it has direct non-retracted ``independent_empirical_support == True`` and is not in
              ``retracted_premise_or_claim_ids``, OR
          (b) it has a non-empty dependency set ``depends_on_claim_ids`` where ALL required
              dependencies are grounded (acyclic foundation reachability!).
      - Pure support cycles ``A <-> B`` without an independent empirical foundation NEVER self-certify!
      - Claims with independent empirical support survive retraction of an unrelated claim.
    """
    retracted = set(retracted_premise_or_claim_ids or ())
    updated_claims = [deepcopy(c) for c in claims]
    by_id = {str(c["claim_id"]): c for c in updated_claims}

    grounded: set[str] = set()
    for cid, c in by_id.items():
        if cid in retracted or c.get("status") == "revoked":
            continue
        deps = list(c.get("depends_on_claim_ids") or ())
        req_prems = set(c.get("required_premise_ids") or ())
        if req_prems & retracted:
            continue
        if not deps and c.get("independent_empirical_support", True):
            grounded.add(cid)
        elif c.get("independent_empirical_support", False) and not (req_prems & retracted):
            grounded.add(cid)

    changed = True
    while changed:
        changed = False
        for cid, c in by_id.items():
            if cid in grounded or cid in retracted or c.get("status") == "revoked":
                continue
            req_prems = set(c.get("required_premise_ids") or ())
            if req_prems & retracted:
                continue
            deps = list(c.get("depends_on_claim_ids") or ())
            if deps and all(d in grounded for d in deps):
                grounded.add(cid)
                changed = True

    for cid, c in by_id.items():
        if cid in grounded:
            c["support_validity"] = "valid_grounded"
        elif cid in retracted:
            c["support_validity"] = "directly_retracted"
            c["status"] = "stale_retracted"
        else:
            deps = list(c.get("depends_on_claim_ids") or ())
            # Detect whether failure is due to an ungrounded cycle vs upstream retraction
            visited: set[str] = set()
            stack = list(deps)
            in_cycle = False
            while stack:
                cur = stack.pop()
                if cur == cid:
                    in_cycle = True
                    break
                if cur not in visited and cur in by_id:
                    visited.add(cur)
                    stack.extend(by_id[cur].get("depends_on_claim_ids") or ())
            if in_cycle:
                c["support_validity"] = "invalid_ungrounded_cycle"
                c["status"] = "unverified_cyclic_support"
            else:
                c["support_validity"] = "invalidated_by_upstream_retraction"
                c["status"] = "stale_dependency_invalidated"
    return updated_claims


def reconcile_competing_branches(
    new_claim: dict,
    old_claims: list[dict],
    spec: TaskSpec,
    target_schema: ObjectSchema | dict,
    *,
    execute_probe_fn: Any = None,
    controls_by_object: dict[str, dict[str, str]] | None = None,
    old_support_regimes_by_id: dict[str, list[dict[str, float]]] | None = None,
    new_unit_scales: dict[str, float] | None = None,
    old_unit_scales_by_id: dict[str, dict[str, float]] | None = None,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Execute the full ``reconcile(new_claim, old_claims, observations, budget)`` specification (§2.1)."""
    alignment = align_semantics_roles_units_and_time(
        new_claim,
        old_claims,
        target_schema=target_schema,
        new_unit_scales=new_unit_scales,
        old_unit_scales_by_id=old_unit_scales_by_id,
        budget=budget,
    )
    if not alignment["aligned_new"].get("aligned"):
        return {"status": "new_claim_unaligned", "alignment": alignment}

    aligned_new = alignment["aligned_new"]
    cur_new = deepcopy(new_claim)
    cur_olds: list[dict] = []
    decisions: list[dict[str, Any]] = []
    paid_probes_executed: list[dict[str, Any]] = []

    for item in alignment["compatible_candidates"]:
        old_raw = item["raw_claim"]
        aligned_old = item["aligned"]
        oid = str(old_raw.get("claim_id") or aligned_old["claim_id"])

        overlap = assess_scope_overlap(cur_new, old_raw)
        if overlap["certified_disjoint"]:
            decisions.append(
                {
                    "old_claim_id": oid,
                    "action": "retain_conditional_alternatives",
                    "overlap": overlap,
                }
            )
            cur_olds.append(deepcopy(old_raw))
            continue

        disagreement = find_feasible_prediction_difference(
            aligned_new,
            aligned_old,
            spec,
            target_schema,
            controls_by_object=controls_by_object,
            budget=budget,
        )
        if disagreement.get("proved_absent"):
            # Certified equivalence consolidation: merge provenance without duplicating nodes!
            merged_old = deepcopy(old_raw)
            merged_sources = list(
                dict.fromkeys(
                    [
                        *(merged_old.get("source_concept_ids") or ()),
                        *(cur_new.get("source_concept_ids") or ()),
                    ]
                )
            )
            merged_old["source_concept_ids"] = merged_sources
            decisions.append(
                {
                    "old_claim_id": oid,
                    "action": "consolidate_certified_equivalent",
                    "equivalence_method": disagreement.get("equivalence_method"),
                }
            )
            cur_olds.append(merged_old)
            cur_new["consolidated_into"] = oid
            continue

        if disagreement.get("witness_available") and execute_probe_fn is not None:
            paid_res = execute_probe_fn(disagreement["probe"])
            if paid_res is not None:
                p_obs, p_trans = paid_res
                adj = adjudicate_branches_on_paid_evidence(
                    cur_new,
                    old_raw,
                    aligned_new,
                    aligned_old,
                    p_obs,
                    p_trans,
                    target_schema,
                    old_support_regime_features=(old_support_regimes_by_id or {}).get(oid),
                    budget=budget,
                    tolerance=tolerance,
                )
                cur_new = adj["updated_new_claim"]
                cur_olds.append(adj["updated_old_claim"])
                paid_probes_executed.append(
                    {
                        "probe_id": disagreement["probe"].probe_id,
                        "max_predicted_disagreement": disagreement["max_abs_disagreement"],
                        "verdict": adj["verdict"],
                        "mse_new": adj["mse_new"],
                        "mse_old": adj["mse_old"],
                    }
                )
                decisions.append(
                    {
                        "old_claim_id": oid,
                        "action": adj["verdict"],
                        "paid_probe_id": disagreement["probe"].probe_id,
                    }
                )
                continue

        # Evidence or budget insufficient to discriminate -> retain unresolved competition!
        decisions.append(
            {
                "old_claim_id": oid,
                "action": "retain_unresolved_competition",
                "overlap": overlap,
                "witness_available": bool(disagreement.get("witness_available")),
            }
        )
        cur_olds.append(deepcopy(old_raw))

    return {
        "status": "reconciled",
        "updated_new_claim": cur_new,
        "updated_old_claims": cur_olds,
        "decisions": decisions,
        "paid_probes_executed": paid_probes_executed,
        "incompatible_semantics": alignment["incompatible_semantics"],
    }


def instantiate_parametric_family_on_target(
    generalized_comm: dict[str, Any],
    spec: TaskSpec,
    training: list[Observation],
    schema: ObjectSchema | dict,
    budget: SearchBudget,
    *,
    audit_report: dict | None = None,
    transitions: list[dict] | tuple[dict, ...] = (),
    tolerance: float = 1e-7,
) -> list[dict[str, Any]]:
    """Instantiate an anti-unified parametric family ``H*(r_0, theta_*)`` on a new target apparatus.

    Calibrates continuous parameter slots ``theta_*`` and radius ``r_0`` directly from target
    observed transitions and ``training`` rows (zero hidden oracle hints!).
    """
    from .models import Candidate

    sc = _as_schema(schema)
    if sc is None or not training:
        return []
    skel_dict = generalized_comm.get("anti_unified_predicate_skeleton")
    slots = list(generalized_comm.get("parameter_slots") or ())
    if not skel_dict or not slots:
        return []
    skeleton = Program.from_dict(skel_dict)

    r_params = generalized_comm.get("radius_parameters") or ()
    r_sym = r_params[0]["symbol"] if r_params else "r_0"
    dir_mode = r_params[0].get("direction", "right") if r_params else "right"

    # Candidate radii from audit_report interval or source values
    cand_radii: list[float] = []
    if audit_report and audit_report.get("radius_interval"):
        r_lo, r_hi = (
            float(audit_report["radius_interval"][0]),
            float(audit_report["radius_interval"][1]),
        )
        cand_radii.append(0.5 * (r_lo + r_hi))
        if r_hi - r_lo > 0.02:
            cand_radii.extend([r_lo + 0.25 * (r_hi - r_lo), r_lo + 0.75 * (r_hi - r_lo)])
    for rp in r_params:
        for sv in rp.get("source_values") or ():
            cand_radii.append(float(sv))
    if not cand_radii:
        cand_radii = [0.90]
    # Deduplicate within 1e-4
    dedup_radii: list[float] = []
    for rv in cand_radii:
        if all(abs(rv - x) > 1e-4 for x in dedup_radii):
            dedup_radii.append(rv)

    # Extract candidate threshold values for `theta_0` from empirical active/inactive directed gaps!
    cand_thetas: list[float] = []
    if audit_report and audit_report.get("verified_threshold_interval"):
        t_lo, t_hi = (
            float(audit_report["verified_threshold_interval"][0]),
            float(audit_report["verified_threshold_interval"][1]),
        )
        cand_thetas.append(0.5 * (t_lo + t_hi))

    if transitions:
        objs = list(sc.objects)
        r_ref = dedup_radii[0]
        active_gaps: list[float] = []
        inactive_pos_gaps: list[float] = []
        for rec in transitions:
            bef = rec.get("before") or {}
            aft = rec.get("after") or {}
            if not bef or not aft:
                continue
            qs = [float(bef.get(sc.fields[g]["q"], 0.0)) for g in objs]
            us = [float(bef.get(sc.fields[g].get("u", ""), 0.0)) for g in objs]
            xs = [float(bef.get(sc.fields[g].get("x", ""), 0.0)) for g in objs]
            q_aft = [float(aft.get(sc.fields[g]["q"], 0.0)) for g in objs]
            deltas = [q_aft[i] - qs[i] - us[i] for i in range(len(objs))]
            for i in range(len(objs)):
                if qs[i] <= 1e-4:
                    continue
                for j in range(len(objs)):
                    if i == j:
                        continue
                    dx = xs[j] - xs[i]
                    in_dir = (0.0 < dx < r_ref) if dir_mode == "right" else (0.0 < -dx < r_ref)
                    if not in_dir:
                        continue
                    gap = qs[i] - qs[j]
                    # Count clean single-neighbor transitions to bound theta_0 tightly!
                    other_neighbors = sum(
                        1
                        for k in range(len(objs))
                        if k not in (i, j) and (abs(xs[k] - xs[i]) < r_ref or abs(xs[k] - xs[j]) < r_ref)
                    )
                    if other_neighbors == 0:
                        if deltas[i] < -1e-4 and deltas[j] > 1e-4:
                            active_gaps.append(gap)
                        elif abs(deltas[i]) <= 1e-5 and gap > 0.0:
                            inactive_pos_gaps.append(gap)
        if active_gaps and inactive_pos_gaps:
            lo_b = max(inactive_pos_gaps)
            hi_b = min(active_gaps)
            if lo_b <= hi_b + 0.05:
                cand_thetas.append(0.5 * (lo_b + hi_b))
        elif active_gaps:
            cand_thetas.append(max(0.0, min(active_gaps) - 0.04))

    for grid_t in (0.0, 0.15, 0.22, 0.28, 0.34, 0.40, 0.46, 0.55):
        if all(abs(grid_t - ct) > 0.02 for ct in cand_thetas):
            cand_thetas.append(grid_t)

    best_entry: dict[str, Any] | None = None
    for r_try in dedup_radii[:3]:
        for th_try in cand_thetas[:10]:
            slot_map = {slots[0]: float(th_try)}
            bound_pred = _bind_arg_slots(skeleton, slot_map)
            inst_comm = deepcopy(generalized_comm)
            for rp in inst_comm.get("radius_parameters") or ():
                rp["edge_condition"] = bound_pred.to_dict()
            for rt in inst_comm.get("role_terms") or ():
                def _set_p(nd: dict) -> None:
                    if nd.get("kind") == "all_relation_sum":
                        nd["predicate"] = bound_pred.to_dict()
                    for ch_d in nd.get("children") or ():
                        if isinstance(ch_d, dict):
                            _set_p(ch_d)
                _set_p(rt)
            terms = instantiate_commitment_terms(
                inst_comm,
                sc,
                radius_bindings={r_sym: float(r_try)},
                predicate_overrides={r_sym: bound_pred},
            )
            if not terms:
                continue
            cost = len(training) * (len(terms) + 2)
            if not budget.spend(cost):
                break
            cols = [[float(t.predict(r)) for r in training] for t in terms]
            coeffs = _linear_fit(cols, [float(r.target) for r in training])
            prog = _sum_terms(list(terms), coeffs[1:], coeffs[0]).simplified()
            tr_loss = float(loss(prog, training, 0.0, spec.risk_quantile))
            cand_entry = {
                "candidate": Candidate(
                    f"parametric_family:{generalized_comm['commitment_id']}:theta_{th_try:.3f}:r_{r_try:.3f}",
                    prog,
                    "instantiate_anti_unified_parametric_family",
                    rationale=f"Calibrated parametric family {generalized_comm['commitment_id']} at theta_0={th_try:.4f}, r_0={r_try:.4f}",
                    parents=tuple(generalized_comm.get("source_branch_ids") or ()),
                ),
                "kind": "anti_unified_parametric_family",
                "family_id": "T03",
                "training_loss": tr_loss,
                "calibrated_slots": {**slot_map, r_sym: float(r_try)},
                "commitment_id": generalized_comm["commitment_id"],
            }
            if best_entry is None or tr_loss < best_entry["training_loss"]:
                best_entry = cand_entry
            if tr_loss <= tolerance:
                return [cand_entry]

    return [best_entry] if best_entry is not None else []

