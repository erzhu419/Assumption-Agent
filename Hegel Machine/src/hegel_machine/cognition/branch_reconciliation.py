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
        "after",
        "target",
        "residual",
        "loss",
    }
)

# Explicit public observable variable manifest (v12 Deliverable A2 / F09):
# Binds every permitted scope-guard variable to its physical unit, temporal availability,
# and whether a calibration pilot batch must be purchased before evaluating the guard!
PUBLIC_OBSERVABLE_GUARD_MANIFEST: dict[str, dict[str, Any]] = {
    "obs_min_active_donor_q": {
        "unit": "quantity",
        "timing": "pilot_calibration_summary",
        "requires_pilot_acquisition": True,
    },
    "obs_max_active_donor_q": {
        "unit": "quantity",
        "timing": "pilot_calibration_summary",
        "requires_pilot_acquisition": True,
    },
    "obs_min_active_gap_dq": {
        "unit": "quantity",
        "timing": "pilot_calibration_summary",
        "requires_pilot_acquisition": True,
    },
    "obs_signed_direction_flux": {
        "unit": "length_times_quantity",
        "timing": "pilot_calibration_summary",
        "requires_pilot_acquisition": True,
    },
    "obs_conservation_residual": {
        "unit": "quantity_squared",
        "timing": "pilot_calibration_summary",
        "requires_pilot_acquisition": True,
    },
    "obs_mean_abs_q": {
        "unit": "quantity",
        "timing": "pre_prediction_state",
        "requires_pilot_acquisition": False,
    },
    "obs_unconditional_block_rate": {
        "unit": "dimensionless",
        "timing": "pilot_calibration_summary",
        "requires_pilot_acquisition": True,
    },
    "obs_negative_donor_transfer_rate": {
        "unit": "dimensionless",
        "timing": "pilot_calibration_summary",
        "requires_pilot_acquisition": True,
    },
    "edge_source_quantity": {
        "unit": "quantity",
        "timing": "pre_prediction_state",
        "requires_pilot_acquisition": False,
    },
    "edge_target_quantity": {
        "unit": "quantity",
        "timing": "pre_prediction_state",
        "requires_pilot_acquisition": False,
    },
    "edge_source_position": {
        "unit": "length",
        "timing": "pre_prediction_state",
        "requires_pilot_acquisition": False,
    },
    "edge_target_position": {
        "unit": "length",
        "timing": "pre_prediction_state",
        "requires_pilot_acquisition": False,
    },
}

OBSERVABLE_REGIME_FEATURES = frozenset(PUBLIC_OBSERVABLE_GUARD_MANIFEST.keys())


def extract_guard_ast_dependencies(guard: Program | dict | None) -> tuple[list[str], bool]:
    """Extract all manifest variables and pilot-acquisition dependency from the COMPLETE guard AST (v13 A2 / Check #5)."""
    if guard is None:
        return [], False
    prog = guard if isinstance(guard, Program) else Program.from_dict(guard)
    seen: list[str] = []

    def _walk(node: Program) -> None:
        if node.op in {"feature", "arg"} and node.name and node.name not in seen:
            seen.append(node.name)
        for ch in node.children:
            _walk(ch)

    _walk(prog)
    requires_pilot = any(
        bool((PUBLIC_OBSERVABLE_GUARD_MANIFEST.get(fn) or {}).get("requires_pilot_acquisition", False))
        for fn in seen
    )
    return seen, requires_pilot


def validate_observable_scope_guard(
    guard: Program | dict | None,
    *,
    pilot_calibration_available: bool = True,
) -> tuple[bool, str]:
    """Verify ``guard`` uses ONLY manifest-declared public observable variables (whitelist + timing check)."""
    if guard is None:
        return True, "universal_scope"
    prog = guard if isinstance(guard, Program) else Program.from_dict(guard)

    def _check(node: Program) -> tuple[bool, str]:
        if node.op == "lag":
            return False, "forbidden_post_step_or_lag_in_scope_guard"
        if node.op in {"feature", "arg"} and node.name:
            low_n = node.name.lower()
            for bad in FORBIDDEN_SCOPE_TOKENS:
                if bad in low_n:
                    return False, f"forbidden_non_observable_token:{node.name}"
            meta = PUBLIC_OBSERVABLE_GUARD_MANIFEST.get(node.name)
            if meta is None:
                return False, f"unmanifested_unknown_scope_variable:{node.name}"
            if meta["requires_pilot_acquisition"] and not pilot_calibration_available:
                return False, f"pilot_calibration_required_before_evaluating:{node.name}"
        for ch in node.children:
            ok, msg = _check(ch)
            if not ok:
                return False, msg
        return True, "ok"

    return _check(prog)


def compute_observable_regime_features(
    transitions: list[dict] | tuple[dict, ...],
    schema: ObjectSchema | dict | None,
    *,
    protocol_id: str = "paid_episode_transitions",
    timing: str = "pilot_calibration_summary",
    source_ref: str | None = None,
) -> dict[str, Any]:
    """Compute strictly observable transition summary features with explicit counts & unknown values (v13 A2 / Check #4).

    Never encodes ``0/0`` ratios or ``min(empty)`` as ``0.0``: missing pilot data, zero valid neighbors,
    or zero negative-donor candidates yield ``None`` (unknown) alongside ``_meta`` counts & protocol identity.
    """
    all_feat_keys = (
        "obs_min_active_donor_q",
        "obs_max_active_donor_q",
        "obs_min_active_gap_dq",
        "obs_signed_direction_flux",
        "obs_conservation_residual",
        "obs_mean_abs_q",
        "obs_unconditional_block_rate",
        "obs_negative_donor_transfer_rate",
    )
    sc = _as_schema(schema)
    if sc is None or not transitions:
        return {
            **{k: None for k in all_feat_keys},
            "_meta": {
                "pilot_calibration_available": False,
                "total_valid_transitions": 0,
                "protocol_id": protocol_id,
                "timing": timing,
                "source_ref": source_ref,
                "valid_counts": {k: 0 for k in all_feat_keys},
                "available": {k: False for k in all_feat_keys},
            },
        }

    objs = list(sc.objects)
    n = len(objs)
    cons_sq: list[float] = []
    signed_flux_votes: list[float] = []
    active_donors: list[float] = []
    active_gaps: list[float] = []
    abs_qs: list[float] = []
    active_pair_dists: list[float] = []
    parsed_steps: list[tuple[list[float], list[float], list[float]]] = []

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
        parsed_steps.append((qs, xs, deltas))

        moved = [i for i in range(n) if abs(deltas[i]) > 1e-5]
        if len(moved) == 2:
            ia, ib = moved[0], moved[1]
            active_pair_dists.append(abs(xs[ib] - xs[ia]))
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

    if not parsed_steps:
        return {
            **{k: None for k in all_feat_keys},
            "_meta": {
                "pilot_calibration_available": False,
                "total_valid_transitions": 0,
                "protocol_id": protocol_id,
                "timing": timing,
                "source_ref": source_ref,
                "valid_counts": {k: 0 for k in all_feat_keys},
                "available": {k: False for k in all_feat_keys},
            },
        }

    mean_flux = (
        float(sum(signed_flux_votes) / len(signed_flux_votes))
        if signed_flux_votes
        else None
    )
    obs_active_r = (
        min(1.15, max(active_pair_dists) * 0.92)
        if active_pair_dists
        else 0.72
    )
    dir_sign = 1.0 if (mean_flux is None or mean_flux >= 0.0) else -1.0
    close_candidates = 0
    blocked_candidates = 0
    neg_donor_candidates = 0
    neg_donor_active = 0
    for qs, xs, deltas in parsed_steps:
        for i in range(n):
            if abs(qs[i]) <= 0.08:
                continue
            directed_nbrs = [
                j
                for j in range(n)
                if j != i and 0.0 < dir_sign * (xs[j] - xs[i]) < obs_active_r
            ]
            other_close = [
                k
                for k in range(n)
                if k != i and k not in directed_nbrs and abs(xs[k] - xs[i]) < obs_active_r * 1.25
            ]
            if len(directed_nbrs) == 1 and not other_close:
                close_candidates += 1
                if abs(deltas[i]) <= 1e-5:
                    blocked_candidates += 1
                if qs[i] < -0.08:
                    neg_donor_candidates += 1
                    if abs(deltas[i]) > 1e-4:
                        neg_donor_active += 1

    block_rate = float(blocked_candidates / close_candidates) if close_candidates > 0 else None
    neg_rate = float(neg_donor_active / neg_donor_candidates) if neg_donor_candidates > 0 else None

    values: dict[str, float | None] = {
        "obs_min_active_donor_q": float(min(active_donors)) if active_donors else None,
        "obs_max_active_donor_q": float(max(active_donors)) if active_donors else None,
        "obs_min_active_gap_dq": float(min(active_gaps)) if active_gaps else None,
        "obs_signed_direction_flux": mean_flux,
        "obs_conservation_residual": float(sum(cons_sq) / len(cons_sq)) if cons_sq else None,
        "obs_mean_abs_q": float(sum(abs_qs) / len(abs_qs)) if abs_qs else None,
        "obs_unconditional_block_rate": block_rate,
        "obs_negative_donor_transfer_rate": neg_rate,
    }
    counts: dict[str, int] = {
        "obs_min_active_donor_q": len(active_donors),
        "obs_max_active_donor_q": len(active_donors),
        "obs_min_active_gap_dq": len(active_gaps),
        "obs_signed_direction_flux": len(signed_flux_votes),
        "obs_conservation_residual": len(cons_sq),
        "obs_mean_abs_q": len(abs_qs),
        "obs_unconditional_block_rate": close_candidates,
        "obs_negative_donor_transfer_rate": neg_donor_candidates,
    }
    return {
        **values,
        "_meta": {
            "pilot_calibration_available": True,
            "total_valid_transitions": len(parsed_steps),
            "protocol_id": protocol_id,
            "timing": timing,
            "source_ref": source_ref,
            "valid_counts": counts,
            "available": {k: (values[k] is not None and counts[k] > 0) for k in all_feat_keys},
        },
    }


def evaluate_scope_guard(
    guard: Program | dict | None,
    obs_features: dict[str, Any] | None,
    *,
    expected_protocol: str | None = None,
) -> str:
    """Three-valued scope/routing guard evaluator returning ``'accept' | 'reject' | 'unknown'`` (v13 A2 / Check #4)."""
    if guard is None:
        return "accept"
    feats = obs_features or {}
    meta = feats.get("_meta") if isinstance(feats, dict) else None
    pilot_avail = bool(meta.get("pilot_calibration_available", True)) if isinstance(meta, dict) else bool(feats)
    ok, reason = validate_observable_scope_guard(guard, pilot_calibration_available=pilot_avail)
    if not ok:
        if reason.startswith("pilot_calibration_required_before_evaluating:"):
            return "unknown"
        raise ValueError(f"Illegal non-observable scope guard: {reason}")
    if (
        expected_protocol is not None
        and isinstance(meta, dict)
        and meta.get("protocol_id") is not None
        and meta.get("protocol_id") != expected_protocol
    ):
        return "unknown"

    prog = guard if isinstance(guard, Program) else Program.from_dict(guard)
    valid_counts = meta.get("valid_counts", {}) if isinstance(meta, dict) else {}

    def _eval_kleene_bool(nd: Program) -> bool | None:
        if nd.op == "and":
            vals = [_eval_kleene_bool(ch) for ch in nd.children]
            if any(v is False for v in vals):
                return False
            if all(v is True for v in vals):
                return True
            return None
        if nd.op == "or":
            vals = [_eval_kleene_bool(ch) for ch in nd.children]
            if any(v is True for v in vals):
                return True
            if all(v is False for v in vals):
                return False
            return None
        if nd.op == "not":
            v0 = _eval_kleene_bool(nd.children[0])
            return None if v0 is None else (not v0)
        if nd.op == "lt":
            a = _eval_num(nd.children[0])
            b = _eval_num(nd.children[1])
            if a is None or b is None:
                return None
            return bool(a < b)
        nv = _eval_num(nd)
        return None if nv is None else bool(nv >= 0.5)

    def _eval_num(nd: Program) -> float | None:
        if nd.op == "const":
            return float(nd.value or 0.0)
        if nd.op in {"feature", "arg"}:
            nm = str(nd.name or "")
            if nm not in feats or feats[nm] is None:
                return None
            if nm in valid_counts and int(valid_counts[nm]) <= 0:
                return None
            val = float(feats[nm])
            return val if isfinite(val) else None
        ch_vals = [_eval_num(ch) for ch in nd.children]
        if any(v is None for v in ch_vals):
            return None
        if nd.op == "add":
            return float(sum(ch_vals))  # type: ignore[arg-type]
        if nd.op == "sub":
            return float(ch_vals[0] - ch_vals[1])  # type: ignore[operator]
        if nd.op == "mul":
            prod = 1.0
            for v in ch_vals:
                prod *= float(v)  # type: ignore[arg-type]
            return prod
        if nd.op in {"and", "or", "not", "lt"}:
            bv = _eval_kleene_bool(nd)
            return None if bv is None else (1.0 if bv else 0.0)
        row_clean = Observation(
            {k: float(v) for k, v in feats.items() if k != "_meta" and isinstance(v, (int, float)) and isfinite(float(v))},
            0.0,
            (),
        )
        return float(nd.predict(row_clean))

    tri = _eval_kleene_bool(prog)
    if tri is True:
        return "accept"
    if tri is False:
        return "reject"
    return "unknown"


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


def _infer_homogeneous_q_degree(
    node: Program,
    q_like_features: set[str],
) -> int | None:
    """Return homogeneous polynomial degree in ``(q, u)`` features, or ``None`` if unsupported/inhomogeneous."""
    if node.op == "const":
        return 0
    if node.op == "feature":
        return 1 if (node.name in q_like_features) else 0
    if node.op in {"window_sum", "event_delta"}:
        return (
            _infer_homogeneous_q_degree(node.children[0], q_like_features)
            if node.children
            else None
        )
    if node.op == "accumulate":
        if not node.children:
            return None
        return _infer_homogeneous_q_degree(node.children[0], q_like_features)
    if node.op == "object_sum":
        if not node.children:
            return 0
        degs = [_infer_homogeneous_q_degree(c, q_like_features) for c in node.children]
        if any(d is None for d in degs) or len(set(degs)) > 1:
            return None
        return degs[0]
    if node.op in {"add", "sub", "neg"}:
        if len(node.children) == 1:
            return _infer_homogeneous_q_degree(node.children[0], q_like_features)
        non_zero_degs: list[int] = []
        for c in node.children:
            if c.op == "const" and abs(float(c.value or 0.0)) <= 1e-15:
                continue
            dc = _infer_homogeneous_q_degree(c, q_like_features)
            if dc is None:
                return None
            non_zero_degs.append(dc)
        if not non_zero_degs:
            return 0
        if len(set(non_zero_degs)) > 1:
            return None
        return non_zero_degs[0]
    if node.op == "mul":
        total_deg = 0
        for c in node.children:
            dc = _infer_homogeneous_q_degree(c, q_like_features)
            if dc is None:
                return None
            total_deg += dc
        return total_deg
    if node.op in {"neighbor_sum", "relation_sum"}:
        offset = 2 if node.op == "relation_sum" else 1
        val_nodes = [node.children[idx + 2] for idx in range(offset, len(node.children), 3)]
        if not val_nodes:
            return None
        degs = [_infer_homogeneous_q_degree(vn, q_like_features) for vn in val_nodes]
        if any(d is None for d in degs) or len(set(degs)) > 1:
            return None
        return degs[0]
    return None


def instantiate_aligned_claim_program(
    claim: dict,
    target_schema: ObjectSchema | dict,
    *,
    unit_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Align a world commitment/claim onto ``target_schema`` with role & physical-unit conversion (Acceptance B1, A3.5).

    If ``q_target = q_scale * q_source``, ``x_target = x_scale * x_source``, and
    ``y_target = y_scale * y_source``:
      - radius in target coordinates becomes ``r_target = x_scale * r_source``;
      - homogeneous degree-``p`` term in ``(q, u)`` scales by ``y_scale / (q_scale ** p)``;
      - inhomogeneous or unsupported nonlinear terms return ``aligned=False`` explicitly!
    """
    sc = _as_schema(target_schema)
    if sc is None:
        return {"aligned": False, "reason": "invalid_target_schema"}
    scales = unit_scales or {}
    q_scale = float(scales.get("q_scale", 1.0))
    x_scale = float(scales.get("x_scale", 1.0))
    y_scale = float(scales.get("y_scale", q_scale))
    if abs(q_scale) <= 1e-12 or abs(x_scale) <= 1e-12:
        return {"aligned": False, "reason": "degenerate_zero_unit_scale"}

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

    # Support explicit custom nonlinear readout terms supplied on claim["explicit_terms"] for A3.5 unit verification
    if claim.get("explicit_terms"):
        terms = tuple(
            t if isinstance(t, Program) else Program.from_dict(t)
            for t in claim["explicit_terms"]
        )

    coeffs = list(claim.get("coefficients") or [0.0, *([1.0] * len(terms))])
    if len(coeffs) != len(terms) + 1:
        coeffs = [0.0, *([1.0] * len(terms))]

    q_like_feats = {
        sc.fields[g][role]
        for g in sc.objects
        for role in ("q", "u")
        if role in sc.fields[g]
    }
    need_q_scaling = abs(q_scale - 1.0) > 1e-12 or abs(y_scale - 1.0) > 1e-12
    adj_weights: list[float] = []
    term_degrees: list[int] = []
    for w_raw, t_prog in zip(coeffs[1:], terms):
        deg_p = _infer_homogeneous_q_degree(t_prog, q_like_feats)
        if deg_p is None:
            if need_q_scaling:
                return {
                    "aligned": False,
                    "reason": "unsupported_inhomogeneous_or_nonlinear_q_unit_scaling",
                }
            deg_p = 1
        term_degrees.append(deg_p)
        scale_factor = (y_scale / (q_scale ** deg_p)) if need_q_scaling else 1.0
        adj_weights.append(float(w_raw) * scale_factor)

    adj_intercept = float(coeffs[0]) * y_scale
    prog = _sum_terms(list(terms), adj_weights, adj_intercept)

    temporal_depth = int(prog.history_depth)
    return {
        "aligned": True,
        "claim_id": claim.get("claim_id") or comm.get("commitment_id"),
        "program": prog,
        "terms": terms,
        "term_q_degrees": term_degrees,
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


def _extract_guard_interval_bounds(
    guard: Program | dict | None,
) -> tuple[dict[str, list[float]], str]:
    """Extract exact 1D interval bounds implied by an ``lt`` / ``and`` scope guard AST."""
    if guard is None:
        return {}, "universal"
    prog = guard if isinstance(guard, Program) else Program.from_dict(guard)
    intervals: dict[str, list[float]] = {}

    def _parse(node: Program) -> bool:
        if node.op == "and":
            return all(_parse(ch) for ch in node.children)
        if node.op == "lt" and len(node.children) == 2:
            left, right = node.children
            if left.op == "const" and right.op in {"feature", "arg"} and right.name:
                lo, hi = intervals.get(right.name, [-1e6, 1e6])
                intervals[right.name] = [max(lo, float(left.value or 0.0)), hi]
                return True
            if left.op in {"feature", "arg"} and left.name and right.op == "const":
                lo, hi = intervals.get(left.name, [-1e6, 1e6])
                intervals[left.name] = [lo, min(hi, float(right.value or 0.0))]
                return True
        return False

    if _parse(prog):
        return intervals, "exact_interval_conjunction"
    return {}, "non_box_guard"


def assess_scope_overlap(claim_a: dict, claim_b: dict) -> dict[str, Any]:
    """Assess scope overlap between two claims without ever defaulting ``unknown`` to ``disjoint`` (Acceptance B2, A2)."""
    guard_a = claim_a.get("scope_guard")
    guard_b = claim_b.get("scope_guard")
    ok_a, msg_a = validate_observable_scope_guard(guard_a)
    ok_b, msg_b = validate_observable_scope_guard(guard_b)
    if not ok_a or not ok_b:
        raise ValueError(f"Non-observable scope guard rejected: {msg_a} / {msg_b}")

    if claim_a.get("status") == "scope_unidentified" or claim_b.get("status") == "scope_unidentified":
        return {
            "status": "unknown_overlap_scope_unidentified",
            "certified_disjoint": False,
            "unknown": True,
        }

    raw_bounds_a = claim_a.get("observable_bounds") or {}
    raw_bounds_b = claim_b.get("observable_bounds") or {}
    guard_bounds_a, mode_a = _extract_guard_interval_bounds(guard_a)
    guard_bounds_b, mode_b = _extract_guard_interval_bounds(guard_b)

    if guard_a is None and guard_b is None and not raw_bounds_a and not raw_bounds_b:
        return {
            "status": "overlapping_universal",
            "certified_disjoint": False,
            "unknown": False,
        }

    if mode_a == "non_box_guard" or mode_b == "non_box_guard":
        return {
            "status": "unknown_overlap_must_test",
            "certified_disjoint": False,
            "unknown": True,
        }

    def _effective_consistent_bounds(
        raw_b: dict[str, Any], g_b: dict[str, list[float]], has_guard: bool
    ) -> tuple[dict[str, list[float]] | None, str]:
        if has_guard and not raw_b:
            # Guard present without explicit bounds metadata -> do not assume disjoint without testing!
            return None, "missing_bounds_metadata"
        eff: dict[str, list[float]] = {}
        for fk, iv in raw_b.items():
            if fk not in OBSERVABLE_REGIME_FEATURES:
                raise ValueError(f"Non-observable scope bound feature forbidden: {fk}")
            r_lo, r_hi = float(iv[0]), float(iv[1])
            if fk in g_b:
                g_lo, g_hi = g_b[fk]
                # `raw_b` must either match `guard` or be a conservative outer envelope (r_lo <= g_lo + 1e-4 and r_hi >= g_hi - 1e-4)
                if r_lo > g_lo + 1e-4 or r_hi < g_hi - 1e-4:
                    return None, f"bounds_contradict_guard_on_{fk}"
            eff[fk] = [r_lo, r_hi]
        for fk, giv in g_b.items():
            if fk not in eff:
                return None, f"guard_feature_missing_from_bounds_{fk}"
        return eff, "consistent"

    bounds_a, cons_a = _effective_consistent_bounds(raw_bounds_a, guard_bounds_a, guard_a is not None)
    bounds_b, cons_b = _effective_consistent_bounds(raw_bounds_b, guard_bounds_b, guard_b is not None)
    if bounds_a is None or bounds_b is None:
        return {
            "status": "unknown_overlap_must_test",
            "certified_disjoint": False,
            "unknown": True,
            "bounds_consistency": {"claim_a": cons_a, "claim_b": cons_b},
        }

    # Certified disjointness ONLY when an explicit observable feature interval is provably disjoint!
    for feat_k in set(bounds_a.keys()) & set(bounds_b.keys()):
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
    aligned_a: dict[str, Any], aligned_b: dict[str, Any], *, exact: bool = True, tol: float = 0.0
) -> bool:
    """Strict canonical AST & parameter identity check (v12 A3.1 / F05: zero tolerance on discrete predicates!)."""
    comm_a = aligned_a["aligned_commitment"]
    comm_b = aligned_b["aligned_commitment"]
    if comm_a.get("nonconservation_mechanism") != comm_b.get("nonconservation_mechanism"):
        return False
    if comm_a.get("latent_state_mechanism") != comm_b.get("latent_state_mechanism"):
        return False

    eff_tol = 0.0 if exact else float(tol)
    rpa = comm_a.get("radius_parameters") or ()
    Rpb = comm_b.get("radius_parameters") or ()
    if len(rpa) != len(Rpb):
        return False
    for pa, pb in zip(rpa, Rpb):
        if pa.get("direction") != pb.get("direction"):
            return False
        ra = float(aligned_a["radius_bindings"].get(pa["symbol"], 1.0))
        rb = float(aligned_b["radius_bindings"].get(pb["symbol"], 1.0))
        # Radius is a spatial step-function threshold: exact equality (up to IEEE754 binary roundoff 1e-15 on exact dyadic scales) required!
        if abs(ra - rb) > max(eff_tol, 1e-15):
            return False
        ca = _canonical_edge_predicate_ast(pa.get("edge_condition")).simplified()
        cb = _canonical_edge_predicate_ast(pb.get("edge_condition")).simplified()
        if not _programs_algebraically_identical(ca, cb, exact=True, tol=0.0):
            return False

    ca_vec = aligned_a["coefficients"]
    cb_vec = aligned_b["coefficients"]
    if len(ca_vec) != len(cb_vec):
        return False
    if any(abs(float(x) - float(y)) > max(eff_tol, 1e-15) for x, y in zip(ca_vec, cb_vec)):
        return False

    ta_list = aligned_a["terms"]
    tb_list = aligned_b["terms"]
    if len(ta_list) != len(tb_list):
        return False
    for ta, tb in zip(ta_list, tb_list):
        if not _programs_algebraically_identical(ta.simplified(), tb.simplified(), exact=exact, tol=eff_tol):
            return False
    return True


def _programs_algebraically_identical(
    pa: Program, pb: Program, *, exact: bool = True, tol: float = 0.0
) -> bool:
    """Exact canonical AST identity (v12 F05: never treats distinct comparator thresholds like 0 vs 5e-10 as identical!)."""
    if pa.op != pb.op or pa.name != pb.name or pa.lag != pb.lag:
        return False
    eff_tol = 0.0 if exact else float(tol)
    if pa.op == "const":
        va = float(pa.value or 0.0)
        vb = float(pb.value or 0.0)
        return (va == vb) if exact else (abs(va - vb) <= eff_tol)
    if pa.value is not None or pb.value is not None:
        if pa.value is None or pb.value is None:
            return False
        va = float(pa.value)
        vb = float(pb.value)
        if (va != vb) if exact else (abs(va - vb) > eff_tol):
            return False
    if len(pa.children) != len(pb.children):
        return False
    # Inside any discrete comparator or Boolean connective, constants MUST be strictly identical!
    child_exact = exact or (pa.op in {"lt", "le", "gt", "ge", "eq", "and", "or", "not", "if_pos", "relation_sum", "neighbor_sum"})
    child_tol = 0.0 if child_exact else eff_tol
    if pa.op in {"add", "mul", "and", "or"} and len(pa.children) == 2:
        direct = all(
            _programs_algebraically_identical(ca, cb, exact=child_exact, tol=child_tol)
            for ca, cb in zip(pa.children, pb.children)
        )
        if direct:
            return True
        return _programs_algebraically_identical(
            pa.children[0], pb.children[1], exact=child_exact, tol=child_tol
        ) and _programs_algebraically_identical(
            pa.children[1], pb.children[0], exact=child_exact, tol=child_tol
        )
    return all(
        _programs_algebraically_identical(ca, cb, exact=child_exact, tol=child_tol)
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
                    q_i, u_i, x_i, f_i = q0_val * scale_t, 0.0, 0.0, 1.0
                elif i_obj == 1:
                    q_i, u_i, x_i, f_i = q1_val * scale_t, 0.0, dx_val, 0.0
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


def _program_semantic_version(prog: Program | dict | None) -> str:
    if prog is None:
        return "none"
    d = prog.to_dict() if isinstance(prog, Program) else prog
    raw = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return f"prog_v1:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def shrink_claim_to_observable_scope(
    refuted_universal_claim: dict,
    retained_support_regime_features: list[dict[str, Any]],
    refuting_regime_features: dict[str, Any] | list[dict[str, Any]],
    *,
    scope_kind: str = "empirical_applicability_scope",
) -> dict:
    """Derive a verified observable ``Program`` guard separating retained support from counterexamples (B5, A2, v13 B1/B2).

    ``scope_kind`` distinguishes:
      - ``'empirical_applicability_scope'``: counterexamples come from actual prediction failure of this claim;
        updates ``scope_guard`` and restricts scientific applicability.
      - ``'routing_scope_candidate'``: negative examples come only from differing structural labels / successful
        neighbor branches without cross-prediction failure; updates ``routing_guard`` ONLY (guides candidate
        ordering without revoking scientific eligibility!).
    """
    updated = deepcopy(refuted_universal_claim)
    refuting_list = (
        [refuting_regime_features]
        if isinstance(refuting_regime_features, dict)
        else list(refuting_regime_features or ())
    )

    if not retained_support_regime_features or not refuting_list:
        if scope_kind == "empirical_applicability_scope":
            updated["previous_status"] = updated.get("status", "universal")
            updated["status"] = "scope_unidentified"
            updated["shrunk"] = False
            updated["scope_guard"] = None
        else:
            updated["routing_shrunk"] = False
            updated["routing_guard"] = None
        updated["scope_verification"] = {
            "status": "scope_unidentified",
            "scope_kind": scope_kind,
            "reason": (
                "missing_retained_support_regimes"
                if not retained_support_regime_features
                else "missing_refuting_regimes"
            ),
            "retained_support_count": len(retained_support_regime_features or ()),
            "excluded_counterexample_count": 0,
        }
        updated.setdefault("scope_revision_history", []).append(
            {
                "event": "scope_unidentified_insufficient_empirical_separation",
                "scope_kind": scope_kind,
                "reason": updated["scope_verification"]["reason"],
                "refuting_regime_features": deepcopy(refuting_regime_features),
            }
        )
        return updated

    candidate_features = (
        "obs_unconditional_block_rate",
        "obs_signed_direction_flux",
        "obs_min_active_donor_q",
        "obs_min_active_gap_dq",
        "obs_negative_donor_transfer_rate",
        "obs_conservation_residual",
    )

    remaining_ce = list(refuting_list)
    chosen_atoms: list[Program] = []
    bounds_field = "observable_bounds" if scope_kind == "empirical_applicability_scope" else "routing_bounds"
    guard_field = "scope_guard" if scope_kind == "empirical_applicability_scope" else "routing_guard"
    chosen_bounds: dict[str, list[float]] = deepcopy(updated.get(bounds_field) or {})
    chosen_features: list[str] = []
    min_halfwidth = float("inf")

    # Keep existing guard atoms if they cover 100% of retained support regimes (`== 'accept'`)
    existing_guard = updated.get(guard_field)
    if existing_guard is not None:
        ex_prog = existing_guard if isinstance(existing_guard, Program) else Program.from_dict(existing_guard)
        if all(evaluate_scope_guard(ex_prog, sup_r) == "accept" for sup_r in retained_support_regime_features):
            chosen_atoms.append(ex_prog)
            remaining_ce = [
                ce_r for ce_r in remaining_ce if evaluate_scope_guard(ex_prog, ce_r) != "reject"
            ]

    for _step in range(3):
        if not remaining_ce:
            break
        best_choice: tuple[int, float, str, Program, list[float]] | None = None
        for feat_name in candidate_features:
            if not all(
                feat_name in r and r[feat_name] is not None
                for r in retained_support_regime_features
            ):
                continue
            valid_ces = [
                ce_r for ce_r in remaining_ce if feat_name in ce_r and ce_r[feat_name] is not None
            ]
            if not valid_ces:
                continue
            sup_vals = [float(r[feat_name]) for r in retained_support_regime_features]
            sup_lo, sup_hi = min(sup_vals), max(sup_vals)

            below_ces = [float(ce_r[feat_name]) for ce_r in valid_ces if float(ce_r[feat_name]) < sup_lo - 1e-5]
            if below_ces:
                closest_below = max(below_ces)
                mid = round(0.5 * (closest_below + sup_lo), 6)
                margin = sup_lo - closest_below
                atom = Program(
                    "lt",
                    children=(Program("const", value=mid), Program("feature", name=feat_name)),
                )
                cand_tuple = (len(below_ces), margin, feat_name, atom, [mid, 1e6])
                if best_choice is None or (cand_tuple[0], cand_tuple[1]) > (best_choice[0], best_choice[1]):
                    best_choice = cand_tuple

            above_ces = [float(ce_r[feat_name]) for ce_r in valid_ces if float(ce_r[feat_name]) > sup_hi + 1e-5]
            if above_ces:
                closest_above = min(above_ces)
                mid = round(0.5 * (sup_hi + closest_above), 6)
                margin = closest_above - sup_hi
                atom = Program(
                    "lt",
                    children=(Program("feature", name=feat_name), Program("const", value=mid)),
                )
                cand_tuple = (len(above_ces), margin, feat_name, atom, [-1e6, mid])
                if best_choice is None or (cand_tuple[0], cand_tuple[1]) > (best_choice[0], best_choice[1]):
                    best_choice = cand_tuple

        if best_choice is None:
            break
        _, margin_c, feat_c, atom_c, bnd_c = best_choice
        chosen_atoms.append(atom_c)
        if feat_c not in chosen_features:
            chosen_features.append(feat_c)
        prev_lo, prev_hi = chosen_bounds.get(feat_c, [-1e6, 1e6])
        chosen_bounds[feat_c] = [max(float(prev_lo), bnd_c[0]), min(float(prev_hi), bnd_c[1])]
        min_halfwidth = min(min_halfwidth, 0.5 * margin_c)
        remaining_ce = [
            ce_r for ce_r in remaining_ce if evaluate_scope_guard(atom_c, ce_r) != "reject"
        ]

    if not chosen_atoms or remaining_ce:
        if scope_kind == "empirical_applicability_scope":
            updated["previous_status"] = updated.get("status", "universal")
            updated["status"] = "scope_unidentified"
            updated["shrunk"] = False
            updated["scope_guard"] = None
        else:
            updated["routing_shrunk"] = False
            updated["routing_guard"] = None
        updated["scope_verification"] = {
            "status": "scope_unidentified",
            "scope_kind": scope_kind,
            "reason": "no_manifest_observable_separates_support_and_counterexample",
            "retained_support_count": len(retained_support_regime_features),
            "unexcluded_counterexample_count": len(remaining_ce),
        }
        updated.setdefault("scope_revision_history", []).append(
            {
                "event": "scope_unidentified_no_separating_observable",
                "scope_kind": scope_kind,
                "refuting_regime_features": deepcopy(refuting_regime_features),
            }
        )
        return updated

    final_guard = chosen_atoms[0]
    for nxt_atom in chosen_atoms[1:]:
        final_guard = Program("and", children=(final_guard, nxt_atom))

    ok, msg = validate_observable_scope_guard(final_guard)
    assert ok, msg
    retained_hits = sum(
        1 for r in retained_support_regime_features if evaluate_scope_guard(final_guard, r) == "accept"
    )
    excluded_hits = sum(
        1 for r in refuting_list if evaluate_scope_guard(final_guard, r) == "reject"
    )
    # Check #5: Compute AST dependencies & pilot requirement from the COMPLETE final_guard AST!
    all_ast_features, requires_pilot = extract_guard_ast_dependencies(final_guard)
    sup_protocols = sorted(
        {
            str((r.get("_meta") or {}).get("protocol_id"))
            for r in retained_support_regime_features
            if isinstance(r.get("_meta"), dict) and (r.get("_meta") or {}).get("protocol_id")
        }
    )
    single_episode_only = len(retained_support_regime_features) == 1
    verif_status = (
        "single_episode_local_scope_candidate"
        if single_episode_only
        else (
            "candidate_empirical_scope_verified_on_observed_regimes"
            if scope_kind == "empirical_applicability_scope"
            else "candidate_routing_preference_verified_on_observed_regimes"
        )
    )
    verif_record = {
        "status": verif_status,
        "scope_kind": scope_kind,
        "epistemic_status": (
            "single_source_local_basis_not_global_verification"
            if single_episode_only
            else "candidate_scope_hypothesis_not_global_proof"
        ),
        "single_source_local_support_only": single_episode_only,
        "protocols_observed": sup_protocols or ["paid_episode_transitions"],
        "retained_support_passed": retained_hits,
        "retained_support_total": len(retained_support_regime_features),
        "excluded_counterexample_passed": excluded_hits,
        "excluded_counterexample_total": len(refuting_list),
        "boundary_uncertainty_halfwidth": round(float(min_halfwidth), 6) if isfinite(min_halfwidth) else None,
        "bounds_consistency": "exact_box_consistent_with_guard",
        "guard_ast_features": all_ast_features,
        "requires_pilot_calibration": requires_pilot,
    }

    if scope_kind == "empirical_applicability_scope":
        updated["previous_status"] = updated.get("status", "universal")
        updated["status"] = "scoped_conditional"
        updated["shrunk"] = True
        updated["scope_kind"] = "empirical_applicability_scope"
        updated["scope_guard"] = final_guard.to_dict()
        updated["observable_bounds"] = chosen_bounds
        eff_features = all_ast_features or chosen_features
        updated["scope_shrink_feature"] = chosen_features[0] if chosen_features else (eff_features[0] if eff_features else None)
        updated["scope_shrink_features"] = eff_features
        updated["scope_verification"] = verif_record
    else:
        updated["routing_shrunk"] = True
        updated["routing_scope_kind"] = "routing_scope_candidate"
        updated["routing_guard"] = final_guard.to_dict()
        updated["routing_bounds"] = chosen_bounds
        updated["routing_features"] = all_ast_features or chosen_features
        updated["routing_verification"] = verif_record

    updated.setdefault("scope_revision_history", []).append(
        {
            "event": (
                "shrunk_universal_scope_after_empirical_counterexample"
                if scope_kind == "empirical_applicability_scope"
                else "updated_empirical_routing_candidate_from_neighbor_regimes"
            ),
            "scope_kind": scope_kind,
            "separating_features": all_ast_features or chosen_features,
            "scope_verification": deepcopy(verif_record),
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
    target_schema: ObjectSchema | dict | None,
    *,
    new_paid_observations: list[Observation] | None = None,
    old_support_regime_features: list[dict[str, Any]] | None = None,
    evidence_ref: dict[str, Any] | None = None,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Adjudicate competing claims on real paid evidence with version-bound counterexamples & pending subquestions (v13 A3, B1)."""
    if budget is not None:
        budget.spend(2 * max(1, len(paid_observations)))

    obs_for_new = new_paid_observations if new_paid_observations is not None else paid_observations
    mse_new = float(loss(aligned_new["program"], obs_for_new, 0.0, 0.5))
    mse_old = float(loss(aligned_old["program"], paid_observations, 0.0, 0.5))
    obs_reg = compute_observable_regime_features(
        paid_transitions,
        target_schema,
        protocol_id=str((evidence_ref or {}).get("protocol_id") or "discriminating_probe"),
        source_ref=str((evidence_ref or {}).get("actual_probe_id") or "paid_evidence"),
    )

    new_prog_ver = _program_semantic_version(aligned_new.get("program"))
    old_prog_ver = _program_semantic_version(aligned_old.get("program"))
    ev_meta = deepcopy(evidence_ref) if evidence_ref is not None else {
        "status": "direct_paid_observations",
        "actual_probe_id": None,
        "observation_count": len(paid_observations),
    }

    new_ok = mse_new <= tolerance
    old_ok = mse_old <= tolerance

    if old_ok and not new_ok:
        # Check #8: Refuted newcomer instance is rejected WITHOUT overwriting valid old claim
        # and WITHOUT automatically refuting the entire parametric model family!
        rej_new = deepcopy(new_claim)
        rej_new["status"] = "refuted"
        rej_new["refutation_scope"] = "specific_parameter_instance_only_not_entire_family"
        rej_new.setdefault("counterexamples", []).append(
            {
                "evidence_kind": "empirical_episode",
                "tested_program_version": new_prog_ver,
                "role_and_parameter_bindings": deepcopy(aligned_new.get("radius_bindings") or {}),
                "declared_scope_guard": deepcopy(new_claim.get("scope_guard")),
                "evidence_ref": ev_meta,
                "instance_only_not_family_refutation": True,
                "mse_on_discriminating_probe": mse_new,
                "observable_regime": obs_reg,
            }
        )
        return {
            "verdict": "new_claim_refuted_old_preserved",
            "mse_new": mse_new,
            "mse_old": mse_old,
            "tested_program_versions": {"new": new_prog_ver, "old": old_prog_ver},
            "updated_new_claim": rej_new,
            "updated_old_claim": deepcopy(old_claim),
            "old_claim_overwritten": False,
            "pending_questions": [],
        }

    if new_ok and not old_ok:
        # Old claim failed on paid probe -> bind counterexample to exact tested version & shrink old applicability scope!
        old_with_ce = deepcopy(old_claim)
        old_with_ce.setdefault("counterexamples", []).append(
            {
                "evidence_kind": "empirical_episode",
                "tested_program_version": old_prog_ver,
                "role_and_parameter_bindings": deepcopy(aligned_old.get("radius_bindings") or {}),
                "declared_scope_guard": deepcopy(old_claim.get("scope_guard")),
                "evidence_ref": ev_meta,
                "instance_only_not_family_refutation": True,
                "mse_on_discriminating_probe": mse_old,
                "observable_regime": obs_reg,
            }
        )
        shrunk_old = shrink_claim_to_observable_scope(
            old_with_ce,
            old_support_regime_features or [],
            obs_reg,
            scope_kind="empirical_applicability_scope",
        )
        if shrunk_old.get("status") == "scope_unidentified":
            pending_new = deepcopy(new_claim)
            pending_new["status"] = "conditional_candidate_scope_pending"
            rq_scope = ResearchQuestion(
                question="Empirical probe refuted universal scope of prior claim, but retained support pilot statistics did not separate an observable scope boundary.",
                evidence={"mse_new": mse_new, "mse_old": mse_old, "refuting_regime": obs_reg},
                operations=("acquire_pilot_regime_summary", "identify_observable_scope_guard"),
            )
            pq_scope = {
                "question_id": f"scope_subq:{old_claim.get('claim_id')}_vs_{new_claim.get('claim_id')}",
                "status": "pending",
                "kind": "scope_boundary_unidentified_after_refutation",
                "tested_program_versions": {"new": new_prog_ver, "old": old_prog_ver},
                "role_and_parameter_bindings": {
                    "new": deepcopy(aligned_new.get("radius_bindings") or {}),
                    "old": deepcopy(aligned_old.get("radius_bindings") or {}),
                },
                "counterexample_evidence_ref": ev_meta,
                "observable_regime": deepcopy(obs_reg),
                "unresolved_parent_goal": "separate_applicable_observable_domains",
                "allowed_return_types": ["pilot_regime_summary", "observable_scope_guard"],
                "target_parent_mutation": "update_branch_scope_guard_or_experiment",
                "research_question": {
                    "question": rq_scope.question,
                    "evidence": rq_scope.evidence,
                    "operations": list(rq_scope.operations),
                },
            }
            return {
                "verdict": "old_refuted_on_probe_scope_unidentified_pending_support",
                "mse_new": mse_new,
                "mse_old": mse_old,
                "tested_program_versions": {"new": new_prog_ver, "old": old_prog_ver},
                "updated_new_claim": pending_new,
                "updated_old_claim": shrunk_old,
                "old_claim_overwritten": False,
                "research_question": rq_scope,
                "pending_question": pq_scope,
                "pending_questions": [pq_scope],
            }

        # Check #12 & B1: Complement of old guard is NEVER auto-confirmed as new_claim's global applicability scope!
        # Third mechanisms outside `old_guard` remain possible; store complement ONLY as an unconfirmed proposal
        # and routing candidate, while keeping `new_claim`'s scientific `scope_guard = None` until tested against a CE of its own!
        scoped_new = deepcopy(new_claim)
        if scoped_new.get("status") in {None, "universal"}:
            scoped_new["status"] = "locally_supported_conditional"
        if shrunk_old.get("scope_shrink_feature") and shrunk_old.get("observable_bounds"):
            feat_s = shrunk_old["scope_shrink_feature"]
            b_old = shrunk_old["observable_bounds"][feat_s]
            if b_old[1] >= 1e5:
                cut = float(b_old[0])
                comp_bounds = {feat_s: [-1e6, cut - 1e-5]}
                comp_prog = Program(
                    "lt",
                    children=(Program("feature", name=feat_s), Program("const", value=cut - 1e-5)),
                )
            else:
                cut = float(b_old[1])
                comp_bounds = {feat_s: [cut + 1e-5, 1e6]}
                comp_prog = Program(
                    "lt",
                    children=(Program("const", value=cut + 1e-5), Program("feature", name=feat_s)),
                )
            old_sup_checks = [
                evaluate_scope_guard(comp_prog, r) for r in (old_support_regime_features or ())
            ]
            new_sup_check = evaluate_scope_guard(comp_prog, obs_reg)
            scoped_new["proposed_complement_scope_candidate"] = {
                "guard": comp_prog.to_dict(),
                "bounds": comp_bounds,
                "confirmed_applicability_scope": False,
                "allows_third_mechanism": True,
                "epistemic_status": "unverified_complement_proposal_third_mechanism_possible",
                "checked_excluded_old_support": all(c == "reject" for c in old_sup_checks) if old_sup_checks else False,
                "checked_covers_new_support": (new_sup_check == "accept"),
            }
            if new_sup_check == "accept":
                scoped_new["routing_guard"] = comp_prog.to_dict()
                scoped_new["routing_bounds"] = comp_bounds
                scoped_new["routing_scope_kind"] = "routing_scope_candidate"
        return {
            "verdict": "old_universal_shrunk_and_conditional_branch_retained",
            "mse_new": mse_new,
            "mse_old": mse_old,
            "tested_program_versions": {"new": new_prog_ver, "old": old_prog_ver},
            "updated_new_claim": scoped_new,
            "updated_old_claim": shrunk_old,
            "old_claim_overwritten": False,
            "pending_questions": [],
        }

    if not new_ok and not old_ok:
        # Checks #6, #7 & A3: BOTH claims refuted on real paid evidence!
        # Restrict BOTH claims' applicability on this regime (preserving valid old source calculations)
        # AND return a concrete parent-bound pending subquestion!
        ce_common = {
            "evidence_kind": "empirical_episode",
            "evidence_ref": ev_meta,
            "observable_regime": obs_reg,
            "instance_only_not_family_refutation": True,
        }
        old_ce_claim = deepcopy(old_claim)
        old_ce_claim.setdefault("counterexamples", []).append(
            {
                **ce_common,
                "tested_program_version": old_prog_ver,
                "role_and_parameter_bindings": deepcopy(aligned_old.get("radius_bindings") or {}),
                "declared_scope_guard": deepcopy(old_claim.get("scope_guard")),
                "mse_on_discriminating_probe": mse_old,
            }
        )
        if old_support_regime_features:
            upd_old = shrink_claim_to_observable_scope(
                old_ce_claim,
                old_support_regime_features,
                obs_reg,
                scope_kind="empirical_applicability_scope",
            )
            if upd_old.get("status") == "scope_unidentified":
                upd_old["status"] = "contested_scope_unidentified"
        else:
            upd_old = old_ce_claim
            upd_old["previous_status"] = upd_old.get("status", "universal")
            upd_old["status"] = "contested_scope_unidentified"

        new_ce_claim = deepcopy(new_claim)
        new_ce_claim.setdefault("counterexamples", []).append(
            {
                **ce_common,
                "tested_program_version": new_prog_ver,
                "role_and_parameter_bindings": deepcopy(aligned_new.get("radius_bindings") or {}),
                "declared_scope_guard": deepcopy(new_claim.get("scope_guard")),
                "mse_on_discriminating_probe": mse_new,
            }
        )
        new_sup_regs = list(new_claim.get("pilot_regime_features") or ())
        if new_sup_regs:
            upd_new = shrink_claim_to_observable_scope(
                new_ce_claim,
                new_sup_regs,
                obs_reg,
                scope_kind="empirical_applicability_scope",
            )
            if upd_new.get("status") == "scope_unidentified":
                upd_new["status"] = "contested_scope_unidentified"
        else:
            upd_new = new_ce_claim
            upd_new["previous_status"] = upd_new.get("status", "universal")
            upd_new["status"] = "contested_refuted_on_probe"

        rq = ResearchQuestion(
            question="Both competing hypotheses failed on paid discriminating probe; investigate missing state, representation, or non-pairwise mechanism.",
            evidence={
                "mse_new": mse_new,
                "mse_old": mse_old,
                "tested_program_versions": {"new": new_prog_ver, "old": old_prog_ver},
                "evidence_ref": ev_meta,
                "observable_regime": obs_reg,
            },
            operations=("expand_latent_state", "propose_open_grammar_revision", "audit_nonconservation"),
        )
        pending_q = {
            "question_id": f"reconcile_subq:{new_claim.get('claim_id')}_vs_{old_claim.get('claim_id')}:{ev_meta.get('actual_probe_id') or 'paid'}",
            "status": "pending",
            "kind": "both_competing_branches_refuted",
            "tested_program_versions": {
                "new_claim_id": new_claim.get("claim_id"),
                "new_program_version": new_prog_ver,
                "old_claim_id": old_claim.get("claim_id"),
                "old_program_version": old_prog_ver,
            },
            "role_and_parameter_bindings": {
                "new": deepcopy(aligned_new.get("radius_bindings") or {}),
                "old": deepcopy(aligned_old.get("radius_bindings") or {}),
            },
            "declared_scopes": {
                "new_scope_guard": deepcopy(new_claim.get("scope_guard")),
                "old_scope_guard": deepcopy(old_claim.get("scope_guard")),
            },
            "counterexample_evidence_ref": ev_meta,
            "observable_regime": deepcopy(obs_reg),
            "mse_new": mse_new,
            "mse_old": mse_old,
            "unresolved_parent_goal": "resolve_target_prediction_and_branch_competition",
            "allowed_return_types": [
                "revised_edge_predicate",
                "expanded_latent_state",
                "nonconservation_mechanism",
                "recalibrated_parameter_or_binding",
                "next_discriminating_probe",
            ],
            "target_parent_mutation": "replace_parent_candidate_or_schedule_discriminating_probe",
            "research_question": {
                "question": rq.question,
                "evidence": deepcopy(rq.evidence),
                "operations": list(rq.operations),
            },
        }
        return {
            "verdict": "both_claims_refuted_open_missing_mechanism_question",
            "mse_new": mse_new,
            "mse_old": mse_old,
            "tested_program_versions": {"new": new_prog_ver, "old": old_prog_ver},
            "updated_new_claim": upd_new,
            "updated_old_claim": upd_old,
            "research_question": rq,
            "pending_question": pending_q,
            "pending_questions": [pending_q],
        }

    return {
        "verdict": "both_consistent_on_probe_retain_competition",
        "mse_new": mse_new,
        "mse_old": mse_old,
        "tested_program_versions": {"new": new_prog_ver, "old": old_prog_ver},
        "updated_new_claim": deepcopy(new_claim),
        "updated_old_claim": deepcopy(old_claim),
        "pending_questions": [],
    }


def _is_theta_slot_arg(node: Program) -> bool:
    return node.op == "arg" and bool(node.name and node.name.startswith("theta_"))


def anti_unify_programs(
    prog_a: Program, prog_b: Program
) -> tuple[Program, list[dict[str, Any]]] | None:
    """Typed AST anti-unification supporting incremental ``theta_*`` slots + canonical preorder numbering (B7, A3.2).

    Each distinct differing AST path receives its own ``theta_k`` slot (never merging two separate
    positions by numerical coincidence!), and existing ``arg('theta_*')`` nodes unify seamlessly
    with new ``const`` or ``arg('theta_*')`` nodes across 3+ branches in any permutation order.
    """
    slots: list[dict[str, Any]] = []

    def _walk(na: Program, nb: Program, path: tuple[int, ...]) -> Program | None:
        if na.op == "const" and nb.op == "const":
            va = float(na.value or 0.0)
            vb = float(nb.value or 0.0)
            if va == vb:
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

        if _is_theta_slot_arg(na) and nb.op == "const":
            vb = float(nb.value or 0.0)
            slot_name = f"theta_{len(slots)}"
            slots.append(
                {
                    "slot": slot_name,
                    "path": path,
                    "source_slot_a": na.name,
                    "value_b": vb,
                }
            )
            return Program("arg", name=slot_name)

        if na.op == "const" and _is_theta_slot_arg(nb):
            va = float(na.value or 0.0)
            slot_name = f"theta_{len(slots)}"
            slots.append(
                {
                    "slot": slot_name,
                    "path": path,
                    "value_a": va,
                    "source_slot_b": nb.name,
                }
            )
            return Program("arg", name=slot_name)

        if _is_theta_slot_arg(na) and _is_theta_slot_arg(nb):
            slot_name = f"theta_{len(slots)}"
            slots.append(
                {
                    "slot": slot_name,
                    "path": path,
                    "source_slot_a": na.name,
                    "source_slot_b": nb.name,
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


def _extract_concrete_slot_env(concrete: Program, skeleton: Program) -> dict[str, float] | None:
    """Extract the exact concrete ``{theta_k: float}`` substitution mapping ``skeleton -> concrete``."""
    if _is_theta_slot_arg(skeleton):
        if concrete.op != "const":
            return None
        assert skeleton.name is not None
        return {skeleton.name: float(concrete.value or 0.0)}
    if concrete.op != skeleton.op or concrete.name != skeleton.name or concrete.lag != skeleton.lag:
        return None
    if len(concrete.children) != len(skeleton.children):
        return None
    if skeleton.op == "const":
        if float(concrete.value or 0.0) != float(skeleton.value or 0.0):
            return None
        return {}
    merged: dict[str, float] = {}
    for cc, cs in zip(concrete.children, skeleton.children):
        sub = _extract_concrete_slot_env(cc, cs)
        if sub is None:
            return None
        merged.update(sub)
    return merged


def anti_unify_program_family(
    named_programs: list[tuple[str, Program]] | dict[str, Program],
) -> dict[str, Any] | None:
    """Incrementally anti-unify ``>= 2`` concrete ``Program`` ASTs into an order-invariant skeleton + per-branch substitutions."""
    items = (
        list(named_programs.items())
        if isinstance(named_programs, dict)
        else list(named_programs)
    )
    if len(items) < 2:
        return None
    skeleton = items[0][1]
    latest_slots: list[dict[str, Any]] = []
    for _, nxt_prog in items[1:]:
        au = anti_unify_programs(skeleton, nxt_prog)
        if au is None:
            return None
        skeleton, latest_slots = au

    # Collect canonical preorder slot list from final `skeleton`
    slot_descriptors: list[dict[str, Any]] = []

    def _collect_slots(nd: Program, path: tuple[int, ...], parent_op: str | None) -> None:
        if _is_theta_slot_arg(nd) and nd.name:
            slot_descriptors.append(
                {"slot": nd.name, "path": list(path), "parent_op": parent_op}
            )
            return
        for idx, ch in enumerate(nd.children):
            _collect_slots(ch, (*path, idx), nd.op)

    _collect_slots(skeleton, (), None)
    branch_substitutions: dict[str, dict[str, float]] = {}
    for b_id, b_prog in items:
        env = _extract_concrete_slot_env(b_prog, skeleton)
        if env is None:
            return None
        reconstructed = _bind_arg_slots(skeleton, env)
        if not _programs_algebraically_identical(reconstructed, b_prog, exact=True):
            return None
        branch_substitutions[b_id] = env

    return {
        "skeleton": skeleton,
        "parameter_slots": [sd["slot"] for sd in slot_descriptors],
        "slot_descriptors": slot_descriptors,
        "branch_substitutions": branch_substitutions,
    }


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


def _has_unbound_theta_slot(prog: Program) -> bool:
    if _is_theta_slot_arg(prog):
        return True
    return any(_has_unbound_theta_slot(ch) for ch in prog.children)


def compute_family_scientific_identity(
    base_commitment: dict[str, Any],
    predicate_skeleton: Program,
    parameter_slots: list[str],
) -> tuple[str, str]:
    """Return ``(commitment_id, predicate_skeleton_id)`` binding full world semantics (v12 A3.4 / F08)."""
    skel_dict = predicate_skeleton.to_dict()
    pred_skel_id = "pred_skel:" + hashlib.sha256(
        json.dumps(skel_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    norm_role_terms = deepcopy(base_commitment.get("role_terms") or [])
    for rt in norm_role_terms:
        def _norm_rt_pred(nd: dict) -> None:
            if nd.get("kind") == "all_relation_sum":
                nd["predicate"] = skel_dict
            for ch_d in nd.get("children") or ():
                if isinstance(ch_d, dict):
                    _norm_rt_pred(ch_d)
        if isinstance(rt, dict):
            _norm_rt_pred(rt)

    full_identity_payload = {
        "predicate_skeleton": skel_dict,
        "parameter_slots": list(parameter_slots),
        "directions": [
            str(rp.get("direction") or "symmetric")
            for rp in (base_commitment.get("radius_parameters") or ())
        ],
        "nonconservation_mechanism": base_commitment.get("nonconservation_mechanism") or "conserved",
        "latent_state_mechanism": base_commitment.get("latent_state_mechanism") or "markov",
        "role_terms_skeleton": norm_role_terms,
        "scope_guard": base_commitment.get("scope_guard"),
        "unit_contract": base_commitment.get("unit_contract")
        or {"q_unit": "quantity", "x_unit": "length", "y_unit": "readout"},
    }
    comm_id = "generalized_family:" + hashlib.sha256(
        json.dumps(full_identity_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return comm_id, pred_skel_id


def propose_anti_unified_generalization(
    branch_records: list[dict[str, Any]],
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
    provisional_approximate_tolerance: float = 1.5e-2,
    require_fresh_confirmation: bool = False,
) -> dict[str, Any]:
    """Propose & strictly verify a common parametric family ``H*(theta)`` across ``>= 2`` branches (B7, B8, A1, A3).

    Confirmation contract (v12 A1 / F01):
      - Pre-declared ``tolerance`` (default ``1e-7``) applies uniformly regardless of whether
        ``parameter_slots`` is empty or non-empty (ZERO implicit ``1.5e-2`` relaxation!).
      - Missing ``fresh_holdout_rows`` is reported as ``fresh_confirmation_status = 'unknown_no_fresh_data'``
        (with ``None`` per-branch fresh MSE, NEVER ``0.0``!).
      - Families with MSE in ``(tolerance, provisional_approximate_tolerance]`` are NEVER marked
        ``accepted=True`` / ``confirmed=True``, but are returned with ``status = 'provisional_approximate_candidate_family'``
        so downstream study can inspect candidate structure while knowing exact confirmation failed.
    """
    if len(branch_records) < 2:
        return {"accepted": False, "confirmed": False, "reason": "need_at_least_two_branches"}
    if budget is not None and not budget.spend(15 * len(branch_records)):
        return {"accepted": False, "confirmed": False, "reason": "budget_exhausted"}

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

    if len(dirs) > 1 or len(noncons) > 1 or len(latents) > 1:
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
            "confirmed": False,
            "status": "rejected_false_merge",
            "reason": "minority_or_branch_region_degraded_under_incompatible_mechanisms",
            "pooled_mse": pooled_mse,
            "per_branch_mse": per_branch_mse,
            "max_branch_mse": max(per_branch_mse.values()) if per_branch_mse else float("inf"),
            "restored_branch_ids": [b["claim_id"] for b in branch_records],
        }

    # 2. Order-invariant incremental AST anti-unification across 2, 3, or more branches!
    named_preds = [
        (
            str(b["claim_id"]),
            _canonical_edge_predicate_ast(
                (b["commitment"].get("radius_parameters") or [{}])[0].get("edge_condition")
            ),
        )
        for b in branch_records
    ]
    family_au = anti_unify_program_family(named_preds)
    if family_au is None:
        return {
            "accepted": False,
            "confirmed": False,
            "status": "rejected_incompatible_predicate_skeleton",
            "reason": "edge_predicate_trees_do_not_anti_unify",
            "restored_branch_ids": [b["claim_id"] for b in branch_records],
        }

    skeleton: Program = family_au["skeleton"]
    slot_names: list[str] = family_au["parameter_slots"]
    branch_substitutions: dict[str, dict[str, float]] = family_au["branch_substitutions"]

    # 3. Verify per-branch source reconstruction, scoped counterexample distinction preservation,
    #    and independent fresh holdout points under fixed `tolerance`!
    per_branch_mse = {}
    per_branch_fresh_mse: dict[str, float | None] = {}
    per_branch_fresh_status: dict[str, str] = {}
    per_branch_bindings: dict[str, dict[str, float]] = {}
    scoped_ce_preserved = True

    generalized_comm = deepcopy(branch_records[0]["commitment"])
    comm_id, pred_skel_id = compute_family_scientific_identity(
        generalized_comm, skeleton, slot_names
    )
    generalized_comm["commitment_id"] = comm_id
    generalized_comm["predicate_skeleton_id"] = pred_skel_id
    generalized_comm["anti_unified_predicate_skeleton"] = skeleton.to_dict()
    generalized_comm["parameter_slots"] = list(slot_names)
    generalized_comm["slot_descriptors"] = deepcopy(family_au["slot_descriptors"])
    generalized_comm["source_slot_values"] = {
        sl: sorted({float(env[sl]) for env in branch_substitutions.values() if sl in env})
        for sl in slot_names
    }
    generalized_comm["source_branch_ids"] = [b["claim_id"] for b in branch_records]
    all_src_concepts: list[str] = []
    all_radii: list[float] = []
    for b in branch_records:
        for sc_id in b["commitment"].get("source_concept_ids") or ():
            if sc_id not in all_src_concepts:
                all_src_concepts.append(sc_id)
        for rp_b in b["commitment"].get("radius_parameters") or ():
            for sv in rp_b.get("source_values") or ():
                if all(abs(float(sv) - x) > 1e-6 for x in all_radii):
                    all_radii.append(float(sv))
    generalized_comm["source_concept_ids"] = all_src_concepts
    if generalized_comm.get("radius_parameters") and all_radii:
        generalized_comm["radius_parameters"][0]["source_values"] = all_radii

    for b in branch_records:
        cid = str(b["claim_id"])
        slot_bind = branch_substitutions.get(cid)
        if slot_bind is None:
            return {
                "accepted": False,
                "confirmed": False,
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
                "confirmed": False,
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

        for ce_row in b.get("scoped_counterexamples") or ():
            ce_err = (float(prog_b.predict(ce_row)) - float(ce_row.target)) ** 2
            if ce_err <= tolerance * 10.0:
                scoped_ce_preserved = False
                break

        if b.get("fresh_holdout_rows"):
            f_rows = list(b["fresh_holdout_rows"])
            f_mse = float(loss(prog_b, f_rows, 0.0, 0.5))
            per_branch_fresh_mse[cid] = f_mse
            per_branch_fresh_status[cid] = (
                "confirmed_fresh" if f_mse <= tolerance else "failed_fresh_tolerance"
            )
        else:
            per_branch_fresh_mse[cid] = None
            per_branch_fresh_status[cid] = "unknown_no_fresh_data"

    measured_fresh = [v for v in per_branch_fresh_mse.values() if v is not None]
    if not measured_fresh:
        fresh_confirmation_status = "unknown_no_fresh_data"
    elif any(v > tolerance for v in measured_fresh):
        fresh_confirmation_status = "failed_fresh_tolerance"
    elif len(measured_fresh) == len(branch_records):
        fresh_confirmation_status = "all_branches_fresh_confirmed"
    else:
        fresh_confirmation_status = "partial_fresh_confirmed"

    # Seed generalized commitment with first branch's concrete constant for compatibility with 1D scanners
    seed_bind = branch_substitutions[str(branch_records[0]["claim_id"])]
    seed_pred = _bind_arg_slots(skeleton, seed_bind)
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

    max_b_mse = max(per_branch_mse.values())
    max_f_mse = max(measured_fresh) if measured_fresh else None
    fresh_failed = (max_f_mse is not None and max_f_mse > tolerance) or (
        require_fresh_confirmation and fresh_confirmation_status != "all_branches_fresh_confirmed"
    )

    if max_b_mse > tolerance or not scoped_ce_preserved or fresh_failed:
        approx_ok = (
            scoped_ce_preserved
            and max_b_mse <= provisional_approximate_tolerance
            and (max_f_mse is None or max_f_mse <= provisional_approximate_tolerance)
            and not (require_fresh_confirmation and not measured_fresh)
        )
        if approx_ok:
            prov_comm = deepcopy(generalized_comm)
            prov_comm["confirmed"] = False
            prov_comm["epistemic_status"] = "provisional_approximate_candidate"
            return {
                "accepted": False,
                "confirmed": False,
                "status": "provisional_approximate_candidate_family",
                "epistemic_status": "provisional_approximate_candidate",
                "reason": "exceeded_exact_confirmation_tolerance_retained_as_provisional_candidate",
                "tolerance_contract": tolerance,
                "provisional_approximate_tolerance": provisional_approximate_tolerance,
                "provisional_candidate_commitment": prov_comm,
                "predicate_skeleton_id": pred_skel_id,
                "parameter_slots": list(slot_names),
                "per_branch_substitutions": branch_substitutions,
                "per_branch_mse": per_branch_mse,
                "per_branch_fresh_mse": per_branch_fresh_mse,
                "per_branch_fresh_status": per_branch_fresh_status,
                "fresh_confirmation_status": fresh_confirmation_status,
                "scoped_counterexamples_preserved": scoped_ce_preserved,
                "restored_branch_ids": [b["claim_id"] for b in branch_records],
            }
        return {
            "accepted": False,
            "confirmed": False,
            "status": "rejected_false_merge",
            "reason": (
                "scoped_counterexample_lost"
                if not scoped_ce_preserved
                else (
                    "missing_required_fresh_confirmation"
                    if (require_fresh_confirmation and not measured_fresh)
                    else "per_branch_or_fresh_mse_exceeded_tolerance"
                )
            ),
            "tolerance_contract": tolerance,
            "per_branch_mse": per_branch_mse,
            "per_branch_fresh_mse": per_branch_fresh_mse,
            "per_branch_fresh_status": per_branch_fresh_status,
            "fresh_confirmation_status": fresh_confirmation_status,
            "scoped_counterexamples_preserved": scoped_ce_preserved,
            "restored_branch_ids": [b["claim_id"] for b in branch_records],
        }

    confirmed_exact = fresh_confirmation_status == "all_branches_fresh_confirmed"
    source_pres_status = (
        "all_sources_preservation_verified"
        if confirmed_exact
        else fresh_confirmation_status
    )
    ep_status = (
        "source_preservation_verified_family"
        if confirmed_exact
        else "source_fit_verified_fresh_unknown"
    )
    generalized_comm["confirmed"] = confirmed_exact
    generalized_comm["epistemic_status"] = ep_status
    generalized_comm["source_preservation_status"] = source_pres_status
    generalized_comm["fresh_confirmation_status"] = fresh_confirmation_status
    generalized_comm["independent_out_of_source_confirmations"] = 0

    semantic_relations = [
        {
            "relation": "special_case_of",
            "child_claim_id": b["claim_id"],
            "parent_family_id": generalized_comm["commitment_id"],
            "parameter_binding": per_branch_bindings[str(b["claim_id"])],
        }
        for b in branch_records
    ]
    return {
        "accepted": True,
        "confirmed": confirmed_exact,
        "epistemic_status": ep_status,
        "source_preservation_status": source_pres_status,
        "independent_out_of_source_confirmations": 0,
        "status": "accepted_parametric_generalization",
        "tolerance_contract": tolerance,
        "generalized_commitment": generalized_comm,
        "predicate_skeleton_id": pred_skel_id,
        "anti_unified_predicate_skeleton": skeleton.to_dict(),
        "parameter_slots": list(slot_names),
        "per_branch_substitutions": branch_substitutions,
        "per_branch_mse": per_branch_mse,
        "per_branch_fresh_mse": per_branch_fresh_mse,
        "per_branch_fresh_status": per_branch_fresh_status,
        "fresh_confirmation_status": fresh_confirmation_status,
        "scoped_counterexamples_preserved": True,
        "semantic_relations": semantic_relations,
    }


def propagate_premise_retraction_invalidation(
    claims: list[dict[str, Any]],
    *,
    retracted_premise_or_claim_ids: set[str] | list[str] = (),
) -> list[dict[str, Any]]:
    """Update dependent claims after premise retraction and reject ungrounded support cycles (B9, A3.6, v13 Check #10).

    Rules:
      - A claim is **grounded** iff either:
          (a) it explicitly declares ``independent_empirical_support is True`` (never assumed by
              default on candidate roots!) and is not retracted/revoked, OR
          (b) it has non-empty ``depends_on_claim_ids`` where at least ``min_grounded_dependencies``
              (default ``len(depends_on_claim_ids)`` for conjunctive claims; ``2`` for multi-branch
              anti-unified families; ``1`` for multi-source shared commitments) dependencies are
              grounded — so retracting one source NEVER globally deletes a shared computation or
              family that retains independent valid support!
      - Candidate roots without ``independent_empirical_support is True`` default to
        ``support_validity = 'unverified_missing_support'``.
      - Pure support cycles ``A <-> B`` without an independent empirical foundation receive
        ``support_validity = 'invalid_ungrounded_cycle'``.
    """
    retracted = set(retracted_premise_or_claim_ids or ())
    updated_claims = [deepcopy(c) for c in claims]
    by_id = {str(c["claim_id"]): c for c in updated_claims}

    for cid, c in by_id.items():
        if c.get("status") == "revoked":
            retracted.add(cid)

    grounded: set[str] = set()
    for cid, c in by_id.items():
        if cid in retracted:
            continue
        req_prems = set(c.get("required_premise_ids") or ())
        if req_prems & retracted:
            continue
        has_direct_support = c.get("independent_empirical_support") is True or (
            bool(c.get("empirical_support_refs"))
            and c.get("independent_empirical_support") is not False
        )
        if has_direct_support:
            grounded.add(cid)

    changed = True
    while changed:
        changed = False
        for cid, c in by_id.items():
            if cid in grounded or cid in retracted:
                continue
            req_prems = set(c.get("required_premise_ids") or ())
            if req_prems & retracted:
                continue
            deps = list(c.get("depends_on_claim_ids") or ())
            if not deps:
                continue
            min_req = int(c.get("min_grounded_dependencies") or len(deps))
            min_req = max(1, min(min_req, len(deps)))
            grounded_deps = [d for d in deps if d in grounded]
            if len(grounded_deps) >= min_req:
                grounded.add(cid)
                c["active_grounded_dependencies"] = grounded_deps
                if len(grounded_deps) < len(deps):
                    c["retracted_dependencies_pruned"] = [d for d in deps if d not in grounded]
                changed = True

    for cid, c in by_id.items():
        if cid in grounded:
            c["support_validity"] = "valid_grounded"
        elif cid in retracted:
            c["support_validity"] = "directly_retracted"
            c["status"] = "stale_retracted"
        else:
            deps = list(c.get("depends_on_claim_ids") or ())
            req_prems = set(c.get("required_premise_ids") or ())
            visited: set[str] = set()
            stack = list(deps)
            in_cycle = False
            touches_retracted = bool(req_prems & retracted)
            while stack:
                cur = stack.pop()
                if cur in retracted:
                    touches_retracted = True
                if cur == cid:
                    in_cycle = True
                    break
                if cur not in visited and cur in by_id:
                    visited.add(cur)
                    cur_req = set(by_id[cur].get("required_premise_ids") or ())
                    if cur_req & retracted:
                        touches_retracted = True
                    stack.extend(by_id[cur].get("depends_on_claim_ids") or ())
            if in_cycle:
                c["support_validity"] = "invalid_ungrounded_cycle"
                c["status"] = "unverified_cyclic_support"
            elif touches_retracted:
                c["support_validity"] = "invalidated_by_upstream_retraction"
                c["status"] = "stale_dependency_invalidated"
            else:
                c["support_validity"] = "unverified_missing_support"
                if c.get("status") not in {
                    "candidate",
                    "provisional_approximate_candidate",
                    "pending_empirical_support",
                }:
                    c["status"] = "pending_empirical_support"
    return updated_claims


def reconcile_competing_branches(
    new_claim: dict,
    old_claims: list[dict],
    spec: TaskSpec | None,
    target_schema: ObjectSchema | dict | None,
    *,
    execute_probe_fn: Any = None,
    controls_by_object: dict[str, dict[str, str]] | None = None,
    old_support_regimes_by_id: dict[str, list[dict[str, Any]]] | None = None,
    new_unit_scales: dict[str, float] | None = None,
    old_unit_scales_by_id: dict[str, dict[str, float]] | None = None,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Execute ``reconcile(new_claim, old_claims, ...)`` with typed probe acquisition & subquestion return (v13 A1, A3)."""
    alignment = align_semantics_roles_units_and_time(
        new_claim,
        old_claims,
        target_schema=target_schema,
        new_unit_scales=new_unit_scales,
        old_unit_scales_by_id=old_unit_scales_by_id,
        budget=budget,
    )
    if not alignment["aligned_new"].get("aligned"):
        return {
            "status": "new_claim_unaligned",
            "alignment": alignment,
            "pending_questions": [],
            "paid_probes_executed": [],
        }

    aligned_new = alignment["aligned_new"]
    cur_new = deepcopy(new_claim)
    cur_olds: list[dict] = []
    decisions: list[dict[str, Any]] = []
    paid_probes_executed: list[dict[str, Any]] = []
    reused_protocol_probes: list[dict[str, Any]] = []
    historical_reanalyses: list[dict[str, Any]] = []
    pending_questions: list[dict[str, Any]] = []
    research_questions: list[Any] = []

    for item in alignment["compatible_candidates"]:
        old_raw = item["raw_claim"]
        aligned_old = item["aligned"]
        oid = str(old_raw.get("claim_id") or aligned_old.get("claim_id") or "old")

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
            merged_old["min_grounded_dependencies"] = 1
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
            req_probe = disagreement["probe"]
            req_id = getattr(req_probe, "probe_id", "requested_probe")
            frozen_vers = [
                _program_semantic_version(aligned_new.get("program")),
                _program_semantic_version(aligned_old.get("program")),
            ]
            try:
                paid_res = execute_probe_fn(req_probe, frozen_program_versions=frozen_vers)
            except TypeError:
                paid_res = execute_probe_fn(req_probe)

            # v13 A1 / Checks #1, #2, #3: ONLY accept typed acquisition contracts!
            # Raw 2-tuples or `unavailable` statuses are NEVER recorded in `paid_probes_executed`!
            if isinstance(paid_res, dict) and paid_res.get("status") in {
                "executed_new",
                "reused_exact_protocol",
            }:
                acq_status = str(paid_res["status"])
                actual_id = str(paid_res.get("actual_probe_id") or req_id)
                p_obs = list(paid_res.get("rows") or ())
                p_trans = list(paid_res.get("transitions") or ())
                if p_obs:
                    ev_ref = {
                        "status": acq_status,
                        "requested_probe_id": req_id,
                        "actual_probe_id": actual_id,
                        "observation_refs": list(paid_res.get("observation_refs") or ()),
                        "transition_refs": list(paid_res.get("transition_refs") or ()),
                        "program_versions_frozen_before_result": list(
                            paid_res.get("program_versions_frozen_before_result") or frozen_vers
                        ),
                        "cost_ref": deepcopy(paid_res.get("cost_ref")),
                        "independent_sample_increment": len(p_obs) if acq_status == "executed_new" else 0,
                    }
                    adj = adjudicate_branches_on_paid_evidence(
                        cur_new,
                        old_raw,
                        aligned_new,
                        aligned_old,
                        p_obs,
                        p_trans,
                        target_schema,
                        old_support_regime_features=(old_support_regimes_by_id or {}).get(oid),
                        evidence_ref=ev_ref,
                        budget=budget,
                        tolerance=tolerance,
                    )
                    cur_new = adj["updated_new_claim"]
                    cur_olds.append(adj["updated_old_claim"])
                    for pq in adj.get("pending_questions") or ():
                        pending_questions.append(pq)
                    if "research_question" in adj and adj["research_question"] is not None:
                        research_questions.append(adj["research_question"])

                    probe_entry = {
                        "probe_id": actual_id,
                        "requested_probe_id": req_id,
                        "acquisition_status": acq_status,
                        "independent_sample_increment": ev_ref["independent_sample_increment"],
                        "max_predicted_disagreement": disagreement["max_abs_disagreement"],
                        "verdict": adj["verdict"],
                        "mse_new": adj["mse_new"],
                        "mse_old": adj["mse_old"],
                    }
                    if acq_status == "executed_new":
                        paid_probes_executed.append(probe_entry)
                    else:
                        reused_protocol_probes.append(probe_entry)
                    decisions.append(
                        {
                            "old_claim_id": oid,
                            "action": adj["verdict"],
                            "acquisition_status": acq_status,
                            "requested_probe_id": req_id,
                            "actual_probe_id": actual_id,
                            "independent_sample_increment": ev_ref["independent_sample_increment"],
                        }
                    )
                    continue

            # Check #2: Optional historical reanalysis ONLY if explicitly supplied under `reanalyzable_existing_evidence`
            # AND never disguised as executing `req_id`!
            reanal = (
                paid_res.get("reanalyzable_existing_evidence")
                if isinstance(paid_res, dict)
                else None
            )
            if isinstance(reanal, dict) and reanal.get("rows"):
                h_rows = list(reanal["rows"])
                h_trans = list(reanal.get("transitions") or ())
                orig_id = str(reanal.get("original_probe_id") or "historical_paid_batch")
                max_h_diff = max(
                    (
                        abs(float(aligned_new["program"].predict(r)) - float(aligned_old["program"].predict(r)))
                        for r in h_rows
                    ),
                    default=0.0,
                )
                if max_h_diff > max(1e-4, 10.0 * tolerance):
                    ev_hist = {
                        "status": "reanalyze_paid_evidence",
                        "requested_probe_id": req_id,
                        "actual_probe_id": orig_id,
                        "executed_requested_probe": False,
                        "observation_refs": list(reanal.get("observation_refs") or ()),
                        "transition_refs": list(reanal.get("transition_refs") or ()),
                        "program_versions_frozen_before_result": frozen_vers,
                        "independent_sample_increment": 0,
                    }
                    adj_h = adjudicate_branches_on_paid_evidence(
                        cur_new,
                        old_raw,
                        aligned_new,
                        aligned_old,
                        h_rows,
                        h_trans,
                        target_schema,
                        old_support_regime_features=(old_support_regimes_by_id or {}).get(oid),
                        evidence_ref=ev_hist,
                        budget=budget,
                        tolerance=tolerance,
                    )
                    cur_new = adj_h["updated_new_claim"]
                    cur_olds.append(adj_h["updated_old_claim"])
                    for pq in adj_h.get("pending_questions") or ():
                        pending_questions.append(pq)
                    if "research_question" in adj_h and adj_h["research_question"] is not None:
                        research_questions.append(adj_h["research_question"])
                    historical_reanalyses.append(
                        {
                            "action": "reanalyze_paid_evidence",
                            "original_probe_id": orig_id,
                            "unexecuted_requested_probe_id": req_id,
                            "executed_requested_probe": False,
                            "independent_sample_increment": 0,
                            "verdict": adj_h["verdict"],
                            "mse_new": adj_h["mse_new"],
                            "mse_old": adj_h["mse_old"],
                        }
                    )
                    decisions.append(
                        {
                            "old_claim_id": oid,
                            "action": "reanalyze_paid_evidence",
                            "adjudication_verdict": adj_h["verdict"],
                            "original_probe_id": orig_id,
                            "requested_probe_id": req_id,
                            "executed_requested_probe": False,
                        }
                    )
                    continue

            decisions.append(
                {
                    "old_claim_id": oid,
                    "action": "retain_unresolved_competition",
                    "probe_acquisition_status": (
                        paid_res.get("status")
                        if isinstance(paid_res, dict)
                        else ("unavailable" if paid_res is None else "rejected_untyped_probe_return")
                    ),
                    "requested_probe_id": req_id,
                    "executed_requested_probe": False,
                    "overlap": overlap,
                    "witness_available": True,
                }
            )
            cur_olds.append(deepcopy(old_raw))
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

    out_rec: dict[str, Any] = {
        "status": "reconciled",
        "updated_new_claim": cur_new,
        "updated_old_claims": cur_olds,
        "decisions": decisions,
        "paid_probes_executed": paid_probes_executed,
        "reused_protocol_probes": reused_protocol_probes,
        "historical_reanalyses": historical_reanalyses,
        "pending_questions": pending_questions,
        "research_questions": research_questions,
        "incompatible_semantics": alignment["incompatible_semantics"],
    }
    if research_questions:
        out_rec["research_question"] = research_questions[0]
    return out_rec


def validate_parametric_family_calibration_support(
    generalized_comm: dict[str, Any],
) -> dict[str, Any]:
    """Check whether ``generalized_comm``'s parameter-slot arity is supported for target calibration (A3.3)."""
    skel_dict = generalized_comm.get("anti_unified_predicate_skeleton")
    slots = list(generalized_comm.get("parameter_slots") or ())
    if not skel_dict:
        return {"supported": False, "status": "missing_predicate_skeleton", "slot_count": len(slots)}
    if not slots:
        return {"supported": True, "status": "zero_slot_concrete_family", "slot_count": 0}
    if len(slots) > 2:
        return {
            "supported": False,
            "status": "unsupported_multi_slot_arity",
            "slot_count": len(slots),
            "max_supported_slots": 2,
        }
    return {"supported": True, "status": "supported_slot_arity", "slot_count": len(slots)}


def _infer_slot_role_in_skeleton(skeleton: Program, slot_name: str) -> str:
    """Inspect ``skeleton`` to classify what observable quantity ``slot_name`` compares against."""
    found_role = "generic_threshold"

    def _visit(nd: Program) -> None:
        nonlocal found_role
        if nd.op == "lt" and len(nd.children) == 2:
            left, right = nd.children
            other = None
            if left.op == "arg" and left.name == slot_name:
                other = right
            elif right.op == "arg" and right.name == slot_name:
                other = left
            if other is not None:
                if other.op == "arg" and other.name == "edge_source_quantity":
                    found_role = "donor_quantity_threshold"
                elif other.op == "arg" and other.name == "edge_target_quantity":
                    found_role = "target_quantity_threshold"
                elif other.op == "sub":
                    found_role = "gap_threshold"
        for ch in nd.children:
            _visit(ch)

    _visit(skeleton)
    return found_role


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
    """Instantiate an anti-unified parametric family ``H*(r_0, theta_*)`` on a new target apparatus (A3.3).

    Supports complete 1-slot and 2-slot calibration from target transitions & ``training`` rows,
    verifies zero unbound ``theta_*`` slots remain before evaluation, and explicitly rejects
    ``len(slots) > 2`` with ``status = 'unsupported_multi_slot_arity'`` (never silently binding only ``slots[0]``!).
    """
    from .models import Candidate

    support_check = validate_parametric_family_calibration_support(generalized_comm)
    generalized_comm["_last_instantiation_diagnostic"] = deepcopy(support_check)
    if not support_check["supported"] or support_check["slot_count"] == 0:
        return []

    sc = _as_schema(schema)
    if sc is None or not training:
        return []
    skel_dict = generalized_comm.get("anti_unified_predicate_skeleton")
    slots = list(generalized_comm.get("parameter_slots") or ())
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
    dedup_radii: list[float] = []
    for rv in cand_radii:
        if all(abs(rv - x) > 1e-4 for x in dedup_radii):
            dedup_radii.append(rv)

    # Extract empirical donor & gap transition bounds + trajectory frame breakpoints
    emp_gap_cands: list[float] = []
    emp_donor_cands: list[float] = []
    gap_cell_by_mid: dict[float, list[float]] = {}

    f_names_sc = tuple(sc.feature_names)
    frames_sc = {
        tuple(fr[nm] for nm in f_names_sc): fr
        for r_obs in training
        for fr in (r_obs.features, *r_obs.history[-4:])
        if all(nm in fr for nm in f_names_sc)
    }
    r_scan_f = max(dedup_radii) + 0.05
    sign_f = 1.0 if dir_mode == "right" else -1.0
    frame_gaps: list[float] = []
    frame_donors: list[float] = []
    for s_obj in sc.objects:
        xs_n, qs_n = sc.fields[s_obj]["x"], sc.fields[s_obj]["q"]
        for t_obj in sc.objects:
            if s_obj == t_obj:
                continue
            xt_n, qt_n = sc.fields[t_obj]["x"], sc.fields[t_obj]["q"]
            for fr in frames_sc.values():
                qs_v = float(fr[qs_n])
                if abs(qs_v) <= 1e-9:
                    continue
                dx_v = sign_f * (float(fr[xt_n]) - float(fr[xs_n]))
                if 0.0 < dx_v <= r_scan_f:
                    frame_gaps.append(qs_v - float(fr[qt_n]))
                    frame_donors.append(qs_v)

    if audit_report and audit_report.get("verified_threshold_interval"):
        t_lo, t_hi = (
            float(audit_report["verified_threshold_interval"][0]),
            float(audit_report["verified_threshold_interval"][1]),
        )
        in_band_g = sorted({g for g in frame_gaps if t_lo - 1e-6 <= g <= t_hi + 1e-6})
        anchors_g = [t_lo, *in_band_g, t_hi]
        for a_g, b_g in zip(anchors_g, anchors_g[1:]):
            if b_g - a_g > 1e-9:
                mid_g = 0.5 * (a_g + b_g)
                emp_gap_cands.append(mid_g)
                gap_cell_by_mid[round(mid_g, 9)] = [float(a_g), float(b_g)]
        emp_gap_cands.append(0.5 * (t_lo + t_hi))

    if transitions:
        objs = list(sc.objects)
        r_ref = dedup_radii[0]
        active_gaps: list[float] = []
        inactive_pos_gaps: list[float] = []
        active_donors: list[float] = []
        inactive_donors: list[float] = []
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
                if abs(qs[i]) <= 1e-4:
                    continue
                for j in range(len(objs)):
                    if i == j:
                        continue
                    dx = xs[j] - xs[i]
                    in_dir = (0.0 < dx < r_ref) if dir_mode == "right" else (0.0 < -dx < r_ref)
                    if not in_dir:
                        continue
                    gap = qs[i] - qs[j]
                    other_neighbors = sum(
                        1
                        for k in range(len(objs))
                        if k not in (i, j) and (abs(xs[k] - xs[i]) < r_ref or abs(xs[k] - xs[j]) < r_ref)
                    )
                    if other_neighbors == 0:
                        if abs(deltas[i]) > 1e-4 and deltas[i] * deltas[j] < 0.0:
                            active_gaps.append(gap)
                            active_donors.append(qs[i])
                        elif abs(deltas[i]) <= 1e-5:
                            if gap > 0.0:
                                inactive_pos_gaps.append(gap)
                            inactive_donors.append(qs[i])
        if active_gaps and inactive_pos_gaps:
            lo_b = max(inactive_pos_gaps)
            hi_b = min(active_gaps)
            if lo_b <= hi_b + 0.05:
                emp_gap_cands.append(0.5 * (lo_b + hi_b))
                in_b2 = sorted({g for g in frame_gaps if lo_b - 1e-6 <= g <= hi_b + 1e-6})
                anc2 = [min(lo_b, hi_b), *in_b2, max(lo_b, hi_b)]
                for a2, b2 in zip(anc2, anc2[1:]):
                    if b2 - a2 > 1e-9:
                        emp_gap_cands.append(0.5 * (a2 + b2))
        elif active_gaps:
            emp_gap_cands.append(max(0.0, min(active_gaps) - 0.04))
        if active_donors:
            below_d = [d for d in inactive_donors if d < min(active_donors) + 0.02]
            if below_d:
                lo_d, hi_d = max(below_d), min(active_donors)
                emp_donor_cands.append(0.5 * (lo_d + hi_d))
                in_bd = sorted({d for d in frame_donors if min(lo_d, hi_d) - 1e-6 <= d <= max(lo_d, hi_d) + 1e-6})
                ancd = [min(lo_d, hi_d), *in_bd, max(lo_d, hi_d)]
                for ad, bd in zip(ancd, ancd[1:]):
                    if bd - ad > 1e-9:
                        emp_donor_cands.append(0.5 * (ad + bd))
            emp_donor_cands.append(max(-0.5, min(active_donors) - 0.04))

    if frame_gaps and not emp_gap_cands:
        pos_fg = sorted({g for g in frame_gaps if -0.05 <= g <= 0.85})
        if len(pos_fg) >= 2:
            step_fg = max(1, (len(pos_fg) - 1) // 14)
            for idx_fg in range(0, len(pos_fg) - 1, step_fg):
                emp_gap_cands.append(0.5 * (pos_fg[idx_fg] + pos_fg[min(len(pos_fg) - 1, idx_fg + step_fg)]))
    if frame_donors and not emp_donor_cands:
        pos_fd = sorted({d for d in frame_donors if -0.25 <= d <= 0.65})
        if len(pos_fd) >= 2:
            step_fd = max(1, (len(pos_fd) - 1) // 10)
            for idx_fd in range(0, len(pos_fd) - 1, step_fd):
                emp_donor_cands.append(0.5 * (pos_fd[idx_fd] + pos_fd[min(len(pos_fd) - 1, idx_fd + step_fd)]))

    src_slot_vals = generalized_comm.get("source_slot_values") or {}

    def _slot_grid(slot_name: str, max_items: int, dedup_tol: float = 1e-5) -> list[float]:
        role = _infer_slot_role_in_skeleton(skeleton, slot_name)
        raw: list[float] = []
        if role == "donor_quantity_threshold":
            raw.extend(emp_donor_cands)
            raw.extend(float(v) for v in (src_slot_vals.get(slot_name) or ()))
            raw.extend([0.0, 0.08, 0.15, 0.22, -0.10, 0.30])
        else:
            raw.extend(emp_gap_cands)
            raw.extend(float(v) for v in (src_slot_vals.get(slot_name) or ()))
            raw.extend([0.0, 0.15, 0.22, 0.28, 0.34, 0.40, 0.46, 0.55])
        dedup: list[float] = []
        for v in raw:
            if all(abs(v - x) > dedup_tol for x in dedup):
                dedup.append(float(v))
        return dedup[:max_items]

    def _eval_slot_map(r_try: float, slot_map: dict[str, float]) -> dict[str, Any] | None:
        bound_pred = _bind_arg_slots(skeleton, slot_map)
        if _has_unbound_theta_slot(bound_pred):
            generalized_comm["_last_instantiation_diagnostic"] = {
                "supported": False,
                "status": "unbound_parameter_slot_detected",
                "slot_map": dict(slot_map),
            }
            return None
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
            return None
        cost = len(training) * (len(terms) + 2)
        if not budget.spend(cost):
            return {"budget_exhausted": True}
        from .world_structure_commitments import _build_commitment_entry

        cols = [[float(t.predict(r)) for r in training] for t in terms]
        coeffs = _linear_fit(cols, [float(r.target) for r in training])
        prog = _sum_terms(list(terms), coeffs[1:], coeffs[0]).simplified()
        tr_loss = float(loss(prog, training, 0.0, spec.risk_quantile))
        slot_tag = "_".join(f"{k}_{slot_map[k]:.3f}" for k in slots)
        slot_desc = ", ".join(f"{k}={slot_map[k]:.4f}" for k in slots)
        r_iv = (
            list(audit_report["radius_interval"])
            if audit_report and audit_report.get("radius_interval")
            else [max(0.05, float(r_try) - 0.02), float(r_try) + 0.02]
        )
        t_iv = (
            gap_cell_by_mid.get(round(float(slot_map[slots[0]]), 9))
            or (
                list(audit_report["verified_threshold_interval"])
                if audit_report and audit_report.get("verified_threshold_interval")
                else None
            )
            if len(slots) == 1
            else None
        )
        inst_comm_local = deepcopy(inst_comm)
        inst_comm_local["source_concept_ids"] = []
        built = _build_commitment_entry(
            spec=spec,
            rows=training,
            schema=sc,
            commitment=inst_comm_local,
            terms=tuple(terms),
            fit={"program": prog, "coefficients": coeffs, "loss": tr_loss, "fit_terms": tuple(terms)},
            fitted_radius=float(r_try),
            radius_interval=r_iv,
            threshold_interval=t_iv,
            active_threshold_probing=True,
            complete=True,
            compute_cost=cost,
            reorganization_type="anti_unified_parametric_family_instantiation",
            scientific_revision=None,
            direction_overrides={r_sym: dir_mode},
            predicate_overrides={r_sym: bound_pred.to_dict()},
            role_terms_override=inst_comm.get("role_terms"),
            nonconservation_mechanism=inst_comm.get("nonconservation_mechanism"),
            latent_state_mechanism=inst_comm.get("latent_state_mechanism"),
            tolerance=tolerance,
        )
        cand_id = f"parametric_family:{generalized_comm['commitment_id']}:{slot_tag}:r_{r_try:.3f}"
        built["candidate"] = Candidate(
            cand_id,
            prog,
            "instantiate_anti_unified_parametric_family",
            rationale=f"Calibrated parametric family {generalized_comm['commitment_id']} at {slot_desc}, {r_sym}={r_try:.4f}",
            parents=tuple(
                generalized_comm.get("source_concept_ids")
                or generalized_comm.get("source_branch_ids")
                or ()
            ),
        )
        built["structure"]["fits"][0]["candidate_id"] = cand_id
        built["kind"] = "anti_unified_parametric_family"
        built["calibrated_slots"] = {**slot_map, r_sym: float(r_try)}
        built["commitment_id"] = generalized_comm["commitment_id"]
        return built

    if len(slots) == 1:
        s0 = slots[0]
        slot_maps = [{s0: v0} for v0 in _slot_grid(s0, 24, dedup_tol=1e-5)]
    else:
        s0, s1 = slots[0], slots[1]
        g0 = _slot_grid(s0, 9, dedup_tol=0.005)
        g1 = _slot_grid(s1, 9, dedup_tol=0.005)
        slot_maps = [{s0: v0, s1: v1} for v0 in g0 for v1 in g1]

    best_entry: dict[str, Any] | None = None
    exhausted = False
    for r_try in dedup_radii[:3]:
        if exhausted:
            break
        for sm in slot_maps:
            res = _eval_slot_map(r_try, sm)
            if res is None:
                continue
            if res.get("budget_exhausted"):
                exhausted = True
                break
            if best_entry is None or res["training_loss"] < best_entry["training_loss"]:
                best_entry = res
            if res["training_loss"] <= tolerance:
                generalized_comm["_last_instantiation_diagnostic"] = {
                    **support_check,
                    "calibrated_exact": True,
                    "calibrated_slots": dict(res["calibrated_slots"]),
                }
                return [res]

    # Local coordinate polish around best point if close but not yet within tolerance
    if best_entry is not None and best_entry["training_loss"] > tolerance and not exhausted:
        r_best = float(best_entry["calibrated_slots"][r_sym])
        base_sm = {k: float(best_entry["calibrated_slots"][k]) for k in slots}
        for delta in (-0.04, 0.04, -0.02, 0.02):
            for k in slots:
                pol_sm = {**base_sm, k: base_sm[k] + delta}
                res = _eval_slot_map(r_best, pol_sm)
                if res is None or res.get("budget_exhausted"):
                    exhausted = True
                    break
                if res["training_loss"] < best_entry["training_loss"]:
                    best_entry = res
                    base_sm = {sk: float(best_entry["calibrated_slots"][sk]) for sk in slots}
                if res["training_loss"] <= tolerance:
                    generalized_comm["_last_instantiation_diagnostic"] = {
                        **support_check,
                        "calibrated_exact": True,
                        "calibrated_slots": dict(res["calibrated_slots"]),
                    }
                    return [res]
            if exhausted:
                break

    if best_entry is not None:
        generalized_comm["_last_instantiation_diagnostic"] = {
            **support_check,
            "calibrated_exact": bool(best_entry["training_loss"] <= tolerance),
            "calibrated_slots": dict(best_entry["calibrated_slots"]),
        }
        return [best_entry]
    return []


def _predicate_operator_skeleton_key(pred: Program) -> tuple[str, bool]:
    """Return ``(operator_skeleton_json, has_constant_or_slot)`` erasing only numeric constants/theta slots."""
    has_param = False

    def _erase(nd: Program) -> dict[str, Any]:
        nonlocal has_param
        if nd.op == "const" or _is_theta_slot_arg(nd):
            has_param = True
            return {"op": "__PARAM_CONST__"}
        return {
            "op": nd.op,
            "name": nd.name,
            "lag": nd.lag,
            "children": [_erase(ch) for ch in nd.children],
        }

    erased = _erase(pred)
    return json.dumps(erased, sort_keys=True, separators=(",", ":")), has_param


def _serialize_obs_rows(rows: list[Observation], limit: int = 512) -> list[dict[str, Any]]:
    deduped = list({id(r): r for r in (rows or ())}.values())
    selected_rows = deduped if len(deduped) <= limit else [*deduped[: limit // 2], *deduped[-(limit - limit // 2) :]]
    return [
        {
            "features": {k: float(v) for k, v in r.features.items()},
            "target": float(r.target),
            "history": [{hk: float(hv) for hk, hv in h.items()} for h in r.history],
        }
        for r in selected_rows
    ]


def _deserialize_obs_rows(raw_list: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[Observation]:
    out: list[Observation] = []
    for item in raw_list or ():
        if isinstance(item, Observation):
            out.append(item)
        elif isinstance(item, dict) and "features" in item:
            out.append(
                Observation(
                    {k: float(v) for k, v in item["features"].items()},
                    float(item.get("target", 0.0)),
                    tuple(
                        {hk: float(hv) for hk, hv in h.items()}
                        for h in (item.get("history") or ())
                        if isinstance(h, dict)
                    ),
                )
            )
    return out


def _branch_regime_signature(comm: dict[str, Any]) -> tuple[str, str, str, str]:
    rp = (comm.get("radius_parameters") or [{}])[0]
    direction = str(rp.get("direction") or "symmetric")
    noncons = str(comm.get("nonconservation_mechanism") or "conserved")
    latent = str(comm.get("latent_state_mechanism") or "markov")
    raw_ec = rp.get("edge_condition")
    if raw_ec in (None, "unconditional"):
        pred_key = "unconditional"
    else:
        pred_ast = _canonical_edge_predicate_ast(raw_ec)
        pred_key, _ = _predicate_operator_skeleton_key(pred_ast)
    return (direction, noncons, latent, pred_key)


def _cross_evaluate_fixed_instance_on_neighbor(
    branch_claim: dict[str, Any],
    neighbor_claim: dict[str, Any],
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Evaluate ``branch_claim`` (as a fixed parameter instance) on ``neighbor_claim``'s paid witness rows (v13 B1 / Check #11).

    Explicitly declares ``evaluation_mode = 'fixed_instance_alignment'`` so different structural signatures
    are NEVER assumed to be mutual scientific counterexamples unless prediction actually fails on paid data!
    """
    rows_j_full = _deserialize_obs_rows(neighbor_claim.get("support_rows_serialized") or ())
    rows_j = [*rows_j_full[:8], *rows_j_full[-16:]] if len(rows_j_full) > 24 else rows_j_full
    sc_j = neighbor_claim.get("schema")
    if not rows_j or not sc_j:
        return {
            "evaluated": False,
            "evaluation_mode": "fixed_instance_alignment",
            "empirically_refuted": False,
            "reason": "missing_neighbor_witness_rows_or_schema",
        }
    al = instantiate_aligned_claim_program(branch_claim, sc_j)
    if not al.get("aligned"):
        return {
            "evaluated": False,
            "evaluation_mode": "fixed_instance_alignment",
            "empirically_refuted": False,
            "reason": al.get("reason", "unaligned"),
        }
    terms_j = al["terms"]
    if budget is not None:
        budget.spend(len(rows_j) * (len(terms_j) + 1))
    cols = [[float(t.predict(r)) for r in rows_j] for t in terms_j]
    coeffs = _linear_fit(cols, [float(r.target) for r in rows_j])
    prog_j = _sum_terms(list(terms_j), coeffs[1:], coeffs[0])
    cross_mse = float(loss(prog_j, rows_j, 0.0, 0.5))
    return {
        "evaluated": True,
        "evaluation_mode": "fixed_instance_alignment",
        "tested_program_version": _program_semantic_version(prog_j),
        "cross_mse": cross_mse,
        "empirically_refuted": bool(cross_mse > tolerance),
    }


def _cross_evaluate_refittable_family_on_neighbor(
    family_comm: dict[str, Any],
    neighbor_claim: dict[str, Any],
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Evaluate re-fittable parametric family ``family_comm`` (with allowed slot recalibration) on ``neighbor_claim`` (v13 B1).

    Never refutes an entire parametric family merely because one fixed parameter instance failed!
    """
    rows_j_full = _deserialize_obs_rows(neighbor_claim.get("support_rows_serialized") or ())
    rows_j = [*rows_j_full[:4], *rows_j_full[-8:]] if len(rows_j_full) > 12 else rows_j_full
    sc_j_raw = neighbor_claim.get("schema")
    sc_j = _as_schema(sc_j_raw)
    if not rows_j or sc_j is None:
        return {
            "evaluated": False,
            "evaluation_mode": "refittable_parametric_family",
            "empirically_refuted": False,
        }
    dummy_spec = TaskSpec(
        task_id="family_cross_scope_check",
        description="Family cross-domain scope evaluation",
        feature_names=tuple(sc_j.feature_names),
        initial_probes=(),
        available_probes=(),
        feature_bounds={fn: (-2.0, 2.0) for fn in sc_j.feature_names},
    )
    nb_comm = neighbor_claim.get("commitment") or {}
    nb_rp = (nb_comm.get("radius_parameters") or [{}])[0]
    nb_r = neighbor_claim.get("radius_value") or (nb_rp.get("source_values") or [None])[-1]
    nb_audit = {"radius_interval": [float(nb_r), float(nb_r)]} if nb_r is not None else None
    probe_b = budget if budget is not None else SearchBudget(25_000)
    entries = instantiate_parametric_family_on_target(
        family_comm,
        dummy_spec,
        rows_j,
        sc_j,
        probe_b,
        audit_report=nb_audit,
        tolerance=tolerance,
    )
    if not entries:
        return {
            "evaluated": True,
            "evaluation_mode": "refittable_parametric_family",
            "cross_mse": float("inf"),
            "empirically_refuted": True,
        }
    best_mse = min(float(e["training_loss"]) for e in entries)
    return {
        "evaluated": True,
        "evaluation_mode": "refittable_parametric_family",
        "cross_mse": best_mse,
        "empirically_refuted": bool(best_mse > tolerance),
    }


def _refine_coexisting_branch_scopes_and_routing(
    active_branches: list[dict[str, Any]],
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> list[dict[str, Any]]:
    """Separate ``empirical_applicability_scope`` (actual cross-prediction failure) from ``routing_scope_candidate`` (v13 B1 / Check #11)."""
    if len(active_branches) < 2:
        return active_branches
    refined: list[dict[str, Any]] = []
    for b_i in active_branches:
        if b_i.get("status") == "refuted":
            refined.append(b_i)
            continue
        cur_b = deepcopy(b_i)
        sig_i = _branch_regime_signature(cur_b["commitment"])
        sup_regs_i = list(cur_b.get("pilot_regime_features") or ())
        empirical_ce_regs: list[dict[str, Any]] = []
        routing_neg_regs: list[dict[str, Any]] = []

        for b_j in active_branches:
            if b_j["claim_id"] == cur_b["claim_id"] or b_j.get("status") == "refuted":
                continue
            regs_j = list(b_j.get("pilot_regime_features") or ())
            if not regs_j:
                continue
            cross_check = _cross_evaluate_fixed_instance_on_neighbor(
                cur_b, b_j, budget=budget, tolerance=tolerance
            )
            if cross_check.get("evaluated") and cross_check.get("empirically_refuted"):
                empirical_ce_regs.extend(regs_j)
                routing_neg_regs.extend(regs_j)
            elif _branch_regime_signature(b_j["commitment"]) != sig_i:
                # Check #11: Different structural signature WITHOUT cross-prediction failure
                # becomes ONLY a `routing_scope_candidate` negative, NEVER a scientific counterexample!
                routing_neg_regs.extend(regs_j)

        if sup_regs_i and empirical_ce_regs and any(
            evaluate_scope_guard(cur_b.get("scope_guard"), ce_r) != "reject"
            for ce_r in empirical_ce_regs
        ):
            cur_b = shrink_claim_to_observable_scope(
                cur_b,
                sup_regs_i,
                empirical_ce_regs,
                scope_kind="empirical_applicability_scope",
            )
        if sup_regs_i and routing_neg_regs and any(
            evaluate_scope_guard(cur_b.get("routing_guard"), rt_r) != "reject"
            for rt_r in routing_neg_regs
        ):
            cur_b = shrink_claim_to_observable_scope(
                cur_b,
                sup_regs_i,
                routing_neg_regs,
                scope_kind="routing_scope_candidate",
            )
        refined.append(cur_b)
    return refined


RECON_ALGO_VERSION = "v13_ordinary_lazy_recon_1"


def reconstruct_parametric_families_from_ordinary_archive(
    library: Any,
    budget: SearchBudget,
    *,
    tolerance: float = 1e-7,
    reconstruct_routing: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """3-layer cached on-demand reconstruction from raw ``meta_priors`` + ``OrdinaryPersistentCommitmentCache`` (v13 B3, B4 / Checks #13, #14).

    Caches the exact 3 products needed by ordinary lazy study without redundant grouped consolidation:
      1. Per-prior commitment & witness extraction preserving individual numeric constants;
      2. Source-set anti-unified parametric families (target bindings excluded from cache key);
      3. Optional empirical scope / routing candidates computed by the identical algorithm from raw evidence.
    First computation debits ``budget`` honestly; unchanged subsequent calls reuse in O(1); adding or
    revising a prior invalidates ONLY the affected prior and its family/routing entry.
    """
    from .premise_commitment_index import (
        OrdinaryPersistentCommitmentCache,
        extract_archive_witness_maps,
    )
    from .world_structure_commitments import consolidate_world_structure_commitments

    start_spent = budget.spent
    do_routing = (
        bool(reconstruct_routing)
        if reconstruct_routing is not None
        else bool(getattr(library, "_allow_on_demand_routing_reconstruction", False))
    )
    priors = [
        p
        for p in (getattr(library, "meta_priors", None) or ())
        if p.get("status") not in {"revoked", "stale_dependency_invalidated"}
    ]
    if not priors:
        return [], {
            "reconstructed_count": 0,
            "reconstructed_branches_count": 0,
            "compute_spent": 0,
            "cache_hit": True,
            "layer1_reused": 0,
            "layer1_extracted": 0,
            "layer2_reused": 0,
            "layer2_built": 0,
            "layer3_routing_reused": False,
        }

    ord_cache = getattr(library, "_ordinary_commitment_cache", None) or getattr(library, "_ordinary_persistent_cache", None)
    if ord_cache is None:
        ord_cache = OrdinaryPersistentCommitmentCache()
    library._ordinary_commitment_cache = ord_cache
    library._ordinary_persistent_cache = ord_cache

    if budget is not None and not budget.spend(1):
        return [], {"reconstructed_count": 0, "compute_spent": 0, "cache_hit": False, "interrupted": True}

    active_fp, stale_cids = ord_cache.active_prior_fingerprints(priors)
    tol_key = round(float(tolerance), 14)

    layer1_reused = 0
    layer1_extracted = 0
    interrupted = False
    extracted_items: list[dict[str, Any]] = []

    for i, p in enumerate(priors):
        cid = str(p.get("source_concept_id") or f"anon_{i}")
        if cid in stale_cids or cid not in active_fp:
            continue
        p_schema = (
            p.get("object_schema")
            or (p.get("structure") or {}).get("schema")
            or (p.get("research_program") or {}).get("source_schema")
        )
        if not p_schema:
            continue
        l1_key = (cid, active_fp[cid], tol_key, RECON_ALGO_VERSION)
        if l1_key in ord_cache._per_prior_extraction_memo:
            extracted_items.append(deepcopy(ord_cache._per_prior_extraction_memo[l1_key]))
            layer1_reused += 1
            continue

        sup_map_1, ce_map_1 = extract_archive_witness_maps([p])
        if cid not in sup_map_1:
            continue
        if budget.remaining < 10:
            interrupted = True
            break
        denied0 = budget.denied_requests
        c_list = consolidate_world_structure_commitments([p], budget=budget, tolerance=tolerance)
        if budget.denied_requests > denied0:
            interrupted = True
            break
        if not c_list:
            continue
        comm_i = deepcopy(c_list[0])
        comm_i["commitment_id"] = f"archive_comm:{cid}"
        rp = (comm_i.get("radius_parameters") or [{}])[0]
        raw_ec = rp.get("edge_condition")
        if raw_ec in (None, "unconditional"):
            has_param = False
            op_skel = "unconditional"
        else:
            pred_ast = _canonical_edge_predicate_ast(raw_ec)
            op_skel, has_param = _predicate_operator_skeleton_key(pred_ast)
        sig = (
            str(rp.get("direction") or "symmetric"),
            str(comm_i.get("nonconservation_mechanism") or "conserved"),
            str(comm_i.get("latent_state_mechanism") or "markov"),
            op_skel,
        )
        rec_entry = {
            "source_concept_id": cid,
            "fingerprint": active_fp[cid],
            "claim_id": f"archive_src:{cid}",
            "commitment": deepcopy(comm_i),
            "schema": deepcopy(p_schema),
            "sig": sig,
            "has_param": bool(has_param),
            "pilot_regime_features": deepcopy(list(p.get("pilot_regime_features") or ())),
            "support_rows_serialized": _serialize_obs_rows(sup_map_1[cid], limit=512),
            "counterexamples_serialized": _serialize_obs_rows(ce_map_1.get(cid, []), limit=128),
            "fresh_rows_serialized": deepcopy(list(p.get("fresh_confirmation_witness_rows") or ())),
        }
        ord_cache._per_prior_extraction_memo[l1_key] = deepcopy(rec_entry)
        extracted_items.append(rec_entry)
        layer1_extracted += 1

    # Layer 2: Source-group anti-unification memoization (target bindings never enter key!)
    by_sig: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in extracted_items:
        if item.get("has_param"):
            by_sig.setdefault(item["sig"], []).append(item)

    layer2_reused = 0
    layer2_built = 0
    families: list[dict[str, Any]] = []
    for sig, grp in by_sig.items():
        if len(grp) < 2:
            continue
        grp_fp_key = tuple(sorted(f"{it['source_concept_id']}:{it['fingerprint']}" for it in grp))
        l2_key = (sig, grp_fp_key, tol_key, RECON_ALGO_VERSION)
        if l2_key in ord_cache._family_au_memo:
            cached_au = ord_cache._family_au_memo[l2_key]
            layer2_reused += 1
            if cached_au.get("accepted") and cached_au.get("family_commitment") is not None:
                families.append(deepcopy(cached_au["family_commitment"]))
            continue

        if interrupted or budget.remaining < 20:
            interrupted = True
            break
        denied0 = budget.denied_requests
        au_inputs = [
            {
                "claim_id": it["claim_id"],
                "commitment": deepcopy(it["commitment"]),
                "schema": deepcopy(it["schema"]),
                "support_rows": _deserialize_obs_rows(it["support_rows_serialized"]),
                "scoped_counterexamples": _deserialize_obs_rows(it["counterexamples_serialized"]),
                "fresh_holdout_rows": _deserialize_obs_rows(it["fresh_rows_serialized"]),
            }
            for it in grp
        ]
        au_res = propose_anti_unified_generalization(
            au_inputs, budget=budget, tolerance=tolerance, require_fresh_confirmation=False
        )
        if budget.denied_requests > denied0:
            interrupted = True
            break
        layer2_built += 1
        if au_res.get("accepted"):
            gf = deepcopy(au_res["generalized_commitment"])
            gf["scope_guard"] = None
            gf["routing_guard"] = None
            gf["observable_bounds"] = {}
            gf["min_grounded_dependencies"] = 2
            gf["reconstructed_on_demand_from_ordinary_archive"] = True
            ord_cache._family_au_memo[l2_key] = {"accepted": True, "family_commitment": deepcopy(gf)}
            families.append(gf)
        else:
            ord_cache._family_au_memo[l2_key] = {"accepted": False, "family_commitment": None}

    # Layer 3: Optional on-demand routing & empirical scope reconstruction (same algorithm, cached cleanly!)
    reconstructed_branches: list[dict[str, Any]] = [
        {
            "claim_id": it["claim_id"],
            "source_concept_id": it["source_concept_id"],
            "source_concept_ids": [it["source_concept_id"]],
            "commitment": deepcopy(it["commitment"]),
            "schema": deepcopy(it["schema"]),
            "status": "universal",
            "independent_empirical_support": True,
            "pilot_regime_features": deepcopy(it["pilot_regime_features"]),
            "support_rows_serialized": deepcopy(it["support_rows_serialized"]),
            "fresh_holdout_rows_serialized": deepcopy(it["fresh_rows_serialized"]),
        }
        for it in extracted_items
    ]
    layer3_reused = False
    layer3_built = False
    if do_routing and extracted_items and not interrupted:
        all_fp_tuple = tuple(sorted(f"{it['source_concept_id']}:{it['fingerprint']}" for it in extracted_items))
        raw_ev_count = len(getattr(library, "_raw_reconciliation_evidence_archive", ()) or ())
        l3_key = (all_fp_tuple, raw_ev_count, tol_key, RECON_ALGO_VERSION)
        if l3_key in ord_cache._routing_scope_memo:
            cached_l3 = ord_cache._routing_scope_memo[l3_key]
            reconstructed_branches = deepcopy(cached_l3["branches"])
            fam_guards = cached_l3.get("family_guards_by_sig") or {}
            for gf in families:
                fsig = _branch_regime_signature(gf)
                if fsig in fam_guards:
                    gf.update(deepcopy(fam_guards[fsig]))
            layer3_reused = True
        elif budget.remaining >= 20:
            denied0 = budget.denied_requests
            raw_b_list = deepcopy(reconstructed_branches)
            reconstructed_branches = _refine_coexisting_branch_scopes_and_routing(
                raw_b_list, budget=budget, tolerance=tolerance
            )
            fam_guards_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
            for gf in families:
                fsig = _branch_regime_signature(gf)
                in_bs = [b for b in reconstructed_branches if _branch_regime_signature(b["commitment"]) == fsig]
                out_bs = [b for b in reconstructed_branches if _branch_regime_signature(b["commitment"]) != fsig]
                fam_sup = [reg for b in in_bs for reg in (b.get("pilot_regime_features") or ())]
                emp_out_regs: list[dict[str, Any]] = []
                rot_out_regs: list[dict[str, Any]] = []
                for ob in out_bs:
                    regs_o = list(ob.get("pilot_regime_features") or ())
                    if not regs_o:
                        continue
                    f_chk = _cross_evaluate_refittable_family_on_neighbor(gf, ob, budget=budget, tolerance=tolerance)
                    if f_chk.get("evaluated") and f_chk.get("empirically_refuted"):
                        emp_out_regs.extend(regs_o)
                        rot_out_regs.extend(regs_o)
                    else:
                        rot_out_regs.extend(regs_o)
                fg_patch: dict[str, Any] = {}
                if fam_sup and emp_out_regs:
                    sc_gf = shrink_claim_to_observable_scope(
                        gf, fam_sup, emp_out_regs, scope_kind="empirical_applicability_scope"
                    )
                    if sc_gf.get("shrunk") and sc_gf.get("scope_guard") is not None:
                        fg_patch["scope_guard"] = deepcopy(sc_gf["scope_guard"])
                        fg_patch["observable_bounds"] = deepcopy(sc_gf["observable_bounds"])
                        fg_patch["scope_verification"] = deepcopy(sc_gf["scope_verification"])
                if fam_sup and rot_out_regs:
                    rt_gf = shrink_claim_to_observable_scope(
                        gf, fam_sup, rot_out_regs, scope_kind="routing_scope_candidate"
                    )
                    if rt_gf.get("routing_shrunk") and rt_gf.get("routing_guard") is not None:
                        fg_patch["routing_guard"] = deepcopy(rt_gf["routing_guard"])
                        fg_patch["routing_bounds"] = deepcopy(rt_gf["routing_bounds"])
                        fg_patch["routing_verification"] = deepcopy(rt_gf["routing_verification"])
                if fg_patch:
                    gf.update(deepcopy(fg_patch))
                    fam_guards_map[fsig] = fg_patch
            if budget.denied_requests == denied0:
                ord_cache._routing_scope_memo[l3_key] = {
                    "branches": deepcopy(reconstructed_branches),
                    "family_guards_by_sig": deepcopy(fam_guards_map),
                }
                layer3_built = True

    all_hit = bool(
        not interrupted
        and layer1_extracted == 0
        and layer2_built == 0
        and (not do_routing or layer3_reused or not extracted_items)
    )
    return families, {
        "reconstructed_count": len(families),
        "reconstructed_branches": reconstructed_branches,
        "reconstructed_branches_count": len(reconstructed_branches),
        "compute_spent": budget.spent - start_spent,
        "cache_hit": all_hit,
        "interrupted": interrupted,
        "layer1_reused": layer1_reused,
        "layer1_extracted": layer1_extracted,
        "layer2_reused": layer2_reused,
        "layer2_built": layer2_built,
        "layer3_routing_reused": layer3_reused,
        "layer3_routing_built": layer3_built,
    }


def dispatch_branches_and_families_for_target(
    *,
    library: Any,
    consistent_comms: list[dict[str, Any]],
    transitions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    schema: Any,
    audit: dict[str, Any] | None,
    task_sig: str,
    comm_budget: SearchBudget,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Unified branch, routing & family dispatcher with explicit ``branch -> commitment -> source/version`` mapping (v13 A4, B1, C2).

    Enforces:
      1. Explicit identity mapping across ``branch:concept_*``, ``world_commitment:*``, and ``source_concept_ids``
         (Fixes N5 / Check #9);
      2. Multi-source preservation: a ``world_commitment:*`` or family backed by multiple independent sources is
         NEVER deleted when only a subset of its sources is invalidated or empirically excluded (Check #10);
      3. Four distinct telemetry buckets: ``not_scheduled_by_router`` (ordering only, never deletes!),
         ``scope_not_established`` (``unknown`` pilot/stat, kept in unverified pool!), ``empirically_excluded_on_context``,
         and ``support_invalidated``;
      4. Single-factor flags ``_enable_scope_and_routing_guards`` and ``_prioritize_generalized_families``.
    """
    from .world_structure_commitments import is_commitment_empirically_consistent

    obs_regime = compute_observable_regime_features(transitions, schema)
    use_guards = bool(getattr(library, "_enable_scope_and_routing_guards", True))
    prioritize_hstar = bool(getattr(library, "_prioritize_generalized_families", True))
    retracted_set = set(getattr(library, "_retracted_premise_or_claim_ids", ()) or ())

    raw_branches = list(getattr(library, "_active_reconciled_branches", ()) or ())
    gen_families = list(getattr(library, "_generalized_branch_commitments", ()) or ())
    on_demand_recon_stats = None

    if (not gen_families or (not raw_branches and getattr(library, "_allow_on_demand_routing_reconstruction", False))) and getattr(
        library, "_allow_on_demand_family_reconstruction", False
    ):
        recon_fams, on_demand_recon_stats = reconstruct_parametric_families_from_ordinary_archive(
            library,
            comm_budget,
            tolerance=tolerance,
            reconstruct_routing=bool(getattr(library, "_allow_on_demand_routing_reconstruction", False)),
        )
        if not gen_families:
            gen_families = recon_fams
        if not raw_branches and on_demand_recon_stats.get("reconstructed_branches"):
            raw_branches = list(on_demand_recon_stats["reconstructed_branches"])

    checked_branches = (
        propagate_premise_retraction_invalidation(
            raw_branches, retracted_premise_or_claim_ids=retracted_set
        )
        if raw_branches
        else []
    )
    branch_by_id = {str(b["claim_id"]): b for b in checked_branches}

    branch_identity_map: list[dict[str, Any]] = []
    support_invalidated_ids: list[str] = []
    empirically_excluded_ids: list[str] = []
    not_scheduled_by_router_ids: list[str] = []
    scope_not_established_ids: list[str] = []

    invalidated_or_excluded_sources: set[str] = set()
    invalidated_or_excluded_comm_ids: set[str] = set()
    deferred_router_sources: set[str] = set()

    matched_branch_comms: list[dict[str, Any]] = []
    unknown_scope_comms: list[dict[str, Any]] = []
    deferred_router_comms: list[dict[str, Any]] = []

    for b in checked_branches:
        bid = str(b["claim_id"])
        b_comm = deepcopy(b.get("commitment") or b)
        b_cid = str(b_comm.get("commitment_id") or bid)
        b_srcs = list(
            dict.fromkeys(
                [
                    *(b.get("source_concept_ids") or ()),
                    *(b_comm.get("source_concept_ids") or ()),
                    *([b["source_concept_id"]] if b.get("source_concept_id") else []),
                ]
            )
        )
        sem_ver = (
            b.get("semantic_version")
            or b_comm.get("semantic_group_fingerprint")
            or b_cid
        )
        branch_identity_map.append(
            {
                "branch_id": bid,
                "commitment_id": b_cid,
                "source_concept_ids": b_srcs,
                "semantic_version": sem_ver,
            }
        )

        if b.get("support_validity") != "valid_grounded" or b.get("status") == "refuted":
            support_invalidated_ids.append(bid)
            invalidated_or_excluded_comm_ids.add(b_cid)
            invalidated_or_excluded_sources.update(b_srcs)
            continue

        if use_guards:
            app_res = evaluate_scope_guard(b.get("scope_guard"), obs_regime)
            if app_res == "reject":
                empirically_excluded_ids.append(bid)
                invalidated_or_excluded_comm_ids.add(b_cid)
                invalidated_or_excluded_sources.update(b_srcs)
                continue
            rot_guard = b.get("routing_guard")
            rot_res = evaluate_scope_guard(rot_guard, obs_regime) if rot_guard is not None else "accept"
            if rot_res == "reject":
                # B1 & A4: Empirical routing miss ONLY defers candidate ordering, NEVER deletes knowledge!
                not_scheduled_by_router_ids.append(bid)
                deferred_router_sources.update(b_srcs)
                if is_commitment_empirically_consistent(b_comm, audit, task_id=task_sig):
                    deferred_router_comms.append(b_comm)
                continue
            if app_res == "unknown" or rot_res == "unknown":
                scope_not_established_ids.append(bid)
                if is_commitment_empirically_consistent(b_comm, audit, task_id=task_sig):
                    unknown_scope_comms.append(b_comm)
                continue

        if is_commitment_empirically_consistent(b_comm, audit, task_id=task_sig):
            matched_branch_comms.append(b_comm)

    # Map branch exclusions/invalidations onto `consistent_comms` (`world_commitment:*`) while preserving
    # any multi-source commitment that retains at least one valid non-excluded independent source (Checks #9, #10)!
    filtered_pool: list[dict[str, Any]] = []
    deferred_pool_tail: list[dict[str, Any]] = []
    seen_comm_keys: set[tuple[str, tuple[str, ...]]] = set()

    for m_comm in [*matched_branch_comms, *unknown_scope_comms]:
        mc_key = (str(m_comm.get("commitment_id")), tuple(m_comm.get("source_concept_ids") or ()))
        if mc_key not in seen_comm_keys:
            seen_comm_keys.add(mc_key)
            filtered_pool.append(m_comm)

    for c in consistent_comms:
        cid = str(c.get("commitment_id"))
        c_srcs = list(c.get("source_concept_ids") or ())
        if c_srcs:
            valid_remaining_srcs = [
                s for s in c_srcs if s not in invalidated_or_excluded_sources and s not in retracted_set
            ]
            if not valid_remaining_srcs:
                continue
            c_copy = deepcopy(c)
            if len(valid_remaining_srcs) < len(c_srcs):
                c_copy["source_concept_ids"] = valid_remaining_srcs
                c_copy["pruned_excluded_source_ids"] = [
                    s for s in c_srcs if s not in valid_remaining_srcs
                ]
        else:
            if cid in invalidated_or_excluded_comm_ids or cid in retracted_set:
                continue
            c_copy = deepcopy(c)
            valid_remaining_srcs = []

        c_key = (cid, tuple(c_copy.get("source_concept_ids") or ()))
        if c_key in seen_comm_keys:
            continue
        seen_comm_keys.add(c_key)
        if valid_remaining_srcs and all(s in deferred_router_sources for s in valid_remaining_srcs):
            deferred_pool_tail.append(c_copy)
        else:
            filtered_pool.append(c_copy)

    for d_comm in [*deferred_pool_tail, *deferred_router_comms]:
        dc_key = (str(d_comm.get("commitment_id")), tuple(d_comm.get("source_concept_ids") or ()))
        if dc_key not in seen_comm_keys:
            seen_comm_keys.add(dc_key)
            filtered_pool.append(d_comm)
        elif d_comm in deferred_router_comms and d_comm not in filtered_pool:
            filtered_pool.append(d_comm)

    # Dispatch generalized parametric families (respecting `_prioritize_generalized_families` & multi-branch grounding!)
    scheduled_families: list[dict[str, Any]] = []
    deferred_families: list[dict[str, Any]] = []
    matched_family_ids: list[str] = []

    if prioritize_hstar:
        for gf in gen_families:
            gf_id = str(gf.get("commitment_id"))
            if gf_id in retracted_set:
                support_invalidated_ids.append(gf_id)
                continue
            dep_bids = list(gf.get("depends_on_claim_ids") or gf.get("source_branch_ids") or ())
            if dep_bids and branch_by_id:
                grounded_dep_bids = [
                    db
                    for db in dep_bids
                    if db in branch_by_id
                    and branch_by_id[db].get("support_validity") == "valid_grounded"
                    and branch_by_id[db].get("status") != "refuted"
                ]
                min_req_f = int(gf.get("min_grounded_dependencies") or min(2, len(dep_bids)))
                if len(grounded_dep_bids) < min_req_f:
                    support_invalidated_ids.append(gf_id)
                    continue

            gf_dir = (gf.get("radius_parameters") or [{}])[0].get("direction")
            if audit and audit.get("verified_direction") and gf_dir and audit["verified_direction"] != gf_dir:
                empirically_excluded_ids.append(gf_id)
                continue

            if use_guards:
                f_app = evaluate_scope_guard(gf.get("scope_guard"), obs_regime)
                if f_app == "reject":
                    empirically_excluded_ids.append(gf_id)
                    continue
                f_rot_guard = gf.get("routing_guard")
                f_rot = evaluate_scope_guard(f_rot_guard, obs_regime) if f_rot_guard is not None else "accept"
                if f_rot == "reject":
                    not_scheduled_by_router_ids.append(gf_id)
                    deferred_families.append(gf)
                    continue
                if f_app == "unknown" or f_rot == "unknown":
                    scope_not_established_ids.append(gf_id)

            matched_family_ids.append(gf_id)
            scheduled_families.append(gf)

    telemetry = {
        "observable_regime_features": obs_regime,
        "scope_and_routing_guards_enabled": use_guards,
        "prioritize_generalized_families": prioritize_hstar,
        "branch_identity_map": branch_identity_map,
        "matched_parametric_family_ids": matched_family_ids,
        "support_invalidated": support_invalidated_ids,
        "empirically_excluded_on_context": empirically_excluded_ids,
        "not_scheduled_by_router": not_scheduled_by_router_ids,
        "scope_not_established": scope_not_established_ids,
        # Retain v12 alias keys for compatibility
        "pruned_by_observable_scope_ids": empirically_excluded_ids,
        "pruned_by_retraction_ids": support_invalidated_ids,
        "on_demand_archive_family_reconstruction": (
            {k: v for k, v in on_demand_recon_stats.items() if k != "reconstructed_branches"}
            if isinstance(on_demand_recon_stats, dict)
            else None
        ),
    }
    return {
        "ordered_consistent_comms": filtered_pool,
        "scheduled_families": scheduled_families,
        "deferred_router_families": deferred_families,
        "telemetry": telemetry,
    }


def propose_threshold_interval_boundary_probe(
    commitment: dict[str, Any],
    threshold_interval: list[float] | tuple[float, float],
    spec: TaskSpec,
    target_schema: ObjectSchema | dict,
    *,
    fitted_radius: float = 0.92,
    controls_by_object: dict[str, dict[str, str]] | None = None,
    budget: SearchBudget | None = None,
    min_actionable_width: float = 0.015,
) -> dict[str, Any]:
    """Synthesize a legal boundary-discriminating ``Probe`` from a residual compatible threshold interval (v13 C3).

    Uses ONLY the agent's own public ``threshold_interval = [lo, hi]`` (NEVER reading hidden true threshold!),
    freezes the two interior boundary models BEFORE acquisition, and tags the probe strictly as
    ``role = 'boundary_refinement_training'`` so a separate unconsumed confirmation batch is required afterwards.
    """
    lo, hi = float(threshold_interval[0]), float(threshold_interval[1])
    width = hi - lo
    if width <= min_actionable_width:
        return {"actionable": False, "reason": "interval_already_tight", "interval_width": width}

    rp0 = (commitment.get("radius_parameters") or [{}])[0]
    r_sym = str(rp0.get("symbol") or "r_0")
    raw_ec = rp0.get("edge_condition")
    if raw_ec in (None, "unconditional"):
        return {"actionable": False, "reason": "unconditional_commitment"}

    pred_ast = _canonical_edge_predicate_ast(raw_ec)
    # Locate rightmost tunable threshold constant leaf (e.g. `0 < q_src AND th < q_src - q_dst`)
    const_paths: list[tuple[int, ...]] = []

    def _collect_consts(nd: Program, path: tuple[int, ...]) -> None:
        if nd.op == "const":
            const_paths.append(path)
        for idx, ch in enumerate(nd.children):
            _collect_consts(ch, (*path, idx))

    _collect_consts(pred_ast, ())
    if not const_paths:
        return {"actionable": False, "reason": "no_numeric_threshold_leaf"}
    target_path = const_paths[-1]

    def _replace_at(nd: Program, path: tuple[int, ...], val: float) -> Program:
        if not path:
            return Program("const", value=float(val))
        idx = path[0]
        new_ch = tuple(
            _replace_at(ch, path[1:], val) if i == idx else ch
            for i, ch in enumerate(nd.children)
        )
        return Program(nd.op, value=nd.value, name=nd.name, lag=nd.lag, children=new_ch)

    th_low_model = lo + 0.2 * width
    th_high_model = hi - 0.2 * width
    pred_lo = _replace_at(pred_ast, target_path, th_low_model)
    pred_hi = _replace_at(pred_ast, target_path, th_high_model)

    sc = _as_schema(target_schema)
    if sc is None:
        return {"actionable": False, "reason": "invalid_schema"}

    comm_lo = deepcopy(commitment)
    comm_hi = deepcopy(commitment)
    if comm_lo.get("radius_parameters"):
        comm_lo["radius_parameters"][0]["edge_condition"] = pred_lo.to_dict()
        comm_hi["radius_parameters"][0]["edge_condition"] = pred_hi.to_dict()
    for rt_d in comm_lo.get("role_terms") or ():
        def _upd_lo(nd: dict) -> None:
            if nd.get("kind") == "all_relation_sum":
                nd["predicate"] = pred_lo.to_dict()
            for ch_d in nd.get("children") or ():
                if isinstance(ch_d, dict):
                    _upd_lo(ch_d)
        _upd_lo(rt_d)
    for rt_d in comm_hi.get("role_terms") or ():
        def _upd_hi(nd: dict) -> None:
            if nd.get("kind") == "all_relation_sum":
                nd["predicate"] = pred_hi.to_dict()
            for ch_d in nd.get("children") or ():
                if isinstance(ch_d, dict):
                    _upd_hi(ch_d)
        _upd_hi(rt_d)

    al_lo = instantiate_aligned_claim_program(
        {"claim_id": f"boundary_lo_{th_low_model:.4f}", "commitment": comm_lo, "radius_value": fitted_radius},
        sc,
    )
    al_hi = instantiate_aligned_claim_program(
        {"claim_id": f"boundary_hi_{th_high_model:.4f}", "commitment": comm_hi, "radius_value": fitted_radius},
        sc,
    )
    if not al_lo.get("aligned") or not al_hi.get("aligned"):
        return {"actionable": False, "reason": "term_instantiation_failed"}

    frozen_lo_ver = _program_semantic_version(al_lo["program"])
    frozen_hi_ver = _program_semantic_version(al_hi["program"])

    diff = find_feasible_prediction_difference(
        al_lo,
        al_hi,
        spec,
        sc,
        controls_by_object=controls_by_object,
        budget=budget,
    )
    if not diff.get("witness_available"):
        return {
            "actionable": False,
            "reason": "no_feasible_boundary_witness_found",
            "interval": [lo, hi],
            "frozen_program_versions": [frozen_lo_ver, frozen_hi_ver],
        }
    return {
        "actionable": True,
        "role": "boundary_refinement_training_requires_fresh_confirmation",
        "interval": [lo, hi],
        "tested_boundary_models": {"th_low": th_low_model, "th_high": th_high_model},
        "frozen_program_versions": [frozen_lo_ver, frozen_hi_ver],
        "probe": diff["probe"],
        "max_abs_disagreement": diff["max_abs_disagreement"],
    }


def autonomous_reconcile_library_episode(
    *,
    library: Any,
    task_id: str,
    spec: TaskSpec,
    schema_dict: dict[str, Any] | None,
    confirmed: bool,
    confirmation_loss: float | None,
    confirmation_rows_list: list[Observation],
    support_rows: list[Observation],
    episode_transitions: list[dict[str, Any]],
    concept_id: str | None,
    execute_probe_fn: Any,
    budget: SearchBudget,
    delivered_program: Program | dict | None = None,
    unconfirmed_candidate_entry: dict[str, Any] | None = None,
    schema_alternatives: list[dict[str, Any]] | None = None,
    reconcile_view_rows: list[Observation] | None = None,
    tolerance: float = 1e-7,
    provisional_approximate_tolerance: float = 1.5e-2,
) -> dict[str, Any]:
    """Autonomous branch reconciliation, failure-driven subquestion generation, & family discovery inside ``UnifiedResearchAgent.run()`` (v13 PR-A/B)."""
    from .premise_commitment_index import compute_semantic_version
    from .world_structure_commitments import consolidate_world_structure_commitments

    start_spent = budget.spent
    for attr_name, default_val in (
        ("_reconciliation_episode_ledger", []),
        ("_provisional_candidate_archive", []),
        ("_active_reconciled_branches", []),
        ("_generalized_branch_commitments", []),
        ("_rejected_merge_records", []),
        ("_branch_semantic_relations", []),
        ("_family_evolution_history", []),
        ("_pending_reconciliation_questions", []),
        ("_raw_reconciliation_evidence_archive", []),
    ):
        if not hasattr(library, attr_name) or getattr(library, attr_name) is None:
            setattr(library, attr_name, deepcopy(default_val))
    if not hasattr(library, "_retracted_premise_or_claim_ids") or library._retracted_premise_or_claim_ids is None:
        library._retracted_premise_or_claim_ids = set()

    has_fresh = len(confirmation_rows_list) > 0
    exact_confirmed = bool(
        confirmed
        and has_fresh
        and confirmation_loss is not None
        and float(confirmation_loss) <= tolerance
        and concept_id is not None
        and schema_dict is not None
    )

    if not exact_confirmed:
        # v13 A3 / N2 / Check #6: When an episode is NOT confirmed AND active branches exist,
        # adjudicate the current candidate + compatible historical branches on the paid observations
        # so mutual failure opens concrete parent-bound `pending_questions` and restricts refuted scopes!
        pending_qs_fail: list[dict[str, Any]] = []
        all_paid_pool = list(
            {
                id(r): r
                for r in (*(reconcile_view_rows or ()), *support_rows, *confirmation_rows_list)
            }.values()
        )
        spec_fset = set(spec.feature_names)
        paid_eval_rows = [r for r in all_paid_pool if spec_fset <= set(r.features.keys())] or list(reconcile_view_rows or ())
        sensor_fit_rows = [r for r in support_rows if spec_fset <= set(r.features.keys())] or paid_eval_rows
        if (
            schema_dict is not None
            and paid_eval_rows
            and library._active_reconciled_branches
            and unconfirmed_candidate_entry is not None
        ):
            cand_obj = unconfirmed_candidate_entry.get("candidate")
            cand_prog = (
                cand_obj.program
                if hasattr(cand_obj, "program")
                else (
                    Program.from_dict(delivered_program)
                    if isinstance(delivered_program, dict)
                    else delivered_program
                )
            )
            if isinstance(cand_prog, Program):
                cand_req = cand_prog.required_features()
                cand_eval_rows = [r for r in all_paid_pool if cand_req <= set(r.features.keys())]
                if not cand_eval_rows:
                    cand_eval_rows = [r for r in (*support_rows, *confirmation_rows_list) if cand_req <= set(r.features.keys())]
                new_unconf_claim = {
                    "claim_id": f"unconfirmed_candidate:{task_id}",
                    "status": "candidate",
                    "independent_empirical_support": False,
                    "schema": deepcopy(schema_dict),
                }
                upd_branches_fail: list[dict[str, Any]] = []
                ev_ref_fail = {
                    "status": "paid_episode_observations",
                    "actual_probe_id": f"paid_episode:{task_id}",
                    "observation_count": len(paid_eval_rows),
                    "program_versions_frozen_before_result": [_program_semantic_version(cand_prog)],
                }
                for ob in library._active_reconciled_branches:
                    if ob.get("status") == "refuted" or not cand_eval_rows:
                        upd_branches_fail.append(ob)
                        continue
                    al_old = instantiate_aligned_claim_program(ob, schema_dict)
                    adj_schema = schema_dict
                    eff_alts = list(schema_alternatives or ())
                    if not al_old.get("aligned") and not eff_alts and isinstance(schema_dict, dict):
                        from itertools import permutations
                        objs_p = list(schema_dict.get("objects") or ())
                        flds_p = schema_dict.get("fields") or {}
                        used_s = {fn for gm in flds_p.values() for fn in gm.values()}
                        rem_s = [
                            fn
                            for fn in spec.feature_names
                            if fn not in used_s
                            and all(float(r.features.get(fn, 0.0)) in (0.0, 1.0) for r in paid_eval_rows[:24])
                        ]
                        if 1 <= len(objs_p) <= 4 and len(rem_s) == len(objs_p) and all("f" not in flds_p.get(g, {}) for g in objs_p):
                            for perm_f in permutations(rem_s):
                                cand_sc = deepcopy(schema_dict)
                                for g_obj, f_sensor in zip(objs_p, perm_f):
                                    cand_sc["fields"][g_obj]["f"] = f_sensor
                                eff_alts.append(cand_sc)
                    if not al_old.get("aligned") and eff_alts:
                        # Only adjudicate as refuted when EVERY complete role completion in `eff_alts` fails!
                        alt_trials = []
                        for alt_sc in eff_alts:
                            al_alt = instantiate_aligned_claim_program(ob, alt_sc)
                            if not al_alt.get("aligned"):
                                continue
                            t_alt = al_alt["terms"]
                            if t_alt and sensor_fit_rows:
                                c_alt = [[float(t.predict(r)) for r in sensor_fit_rows] for t in t_alt]
                                cf_alt = _linear_fit(c_alt, [float(r.target) for r in sensor_fit_rows])
                                al_alt = {**al_alt, "program": _sum_terms(list(t_alt), cf_alt[1:], cf_alt[0])}
                            mse_alt = float(loss(al_alt["program"], paid_eval_rows, 0.0, 0.5))
                            alt_trials.append((mse_alt, alt_sc, al_alt))
                        if alt_trials and all(m > tolerance for m, _, _ in alt_trials):
                            _, adj_schema, al_old = min(alt_trials, key=lambda item: item[0])
                    elif al_old.get("aligned"):
                        terms_o = al_old["terms"]
                        if terms_o and sensor_fit_rows:
                            cols_o = [[float(t.predict(r)) for r in sensor_fit_rows] for t in terms_o]
                            cf_o = _linear_fit(cols_o, [float(r.target) for r in sensor_fit_rows])
                            al_old = {**al_old, "program": _sum_terms(list(terms_o), cf_o[1:], cf_o[0])}
                    if not al_old.get("aligned"):
                        upd_branches_fail.append(ob)
                        continue
                    adj_f = adjudicate_branches_on_paid_evidence(
                        new_unconf_claim,
                        ob,
                        {"program": cand_prog, "radius_bindings": {}},
                        al_old,
                        paid_eval_rows,
                        episode_transitions,
                        adj_schema,
                        new_paid_observations=cand_eval_rows,
                        old_support_regime_features=list(ob.get("pilot_regime_features") or ()),
                        evidence_ref=ev_ref_fail,
                        budget=budget,
                        tolerance=tolerance,
                    )
                    upd_branches_fail.append(adj_f["updated_old_claim"])
                    for pq in adj_f.get("pending_questions") or ():
                        pending_qs_fail.append(pq)
                library._active_reconciled_branches = upd_branches_fail
                if pending_qs_fail:
                    library._pending_reconciliation_questions.extend(deepcopy(pending_qs_fail))

        if not has_fresh:
            ep_status = "unknown_no_fresh_data"
        elif (
            confirmation_loss is not None
            and tolerance < float(confirmation_loss) <= provisional_approximate_tolerance
        ):
            ep_status = "provisional_approximate_candidate"
        else:
            ep_status = "failed_confirmation_unresolved"
        fail_rec = {
            "task_id": task_id,
            "confirmed": False,
            "epistemic_status": ep_status,
            "tolerance_contract": tolerance,
            "provisional_approximate_tolerance": provisional_approximate_tolerance,
            "confirmation_loss": float(confirmation_loss) if confirmation_loss is not None else None,
            "confirmation_rows": len(confirmation_rows_list),
            "independent_empirical_support": False,
            "pending_questions_count": len(pending_qs_fail),
        }
        library._reconciliation_episode_ledger.append(fail_rec)
        if ep_status == "provisional_approximate_candidate":
            library._provisional_candidate_archive.append(
                {
                    "claim_id": f"provisional:{task_id}:{len(library._provisional_candidate_archive)}",
                    "status": "provisional_approximate_candidate",
                    "confirmed": False,
                    "independent_empirical_support": False,
                    **fail_rec,
                }
            )
        return {
            "action": "autonomous_branch_reconciliation",
            "task_id": task_id,
            "episode_confirmed": False,
            "epistemic_status": ep_status,
            "pending_questions": pending_qs_fail,
            "compute_cost": budget.spent - start_spent,
        }

    assert schema_dict is not None and concept_id is not None
    new_prior = next(
        (p for p in library.meta_priors if p.get("source_concept_id") == concept_id),
        None,
    )
    if new_prior is None:
        return {
            "action": "autonomous_branch_reconciliation",
            "task_id": task_id,
            "episode_confirmed": False,
            "epistemic_status": "missing_stored_prior",
            "pending_questions": [],
            "compute_cost": budget.spent - start_spent,
        }

    obs_regime = compute_observable_regime_features(episode_transitions, schema_dict, source_ref=concept_id)
    ser_support = _serialize_obs_rows(support_rows, limit=512)
    ser_fresh = _serialize_obs_rows(confirmation_rows_list, limit=128)
    new_prior["support_witness_rows"] = deepcopy(ser_support)
    new_prior["fresh_confirmation_witness_rows"] = deepcopy(ser_fresh)
    new_prior["pilot_regime_features"] = [deepcopy(obs_regime)]
    new_prior["independent_empirical_support"] = True
    new_prior["epistemic_status"] = "confirmed_exact"
    new_prior["evidence_kind"] = "empirical_episode"
    new_prior["empirical_confirmation"] = True

    comm_list = consolidate_world_structure_commitments(
        [new_prior], budget=budget, tolerance=tolerance
    )
    if not comm_list:
        library._reconciliation_episode_ledger.append(
            {
                "task_id": task_id,
                "concept_id": concept_id,
                "confirmed": True,
                "epistemic_status": "confirmed_non_parametric_structure",
            }
        )
        return {
            "action": "autonomous_branch_reconciliation",
            "task_id": task_id,
            "episode_confirmed": True,
            "epistemic_status": "confirmed_non_parametric_structure",
            "pending_questions": [],
            "compute_cost": budget.spent - start_spent,
        }

    new_comm = deepcopy(comm_list[0])
    new_comm["commitment_id"] = f"branch_comm:{concept_id}"
    rp0 = (new_comm.get("radius_parameters") or [{"symbol": "r_0", "source_values": [0.95]}])[0]
    r_sym = str(rp0.get("symbol") or "r_0")
    r_val = float((rp0.get("source_values") or [0.95])[-1])
    raw_ec = rp0.get("edge_condition")
    pred_ov = (
        {r_sym: _canonical_edge_predicate_ast(raw_ec)}
        if raw_ec not in (None, "unconditional")
        else None
    )
    terms_fit = instantiate_commitment_terms(
        new_comm, schema_dict, radius_bindings={r_sym: r_val}, predicate_overrides=pred_ov
    )
    if terms_fit and support_rows:
        if budget is not None:
            budget.spend(len(support_rows) * (len(terms_fit) + 1))
        cols = [[float(t.predict(r)) for r in support_rows] for t in terms_fit]
        coeffs = list(_linear_fit(cols, [float(r.target) for r in support_rows]))
        assembled_prog = _sum_terms(list(terms_fit), coeffs[1:], coeffs[0])
    else:
        coeffs = [0.0, *([1.0] * len(terms_fit or ()))]
        assembled_prog = None

    assembled_ver = _program_semantic_version(assembled_prog)
    delivered_ver = _program_semantic_version(delivered_program)

    new_claim = {
        "claim_id": f"branch:{concept_id}",
        "source_concept_id": concept_id,
        "source_concept_ids": [concept_id],
        "commitment": deepcopy(new_comm),
        "radius_value": r_val,
        "coefficients": coeffs,
        "assembled_program_version": assembled_ver,
        "delivered_program_version": delivered_ver,
        "assembled_matches_delivered_version": (assembled_ver == delivered_ver),
        "semantic_version": compute_semantic_version(new_prior),
        "schema": deepcopy(schema_dict),
        "status": "universal",
        "confirmed": True,
        "epistemic_status": "confirmed_exact",
        "independent_empirical_support": True,
        "empirical_support_refs": [
            {
                "task_id": task_id,
                "concept_id": concept_id,
                "confirmation_loss": float(confirmation_loss),
                "confirmation_rows": len(confirmation_rows_list),
                "semantic_version": compute_semantic_version(new_prior),
            }
        ],
        "pilot_regime_features": [deepcopy(obs_regime)],
        "support_rows_serialized": ser_support,
        "fresh_holdout_rows_serialized": ser_fresh,
    }

    old_branches = list(library._active_reconciled_branches)
    decisions: list[dict[str, Any]] = []
    paid_probes_executed: list[dict[str, Any]] = []
    pending_questions: list[dict[str, Any]] = []
    delivered_program_refuted_on_probe = False

    def _recording_probe_fn(probe_obj: Any, **kw: Any) -> Any:
        res_p = execute_probe_fn(probe_obj, **kw) if kw else execute_probe_fn(probe_obj)
        if isinstance(res_p, dict) and res_p.get("status") == "executed_new" and res_p.get("rows"):
            # v13 B4: Share reconciliation-acquired observations in raw archive so ordinary baseline has identical raw history!
            p_ser = _serialize_obs_rows(list(res_p["rows"]), limit=128)
            library._raw_reconciliation_evidence_archive.append(
                {
                    "task_id": task_id,
                    "source_concept_id": concept_id,
                    "probe_id": res_p.get("actual_probe_id"),
                    "rows_serialized": p_ser,
                    "transitions_count": len(res_p.get("transitions") or ()),
                }
            )
            # Check A3 version guard: if `delivered_program` differs from `assembled_prog`, check `delivered_program` directly!
            nonlocal delivered_program_refuted_on_probe
            if delivered_program is not None:
                dp = delivered_program if isinstance(delivered_program, Program) else Program.from_dict(delivered_program)
                dp_mse = float(loss(dp, list(res_p["rows"]), 0.0, 0.5))
                if dp_mse > tolerance:
                    delivered_program_refuted_on_probe = True
        return res_p

    if old_branches:
        old_regimes_by_id = {
            str(b["claim_id"]): list(b.get("pilot_regime_features") or ())
            for b in old_branches
        }
        rec_res = reconcile_competing_branches(
            new_claim,
            old_branches,
            spec,
            schema_dict,
            execute_probe_fn=_recording_probe_fn if execute_probe_fn is not None else None,
            controls_by_object=schema_dict.get("controls"),
            old_support_regimes_by_id=old_regimes_by_id,
            budget=budget,
            tolerance=tolerance,
        )
        decisions = list(rec_res.get("decisions") or ())
        paid_probes_executed = list(rec_res.get("paid_probes_executed") or ())
        pending_questions = list(rec_res.get("pending_questions") or ())
        if pending_questions:
            library._pending_reconciliation_questions.extend(deepcopy(pending_questions))
        upd_new = rec_res.get("updated_new_claim") or new_claim
        upd_olds = list(rec_res.get("updated_old_claims") or old_branches)
        cons_into = upd_new.get("consolidated_into")
        if cons_into:
            for ob in upd_olds:
                if str(ob.get("claim_id")) == str(cons_into):
                    ob.setdefault("pilot_regime_features", []).append(deepcopy(obs_regime))
                    ob.setdefault("empirical_support_refs", []).extend(
                        upd_new.get("empirical_support_refs") or ()
                    )
            active_branches = upd_olds
        elif upd_new.get("status") in {"refuted", "contested_refuted_on_probe"}:
            # Check A3 version guard: only revoke `new_prior` if `delivered_program` itself failed or matched `assembled_ver`!
            if delivered_program_refuted_on_probe or assembled_ver == delivered_ver:
                new_prior["status"] = "contested_counterexample_on_probe"
                delivered_program_refuted_on_probe = True
            active_branches = upd_olds
        else:
            active_branches = [*upd_olds, upd_new]
    else:
        active_branches = [new_claim]

    # v13 B1 / Check #11: Separate empirical applicability counterexamples (verified cross-prediction failure)
    # from structural routing preferences (`routing_scope_candidate`)!
    active_branches = _refine_coexisting_branch_scopes_and_routing(
        active_branches, budget=budget, tolerance=tolerance
    )
    library._active_reconciled_branches = active_branches

    # Autonomous anti-unification & cross-mechanism false-merge screening over grounded parameterized branches
    param_branches: list[tuple[str, tuple[str, str, str, str], dict[str, Any]]] = []
    for b in active_branches:
        if b.get("status") in {"refuted", "contested_refuted_on_probe"} or b.get("independent_empirical_support") is not True:
            continue
        rp_b = (b["commitment"].get("radius_parameters") or [{}])[0]
        raw_ec_b = rp_b.get("edge_condition")
        if raw_ec_b in (None, "unconditional"):
            continue
        pred_b = _canonical_edge_predicate_ast(raw_ec_b)
        op_skel_b, has_param_b = _predicate_operator_skeleton_key(pred_b)
        if not has_param_b:
            continue
        full_sig_b = _branch_regime_signature(b["commitment"])
        param_branches.append((op_skel_b, full_sig_b, b))

    def _to_au_record(b_item: dict[str, Any]) -> dict[str, Any]:
        return {
            "claim_id": b_item["claim_id"],
            "commitment": deepcopy(b_item["commitment"]),
            "schema": deepcopy(b_item["schema"]),
            "support_rows": _deserialize_obs_rows(b_item.get("support_rows_serialized") or ()),
            "scoped_counterexamples": [],
            "fresh_holdout_rows": _deserialize_obs_rows(
                b_item.get("fresh_holdout_rows_serialized") or ()
            ),
        }

    # 1) Check shared predicate-skeleton super-groups across conflicting directions/mechanisms
    by_op_skel: dict[str, list[tuple[tuple[str, str, str, str], dict[str, Any]]]] = {}
    for op_s, f_sig, b_item in param_branches:
        by_op_skel.setdefault(op_s, []).append((f_sig, b_item))

    for op_s, items_s in by_op_skel.items():
        distinct_mechs = {f_sig[:3] for f_sig, _ in items_s}
        if len(distinct_mechs) > 1 and budget.remaining > 0:
            involved_ids = sorted(str(b_it["claim_id"]) for _, b_it in items_s)
            already_tested = any(
                sorted(r.get("tested_branch_ids") or ()) == involved_ids
                for r in library._rejected_merge_records
            )
            if not already_tested:
                cross_res = propose_anti_unified_generalization(
                    [_to_au_record(b_it) for _, b_it in items_s],
                    budget=budget,
                    tolerance=tolerance,
                    require_fresh_confirmation=True,
                )
                if not cross_res.get("accepted"):
                    library._rejected_merge_records.append(
                        {
                            "task_id": task_id,
                            "tested_branch_ids": involved_ids,
                            "status": cross_res.get("status"),
                            "reason": cross_res.get("reason"),
                            "pooled_mse": cross_res.get("pooled_mse"),
                            "per_branch_mse": cross_res.get("per_branch_mse"),
                            "max_branch_mse": cross_res.get("max_branch_mse"),
                            "restored_branch_ids": cross_res.get("restored_branch_ids"),
                        }
                    )

    # 2) Incrementally anti-unify compatible groups (`>= 2` branches, including 3+ branch updates!)
    by_full_sig: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for _, f_sig, b_item in param_branches:
        by_full_sig.setdefault(f_sig, []).append(b_item)

    updated_families_by_sig: dict[tuple[str, str, str, str], dict[str, Any]] = {
        _branch_regime_signature(gf): deepcopy(gf)
        for gf in library._generalized_branch_commitments
    }
    new_family_ids: list[str] = []

    for f_sig, group_bs in by_full_sig.items():
        if len(group_bs) < 2 or budget.remaining <= 0:
            continue
        group_ids = [str(b["claim_id"]) for b in group_bs]
        existing_gf = updated_families_by_sig.get(f_sig)
        if existing_gf and list(existing_gf.get("source_branch_ids") or ()) == group_ids:
            continue

        au_res = propose_anti_unified_generalization(
            [_to_au_record(b) for b in group_bs],
            budget=budget,
            tolerance=tolerance,
            require_fresh_confirmation=True,
        )
        if au_res.get("accepted"):
            gf = deepcopy(au_res["generalized_commitment"])
            gf["independent_empirical_support"] = False
            gf["depends_on_claim_ids"] = list(group_ids)
            gf["min_grounded_dependencies"] = 2
            gf["per_branch_substitutions"] = deepcopy(au_res.get("per_branch_substitutions") or {})

            # v13 B1: Distinguish empirical family applicability failure (even after theta recalibration!)
            # from routing preference against out-of-family active branches!
            fam_sup_regs = [
                reg for b in group_bs for reg in (b.get("pilot_regime_features") or ())
            ]
            emp_out_fam_regs: list[dict[str, Any]] = []
            rot_out_fam_regs: list[dict[str, Any]] = []
            for b_out in active_branches:
                if b_out["claim_id"] in set(group_ids) or b_out.get("status") == "refuted":
                    continue
                regs_out = list(b_out.get("pilot_regime_features") or ())
                if not regs_out:
                    continue
                fam_chk = _cross_evaluate_refittable_family_on_neighbor(
                    gf, b_out, budget=budget, tolerance=tolerance
                )
                if fam_chk.get("evaluated") and fam_chk.get("empirically_refuted"):
                    emp_out_fam_regs.extend(regs_out)
                    rot_out_fam_regs.extend(regs_out)
                else:
                    rot_out_fam_regs.extend(regs_out)

            if fam_sup_regs and emp_out_fam_regs:
                scoped_gf = shrink_claim_to_observable_scope(
                    gf, fam_sup_regs, emp_out_fam_regs, scope_kind="empirical_applicability_scope"
                )
                if scoped_gf.get("shrunk") and scoped_gf.get("scope_guard") is not None:
                    gf["scope_guard"] = deepcopy(scoped_gf["scope_guard"])
                    gf["observable_bounds"] = deepcopy(scoped_gf["observable_bounds"])
                    gf["scope_verification"] = deepcopy(scoped_gf["scope_verification"])
                    new_cid, pred_sid = compute_family_scientific_identity(
                        gf,
                        Program.from_dict(gf["anti_unified_predicate_skeleton"]),
                        list(gf["parameter_slots"]),
                    )
                    gf["commitment_id"] = new_cid
                    gf["predicate_skeleton_id"] = pred_sid
            if fam_sup_regs and rot_out_fam_regs:
                routed_gf = shrink_claim_to_observable_scope(
                    gf, fam_sup_regs, rot_out_fam_regs, scope_kind="routing_scope_candidate"
                )
                if routed_gf.get("routing_shrunk") and routed_gf.get("routing_guard") is not None:
                    gf["routing_guard"] = deepcopy(routed_gf["routing_guard"])
                    gf["routing_bounds"] = deepcopy(routed_gf["routing_bounds"])
                    gf["routing_verification"] = deepcopy(routed_gf["routing_verification"])

            updated_families_by_sig[f_sig] = gf
            new_family_ids.append(gf["commitment_id"])
            library._family_evolution_history.append(
                {
                    "task_id": task_id,
                    "family_id": gf["commitment_id"],
                    "predicate_skeleton_id": gf.get("predicate_skeleton_id"),
                    "source_branch_ids": list(group_ids),
                    "parameter_slots": list(gf.get("parameter_slots") or ()),
                    "per_branch_substitutions": deepcopy(gf.get("per_branch_substitutions") or {}),
                    "per_branch_mse": deepcopy(au_res.get("per_branch_mse") or {}),
                    "per_branch_fresh_mse": deepcopy(au_res.get("per_branch_fresh_mse") or {}),
                    "source_preservation_status": au_res.get("source_preservation_status"),
                    "fresh_confirmation_status": au_res.get("fresh_confirmation_status"),
                    "scope_guard": deepcopy(gf.get("scope_guard")),
                    "routing_guard": deepcopy(gf.get("routing_guard")),
                }
            )
            child_set = set(group_ids)
            library._branch_semantic_relations = [
                rel
                for rel in library._branch_semantic_relations
                if rel.get("child_claim_id") not in child_set
            ]
            for b_c in group_bs:
                library._branch_semantic_relations.append(
                    {
                        "relation": "special_case_of",
                        "child_claim_id": b_c["claim_id"],
                        "parent_family_id": gf["commitment_id"],
                        "parameter_binding": (gf.get("per_branch_substitutions") or {}).get(
                            str(b_c["claim_id"])
                        ),
                    }
                )

    library._generalized_branch_commitments = list(updated_families_by_sig.values())
    ep_confirmed_final = not delivered_program_refuted_on_probe
    ep_record = {
        "task_id": task_id,
        "concept_id": concept_id,
        "confirmed": ep_confirmed_final,
        "epistemic_status": "confirmed_exact" if ep_confirmed_final else "revoked_after_discriminating_probe_failure",
        "tolerance_contract": tolerance,
        "confirmation_loss": float(confirmation_loss),
        "confirmation_rows": len(confirmation_rows_list),
        "observable_regime_features": obs_regime,
        "reconciliation_decisions": decisions,
        "paid_probes_executed": paid_probes_executed,
        "pending_questions_count": len(pending_questions),
        "active_branch_count": len(library._active_reconciled_branches),
        "generalized_family_ids": [g["commitment_id"] for g in library._generalized_branch_commitments],
        "newly_updated_family_ids": new_family_ids,
        "rejected_merge_count": len(library._rejected_merge_records),
        "compute_cost": budget.spent - start_spent,
    }
    library._reconciliation_episode_ledger.append(ep_record)
    return {
        "action": "autonomous_branch_reconciliation",
        "episode_confirmed": ep_confirmed_final,
        "delivered_program_refuted_on_probe": delivered_program_refuted_on_probe,
        "pending_questions": pending_questions,
        **ep_record,
    }


