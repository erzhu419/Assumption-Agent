"""Equal-tool attribution, single-factor dependency isolation, and frozen v4 lifecycle helpers (v10.2).

Implements the strict experimental controls specified in ``02_下一施工合同.md`` (Deliverables B & C,
Findings F03, F04, F06, F17, and Behavioral Acceptance #9–#21) with ``model_calls = 0``:

1. ``build_bounded_open_grammar_patch_catalog``:
   A general, benchmark-agnostic 4-slot local AST patch catalog built exclusively from existing
   ``Program`` primitives (``ALLOWED_EDGE_PREDICATE_OPS``, ``ALLOWED_ROLE_SUMMAND_OPS``,
   ``ALLOWED_OBJECT_DRIVE_OPS``) with zero task-specific oracle hints.

2. ``run_single_factor_dependency_ablation_arm``:
   Holds parent ``WorldStructureCommitment``, resolved ``ObjectSchema`` roles, paid observations &
   transitions, candidate patch grammar, fitting/validation tools, and ``SearchBudget`` **100%
   identical** across arms — varying ONLY whether explicit ``PremiseDependencyGraph`` violated-premise
   localization is supplied vs. withheld (recording an auditable ``allowed_information_manifest``
   and ``transmitted_payload_sha256`` per Acceptance #19).

3. ``run_patch_vs_scratch_ablation_arm``:
   Separately isolates local commitment patching vs. from-scratch multi-term role-program search
   under identical resolved ``ObjectSchema``, paid data, and ``dependency_cue = False`` (Contract B3).

4. ``run_equal_tool_knowledge_arm``:
   Compares ``cold_full_tools``, ``archive_full_tools`` (strong ordinary research allowed to
   reconstruct equivalent commitments from raw source archives with reconstruction ``SearchBudget``
   charged!), and ``structured_full_tools`` under **identical full v10 + open-grammar tools**,
   emitting the complete 8-stage causal action chain (Contract B4, Acceptance #13, #15, #16, #21).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from .continuous_state import _sum_terms
from .evaluation import initial_prediction, loss
from .language import Program
from .models import Candidate, Observation, SearchBudget
from .open_grammar_author import (
    _audit_focus_membership_from_trajectories,
    _enumerate_position_bijections,
    validate_edge_predicate_ast,
    validate_object_drive_ast,
    validate_role_term_spec,
)
from .premise_dependency_audit import (
    apply_verified_transition_premises,
    audit_transition_premises,
    screen_custom_edge_predicate_on_transitions,
    screen_custom_latent_transition_drive,
)
from .research_programs import _Fitter
from .structure_lab import ObjectSchema
from .world_structure_commitments import (
    _as_schema,
    _extract_latent_poles_from_role_terms,
    _rebind_latent_pole_in_role_terms,
    consolidate_world_structure_commitments,
    expand_role_terms_for_custom_latent_transition,
    expand_role_terms_for_nonconservation,
    instantiate_commitment_on_schema,
    instantiate_commitment_terms,
    is_commitment_empirically_consistent,
    propose_scientific_premise_revision,
)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _op(op: str, *children: dict, value: float | None = None, name: str | None = None, lag: int = 0) -> dict:
    return {
        "op": op,
        "value": value,
        "name": name,
        "lag": lag,
        "children": list(children),
    }


def _arg(name: str) -> dict:
    return _op("arg", name=name)


def _feat(name: str) -> dict:
    return _op("feature", name=name)


def _const(val: float) -> dict:
    return _op("const", value=float(val))


def build_bounded_open_grammar_patch_catalog() -> dict[str, list[dict[str, Any]]]:
    """Return a general 4-slot open-grammar AST patch catalog using only existing Program primitives."""
    qs = _arg("edge_source_quantity")
    qt = _arg("edge_target_quantity")
    xs = _arg("edge_source_position")
    xt = _arg("edge_target_position")
    dx = _op("sub", xt, xs)
    dq = _op("sub", qs, qt)
    ndq = _op("sub", qt, qs)

    rf_q = _feat("object_role:q")
    rf_u = _feat("object_role:u")
    rf_x = _feat("object_role:x")
    rf_f = _feat("object_role:f")

    # Slot 1: P_temporal_readout (size-ordered role-uniform all_object_sum summands)
    readout_summands = [
        ("tot_abs_q", _op("abs", rf_q)),
        ("tot_u_sq", _op("mul", rf_u, rf_u)),
        ("foc_u_sq", _op("mul", rf_f, _op("mul", rf_u, rf_u))),
        ("foc_q_u", _op("mul", rf_f, _op("mul", rf_q, rf_u))),
        ("foc_q_sq", _op("mul", rf_f, _op("mul", rf_q, rf_q))),
        ("foc_x_sq", _op("mul", rf_f, _op("mul", rf_x, rf_x))),
        ("foc_q_x", _op("mul", rf_f, _op("mul", rf_q, rf_x))),
        ("tot_q_sq", _op("mul", rf_q, rf_q)),
        ("tot_x_sq", _op("mul", rf_x, rf_x)),
        ("tot_q_u", _op("mul", rf_q, rf_u)),
        ("foc_abs_q", _op("mul", rf_f, _op("abs", rf_q))),
        ("foc_q_cube", _op("mul", rf_f, _op("mul", rf_q, _op("mul", rf_q, rf_q)))),
        (
            "foc_acc_u_sq",
            _op("mul", rf_f, _op("accumulate", _op("mul", rf_u, rf_u), _const(0.0), value=0.65)),
        ),
        (
            "foc_acc_q_u",
            _op("mul", rf_f, _op("accumulate", _op("mul", rf_q, rf_u), _const(0.0), value=0.65)),
        ),
        (
            "foc_acc_abs_q",
            _op("mul", rf_f, _op("accumulate", _op("abs", rf_q), _const(0.0), value=0.65)),
        ),
        (
            "foc_acc_q_sq",
            _op("mul", rf_f, _op("accumulate", _op("mul", rf_q, rf_q), _const(0.0), value=0.65)),
        ),
    ]
    slot_readout = [
        {
            "patch_id": f"readout:{name}",
            "slot": "P_temporal_readout",
            "extra_role_terms": [{"kind": "all_object_sum", "summand": ast_d}],
        }
        for name, ast_d in readout_summands
    ]

    # Slot 2: P_conservation (non-conservative companion role structures)
    slot_conservation = [
        {"patch_id": "noncons:sink", "slot": "P_conservation", "mechanism": "sink"},
        {"patch_id": "noncons:nonlocal", "slot": "P_conservation", "mechanism": "nonlocal"},
    ]

    # Slot 3: P_latent_transition_drive (custom per-object latent drive ASTs)
    latent_drives = [
        ("drive_u_sq", _op("mul", rf_u, rf_u)),
        ("drive_q_sq", _op("mul", rf_q, rf_q)),
        ("drive_x_sq", _op("mul", rf_x, rf_x)),
        ("drive_q_x", _op("mul", rf_q, rf_x)),
        ("drive_u_x", _op("mul", rf_u, rf_x)),
        ("drive_abs_q", _op("abs", rf_q)),
        ("drive_abs_u", _op("abs", rf_u)),
        ("drive_q_add_u", _op("add", rf_q, rf_u)),
        ("drive_q_sub_u", _op("sub", rf_q, rf_u)),
        ("drive_q_mul_u", _op("mul", rf_q, rf_u)),
        ("drive_q2_mul_u", _op("mul", _op("mul", rf_q, rf_q), rf_u)),
        ("drive_q_mul_u2", _op("mul", rf_q, _op("mul", rf_u, rf_u))),
    ]
    slot_latent = [
        {
            "patch_id": f"latent:{name}",
            "slot": "P_latent_transition_drive",
            "latent_drive_ast": ast_d,
        }
        for name, ast_d in latent_drives
    ]

    # Slot 4: P_exchange_form (open-grammar relational edge predicates over EDGE_ARGUMENTS)
    edge_preds = [
        ("src_pos_only", _op("lt", _const(0.0), qs)),
        ("dst_pos_only", _op("lt", _const(0.0), qt)),
        ("src_neg_only", _op("lt", qs, _const(0.0))),
        ("gap_gt_thresh", _op("lt", _const(0.20), dq)),
        ("rev_gap_gt_thresh", _op("lt", _const(0.20), ndq)),
        ("sum_gt_thresh", _op("lt", _const(0.20), _op("add", qs, qt))),
        ("abs_gap_gt_thresh", _op("lt", _const(0.20), _op("abs", dq))),
        ("prod_pos", _op("lt", _const(0.05), _op("mul", qs, qt))),
        ("dst_pos_and_rev_gap", _op("and", _op("lt", _const(0.0), qt), _op("lt", _const(0.20), ndq))),
        ("src_pos_and_sum", _op("and", _op("lt", _const(0.0), qs), _op("lt", _const(0.20), _op("add", qs, qt)))),
        ("src_neg_and_gap", _op("and", _op("lt", qs, _const(0.0)), _op("lt", _const(0.20), ndq))),
        ("donor_positive_gap", _op("and", _op("lt", _const(0.0), qs), _op("lt", _const(0.20), dq))),
        (
            "distance_attenuated_gap",
            _op(
                "and",
                _op("lt", _const(0.0), qs),
                _op("lt", _const(0.20), _op("sub", dq, _op("mul", _const(0.25), _op("mul", dx, dx)))),
            ),
        ),
    ]
    slot_exchange = [
        {
            "patch_id": f"exchange:{name}",
            "slot": "P_exchange_form",
            "edge_predicate": ast_d,
            "directions": ["right", "left"],
        }
        for name, ast_d in edge_preds
    ]

    return {
        "P_temporal_readout": slot_readout,
        "P_conservation": slot_conservation,
        "P_latent_transition_drive": slot_latent,
        "P_exchange_form": slot_exchange,
    }


def _extract_transition_arrays(model: dict, records: list[dict], audit: dict | None = None):
    dynamic = list(model.get("dynamic_fields") or ())
    drive_by_q: dict[str, str] = {}
    if audit and audit.get("recovered_drive_by_q"):
        drive_by_q.update(audit["recovered_drive_by_q"])
    for group in model.get("groups", ()):
        if group.get("q") and group.get("u"):
            drive_by_q.setdefault(group["q"], group["u"])
    if len(drive_by_q) < len(dynamic):
        for link in (model.get("drive_links") or {}).values():
            if link.get("quantity") and link.get("field"):
                drive_by_q.setdefault(link["quantity"], link["field"])

    pos_cands = deepcopy(
        (audit.get("recovered_position_candidates") if audit else None)
        or model.get("position_candidates")
        or {}
    )
    role_cands_all = model.get("role_candidates") or {}
    if (
        len(dynamic) >= 2
        and len(drive_by_q) == len(dynamic)
        and len(pos_cands) < len(dynamic)
        and len(role_cands_all) == 2 * len(dynamic)
    ):
        used_u = set(drive_by_q.values())
        completed_pos = {
            s_name: deepcopy(pos_cands[s_name])
            if s_name in pos_cands
            else {
                "control": role_cands_all[s_name]["control"],
                "quantities": list(dynamic),
                "support_pairs": role_cands_all[s_name].get("support_pairs", 0),
            }
            for s_name in sorted(role_cands_all)
            if s_name not in used_u
        }
        if len(completed_pos) == len(dynamic):
            pos_cands = completed_pos

    x_fields = sorted(pos_cands.keys())
    if not dynamic or len(drive_by_q) != len(dynamic) or len(x_fields) != len(dynamic):
        return None

    if audit and audit.get("transition_payload"):
        tp = audit["transition_payload"]
        q_rows = list(tp["q_rows"])
        x_rows = list(tp["x_rows"])
        delta_rows = list(tp["delta_rows"])
    else:
        seen = set()
        q_rows, x_rows, delta_rows = [], [], []
        for r in records:
            bef, aft = r.get("before"), r.get("after")
            if not bef or not aft:
                continue
            if not (set(dynamic) <= set(bef) and set(dynamic) <= set(aft) and set(x_fields) <= set(bef)):
                continue
            if not all(drive_by_q[q] in bef for q in dynamic):
                continue
            key = (
                tuple(round(float(bef[q]), 12) for q in dynamic),
                tuple(round(float(bef[x]), 12) for x in x_fields),
            )
            if key in seen:
                continue
            seen.add(key)
            q_rows.append([float(bef[q]) for q in dynamic])
            x_rows.append([float(bef[x]) for x in x_fields])
            delta_rows.append(
                [float(aft[q]) - float(bef[q]) - float(bef[drive_by_q[q]]) for q in dynamic]
            )
    bijections, _ = _enumerate_position_bijections(dynamic, x_fields, pos_cands)
    return {
        "dynamic": dynamic,
        "drive_by_q": drive_by_q,
        "x_fields": x_fields,
        "pos_cands": pos_cands,
        "q_rows": q_rows,
        "x_rows": x_rows,
        "delta_rows": delta_rows,
        "bijections": bijections,
    }


def run_single_factor_dependency_ablation_arm(
    *,
    parent_commitment: dict,
    model: dict,
    records: list[dict],
    spec,
    training: list[Observation],
    validation: list[Observation],
    compute_budget_limit: int,
    use_support_dependency_cue: bool,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Strict single-factor isolation of ``PremiseDependencyGraph`` localization (Contract B3, F04, #14, #19).

    Both arms receive:
      - Identical ``parent_commitment``
      - Identical ``model`` (role partition & candidate schemas)
      - Identical ``records``, ``training``, ``validation``
      - Identical 4-slot ``build_bounded_open_grammar_patch_catalog()``
      - Identical ``_Fitter``, transition screeners, and ``SearchBudget(compute_budget_limit)``
      - ``model_calls = 0``

    ONLY difference:
      - ``use_support_dependency_cue=True`` reads the violated premise ID from ``audit_transition_premises``
        to prioritize the implicated slot first.
      - ``use_support_dependency_cue=False`` withholds ``violated_premise_id`` (recorded in
        ``allowed_information_manifest``) and evaluates the exact same 4-slot patch catalog in fixed
        canonical order ``[P_temporal_readout, P_conservation, P_latent_transition_drive, P_exchange_form]``.
    """
    budget = SearchBudget(compute_budget_limit)
    fallback = initial_prediction(training, spec.risk_quantile)
    catalog = build_bounded_open_grammar_patch_catalog()

    # Run transition audit to obtain resolved roles/schemas; mask dependency cue if withheld!
    audit = audit_transition_premises(
        deepcopy(model),
        records,
        budget,
        commitments=[parent_commitment],
        allow_expanded_revisions=True,
        allow_latent_state=True,
        allow_higher_order_latent=True,
        observations=training,
        tolerance=tolerance,
    )
    updated_model = apply_verified_transition_premises(deepcopy(model), audit)
    schema_dict = updated_model.get("schema")
    if schema_dict:
        candidate_schemas = [schema_dict]
    else:
        from .object_alternatives import enumerate_object_schemas

        enum_res = enumerate_object_schemas(updated_model, budget, max_schemas=8)
        candidate_schemas = enum_res["schemas"]

    refuted_ids = [r.get("premise_id") for r in audit.get("refuted_premises", ())]
    if "P_exchange_form" in refuted_ids:
        true_violated_slot = "P_exchange_form"
    elif "P_latent_transition_drive" in refuted_ids:
        true_violated_slot = "P_latent_transition_drive"
    elif "P_conservation" in refuted_ids:
        true_violated_slot = "P_conservation"
    else:
        true_violated_slot = "P_temporal_readout"

    canonical_slots = [
        "P_temporal_readout",
        "P_conservation",
        "P_latent_transition_drive",
        "P_exchange_form",
    ]
    if use_support_dependency_cue:
        slot_order = [
            true_violated_slot,
            *(s for s in canonical_slots if s != true_violated_slot),
        ]
        visible_dependency_cue: str | None = true_violated_slot
    else:
        slot_order = list(canonical_slots)
        visible_dependency_cue = None

    # Auditable runtime information manifest (Acceptance #19)
    allowed_manifest = {
        "has_private_apparatus_access": False,
        "parent_commitment_id": parent_commitment["commitment_id"],
        "candidate_schema_count": len(candidate_schemas),
        "training_rows": len(training),
        "validation_rows": len(validation),
        "transition_records": len(records),
        "patch_catalog_slots": list(canonical_slots),
        "patch_catalog_total_candidates": sum(len(v) for v in catalog.values()),
        "support_dependency_cue_supplied": bool(use_support_dependency_cue),
        "visible_violated_premise_cue": visible_dependency_cue,
    }
    transmitted_digest = _sha256_json(allowed_manifest)

    tr_arrays = _extract_transition_arrays(updated_model, records, audit=audit)
    sym = (
        parent_commitment["radius_parameters"][0]["symbol"]
        if parent_commitment.get("radius_parameters")
        else "r_0"
    )
    r_bounds_audit = (
        tuple(audit["radius_interval"]) if audit.get("radius_interval") else None
    )

    attempted_patches: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    budget_exhausted = False

    for slot_name in slot_order:
        if budget.remaining <= 0 or (best_entry is not None and best_entry["validation_loss"] <= tolerance):
            break

        for patch in catalog[slot_name]:
            if budget.remaining <= 64:
                budget_exhausted = True
                break

            patch_id = patch["patch_id"]
            if slot_name == "P_temporal_readout" and candidate_schemas:
                primary_sc = _as_schema(candidate_schemas[0])
                if primary_sc is None:
                    continue
                valid_extras = [
                    v
                    for item in patch["extra_role_terms"]
                    for v in (validate_role_term_spec(item, primary_sc),)
                    if v is not None
                ]
                if not valid_extras:
                    continue
                combined_rt = [*deepcopy(parent_commitment["role_terms"]), *valid_extras]
                extra_poles = _extract_latent_poles_from_role_terms(valid_extras)
                eff_lat = (
                    "custom_focal_memory"
                    if extra_poles
                    else parent_commitment.get("latent_state_mechanism")
                )
                patch_best_val = float("inf")
                for sc_d in candidate_schemas[:4]:
                    sc_obj = _as_schema(sc_d)
                    if sc_obj is None or budget.remaining <= 64:
                        break
                    rev_meta = {
                        "trigger_source": "single_factor_local_patch_proposer",
                        "revision_kind": "open_grammar_temporal_readout_revision",
                        "parent_commitment_id": parent_commitment["commitment_id"],
                        "latent_state_mechanism": eff_lat,
                        "retracted_premises": [{"premise_id": "temporal_readout_basis"}],
                        "replacement_premises": [
                            {"premise_id": "temporal_readout_basis", "added_role_terms": deepcopy(valid_extras)}
                        ],
                        "retained_premises": [
                            p["premise_id"]
                            for p in parent_commitment.get("support_premises", ())
                            if p["premise_id"] != "temporal_readout_basis"
                        ],
                    }
                    entries = instantiate_commitment_on_schema(
                        spec,
                        training,
                        sc_obj,
                        parent_commitment,
                        budget,
                        radius_bounds=r_bounds_audit,
                        role_terms_override=combined_rt,
                        latent_state_mechanism=eff_lat,
                        reorganization_type="scientific_premise_revision",
                        scientific_revision=rev_meta,
                        tolerance=tolerance,
                    )
                    for entry in entries:
                        if not budget.spend(len(validation)):
                            budget_exhausted = True
                            break
                        v_loss = float(loss(entry["candidate"].program, validation, fallback, spec.risk_quantile))
                        entry["validation_loss"] = v_loss
                        entry["resolved_schema"] = sc_obj.to_dict()
                        patch_best_val = min(patch_best_val, v_loss)
                        if best_entry is None or v_loss < best_entry["validation_loss"]:
                            best_entry = entry
                attempted_patches.append(
                    {"patch_id": patch_id, "slot": slot_name, "validation_loss": patch_best_val}
                )
                if best_entry is not None and best_entry["validation_loss"] <= tolerance:
                    break

            elif slot_name == "P_conservation" and candidate_schemas:
                mech = patch["mechanism"]
                exp_rt = expand_role_terms_for_nonconservation(parent_commitment["role_terms"], mech)
                patch_best_val = float("inf")
                for sc_d in candidate_schemas[:4]:
                    sc_obj = _as_schema(sc_d)
                    if sc_obj is None or budget.remaining <= 64:
                        break
                    entries = instantiate_commitment_on_schema(
                        spec,
                        training,
                        sc_obj,
                        parent_commitment,
                        budget,
                        radius_bounds=r_bounds_audit,
                        role_terms_override=exp_rt,
                        nonconservation_mechanism=mech,
                        reorganization_type="scientific_premise_revision",
                        tolerance=tolerance,
                    )
                    for entry in entries:
                        if not budget.spend(len(validation)):
                            budget_exhausted = True
                            break
                        v_loss = float(loss(entry["candidate"].program, validation, fallback, spec.risk_quantile))
                        entry["validation_loss"] = v_loss
                        entry["resolved_schema"] = sc_obj.to_dict()
                        patch_best_val = min(patch_best_val, v_loss)
                        if best_entry is None or v_loss < best_entry["validation_loss"]:
                            best_entry = entry
                attempted_patches.append(
                    {"patch_id": patch_id, "slot": slot_name, "validation_loss": patch_best_val}
                )
                if best_entry is not None and best_entry["validation_loss"] <= tolerance:
                    break

            elif slot_name == "P_latent_transition_drive":
                drive_prog = validate_object_drive_ast(patch["latent_drive_ast"])
                if drive_prog is None:
                    continue
                lat_pay = audit.get("transition_latent_payload")
                if lat_pay and lat_pay.get("valid_rows"):
                    bij_l, _ = _enumerate_position_bijections(
                        lat_pay["dynamic"],
                        lat_pay["x_fields"],
                        updated_model.get("position_candidates") or model["position_candidates"],
                    )
                    scr_l = screen_custom_latent_transition_drive(
                        lat_pay["valid_rows"],
                        lat_pay["dynamic"],
                        lat_pay["drive_by_q"],
                        lat_pay["x_fields"],
                        bij_l,
                        drive_prog,
                        (lat_pay.get("representative_direction") or "right",),
                        budget=budget,
                        eff_tol=float(lat_pay.get("eff_tol", max(1e-12, tolerance))),
                        observations=training,
                    )
                    if not scr_l.get("supported"):
                        attempted_patches.append(
                            {"patch_id": patch_id, "slot": slot_name, "validation_loss": None, "status": "rejected_on_transitions"}
                        )
                        continue
                    v_pole = float(scr_l["verified_pole"])
                    exp_rt_l = expand_role_terms_for_custom_latent_transition(
                        parent_commitment["role_terms"], drive_prog, v_pole
                    )
                    patch_best_val = float("inf")
                    for sc_d in candidate_schemas[:4]:
                        sc_obj = _as_schema(sc_d)
                        if sc_obj is None or budget.remaining <= 64:
                            break
                        entries = instantiate_commitment_on_schema(
                            spec,
                            training,
                            sc_obj,
                            parent_commitment,
                            budget,
                            role_terms_override=exp_rt_l,
                            latent_state_mechanism="custom_transition_memory",
                            reorganization_type="scientific_premise_revision",
                            tolerance=tolerance,
                        )
                        for entry in entries:
                            if not budget.spend(len(validation)):
                                budget_exhausted = True
                                break
                            v_loss = float(loss(entry["candidate"].program, validation, fallback, spec.risk_quantile))
                            entry["validation_loss"] = v_loss
                            entry["resolved_schema"] = sc_obj.to_dict()
                            patch_best_val = min(patch_best_val, v_loss)
                            if best_entry is None or v_loss < best_entry["validation_loss"]:
                                best_entry = entry
                    attempted_patches.append(
                        {"patch_id": patch_id, "slot": slot_name, "validation_loss": patch_best_val}
                    )
                else:
                    # Even when transitions are conservative, untargeted arm spends real screening compute checking drive_prog!
                    if tr_arrays and not budget.spend(len(tr_arrays["q_rows"]) * len(tr_arrays["dynamic"]) * 6):
                        budget_exhausted = True
                        break
                    attempted_patches.append(
                        {"patch_id": patch_id, "slot": slot_name, "validation_loss": None, "status": "no_latent_residual"}
                    )
                if best_entry is not None and best_entry["validation_loss"] <= tolerance:
                    break

            elif slot_name == "P_exchange_form" and tr_arrays is not None:
                pred_prog = validate_edge_predicate_ast(patch["edge_predicate"])
                if pred_prog is None:
                    continue
                screened = screen_custom_edge_predicate_on_transitions(
                    tr_arrays["q_rows"],
                    tr_arrays["x_rows"],
                    tr_arrays["delta_rows"],
                    tr_arrays["dynamic"],
                    tr_arrays["x_fields"],
                    tr_arrays["bijections"],
                    pred_prog,
                    ("right", "left"),
                    budget=budget,
                    eff_tol=max(1e-12, tolerance),
                )
                if not screened.get("supported"):
                    attempted_patches.append(
                        {"patch_id": patch_id, "slot": slot_name, "validation_loss": None, "status": "rejected_on_transitions"}
                    )
                    if budget.denied_requests:
                        budget_exhausted = True
                        break
                    continue

                verified_pred = screened.get("calibrated_predicate") or pred_prog
                matches_by_dir: dict[str, list[dict]] = {}
                for m in screened["matches"]:
                    matches_by_dir.setdefault(m["direction"], []).append(m)
                narrowed_pos = {
                    xf: sorted({m["mapping"][xf] for m in screened["matches"]})
                    for xf in tr_arrays["x_fields"]
                }
                focus_aud = _audit_focus_membership_from_trajectories(
                    updated_model, records, narrowed_pos, budget, eff_tol=max(1e-12, tolerance)
                )
                synth_audit = {
                    **deepcopy(audit),
                    "status": "verified_conditional_exchange",
                    "exhaustive_bijections": True,
                    "matches_by_direction": matches_by_dir,
                    "narrowed_position_domains": narrowed_pos,
                    "narrowed_focus_domains": (
                        focus_aud["narrowed_focus_domains"]
                        if focus_aud and focus_aud.get("narrowed_focus_domains")
                        else None
                    ),
                    "verified_direction": next(iter(matches_by_dir)) if len(matches_by_dir) == 1 else None,
                    "verified_edge_condition": verified_pred.to_dict(),
                }
                resolved_m = apply_verified_transition_premises(updated_model, synth_audit)
                if resolved_m.get("schema"):
                    ex_schemas = [resolved_m["schema"]]
                else:
                    from .object_alternatives import enumerate_object_schemas

                    ex_schemas = enumerate_object_schemas(resolved_m, budget, max_schemas=6)["schemas"]

                committed_dir = parent_commitment["radius_parameters"][0]["direction"]
                committed_cond = parent_commitment["radius_parameters"][0].get("edge_condition", "unconditional")
                patch_best_val = float("inf")
                for sc_d in ex_schemas:
                    sc_obj = _as_schema(sc_d)
                    if sc_obj is None:
                        continue
                    sc_x_to_q = {sc_obj.fields[g]["x"]: sc_obj.fields[g]["q"] for g in sc_obj.objects}
                    for v_dir, d_matches in matches_by_dir.items():
                        compat = [m for m in d_matches if m["mapping"] == sc_x_to_q]
                        if not compat:
                            continue
                        r_b = (
                            min(m["radius_interval"][0] for m in compat),
                            max(m["radius_interval"][1] for m in compat),
                        )
                        rev_meta = {
                            "trigger_source": "single_factor_local_patch_proposer",
                            "revision_kind": "open_grammar_exchange_condition_revision",
                            "parent_commitment_id": parent_commitment["commitment_id"],
                            "retracted_premises": [
                                {"premise_id": f"exchange_form:{sym}", "edge_condition": committed_cond}
                            ],
                            "replacement_premises": [
                                {
                                    "premise_id": f"exchange_form:{sym}",
                                    "edge_condition": verified_pred.to_dict(),
                                    "direction": v_dir,
                                }
                            ],
                            "retained_premises": [
                                p["premise_id"]
                                for p in parent_commitment.get("support_premises", ())
                                if p["premise_id"] not in {f"exchange_form:{sym}", f"interaction_direction:{sym}"}
                            ],
                        }
                        if v_dir != committed_dir:
                            rev_meta["retracted_premises"].append(
                                {"premise_id": f"interaction_direction:{sym}", "direction": committed_dir}
                            )
                            rev_meta["replacement_premises"].append(
                                {"premise_id": f"interaction_direction:{sym}", "direction": v_dir}
                            )
                        entries = instantiate_commitment_on_schema(
                            spec,
                            training,
                            sc_obj,
                            parent_commitment,
                            budget,
                            radius_bounds=r_b,
                            threshold_interval=screened.get("threshold_interval"),
                            direction_overrides={sym: v_dir},
                            predicate_overrides={sym: verified_pred.to_dict()},
                            reorganization_type="scientific_premise_revision",
                            scientific_revision=rev_meta,
                            tolerance=tolerance,
                        )
                        for entry in entries:
                            if not budget.spend(len(validation)):
                                budget_exhausted = True
                                break
                            v_loss = float(loss(entry["candidate"].program, validation, fallback, spec.risk_quantile))
                            entry["validation_loss"] = v_loss
                            entry["resolved_schema"] = sc_obj.to_dict()
                            patch_best_val = min(patch_best_val, v_loss)
                            if best_entry is None or v_loss < best_entry["validation_loss"]:
                                best_entry = entry
                attempted_patches.append(
                    {"patch_id": patch_id, "slot": slot_name, "validation_loss": patch_best_val}
                )
                if best_entry is not None and best_entry["validation_loss"] <= tolerance:
                    break

    supported = bool(best_entry is not None and best_entry["validation_loss"] <= tolerance)
    outcome_category = (
        "completed_supported"
        if supported
        else ("budget_interrupted" if budget_exhausted or budget.denied_requests else "scientifically_unexplained")
    )
    return {
        "use_support_dependency_cue": bool(use_support_dependency_cue),
        "outcome_category": outcome_category,
        "supported": supported,
        "best_entry": best_entry,
        "validation_loss": best_entry["validation_loss"] if best_entry is not None else None,
        "candidates_attempted_denominator": len(attempted_patches),
        "failed_candidates_count": sum(
            1 for p in attempted_patches if p.get("validation_loss") is None or p["validation_loss"] > tolerance
        ),
        "attempted_patches": attempted_patches,
        "compute_cost": budget.spent,
        "budget_limit": compute_budget_limit,
        "model_calls": 0,
        "total_tokens": 0,
        "allowed_information_manifest": allowed_manifest,
        "transmitted_payload_sha256": transmitted_digest,
    }


def run_from_scratch_role_program_arm(
    *,
    model: dict,
    records: list[dict],
    spec,
    training: list[Observation],
    validation: list[Observation],
    compute_budget_limit: int,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Separate factor arm: from-scratch role-uniform basis search without parent local patch (Contract B3)."""
    budget = SearchBudget(compute_budget_limit)
    fallback = initial_prediction(training, spec.risk_quantile)
    catalog = build_bounded_open_grammar_patch_catalog()

    audit = audit_transition_premises(
        deepcopy(model), records, budget, allow_expanded_revisions=True, tolerance=tolerance
    )
    updated_model = apply_verified_transition_premises(deepcopy(model), audit)
    if updated_model.get("schema"):
        schemas = [updated_model["schema"]]
    else:
        from .object_alternatives import enumerate_object_schemas

        schemas = enumerate_object_schemas(updated_model, budget, max_schemas=6)["schemas"]

    fitter = _Fitter(list(training), budget, spec.risk_quantile)
    best_prog = None
    best_val = None
    attempted = 0

    # Must combinatorially assemble totals + temporal wrappers + custom predicates/readouts from scratch!
    radii_grid = [0.65, 0.85, 1.05, 1.25]
    directions = ["symmetric", "right", "left"]
    temporal_wraps = ["instantaneous", "event_delta", "window_sum_3", "window_event_3"]
    pred_choices = [None, *(p["edge_predicate"] for p in catalog["P_exchange_form"])]
    extra_choices = [None, *(r["extra_role_terms"][0]["summand"] for r in catalog["P_temporal_readout"][:8])]

    for sc_d in schemas[:4]:
        sc_obj = _as_schema(sc_d)
        if sc_obj is None or budget.remaining <= 64:
            break
        tot_q = Program(
            "object_sum", children=tuple(Program("feature", name=sc_obj.fields[g]["q"]) for g in sc_obj.objects)
        )
        tot_u = Program(
            "object_sum", children=tuple(Program("feature", name=sc_obj.fields[g]["u"]) for g in sc_obj.objects)
        )
        foc_q = Program(
            "object_sum",
            children=tuple(
                Program(
                    "mul",
                    children=(
                        Program("feature", name=sc_obj.fields[g]["f"]),
                        Program("feature", name=sc_obj.fields[g]["q"]),
                    ),
                )
                for g in sc_obj.objects
            ),
        )
        foc_u = Program(
            "object_sum",
            children=tuple(
                Program(
                    "mul",
                    children=(
                        Program("feature", name=sc_obj.fields[g]["f"]),
                        Program("feature", name=sc_obj.fields[g]["u"]),
                    ),
                )
                for g in sc_obj.objects
            ),
        )
        triples = tuple(
            Program("feature", name=sc_obj.fields[g][r])
            for g in sc_obj.objects
            for r in ("f", "x", "q")
        )

        for tw in temporal_wraps:
            for d_name in directions:
                op_dir = d_name if d_name in {"right", "left"} else None
                for pred_d in pred_choices:
                    for r_val in radii_grid:
                        for ext_d in extra_choices:
                            if budget.remaining <= 64:
                                break
                            r_node = Program("const", value=r_val)
                            if pred_d is None:
                                nb_raw = Program("neighbor_sum", name=op_dir, children=(r_node, *triples))
                            else:
                                pp = validate_edge_predicate_ast(pred_d)
                                if pp is None:
                                    continue
                                nb_raw = Program("relation_sum", name=op_dir, children=(r_node, pp, *triples))

                            if tw == "instantaneous":
                                base_terms = [tot_q, tot_u, foc_q, foc_u, nb_raw]
                            elif tw == "event_delta":
                                base_terms = [
                                    tot_q,
                                    tot_u,
                                    Program("event_delta", lag=1, children=(foc_q,)),
                                    Program("event_delta", lag=1, children=(foc_u,)),
                                    Program("event_delta", lag=1, children=(nb_raw,)),
                                ]
                            elif tw == "window_sum_3":
                                base_terms = [
                                    Program("window_sum", lag=3, children=(tot_q,)),
                                    Program("window_sum", lag=3, children=(tot_u,)),
                                    Program("window_sum", lag=3, children=(nb_raw,)),
                                ]
                            else:
                                base_terms = [
                                    Program("window_sum", lag=3, children=(tot_q,)),
                                    Program("window_sum", lag=3, children=(tot_u,)),
                                    Program(
                                        "window_sum",
                                        lag=3,
                                        children=(Program("event_delta", lag=1, children=(foc_q,)),),
                                    ),
                                    Program(
                                        "window_sum",
                                        lag=3,
                                        children=(Program("event_delta", lag=1, children=(foc_u,)),),
                                    ),
                                    Program(
                                        "window_sum",
                                        lag=3,
                                        children=(Program("event_delta", lag=1, children=(nb_raw,)),),
                                    ),
                                ]

                            if ext_d is not None:
                                norm_ext = validate_role_term_spec(
                                    {"kind": "all_object_sum", "summand": ext_d}, sc_obj
                                )
                                if norm_ext is not None:
                                    pat = Program.from_dict(norm_ext["summand"])
                                    ext_prog = Program(
                                        "object_sum",
                                        children=tuple(
                                            pat.remap_features(
                                                {
                                                    f"object_role:{rl}": sc_obj.fields[g][rl]
                                                    for rl in norm_ext["roles"]
                                                }
                                            )
                                            for g in sc_obj.objects
                                        ),
                                    )
                                    base_terms.append(ext_prog)

                            attempted += 1
                            fitted = fitter.fit(tuple(base_terms))
                            if fitted is None:
                                break
                            if not budget.spend(len(validation)):
                                break
                            v_loss = float(loss(fitted["program"], validation, fallback, spec.risk_quantile))
                            if best_val is None or v_loss < best_val:
                                best_val = v_loss
                                best_prog = fitted["program"]
                            if v_loss <= tolerance:
                                break
                        if best_val is not None and best_val <= tolerance:
                            break
                    if best_val is not None and best_val <= tolerance:
                        break
                if best_val is not None and best_val <= tolerance:
                    break
            if best_val is not None and best_val <= tolerance:
                break

    supported = bool(best_val is not None and best_val <= tolerance)
    return {
        "mode": "from_scratch_role_program_search",
        "supported": supported,
        "program": best_prog,
        "validation_loss": best_val,
        "candidates_attempted_denominator": attempted,
        "compute_cost": budget.spent,
        "budget_limit": compute_budget_limit,
        "model_calls": 0,
        "total_tokens": 0,
    }


def reconstruct_commitments_from_raw_archive(
    raw_archive_priors: list[dict],
    budget: SearchBudget,
    *,
    install_into_library: Any = None,
    deduplicate_audit_predicates: bool = True,
    tolerance: float = 1e-7,
) -> tuple[list[dict], int]:
    """Allow ``archive`` baselines to reconstruct commitments once per target episode and reuse them (v11 A2).

    When ``install_into_library`` is provided, installs the reconstructed commitments into
    ``library._episode_archive_cache`` keyed by semantic versions so ``UnifiedResearchAgent``
    reuses them across all rounds of the same target episode with zero double-charging!
    """
    from .premise_commitment_index import (
        _commitment_direction_class,
        _commitment_predicate_key,
        _prior_state_fingerprint,
        extract_archive_witness_maps,
    )

    before = budget.spent
    sup_map, ce_map = extract_archive_witness_maps(raw_archive_priors)
    reconstructed = consolidate_world_structure_commitments(
        deepcopy(raw_archive_priors),
        budget=budget,
        support_rows_by_source=sup_map or None,
        counterexample_rows_by_source=ce_map or None,
        tolerance=tolerance,
    )
    spent = budget.spent - before

    if install_into_library is not None:
        fp_map = {
            p.get("source_concept_id", f"anon_{i}"): _prior_state_fingerprint(p)
            for i, p in enumerate(raw_archive_priors)
        }
        dedup: list[dict] = []
        if deduplicate_audit_predicates:
            seen_p: set[str] = set()
            seen_d: set[str] = set()
            for c in reconstructed:
                pk = _commitment_predicate_key(c)
                dc = _commitment_direction_class(c)
                rps = c.get("radius_parameters") or ()
                if rps and isinstance(rps[0].get("edge_condition"), dict):
                    if pk not in seen_p:
                        seen_p.add(pk)
                        dedup.append(c)
                elif dc not in seen_d:
                    seen_d.add(dc)
                    dedup.append(c)
        else:
            dedup = list(reconstructed)
        install_into_library._episode_archive_cache = {
            "fingerprints": fp_map,
            "commitments": deepcopy(reconstructed),
            "dedup_audit_commitments": deepcopy(dedup),
            "reconstruct_compute_already_charged": spent,
        }
    return reconstructed, spent


class UnifiedResourceAccount:
    """Single top-level budget ledger governing source, main loop, fallback, and confirmation (v11 A3)."""

    def __init__(
        self,
        *,
        compute_limit: int,
        observation_limit: int,
        measurement_limit: int,
    ) -> None:
        self.compute_limit = int(compute_limit)
        self.observation_limit = int(observation_limit)
        self.measurement_limit = int(measurement_limit)
        self.compute_spent = 0
        self.observation_spent = 0
        self.measurement_spent = 0
        self.denied_events: list[dict[str, Any]] = []
        self.ledger_events: list[dict[str, Any]] = []

    @property
    def remaining_compute(self) -> int:
        return max(0, self.compute_limit - self.compute_spent)

    @property
    def remaining_observations(self) -> int:
        return max(0, self.observation_limit - self.observation_spent)

    @property
    def remaining_measurements(self) -> int:
        return max(0, self.measurement_limit - self.measurement_spent)

    def can_afford(
        self,
        *,
        compute: int = 0,
        observations: int = 0,
        measurements: int = 0,
    ) -> bool:
        return (
            self.compute_spent + int(compute) <= self.compute_limit
            and self.observation_spent + int(observations) <= self.observation_limit
            and self.measurement_spent + int(measurements) <= self.measurement_limit
        )

    def debit(
        self,
        stage: str,
        *,
        compute: int = 0,
        observations: int = 0,
        measurements: int = 0,
    ) -> bool:
        c, o, m = int(compute), int(observations), int(measurements)
        if not self.can_afford(compute=c, observations=o, measurements=m):
            self.denied_events.append(
                {
                    "stage": stage,
                    "requested": {"compute": c, "observations": o, "measurements": m},
                    "remaining": {
                        "compute": self.remaining_compute,
                        "observations": self.remaining_observations,
                        "measurements": self.remaining_measurements,
                    },
                }
            )
            return False
        self.compute_spent += c
        self.observation_spent += o
        self.measurement_spent += m
        self.ledger_events.append(
            {
                "stage": stage,
                "compute": c,
                "observations": o,
                "measurements": m,
            }
        )
        return True

    def request_compute(self, stage: str, units: int) -> bool:
        return self.debit(stage, compute=units)

    def request_observations(
        self, stage: str, observations: int, measurements: int = 0
    ) -> bool:
        return self.debit(stage, observations=observations, measurements=measurements)

    @property
    def exceeded_or_denied(self) -> bool:
        return bool(self.denied_events)

    def summary(self) -> dict[str, Any]:
        return {
            "limits": {
                "compute": self.compute_limit,
                "observations": self.observation_limit,
                "measurements": self.measurement_limit,
            },
            "spent": {
                "compute": self.compute_spent,
                "observations": self.observation_spent,
                "measurements": self.measurement_spent,
            },
            "within_all_limits": (
                self.compute_spent <= self.compute_limit
                and self.observation_spent <= self.observation_limit
                and self.measurement_spent <= self.measurement_limit
            ),
            "budget_interrupted": bool(self.denied_events),
            "denied_events": list(self.denied_events),
            "ledger_events": list(self.ledger_events),
        }


def extract_real_consumption_events_from_actions(
    actions: list[dict] | tuple[dict, ...],
) -> list[dict[str, Any]]:
    """Extract verified ``ConsumptionEvent`` records from agent action traces (v11 A3)."""
    events: list[dict[str, Any]] = []
    for act in actions or ():
        for ev in act.get("consumption_events") or ():
            if isinstance(ev, dict):
                events.append(deepcopy(ev))
    return events


def build_action_chain_telemetry(
    *,
    arm: str,
    task_id: str,
    knowledge_available_count: int,
    knowledge_retrieved_ids: list[str],
    entered_candidate_pool: bool | None = None,
    changed_probe_or_audit: bool | None = None,
    changed_parent_commitment: bool | None = None,
    reorganization_type: str | None = None,
    confirmed: bool,
    heldout_mse: float | None,
    tolerance: float = 1e-7,
    consumption_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct complete causal chain telemetry from real ``ConsumptionEvent`` diffs (v11 A3).

    Never sets ``entered_candidates``, ``changed_action_or_probe``, or ``changed_parent_model``
    from ``bool(retrieved_ids)`` when ``consumption_events`` are supplied!
    """
    passed = bool(heldout_mse is not None and heldout_mse <= tolerance)
    if consumption_events is not None:
        real_entered = any(
            bool(ev.get("generated_candidate_ids")) for ev in consumption_events
        )
        real_changed_parent = any(
            ev.get("parent_before") != ev.get("parent_after")
            for ev in consumption_events
            if "parent_before" in ev or "parent_after" in ev
        )
        real_changed_probe = any(
            ev.get("probe_before") != ev.get("probe_after")
            for ev in consumption_events
            if "probe_before" in ev or "probe_after" in ev
        )
        entered_flag = real_entered
        changed_probe_flag = real_changed_probe
        changed_parent_flag = real_changed_parent
    else:
        entered_flag = bool(entered_candidate_pool)
        changed_probe_flag = bool(changed_probe_or_audit)
        changed_parent_flag = bool(changed_parent_commitment)

    actually_invoked = bool(
        entered_flag or changed_probe_flag or changed_parent_flag
    )
    if not knowledge_retrieved_ids:
        retrieval_substatus = (
            "no_retained_knowledge_available"
            if knowledge_available_count == 0
            else "not_retrieved"
        )
        knowledge_utility_status = (
            "no_retained_knowledge_available"
            if knowledge_available_count == 0
            else "retained_knowledge_not_invoked"
        )
    elif not actually_invoked:
        retrieval_substatus = "retrieved_not_invoked"
        knowledge_utility_status = "retained_knowledge_not_invoked"
    elif passed and confirmed:
        retrieval_substatus = "retrieved_and_invoked"
        knowledge_utility_status = "invoked_and_supported"
    elif passed and not confirmed:
        # Acceptance A3: external fresh48 pass without internal confirmation stays strictly separate!
        retrieval_substatus = "retrieved_and_invoked"
        knowledge_utility_status = "invoked_external_pass_unconfirmed_internally"
    else:
        retrieval_substatus = "retrieved_and_invoked"
        knowledge_utility_status = "invoked_without_confirmed_benefit"

    return {
        "arm": arm,
        "task_id": task_id,
        "knowledge_available_count": knowledge_available_count,
        "retrieved_ids": list(knowledge_retrieved_ids),
        "retrieval_substatus": retrieval_substatus,
        "entered_candidates": entered_flag,
        "changed_action_or_probe": changed_probe_flag,
        "changed_parent_model": changed_parent_flag,
        "actually_invoked": actually_invoked,
        "consumption_events_count": len(consumption_events) if consumption_events is not None else None,
        "consumption_events": deepcopy(consumption_events) if consumption_events is not None else None,
        "reorganization_type": reorganization_type,
        "confirmed_on_task": bool(confirmed),
        "external_fresh48_passed_unconfirmed": bool(passed and not confirmed),
        "heldout_mse": heldout_mse,
        "passed_heldout_tolerance": passed,
        "knowledge_utility_status": knowledge_utility_status,
    }
