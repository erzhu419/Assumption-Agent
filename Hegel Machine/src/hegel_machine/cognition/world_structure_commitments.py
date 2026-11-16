"""Consolidate and instantiate arity-parametric world-structure commitments (v10).

Inspired by AWM's exploration -> executable world model -> consolidation ->
predicate-guided verification cycle and adapted to Hegel Machine's strict
closed-ledger epistemic contracts:

1. ``Equivalence Reorganization`` consolidates confirmed object-relational
   priors into arity-parametric world-structure commitments ONLY when complete
   fitted columns (including zero-weight coordinates), symbolic radius-sharing
   groups, operator semantics, paid-row predictions, and counterexample
   rejection sets ``Rej(T) >= Rej(S)`` are preserved.
2. ``Cross-Scale Instantiation`` binds a verified commitment onto any complete
   target ``ObjectSchema`` (e.g. ``N=4 -> N=5``), re-identifying locality
   radii on target observed distances (narrowed by verified transition bounds
   when available) and refitting linear readout weights under ``SearchBudget``.
3. ``Scientific Premise Revision`` is strictly separated from equivalence
   consolidation: when empirical evidence refutes a specific support premise
   (such as ``interaction_direction``), it records the retracted premise and
   counterexample while testing minimal conservative replacements.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import json
from math import sqrt
from random import Random

from .continuous_state import _linear_fit, _sum_terms
from .evaluation import loss
from .language import Program
from .models import Candidate, Observation, SearchBudget
from .premise_dependency_audit import (
    VERIFIED_EXCHANGE_STATUSES,
    _edge_condition_holds,
    _replace_const_at_path,
    _replace_node_at_path,
    describe_comparator_threshold_parameter,
    describe_edge_condition,
    edge_condition_predicate,
    extract_comparator_threshold_interval,
    screen_custom_edge_predicate_on_transitions,
)
from .relation_conditions import TRUE
from .research_programs import _Fitter, _plan
from .structural_discovery import StructuralDiscovery
from .structural_research_program import (
    attach_structural_research_program,
    research_program_fit_contract,
    structural_program_fit_contract,
)
from .structure_lab import ObjectSchema
from .transfer_partitions import _saved_binding, transfer_partition_view


_FAMILY_IDS = {
    "conserved_total": "T15",
    "conserved_combination": "T15",
    "focal_coordinates": "T04",
    "object_specific": "T04",
    "local_exchange": "T08",
    "local_and_global": "T08",
    "object_specific_exchange": "T08",
    "event_change": "T11",
    "window_exchange": "T12",
    "joint_current": "T08",
    "joint_event": "T11",
    "joint_window": "T12",
    "joint_window_event": "T12",
    "world_commitment": "T12",
}


def _as_schema(value) -> ObjectSchema | None:
    if value is None:
        return None
    if isinstance(value, ObjectSchema):
        return value
    payload = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, dict) or not payload.get("objects") or not payload.get("fields"):
        return None
    return ObjectSchema.from_dict(payload)


def _whole_view(schema: ObjectSchema | None) -> bool:
    if schema is None or not schema.objects:
        return False
    members = [member for group in schema.objects for member in schema.members.get(group, ())]
    return (
        len(set(schema.objects)) == len(schema.objects)
        and len(members) == len(set(members))
        and set(members) == set(schema.controls)
        and all({"q", "u", "x", "f"} <= set(schema.fields.get(group, ())) for group in schema.objects)
    )


def _extract_source_schema(prior: dict) -> ObjectSchema | None:
    for candidate in (
        prior.get("object_schema"),
        (prior.get("structure") or {}).get("schema"),
        (prior.get("research_program") or {}).get("source_schema"),
    ):
        schema = _as_schema(candidate)
        if _whole_view(schema):
            return schema
    return None


def _extract_fit_and_symbols(prior: dict):
    """Recover complete fitted columns and explicit symbolic radius bindings."""
    plan = prior.get("research_program")
    structure = prior.get("structure")
    fit = None
    symbolic_terms = None
    symbol_records = {}
    if plan:
        try:
            fit = research_program_fit_contract(plan)
        except ValueError:
            fit = None
        if fit is not None:
            view = transfer_partition_view(plan)
            if view["status"] == "ready":
                saved = _saved_binding(plan)
                symbolic_terms = tuple(Program.from_dict(term) for term in saved["parameter_terms"])
                for item in saved["parameters"]:
                    symbol_records[item["symbol"]] = {
                        "symbol": item["symbol"],
                        "source_value": float(item["radius"]),
                        "source_intervals": deepcopy(item.get("radius_intervals", [])),
                        "binding": "saved_parameter_symbol",
                    }
    if fit is None and structure and structure.get("candidates") and structure.get("fits"):
        program = Program.from_dict(structure["candidates"][0]["program"])
        fit = structural_program_fit_contract(
            program, structure, loss=float(structure["fits"][0].get("training_loss", 0.0))
        )
    if fit is None:
        return None
    shared_structural_radius = None
    shared_interval = None
    if plan and isinstance(plan.get("structural_replay"), dict):
        shared_structural_radius = plan["structural_replay"].get("radius")
        shared_interval = plan["structural_replay"].get("radius_interval")
    elif structure and structure.get("fits"):
        shared_structural_radius = structure["fits"][0].get("radius")
        shared_interval = structure.get("radius_interval") or structure["fits"][0].get("radius_interval")
    return {
        "fit": fit,
        "symbolic_terms": symbolic_terms,
        "symbol_records": symbol_records,
        "shared_structural_radius": shared_structural_radius,
        "shared_structural_interval": shared_interval,
    }


def _direction_label(name: str | None) -> str:
    return name if name in {"right", "left"} else "symmetric"


def _direction_name(label: str) -> str | None:
    return label if label in {"right", "left"} else None


def extract_parametric_role_pattern(prior: dict) -> dict | None:
    """Lift a confirmed whole-object prior to an arity-parametric role pattern.

    Returns ``None`` whenever any leaf references a bare object-specific field,
    partial object subset, or asymmetric per-object constant. Shared radius
    symbols come strictly from ``resolve_relation_intervals`` or a single
    structural discovery fit, never from coincidental float equality.
    """
    if prior.get("status") not in {"supported", "conditional"}:
        return None
    source = _extract_source_schema(prior)
    if source is None:
        return None
    recovered = _extract_fit_and_symbols(prior)
    if recovered is None:
        return None
    fit = recovered["fit"]
    raw_terms = recovered["symbolic_terms"] or fit["fit_terms"]
    if not raw_terms or len(raw_terms) != len(fit["fit_terms"]):
        return None

    owners = {
        name: (group, role)
        for group in source.objects
        for role, name in source.fields[group].items()
        if role in {"q", "u", "x", "f"}
    }
    expected_count = 4 * len(source.objects)
    if len(owners) != expected_count:
        return None

    symbol_records = deepcopy(recovered["symbol_records"])
    shared_radius = recovered["shared_structural_radius"]
    shared_interval = recovered["shared_structural_interval"]
    next_fresh_index = 0
    reductions = []

    def allocate_symbol(radius_node: Program, actual_radius_node: Program) -> str | None:
        nonlocal next_fresh_index
        if radius_node.op == "arg":
            sym = radius_node.name
            if sym not in symbol_records:
                if actual_radius_node.op != "const":
                    return None
                symbol_records[sym] = {
                    "symbol": sym,
                    "source_value": float(actual_radius_node.value),
                    "source_intervals": [list(shared_interval)] if shared_interval else [],
                    "binding": "saved_parameter_symbol",
                }
            return sym
        if radius_node.op != "const":
            return None
        value = float(radius_node.value)
        if shared_radius is not None and abs(value - float(shared_radius)) == 0.0:
            sym = "r_0"
            symbol_records.setdefault(
                sym,
                {
                    "symbol": sym,
                    "source_value": value,
                    "source_intervals": [list(shared_interval)] if shared_interval else [],
                    "binding": "single_structural_family_radius",
                },
            )
            return sym
        while f"r_{next_fresh_index}" in symbol_records:
            next_fresh_index += 1
        sym = f"r_{next_fresh_index}"
        next_fresh_index += 1
        symbol_records[sym] = {
            "symbol": sym,
            "source_value": value,
            "source_intervals": [],
            "binding": "occurrence_specific_const_radius",
        }
        return sym

    def lift_object_sum(node: Program, path: tuple) -> dict | None:
        groups, patterns = [], []
        for child in node.children:
            fields = child.required_features()
            if not fields or not fields <= owners.keys():
                return None
            local = {owners[name][0] for name in fields}
            if len(local) != 1:
                return None
            group = next(iter(local))
            groups.append(group)
            patterns.append(
                child.remap_features(
                    {name: f"object_role:{role}" for role, name in source.fields[group].items()}
                )
            )
        if (
            len(groups) != len(source.objects)
            or set(groups) != set(source.objects)
            or any(pattern != patterns[0] for pattern in patterns[1:])
        ):
            return None
        roles = sorted(name.removeprefix("object_role:") for name in patterns[0].required_features())
        reductions.append({"path": list(path), "kind": "all_object_sum", "roles": roles})
        return {
            "kind": "all_object_sum",
            "summand": patterns[0].to_dict(),
            "roles": roles,
        }

    def lift_neighborhood(node: Program, actual_node: Program, path: tuple) -> dict | None:
        if actual_node.op != node.op or len(actual_node.children) != len(node.children):
            return None
        offset = 2 if node.op == "relation_sum" else 1
        triples = [node.children[i : i + 3] for i in range(offset, len(node.children), 3)]
        expected = [
            tuple(Program("feature", name=source.fields[group][role]) for role in ("f", "x", "q"))
            for group in source.objects
        ]
        value_summand_dict = None
        value_roles = None
        if len(triples) != len(expected):
            return None
        if set(triples) != set(expected):
            matched_groups = []
            val_patterns = []
            for sel_n, pos_n, val_n in triples:
                if sel_n.op != "feature" or pos_n.op != "feature":
                    return None
                own_s = owners.get(sel_n.name)
                own_p = owners.get(pos_n.name)
                if not own_s or not own_p or own_s[1] != "f" or own_p[1] != "x" or own_s[0] != own_p[0]:
                    return None
                grp = own_s[0]
                req_f = val_n.required_features()
                if not req_f or not req_f <= set(source.fields[grp].values()):
                    return None
                matched_groups.append(grp)
                val_patterns.append(
                    val_n.remap_features(
                        {name: f"object_role:{role}" for role, name in source.fields[grp].items()}
                    )
                )
            if set(matched_groups) != set(source.objects) or any(
                vp != val_patterns[0] for vp in val_patterns[1:]
            ):
                return None
            value_summand_dict = val_patterns[0].to_dict()
            value_roles = sorted(
                name.removeprefix("object_role:") for name in val_patterns[0].required_features()
            )
        sym = allocate_symbol(node.children[0], actual_node.children[0])
        if sym is None:
            return None
        direction = _direction_label(node.name)
        predicate = actual_node.children[1].to_dict() if node.op == "relation_sum" else None
        reductions.append(
            {
                "path": list(path),
                "kind": f"all_{node.op}",
                "direction": direction,
                "symbol": sym,
                "predicate": predicate,
            }
        )
        out_nb = {
            "kind": f"all_{node.op}",
            "direction": direction,
            "symbol": sym,
            "predicate": predicate,
        }
        if value_summand_dict is not None:
            out_nb["value_summand"] = value_summand_dict
            out_nb["value_roles"] = value_roles
        return out_nb

    def lift(node: Program, actual_node: Program, path: tuple) -> dict | None:
        if node.op == "object_sum":
            lifted_sum = lift_object_sum(node, path)
            if lifted_sum is not None:
                return lifted_sum
        if node.op in {"neighbor_sum", "relation_sum"}:
            return lift_neighborhood(node, actual_node, path)
        if node.op in {"feature", "lag", "arg", "call"}:
            if node.op == "const":
                return {"kind": "program_op", "op": "const", "value": node.value, "children": []}
            return None
        if node.op == "const":
            return {"kind": "program_op", "op": "const", "value": float(node.value), "children": []}
        if actual_node.op != node.op or len(actual_node.children) != len(node.children):
            return None
        children = []
        for idx, (child, actual_child) in enumerate(zip(node.children, actual_node.children)):
            lifted_child = lift(child, actual_child, (*path, idx))
            if lifted_child is None:
                return None
            children.append(lifted_child)
        return {
            "kind": "program_op",
            "op": node.op,
            "name": node.name,
            "value": node.value,
            "lag": node.lag,
            "children": children,
        }

    role_terms = []
    for index, (symbolic_term, actual_term) in enumerate(zip(raw_terms, fit["fit_terms"])):
        lifted = lift(symbolic_term, actual_term, (index,))
        if lifted is None:
            return None
        role_terms.append(lifted)
    if not reductions:
        return None

    # Order radius parameters deterministically by first appearance in role_terms.
    ordered_symbols = []
    for item in reductions:
        sym = item.get("symbol")
        if sym and sym not in ordered_symbols:
            ordered_symbols.append(sym)
    radius_parameters = []
    for sym in ordered_symbols:
        directions = sorted({item["direction"] for item in reductions if item.get("symbol") == sym})
        predicates = [item["predicate"] for item in reductions if item.get("symbol") == sym]
        first_pred = predicates[0]
        if len(directions) != 1 or any(pred != first_pred for pred in predicates[1:]):
            # One shared symbol bound to conflicting directions/predicates is kept explicit.
            direction = directions[0]
        else:
            direction = directions[0]
        meta = symbol_records[sym]
        radius_parameters.append(
            {
                "symbol": sym,
                "direction": direction,
                "edge_condition": describe_edge_condition(first_pred),
                "binding": meta["binding"],
                "source_value": meta["source_value"],
                "source_intervals": meta["source_intervals"],
            }
        )

    app_meta = prior.get("world_commitment_application") or (
        (prior.get("research_program") or {}).get("world_commitment_application") or {}
    )
    sci_rev = prior.get("scientific_premise_revision") or (
        (prior.get("research_program") or {}).get("scientific_premise_revision") or {}
    )
    nonconservation_mechanism = (
        app_meta.get("nonconservation_mechanism")
        or sci_rev.get("nonconservation_mechanism")
    )
    latent_poles = _extract_latent_poles_from_role_terms(role_terms)
    ar2_params_list = _extract_ar2_params_from_role_terms(role_terms)
    ar2_params = list(ar2_params_list[0]) if ar2_params_list else None
    vdp_params_list = _extract_vdp_params_from_role_terms(role_terms)
    vdp_params = list(vdp_params_list[0]) if vdp_params_list else None
    latent_state_mechanism = (
        app_meta.get("latent_state_mechanism")
        or sci_rev.get("latent_state_mechanism")
        or (
            "van_der_pol_transition_memory"
            if vdp_params
            else ("oscillatory_focal_memory" if ar2_params else ("focal_memory" if latent_poles else None))
        )
    )
    generation = int(sci_rev.get("generation") or app_meta.get("generation") or 0)
    revision_lineage = list(
        deepcopy(sci_rev.get("revision_lineage") or app_meta.get("revision_lineage") or [])
    )
    if sci_rev and (
        not revision_lineage
        or (
            revision_lineage[-1].get("generation"),
            revision_lineage[-1].get("concept_id"),
        )
        != (generation, prior["source_concept_id"])
    ):
        revision_lineage.append(
            {
                "generation": generation,
                "concept_id": prior["source_concept_id"],
                "task_id": prior.get("task_id"),
                "parent_commitment_id": sci_rev.get("parent_commitment_id"),
                "retracted_premises": [
                    p.get("premise_id") for p in sci_rev.get("retracted_premises", ())
                ],
                "replacement_premises": deepcopy(sci_rev.get("replacement_premises", ())),
            }
        )

    # Identify explicit support premises carried by this confirmed law.
    has_total_q = any(
        _contains_total_role(term, "q") for term in role_terms
    )
    has_total_u = any(
        _contains_total_role(term, "u") for term in role_terms
    )
    structure_conservation = any(
        h.get("kind") == "conservation" and h.get("status") in {"proposed", "supported"}
        for h in ((prior.get("structure") or {}).get("hypotheses") or ())
    )
    conservation_supported = bool(
        not nonconservation_mechanism and ((has_total_q and has_total_u) or structure_conservation)
    )
    temporal_signature = sorted(
        {
            op_sig
            for term in role_terms
            for op_sig in _collect_temporal_ops(term)
        }
    )
    support_premises = [
        {
            "premise_id": "uniform_object_symmetry",
            "kind": "symmetry",
            "claim": "All objects participate symmetrically via role-uniform reductions",
            "test_layer": "role_and_scope_structure",
        },
        {
            "premise_id": "markov_state_sufficiency",
            "kind": "state_sufficiency",
            "expected": "refuted" if latent_state_mechanism else "supported",
            "latent_state_mechanism": latent_state_mechanism,
            "latent_poles": latent_poles,
            "claim": (
                f"Per-object unobserved latent state ({latent_state_mechanism}) augments observed roles"
                if latent_state_mechanism
                else "Observed (Q, U, X, F) roles are Markov-sufficient without internal hidden memory"
            ),
            "test_layer": "state_sufficiency",
        },
    ]
    if nonconservation_mechanism:
        support_premises.append(
            {
                "premise_id": "conservation",
                "kind": "conservation",
                "expected": "refuted",
                "nonconservation_mechanism": nonconservation_mechanism,
                "claim": f"Non-conservative {nonconservation_mechanism} coupling accompanies pairwise exchange",
                "test_layer": "state_transitions",
            }
        )
    elif conservation_supported or radius_parameters:
        support_premises.append(
            {
                "premise_id": "conservation",
                "kind": "conservation",
                "expected": "supported",
                "claim": "Pairwise state exchange conserves sum_i (q'_i - q_i - u_i) = 0",
                "test_layer": "state_transitions",
            }
        )
    for param in radius_parameters:
        support_premises.append(
            {
                "premise_id": f"exchange_form:{param['symbol']}",
                "kind": "exchange_form",
                "symbol": param["symbol"],
                "edge_condition": param["edge_condition"],
                "claim": f"Pairwise transport for {param['symbol']} uses edge_condition={param['edge_condition']}",
                "test_layer": "state_transitions",
            }
        )
        support_premises.append(
            {
                "premise_id": f"interaction_direction:{param['symbol']}",
                "kind": "interaction_direction",
                "symbol": param["symbol"],
                "direction": param["direction"],
                "edge_condition": param["edge_condition"],
                "claim": f"Pairwise interaction for {param['symbol']} has direction={param['direction']}",
                "test_layer": "state_transitions",
            }
        )
        support_premises.append(
            {
                "premise_id": f"locality_radius:{param['symbol']}",
                "kind": "locality_radius",
                "symbol": param["symbol"],
                "reestimate_across_arity": True,
                "source_value": param["source_value"],
                "claim": f"Interaction range {param['symbol']} is positive and re-identifiable on target geometry",
                "test_layer": "state_transitions_and_response",
            }
        )
    support_premises.append(
        {
            "premise_id": "temporal_readout_basis",
            "kind": "temporal_readout",
            "temporal_operators": temporal_signature,
            "column_count": len(role_terms),
            "claim": "Linear readout over the saved role-uniform temporal basis explains target response",
            "test_layer": "trajectory_response",
        }
    )

    family_name = (
        ((prior.get("structure") or {}).get("fits") or [{}])[0].get("family")
        or ((prior.get("research_program") or {}).get("structural_replay") or {}).get("family")
        or "world_commitment"
    )
    return {
        "source_concept_id": prior["source_concept_id"],
        "source_task_id": prior.get("task_id"),
        "source_status": prior.get("status", "supported"),
        "source_schema": source,
        "source_arity": len(source.objects),
        "fit_contract": fit,
        "role_terms": role_terms,
        "radius_parameters": radius_parameters,
        "reductions": reductions,
        "support_premises": support_premises,
        "nonconservation_mechanism": nonconservation_mechanism,
        "latent_state_mechanism": latent_state_mechanism,
        "latent_poles": latent_poles,
        "ar2_params": ar2_params,
        "vdp_params": vdp_params,
        "generation": generation,
        "revision_lineage": revision_lineage,
        "family": family_name,
        "family_id": prior.get("family_id") or _FAMILY_IDS.get(family_name, "T12"),
        "counterexamples": deepcopy(prior.get("counterexamples", [])),
    }


def _contains_total_role(node: dict, role: str) -> bool:
    if node["kind"] == "all_object_sum":
        return node["summand"] == Program("feature", name=f"object_role:{role}").to_dict()
    for child in node.get("children", ()):
        if _contains_total_role(child, role):
            return True
    return False


def _collect_prog_accumulates(prog_dict: dict) -> list[float]:
    poles = []
    if (
        prog_dict.get("op") == "accumulate"
        and prog_dict.get("name") is None
        and prog_dict.get("value") is not None
    ):
        poles.append(float(prog_dict["value"]))
    for ch in prog_dict.get("children", ()):
        poles.extend(_collect_prog_accumulates(ch))
    return poles


def _rebind_prog_accumulates(prog_dict: dict, new_pole: float | str) -> dict:
    updated = deepcopy(prog_dict)
    if updated.get("op") == "accumulate" and updated.get("name") is None:
        updated["value"] = new_pole
    if "children" in updated:
        updated["children"] = [_rebind_prog_accumulates(ch, new_pole) for ch in updated["children"]]
    return updated


def _collect_prog_ar2_params(prog_dict: dict) -> list[tuple[float, float]]:
    found = []
    if (
        prog_dict.get("op") == "accumulate"
        and prog_dict.get("name") in {"ar2", "ar2_lag1"}
        and prog_dict.get("value") is not None
        and len(prog_dict.get("children", ())) == 2
        and prog_dict["children"][1].get("op") == "const"
        and prog_dict["children"][1].get("value") is not None
    ):
        found.append((float(prog_dict["value"]), float(prog_dict["children"][1]["value"])))
    for ch in prog_dict.get("children", ()):
        found.extend(_collect_prog_ar2_params(ch))
    return found


def _rebind_prog_ar2_params(prog_dict: dict, a1: float | str, a2: float | str) -> dict:
    updated = deepcopy(prog_dict)
    if updated.get("op") == "accumulate" and updated.get("name") in {"ar2", "ar2_lag1"}:
        updated["value"] = a1
        chs = list(updated.get("children", ()))
        if len(chs) == 2:
            chs[1] = {"op": "const", "value": a2, "name": None, "lag": 0, "children": []}
            updated["children"] = chs
    if "children" in updated:
        updated["children"] = [_rebind_prog_ar2_params(ch, a1, a2) for ch in updated["children"]]
    return updated


def _extract_latent_poles_from_role_terms(role_terms: list[dict]) -> list[float]:
    poles: list[float] = []

    def _visit(node: dict):
        if node["kind"] == "all_object_sum":
            for p in _collect_prog_accumulates(node["summand"]):
                if all(abs(p - existing) > 1e-12 for existing in poles):
                    poles.append(p)
        elif node["kind"] in {"all_neighbor_sum", "all_relation_sum"} and "value_summand" in node:
            for p in _collect_prog_accumulates(node["value_summand"]):
                if all(abs(p - existing) > 1e-12 for existing in poles):
                    poles.append(p)
        elif node["kind"] == "program_op":
            if node["op"] == "accumulate" and node.get("name") is None and node.get("value") is not None:
                p = float(node["value"])
                if all(abs(p - existing) > 1e-12 for existing in poles):
                    poles.append(p)
            for child in node.get("children", ()):
                _visit(child)

    for term in role_terms:
        _visit(term)
    return poles


def _rebind_latent_pole_in_role_terms(role_terms: list[dict], new_pole: float | str) -> list[dict]:
    def _visit(node: dict) -> dict:
        if node["kind"] == "all_object_sum":
            return {
                **deepcopy(node),
                "summand": _rebind_prog_accumulates(node["summand"], new_pole),
            }
        if node["kind"] in {"all_neighbor_sum", "all_relation_sum"} and "value_summand" in node:
            return {
                **deepcopy(node),
                "value_summand": _rebind_prog_accumulates(node["value_summand"], new_pole),
            }
        if node["kind"] == "program_op":
            updated = deepcopy(node)
            if updated["op"] == "accumulate" and updated.get("name") is None:
                updated["value"] = new_pole
            updated["children"] = [_visit(ch) for ch in updated.get("children", ())]
            return updated
        return deepcopy(node)

    return [_visit(term) for term in role_terms]


def _extract_ar2_params_from_role_terms(role_terms: list[dict]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []

    def _visit(node: dict):
        if node["kind"] == "all_object_sum":
            for a1, a2 in _collect_prog_ar2_params(node["summand"]):
                if all(abs(a1 - ex1) > 1e-12 or abs(a2 - ex2) > 1e-12 for ex1, ex2 in pairs):
                    pairs.append((a1, a2))
        elif node["kind"] in {"all_neighbor_sum", "all_relation_sum"} and "value_summand" in node:
            for a1, a2 in _collect_prog_ar2_params(node["value_summand"]):
                if all(abs(a1 - ex1) > 1e-12 or abs(a2 - ex2) > 1e-12 for ex1, ex2 in pairs):
                    pairs.append((a1, a2))
        elif node["kind"] == "program_op":
            for child in node.get("children", ()):
                _visit(child)

    for term in role_terms:
        _visit(term)
    return pairs


def _rebind_ar2_params_in_role_terms(
    role_terms: list[dict], a1: float | str, a2: float | str
) -> list[dict]:
    def _visit(node: dict) -> dict:
        if node["kind"] == "all_object_sum":
            return {
                **deepcopy(node),
                "summand": _rebind_prog_ar2_params(node["summand"], a1, a2),
            }
        if node["kind"] in {"all_neighbor_sum", "all_relation_sum"} and "value_summand" in node:
            return {
                **deepcopy(node),
                "value_summand": _rebind_prog_ar2_params(node["value_summand"], a1, a2),
            }
        if node["kind"] == "program_op":
            updated = deepcopy(node)
            updated["children"] = [_visit(ch) for ch in updated.get("children", ())]
            return updated
        return deepcopy(node)

    return [_visit(term) for term in role_terms]


def _collect_prog_vdp_params(prog_dict: dict) -> list[tuple[float, float, float]]:
    found = []
    if (
        prog_dict.get("op") == "accumulate"
        and prog_dict.get("name") in {"vdp", "vdp_lag1"}
        and prog_dict.get("value") is not None
        and len(prog_dict.get("children", ())) == 3
        and prog_dict["children"][1].get("op") == "const"
        and prog_dict["children"][2].get("op") == "const"
        and prog_dict["children"][1].get("value") is not None
        and prog_dict["children"][2].get("value") is not None
    ):
        found.append(
            (
                float(prog_dict["value"]),
                float(prog_dict["children"][1]["value"]),
                float(prog_dict["children"][2]["value"]),
            )
        )
    for ch in prog_dict.get("children", ()):
        found.extend(_collect_prog_vdp_params(ch))
    return found


def _rebind_prog_vdp_params(
    prog_dict: dict, a1: float | str, a2: float | str, beta: float | str
) -> dict:
    updated = deepcopy(prog_dict)
    if updated.get("op") == "accumulate" and updated.get("name") in {"vdp", "vdp_lag1"}:
        updated["value"] = a1
        chs = list(updated.get("children", ()))
        if len(chs) == 3:
            chs[1] = {"op": "const", "value": a2, "name": None, "lag": 0, "children": []}
            chs[2] = {"op": "const", "value": beta, "name": None, "lag": 0, "children": []}
            updated["children"] = chs
    if "children" in updated:
        updated["children"] = [_rebind_prog_vdp_params(ch, a1, a2, beta) for ch in updated["children"]]
    return updated


def _extract_vdp_params_from_role_terms(role_terms: list[dict]) -> list[tuple[float, float, float]]:
    triples: list[tuple[float, float, float]] = []

    def _visit(node: dict):
        if node["kind"] == "all_object_sum":
            for a1, a2, bt in _collect_prog_vdp_params(node["summand"]):
                if all(
                    abs(a1 - e1) > 1e-12 or abs(a2 - e2) > 1e-12 or abs(bt - eb) > 1e-12
                    for e1, e2, eb in triples
                ):
                    triples.append((a1, a2, bt))
        elif node["kind"] in {"all_neighbor_sum", "all_relation_sum"} and "value_summand" in node:
            for a1, a2, bt in _collect_prog_vdp_params(node["value_summand"]):
                if all(
                    abs(a1 - e1) > 1e-12 or abs(a2 - e2) > 1e-12 or abs(bt - eb) > 1e-12
                    for e1, e2, eb in triples
                ):
                    triples.append((a1, a2, bt))
        elif node["kind"] == "program_op":
            for child in node.get("children", ()):
                _visit(child)

    for term in role_terms:
        _visit(term)
    return triples


def _rebind_vdp_params_in_role_terms(
    role_terms: list[dict], a1: float | str, a2: float | str, beta: float | str
) -> list[dict]:
    def _visit(node: dict) -> dict:
        if node["kind"] == "all_object_sum":
            return {
                **deepcopy(node),
                "summand": _rebind_prog_vdp_params(node["summand"], a1, a2, beta),
            }
        if node["kind"] in {"all_neighbor_sum", "all_relation_sum"} and "value_summand" in node:
            return {
                **deepcopy(node),
                "value_summand": _rebind_prog_vdp_params(node["value_summand"], a1, a2, beta),
            }
        if node["kind"] == "program_op":
            updated = deepcopy(node)
            updated["children"] = [_visit(ch) for ch in updated.get("children", ())]
            return updated
        return deepcopy(node)

    return [_visit(term) for term in role_terms]


def _normalize_edge_condition_for_grouping(cond: str | dict) -> str | dict:
    if not isinstance(cond, dict):
        return cond
    prog = Program.from_dict(cond)
    from .curved_hypersurface_probing import normalize_hypersurface_for_grouping

    hs_norm = normalize_hypersurface_for_grouping(prog)
    if hs_norm is not None:
        return hs_norm.to_dict()
    desc = describe_comparator_threshold_parameter(prog)
    if desc is not None:
        return _replace_node_at_path(
            prog, tuple(desc["path"]), Program("arg", name="__THRESHOLD__")
        ).to_dict()
    return cond


def _collect_temporal_ops(node: dict) -> list[str]:
    found = []
    if node["kind"] == "program_op":
        if node["op"] in {"window_sum", "event_delta"}:
            found.append(f"{node['op']}:lag={node['lag']}")
        elif node["op"] == "accumulate":
            found.append("accumulate")
    elif node["kind"] == "all_object_sum" and _collect_prog_accumulates(node["summand"]):
        found.append("per_object_accumulate")
    for child in node.get("children", ()):
        found.extend(_collect_temporal_ops(child))
    return found


def _focused_role_term(role: str) -> dict:
    return {
        "kind": "all_object_sum",
        "summand": Program(
            "mul",
            children=(
                Program("feature", name="object_role:f"),
                Program("feature", name=f"object_role:{role}"),
            ),
        ).to_dict(),
        "roles": ["f", role],
    }


def _focused_q_role_term() -> dict:
    return _focused_role_term("q")


def _total_q_role_term() -> dict:
    return {
        "kind": "all_object_sum",
        "summand": Program("feature", name="object_role:q").to_dict(),
        "roles": ["q"],
    }


def _focused_accumulated_role_term(role: str, pole: float) -> dict:
    return {
        "kind": "all_object_sum",
        "summand": Program(
            "mul",
            children=(
                Program("feature", name="object_role:f"),
                Program(
                    "accumulate",
                    value=float(pole),
                    children=(
                        Program("feature", name=f"object_role:{role}"),
                        Program("const", value=0.0),
                    ),
                ),
            ),
        ).to_dict(),
        "roles": ["f", role],
    }


def _total_accumulated_role_term(role: str, pole: float) -> dict:
    return {
        "kind": "all_object_sum",
        "summand": Program(
            "accumulate",
            value=float(pole),
            children=(
                Program("feature", name=f"object_role:{role}"),
                Program("const", value=0.0),
            ),
        ).to_dict(),
        "roles": [role],
    }


def _replace_neighborhood_in_role_term(node: dict, replacement: dict) -> tuple[dict, bool]:
    if node["kind"] in {"all_neighbor_sum", "all_relation_sum"}:
        return deepcopy(replacement), True
    if node["kind"] == "program_op":
        changed = False
        children = []
        for child in node.get("children", ()):
            new_child, child_changed = _replace_neighborhood_in_role_term(child, replacement)
            changed = changed or child_changed
            children.append(new_child)
        if not changed:
            return deepcopy(node), False
        return {**deepcopy(node), "children": children}, True
    return deepcopy(node), False


def expand_role_terms_for_nonconservation(
    role_terms: list[dict], mechanism: str | None
) -> list[dict]:
    """Append deduplicated companion flow terms for ``sink`` or ``nonlocal``."""
    if mechanism == "sink":
        leaf = _focused_q_role_term()
    elif mechanism == "nonlocal":
        leaf = _total_q_role_term()
    else:
        return deepcopy(role_terms)
    expanded = deepcopy(role_terms)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}
    for term in role_terms:
        companion, changed = _replace_neighborhood_in_role_term(term, leaf)
        if changed:
            key = json.dumps(companion, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                expanded.append(companion)
    return expanded


def _replace_total_role_in_role_term(node: dict, role: str, replacement: dict) -> tuple[dict, bool]:
    if (
        node["kind"] == "all_object_sum"
        and node["summand"] == Program("feature", name=f"object_role:{role}").to_dict()
    ):
        return deepcopy(replacement), True
    if node["kind"] == "program_op":
        changed = False
        children = []
        for child in node.get("children", ()):
            new_ch, ch_changed = _replace_total_role_in_role_term(child, role, replacement)
            changed = changed or ch_changed
            children.append(new_ch)
        if not changed:
            return deepcopy(node), False
        return {**deepcopy(node), "children": children}, True
    return deepcopy(node), False


def expand_role_terms_for_latent_transition(
    role_terms: list[dict],
    pole: float,
    drive_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Expand ``role_terms`` with per-object unobserved latent transition terms."""
    active_roles = [
        r for r in ("u", "q") if abs((drive_weights or {}).get(r, 1.0 if r == "u" else 0.0)) > 1e-6
    ]
    if not active_roles:
        active_roles = ["u"]
    expanded = deepcopy(role_terms)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}

    def _add(term_dict: dict):
        key = json.dumps(term_dict, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            expanded.append(term_dict)

    for r_name in active_roles:
        tot_acc = _total_accumulated_role_term(r_name, pole)
        foc_acc = _focused_accumulated_role_term(r_name, pole)
        foc_inst = _focused_role_term(r_name)
        for term in role_terms:
            tot_comp, tot_changed = _replace_total_role_in_role_term(term, "q", tot_acc)
            if tot_changed:
                _add(tot_comp)
            nb_acc, nb_changed = _replace_neighborhood_in_role_term(term, foc_acc)
            if nb_changed:
                _add(nb_acc)
                nb_inst, _ = _replace_neighborhood_in_role_term(term, foc_inst)
                _add(nb_inst)
    return expanded


def expand_role_terms_for_latent_focal_readout(
    role_terms: list[dict], pole: float
) -> list[dict]:
    """Expand ``role_terms`` with per-object focal reservoir terms ``f_i * accumulate(pole, role_i, 0)``."""
    expanded = deepcopy(role_terms)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}
    for r_name in ("q", "u"):
        cand = _focused_accumulated_role_term(r_name, pole)
        key = json.dumps(cand, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            expanded.append(cand)
    return expanded


def _ar2_role_term(role: str, a1: float, a2: float, *, mode: str = "ar2", focused: bool = True) -> dict:
    acc_prog = Program(
        "accumulate",
        value=float(a1),
        name=mode,
        children=(
            Program("feature", name=f"object_role:{role}"),
            Program("const", value=float(a2)),
        ),
    )
    if focused:
        summand = Program(
            "mul",
            children=(Program("feature", name="object_role:f"), acc_prog),
        )
        roles = ["f", role]
    else:
        summand = acc_prog
        roles = [role]
    return {"kind": "all_object_sum", "summand": summand.to_dict(), "roles": roles}


def expand_role_terms_for_ar2_latent_transition(
    role_terms: list[dict],
    a1: float,
    a2: float,
    drive_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Expand ``role_terms`` with 2nd-order oscillatory AR(2) ``ar2_lag1`` transition terms."""
    active_roles = [
        r for r in ("u", "q") if abs((drive_weights or {}).get(r, 1.0 if r == "u" else 0.0)) > 1e-6
    ]
    if not active_roles:
        active_roles = ["u"]
    expanded = deepcopy(role_terms)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}

    def _add(term_dict: dict):
        key = json.dumps(term_dict, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            expanded.append(term_dict)

    for r_name in active_roles:
        tot_ar2 = _ar2_role_term(r_name, a1, a2, mode="ar2_lag1", focused=False)
        foc_ar2 = _ar2_role_term(r_name, a1, a2, mode="ar2_lag1", focused=True)
        for term in role_terms:
            tot_comp, tot_changed = _replace_total_role_in_role_term(term, "q", tot_ar2)
            if tot_changed:
                _add(tot_comp)
            nb_comp, nb_changed = _replace_neighborhood_in_role_term(term, foc_ar2)
            if nb_changed:
                _add(nb_comp)
    return expanded


def _vdp_role_term(
    role: str, a1: float, a2: float, beta: float, *, mode: str = "vdp_lag1", focused: bool = True
) -> dict:
    acc_prog = Program(
        "accumulate",
        value=float(a1),
        name=mode,
        children=(
            Program("feature", name=f"object_role:{role}"),
            Program("const", value=float(a2)),
            Program("const", value=float(beta)),
        ),
    )
    if focused:
        summand = Program(
            "mul",
            children=(Program("feature", name="object_role:f"), acc_prog),
        )
        roles = ["f", role]
    else:
        summand = acc_prog
        roles = [role]
    return {"kind": "all_object_sum", "summand": summand.to_dict(), "roles": roles}


def expand_role_terms_for_vdp_latent_transition(
    role_terms: list[dict],
    a1: float,
    a2: float,
    beta: float,
    drive_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Expand ``role_terms`` with supercritical Van der Pol ``vdp_lag1`` transition terms."""
    active_roles = [
        r for r in ("u", "q") if abs((drive_weights or {}).get(r, 1.0 if r == "u" else 0.0)) > 1e-6
    ]
    if not active_roles:
        active_roles = ["u"]
    expanded = deepcopy(role_terms)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}

    def _add(term_dict: dict):
        key = json.dumps(term_dict, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            expanded.append(term_dict)

    for r_name in active_roles:
        tot_vdp = _vdp_role_term(r_name, a1, a2, beta, mode="vdp_lag1", focused=False)
        foc_vdp = _vdp_role_term(r_name, a1, a2, beta, mode="vdp_lag1", focused=True)
        for term in role_terms:
            tot_comp, tot_changed = _replace_total_role_in_role_term(term, "q", tot_vdp)
            if tot_changed:
                _add(tot_comp)
            nb_comp, nb_changed = _replace_neighborhood_in_role_term(term, foc_vdp)
            if nb_changed:
                _add(nb_comp)
    return expanded


def expand_role_terms_for_ar2_focal_readout(
    role_terms: list[dict], a1: float, a2: float
) -> list[dict]:
    """Expand ``role_terms`` with 2nd-order oscillatory AR(2) focal reservoir terms."""
    expanded = deepcopy(role_terms)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}
    for r_name in ("q", "u"):
        cand = _ar2_role_term(r_name, a1, a2, mode="ar2", focused=True)
        key = json.dumps(cand, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            expanded.append(cand)
    return expanded


def expand_role_terms_for_coupled_latent_transition(
    role_terms: list[dict],
    pole: float,
    symbol: str,
    direction: str,
    drive_weights: dict[str, float] | None = None,
) -> list[dict]:
    """Expand ``role_terms`` with both local and spatially coupled neighbor latent transport."""
    expanded = expand_role_terms_for_latent_transition(role_terms, pole, drive_weights)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}

    def _add(term_dict: dict):
        key = json.dumps(term_dict, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            expanded.append(term_dict)

    active_roles = [
        r for r in ("u", "q") if abs((drive_weights or {}).get(r, 1.0 if r == "u" else 0.0)) > 1e-6
    ] or ["u"]
    for r_name in active_roles:
        nb_acc_leaf = {
            "kind": "all_neighbor_sum",
            "direction": direction,
            "symbol": symbol,
            "predicate": None,
            "value_summand": Program(
                "accumulate",
                value=float(pole),
                children=(
                    Program("feature", name=f"object_role:{r_name}"),
                    Program("const", value=0.0),
                ),
            ).to_dict(),
            "value_roles": [r_name],
        }
        nb_inst_leaf = {
            "kind": "all_neighbor_sum",
            "direction": direction,
            "symbol": symbol,
            "predicate": None,
            "value_summand": Program("feature", name=f"object_role:{r_name}").to_dict(),
            "value_roles": [r_name],
        }
        for term in role_terms:
            comp_acc, ch1 = _replace_neighborhood_in_role_term(term, nb_acc_leaf)
            if ch1:
                _add(comp_acc)
                comp_inst, _ = _replace_neighborhood_in_role_term(term, nb_inst_leaf)
                _add(comp_inst)
    return expanded


def expand_role_terms_for_coupled_focal_readout(
    role_terms: list[dict],
    pole: float,
    symbol: str,
    direction: str,
) -> list[dict]:
    """Expand ``role_terms`` with local + spatially coupled neighbor focal reservoir terms."""
    expanded = expand_role_terms_for_latent_focal_readout(role_terms, pole)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}
    for r_name in ("q", "u"):
        nb_term = {
            "kind": "all_neighbor_sum",
            "direction": direction,
            "symbol": symbol,
            "predicate": None,
            "value_summand": Program(
                "accumulate",
                value=float(pole),
                children=(
                    Program("feature", name=f"object_role:{r_name}"),
                    Program("const", value=0.0),
                ),
            ).to_dict(),
            "value_roles": [r_name],
        }
        key = json.dumps(nb_term, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            expanded.append(nb_term)
    return expanded


def expand_role_terms_for_custom_latent_transition(
    role_terms: list[dict],
    drive_program: Program,
    pole: float,
) -> list[dict]:
    """Expand ``role_terms`` for an open-grammar per-object latent transition drive ``drive_program``."""
    d_roles = sorted(
        name.removeprefix("object_role:") for name in drive_program.required_features()
    )
    acc_prog = Program(
        "accumulate",
        value=float(pole),
        children=(drive_program, Program("const", value=0.0)),
    )
    tot_acc = {"kind": "all_object_sum", "summand": acc_prog.to_dict(), "roles": d_roles}
    tot_inst = {"kind": "all_object_sum", "summand": drive_program.to_dict(), "roles": d_roles}
    foc_roles = sorted({"f", *d_roles})
    foc_acc = {
        "kind": "all_object_sum",
        "summand": Program(
            "mul", children=(Program("feature", name="object_role:f"), acc_prog)
        ).to_dict(),
        "roles": foc_roles,
    }
    foc_inst = {
        "kind": "all_object_sum",
        "summand": Program(
            "mul", children=(Program("feature", name="object_role:f"), drive_program)
        ).to_dict(),
        "roles": foc_roles,
    }
    expanded = deepcopy(role_terms)
    seen = {json.dumps(t, sort_keys=True, separators=(",", ":")) for t in expanded}

    def _add(term_dict: dict):
        key = json.dumps(term_dict, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            expanded.append(term_dict)

    for term in role_terms:
        tc_acc, ch_t = _replace_total_role_in_role_term(term, "q", tot_acc)
        if ch_t:
            _add(tc_acc)
            tc_inst, _ = _replace_total_role_in_role_term(term, "q", tot_inst)
            _add(tc_inst)
        nb_acc, ch_n = _replace_neighborhood_in_role_term(term, foc_acc)
        if ch_n:
            _add(nb_acc)
            nb_inst, _ = _replace_neighborhood_in_role_term(term, foc_inst)
            _add(nb_inst)
    return expanded


def instantiate_commitment_terms(
    commitment: dict,
    target_schema: ObjectSchema | dict | str,
    *,
    radius_bindings: dict[str, float] | None = None,
    direction_overrides: dict[str, str] | None = None,
    predicate_overrides: dict[str, dict | str | Program | None] | None = None,
    role_terms_override: list[dict] | None = None,
    symbolic_radii: bool = False,
) -> tuple[Program, ...] | None:
    """Instantiate arity-parametric role terms onto ``target_schema`` without fitting."""
    target = _as_schema(target_schema)
    if not _whole_view(target):
        return None
    defaults = {
        param["symbol"]: float(
            param["source_values"][-1] if "source_values" in param else param["source_value"]
        )
        for param in commitment.get("radius_parameters", ())
    }
    bindings = {**defaults, **(radius_bindings or {})}
    overrides = direction_overrides or {}
    pred_overrides = predicate_overrides or {}
    active_role_terms = (
        role_terms_override
        if role_terms_override is not None
        else commitment.get("role_terms", ())
    )

    def build(node: dict) -> Program | None:
        kind = node["kind"]
        if kind == "all_object_sum":
            pattern = Program.from_dict(node["summand"])
            roles = node["roles"]
            if any(not set(roles) <= set(target.fields[group]) for group in target.objects):
                return None
            children = tuple(
                pattern.remap_features(
                    {f"object_role:{role}": target.fields[group][role] for role in roles}
                )
                for group in target.objects
            )
            return Program("object_sum", children=children)
        if kind in {"all_neighbor_sum", "all_relation_sum"}:
            sym = node["symbol"]
            if not symbolic_radii and sym not in bindings:
                return None
            radius_expr = (
                Program("arg", name=sym)
                if symbolic_radii
                else Program("const", value=float(bindings[sym]))
            )
            direction = overrides.get(sym, node["direction"])
            op_name = _direction_name(direction)
            if "value_summand" in node:
                val_pat = Program.from_dict(node["value_summand"])
                val_roles = node.get("value_roles") or []
                if any(not set(val_roles) <= set(target.fields[group]) for group in target.objects):
                    return None
                triples = tuple(
                    part
                    for group in target.objects
                    for part in (
                        Program("feature", name=target.fields[group]["f"]),
                        Program("feature", name=target.fields[group]["x"]),
                        val_pat.remap_features(
                            {f"object_role:{r}": target.fields[group][r] for r in val_roles}
                        ),
                    )
                )
            else:
                triples = tuple(
                    Program("feature", name=target.fields[group][role])
                    for group in target.objects
                    for role in ("f", "x", "q")
                )
            if sym in pred_overrides:
                override_val = pred_overrides[sym]
                if override_val is None or override_val == "unconditional":
                    return Program("neighbor_sum", name=op_name, children=(radius_expr, *triples))
                pred_prog = edge_condition_predicate(override_val)
                if pred_prog == TRUE:
                    return Program("neighbor_sum", name=op_name, children=(radius_expr, *triples))
                return Program(
                    "relation_sum", name=op_name, children=(radius_expr, pred_prog, *triples)
                )
            if kind == "all_neighbor_sum":
                return Program("neighbor_sum", name=op_name, children=(radius_expr, *triples))
            predicate = Program.from_dict(node["predicate"])
            return Program(
                "relation_sum", name=op_name, children=(radius_expr, predicate, *triples)
            )
        if kind == "program_op":
            children = tuple(build(child) for child in node.get("children", ()))
            if any(child is None for child in children):
                return None
            return Program(
                node["op"],
                value=node.get("value"),
                name=node.get("name"),
                lag=node.get("lag", 0),
                children=children,
            )
        return None

    built = tuple(build(term) for term in active_role_terms)
    if not built or any(term is None for term in built):
        return None
    return built


def _synthetic_probe_rows(schema: ObjectSchema, seed: int = 104729, count: int = 10) -> list[Observation]:
    rng = Random(seed)
    history, rows = [], []
    objects = list(schema.objects)
    for step in range(count):
        focused_group = objects[step % len(objects)]
        frame = {}
        for group in objects:
            fields = schema.fields[group]
            frame[fields["q"]] = rng.uniform(-1.2, 1.2)
            frame[fields["u"]] = rng.uniform(-0.25, 0.25)
            frame[fields["x"]] = rng.uniform(-2.0, 2.0)
            frame[fields["f"]] = 1.0 if group == focused_group else 0.0
        rows.append(Observation(frame, 0.0, tuple(history[-4:])))
        history.append(dict(frame))
    return rows


def _canonical_term_key(program: Program):
    if program.op in {"neighbor_sum", "relation_sum"}:
        offset = 2 if program.op == "relation_sum" else 1
        triples = tuple(
            sorted(
                tuple(child.describe() for child in program.children[idx : idx + 3])
                for idx in range(offset, len(program.children), 3)
            )
        )
        prefix = (program.children[0].describe(),) + (
            (program.children[1].describe(),) if offset == 2 else ()
        )
        return (program.op, program.name, program.lag, prefix, triples)
    if program.op == "object_sum":
        return (
            program.op,
            program.name,
            program.lag,
            tuple(sorted(_canonical_term_key(child) for child in program.children)),
        )
    return (
        program.op,
        program.name,
        program.value,
        program.lag,
        tuple(_canonical_term_key(child) for child in program.children),
    )


def _basis_min_loss(terms: tuple[Program, ...], rows: list[Observation], budget: SearchBudget | None) -> float | None:
    if not rows or not terms:
        return None
    cost = len(rows) * (len(terms) + 2)
    if budget is not None and not budget.spend(cost):
        return None
    columns = [[term.predict(row) for row in rows] for term in terms]
    targets = [row.target for row in rows]
    coeffs = _linear_fit(columns, targets)
    program = _sum_terms(terms, coeffs[1:], coeffs[0])
    return loss(program, rows, 0.0, 0.5)


def check_equivalence_reorganization(
    sources: list[dict],
    candidate_commitment: dict,
    *,
    support_rows_by_source: dict[str, list[Observation]] | None = None,
    counterexample_rows_by_source: dict[str, list[Observation]] | None = None,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-9,
) -> dict:
    """Verify whether ``candidate_commitment`` is a strict equivalence reorganization.

    Checks all three GPT6/AWM preservation invariants:
    1. Exact fit-contract column preservation (including zero-weight readout
       columns) and symbolic radius-group preservation on every source schema.
    2. Numerical prediction agreement across paid support rows (or deterministic
       multi-step probe frames when raw support rows were not persisted).
    3. Rejection-set preservation ``Rej(T) >= Rej(S)``: ``candidate_commitment``
       must NOT fit/accept counterexample trajectories that the source basis
       rejected, nor collapse directional/predicate distinctions.
    """
    start = budget.spent if budget is not None else 0
    checks = {
        "fit_contract_preserved": True,
        "symbolic_groups_preserved": True,
        "prediction_agreement": True,
        "rejection_preserved": True,
        "source_details": [],
    }
    if not sources or not candidate_commitment.get("role_terms"):
        return {
            "accepted": False,
            "reason": "empty_sources_or_commitment",
            "checks": checks,
            "compute_cost": (budget.spent - start) if budget is not None else 0,
        }

    expected_symbols = [p["symbol"] for p in candidate_commitment.get("radius_parameters", ())]
    expected_directions = {
        p["symbol"]: (p["direction"], p["edge_condition"])
        for p in candidate_commitment.get("radius_parameters", ())
    }

    for source in sources:
        source_id = source["source_concept_id"]
        schema = source["source_schema"]
        source_fit = source["fit_contract"]
        source_params = {p["symbol"]: p for p in source.get("radius_parameters", ())}
        if list(source_params) != expected_symbols:
            checks["symbolic_groups_preserved"] = False
            return {
                "accepted": False,
                "reason": "symbolic_parameter_groups_mismatch",
                "checks": checks,
                "compute_cost": (budget.spent - start) if budget is not None else 0,
            }
        for sym, meta in source_params.items():
            exp_dir, exp_cond = expected_directions[sym]
            if (
                meta["direction"] != exp_dir
                or _normalize_edge_condition_for_grouping(meta["edge_condition"])
                != _normalize_edge_condition_for_grouping(exp_cond)
            ):
                checks["fit_contract_preserved"] = False
                return {
                    "accepted": False,
                    "reason": "operator_direction_or_condition_changed",
                    "checks": checks,
                    "compute_cost": (budget.spent - start) if budget is not None else 0,
                }

        radii = {sym: meta["source_value"] for sym, meta in source_params.items()}
        src_pred_overrides = {
            sym: meta["edge_condition"]
            for sym, meta in source_params.items()
            if isinstance(meta.get("edge_condition"), dict)
        }
        src_role_terms = None
        src_poles = source.get("latent_poles") or []
        src_ar2 = source.get("ar2_params")
        src_vdp = source.get("vdp_params")
        if len(src_poles) == 1:
            src_role_terms = _rebind_latent_pole_in_role_terms(
                candidate_commitment["role_terms"], float(src_poles[0])
            )
        elif src_vdp and len(src_vdp) == 3:
            src_role_terms = _rebind_vdp_params_in_role_terms(
                candidate_commitment["role_terms"],
                float(src_vdp[0]),
                float(src_vdp[1]),
                float(src_vdp[2]),
            )
        elif src_ar2 and len(src_ar2) == 2:
            src_role_terms = _rebind_ar2_params_in_role_terms(
                candidate_commitment["role_terms"], float(src_ar2[0]), float(src_ar2[1])
            )
        instantiated = instantiate_commitment_terms(
            candidate_commitment,
            schema,
            radius_bindings=radii,
            predicate_overrides=src_pred_overrides or None,
            role_terms_override=src_role_terms,
        )
        if instantiated is None or len(instantiated) != len(source_fit["fit_terms"]):
            checks["fit_contract_preserved"] = False
            return {
                "accepted": False,
                "reason": "column_count_or_instantiation_mismatch",
                "checks": checks,
                "compute_cost": (budget.spent - start) if budget is not None else 0,
            }

        for cand_term, orig_term in zip(instantiated, source_fit["fit_terms"]):
            if _canonical_term_key(cand_term) != _canonical_term_key(orig_term):
                checks["fit_contract_preserved"] = False
                return {
                    "accepted": False,
                    "reason": "fit_contract_columns_not_preserved",
                    "checks": checks,
                    "compute_cost": (budget.spent - start) if budget is not None else 0,
                }

        # Check numerical column & reconstructed program agreement on support rows.
        rows = (support_rows_by_source or {}).get(source_id) or _synthetic_probe_rows(schema)
        eval_cost = len(rows) * (2 * len(instantiated) + 2)
        if budget is not None and not budget.spend(eval_cost):
            return {
                "accepted": False,
                "reason": "equivalence_check_compute_exhausted",
                "checks": checks,
                "compute_cost": budget.spent - start,
            }
        rebuilt_program = _sum_terms(
            instantiated, source_fit["coefficients"][1:], source_fit["coefficients"][0]
        )
        max_col_diff = 0.0
        for cand_term, orig_term in zip(instantiated, source_fit["fit_terms"]):
            for row in rows:
                max_col_diff = max(max_col_diff, abs(cand_term.predict(row) - orig_term.predict(row)))
        max_prog_diff = max(
            abs(rebuilt_program.predict(row) - source_fit["program"].predict(row)) for row in rows
        )
        if max_col_diff > tolerance or max_prog_diff > tolerance:
            checks["prediction_agreement"] = False
            return {
                "accepted": False,
                "reason": "prediction_disagreement_on_support_rows",
                "checks": checks,
                "compute_cost": (budget.spent - start) if budget is not None else 0,
            }

        # Check Rej(T) >= Rej(S) on any supplied counterexample trajectories.
        ce_rows = list((counterexample_rows_by_source or {}).get(source_id, ()))
        for item in source.get("counterexamples", ()):
            obs = item.get("observation")
            if isinstance(obs, dict) and "features" in obs and "target" in obs:
                ce_rows.append(
                    Observation(
                        dict(obs["features"]),
                        float(obs["target"]),
                        tuple(dict(past) for past in obs.get("history", ())),
                    )
                )
        source_ce_loss, cand_ce_loss = None, None
        if ce_rows:
            source_ce_loss = _basis_min_loss(source_fit["fit_terms"], ce_rows, budget)
            cand_ce_loss = _basis_min_loss(instantiated, ce_rows, budget)
            if source_ce_loss is None or cand_ce_loss is None:
                return {
                    "accepted": False,
                    "reason": "equivalence_check_compute_exhausted",
                    "checks": checks,
                    "compute_cost": budget.spent - start,
                }
            if source_ce_loss > tolerance and cand_ce_loss <= tolerance:
                checks["rejection_preserved"] = False
                return {
                    "accepted": False,
                    "reason": "rejection_set_shrunk_on_counterexamples",
                    "checks": checks,
                    "compute_cost": (budget.spent - start) if budget is not None else 0,
                }

        checks["source_details"].append(
            {
                "source_concept_id": source_id,
                "max_column_discrepancy": max_col_diff,
                "max_program_discrepancy": max_prog_diff,
                "source_counterexample_loss": source_ce_loss,
                "commitment_counterexample_loss": cand_ce_loss,
            }
        )

    return {
        "accepted": True,
        "reason": "strict_equivalence_preserved",
        "checks": checks,
        "compute_cost": (budget.spent - start) if budget is not None else 0,
    }


def consolidate_world_structure_commitments(
    priors: list[dict],
    *,
    budget: SearchBudget | None = None,
    support_rows_by_source: dict[str, list[Observation]] | None = None,
    counterexample_rows_by_source: dict[str, list[Observation]] | None = None,
    tolerance: float = 1e-9,
) -> list[dict]:
    """Consolidate confirmed whole-object priors into gated world commitments."""
    extracted = []
    for prior in priors:
        pattern = extract_parametric_role_pattern(prior)
        if pattern is not None:
            extracted.append(pattern)
    if not extracted:
        return []

    groups: dict[str, list[dict]] = {}
    for item in extracted:
        norm_role_terms = (
            _rebind_latent_pole_in_role_terms(item["role_terms"], "__LATENT_POLE__")
            if len(item.get("latent_poles") or ()) == 1
            else (
                _rebind_vdp_params_in_role_terms(
                    item["role_terms"], "__VDP_A1__", "__VDP_A2__", "__VDP_BETA__"
                )
                if item.get("vdp_params")
                else (
                    _rebind_ar2_params_in_role_terms(item["role_terms"], "__AR2_A1__", "__AR2_A2__")
                    if item.get("ar2_params")
                    else item["role_terms"]
                )
            )
        )
        # Also normalize 1D custom threshold inside all_relation_sum role_terms
        norm_role_terms_clean = deepcopy(norm_role_terms)
        for rt in norm_role_terms_clean:
            def _norm_rt(n_dict: dict):
                if n_dict.get("kind") == "all_relation_sum" and isinstance(n_dict.get("predicate"), dict):
                    n_dict["predicate"] = _normalize_edge_condition_for_grouping(n_dict["predicate"])
                for ch in n_dict.get("children", ()):
                    _norm_rt(ch)
            _norm_rt(rt)
        signature = json.dumps(
            {
                "role_terms": norm_role_terms_clean,
                "nonconservation_mechanism": item.get("nonconservation_mechanism"),
                "latent_state_mechanism": item.get("latent_state_mechanism"),
                "params": [
                    {
                        "symbol": p["symbol"],
                        "direction": p["direction"],
                        "edge_condition": _normalize_edge_condition_for_grouping(p["edge_condition"]),
                    }
                    for p in item["radius_parameters"]
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        groups.setdefault(signature, []).append(item)

    commitments = []
    for idx, members in enumerate(reversed(list(groups.values()))):
        representative = members[-1]
        merged_params = []
        for p_idx, param in enumerate(representative["radius_parameters"]):
            merged_params.append(
                {
                    "symbol": param["symbol"],
                    "direction": param["direction"],
                    "edge_condition": param["edge_condition"],
                    "binding": param["binding"],
                    "source_values": [m["radius_parameters"][p_idx]["source_value"] for m in members],
                    "source_intervals": [
                        interval
                        for m in members
                        for interval in m["radius_parameters"][p_idx].get("source_intervals", ())
                    ],
                }
            )
        merged_latent_poles = [
            pole
            for m in members
            for pole in (m.get("latent_poles") or ())
        ]
        merged_ar2_params = [
            list(m["ar2_params"])
            for m in members
            if m.get("ar2_params")
        ]
        merged_vdp_params = [
            list(m["vdp_params"])
            for m in members
            if m.get("vdp_params")
        ]
        merged_counterexamples = [
            deepcopy(ce)
            for m in members
            for ce in (m.get("counterexamples") or ())
        ]
        refuted_task_ids = list(
            dict.fromkeys(
                ce.get("task_id")
                for ce in merged_counterexamples
                if isinstance(ce, dict) and ce.get("task_id")
            )
        )
        candidate = {
            "commitment_id": f"world_commitment:{idx:04d}",
            "kind": "world_structure_commitment",
            "reorganization_type": "equivalence_reorganization",
            "source_concept_ids": list(dict.fromkeys(m["source_concept_id"] for m in members)),
            "source_task_ids": list(
                dict.fromkeys(m["source_task_id"] for m in members if m.get("source_task_id"))
            ),
            "source_statuses": {
                m["source_concept_id"]: m.get("source_status", "supported") for m in members
            },
            "refuted_task_ids": refuted_task_ids,
            "counterexamples": merged_counterexamples,
            "source_arities": sorted({m["source_arity"] for m in members}),
            "family": representative["family"],
            "family_id": representative["family_id"],
            "role_terms": deepcopy(representative["role_terms"]),
            "radius_parameters": merged_params,
            "support_premises": deepcopy(representative["support_premises"]),
            "declared_premises": deepcopy(representative["support_premises"]),
            "nonconservation_mechanism": representative.get("nonconservation_mechanism"),
            "latent_state_mechanism": representative.get("latent_state_mechanism"),
            "source_latent_poles": merged_latent_poles,
            "source_ar2_params": merged_ar2_params,
            "source_vdp_params": merged_vdp_params,
            "generation": max((int(m.get("generation", 0)) for m in members), default=0),
            "revision_lineage": deepcopy(representative.get("revision_lineage", [])),
        }
        audit = check_equivalence_reorganization(
            members,
            candidate,
            support_rows_by_source=support_rows_by_source,
            counterexample_rows_by_source=counterexample_rows_by_source,
            budget=budget,
            tolerance=tolerance,
        )
        if audit["accepted"]:
            candidate["equivalence_audit"] = audit
            commitments.append(candidate)

    # Prefer latest scientifically evolved generation first, then more independent sources.
    commitments.sort(
        key=lambda item: (int(item.get("generation", 0)), len(item["source_concept_ids"])),
        reverse=True,
    )
    return commitments


def _directional_relation_distances(
    rows: list[Observation],
    schema: ObjectSchema,
    direction: str,
    budget: SearchBudget,
    *,
    edge_condition: str | dict | Program = "unconditional",
) -> tuple[list[float], bool]:
    """Compute identifiable pairwise distances under ``direction`` & ``edge_condition``."""
    names = tuple(schema.feature_names)
    frames = {
        tuple(frame[name] for name in names): frame
        for row in rows
        for frame in (row.features, *row.history[-4:])
    }
    distances: list[float] = []
    complete = True
    if direction in {"right", "left"}:
        sign = 1.0 if direction == "right" else -1.0
        for source in schema.objects:
            for target in schema.objects:
                if source == target:
                    continue
                if not budget.spend(len(frames)):
                    complete = False
                    break
                fx_s, x_s, q_s = (schema.fields[source][r] for r in ("f", "x", "q"))
                fx_t, x_t, q_t = (schema.fields[target][r] for r in ("f", "x", "q"))
                for frame in frames.values():
                    delta_x = sign * (frame[x_t] - frame[x_s])
                    if (
                        delta_x > 0.0
                        and (frame[fx_t] - frame[fx_s]) * frame[q_s] != 0.0
                        and _edge_condition_holds(
                            edge_condition, frame[x_s], frame[q_s], frame[x_t], frame[q_t]
                        )
                    ):
                        distances.append(delta_x)
            if not complete:
                break
    else:
        for left, right in combinations(schema.objects, 2):
            if not budget.spend(len(frames)):
                complete = False
                break
            fl, xl, ql = (schema.fields[left][r] for r in ("f", "x", "q"))
            fr, xr, qr = (schema.fields[right][r] for r in ("f", "x", "q"))
            for frame in frames.values():
                if (frame[fl] - frame[fr]) * (frame[ql] - frame[qr]) != 0.0:
                    dist = abs(frame[xl] - frame[xr])
                    if dist > 0.0:
                        distances.append(dist)
    return sorted(set(distances)), complete


def _candidate_radii_from_distances(
    distances: list[float],
    *,
    radius_bounds: tuple[float, float] | list[float] | None = None,
    prior_values: list[float] | tuple[float, ...] = (),
) -> list[float]:
    """Select coarse-to-fine candidate radii, prioritizing verified ``radius_bounds``."""
    if not distances and not radius_bounds and not prior_values:
        return []
    boundaries = [(left + right) / 2.0 for left, right in zip(distances, distances[1:])]
    ordered = []

    def add(val: float):
        if val > 0.0 and all(abs(val - existing) > 1e-12 for existing in ordered):
            ordered.append(float(val))

    if radius_bounds is not None:
        low, high = float(radius_bounds[0]), float(radius_bounds[1])
        if high > low:
            add((low + high) / 2.0)
            add(low + 0.25 * (high - low))
            add(low + 0.75 * (high - low))
            for boundary in boundaries:
                if low <= boundary <= high:
                    add(boundary)
        elif low > 0.0:
            add(low)

    for value in reversed(list(prior_values)):
        if radius_bounds is None or (radius_bounds[0] - 1e-9 <= value <= radius_bounds[1] + 1e-9):
            add(float(value))

    if radius_bounds is None:
        if boundaries:
            samples = min(16, len(boundaries))
            for idx in range(samples):
                pos = round(idx * (len(boundaries) - 1) / max(1, samples - 1))
                add(boundaries[pos])
        if distances:
            add(distances[-1] + 1e-6)
            if distances[0] > 1e-6:
                add(distances[0] * 0.5)
    return ordered


def _attach_symbolic_parameter_step(
    entry: dict,
    commitment: dict,
    schema: ObjectSchema,
    coefficients: list[float],
    fitted_radius: float | None,
    radius_interval: list[float] | None,
    direction_overrides: dict[str, str] | None = None,
    predicate_overrides: dict[str, dict | str | Program | None] | None = None,
    role_terms_override: list[dict] | None = None,
    *,
    threshold_interval: list[float] | None = None,
    active_threshold_probing: bool = True,
) -> dict:
    """Attach ``resolve_relation_intervals`` metadata for radii and open-grammar thresholds."""
    params = commitment.get("radius_parameters", ())
    if not params or fitted_radius is None:
        return entry
    overrides = direction_overrides or {}
    pred_overrides = dict(predicate_overrides or {})
    symbolic_pred_overrides = dict(pred_overrides)
    step_params = []
    intervals = [list(radius_interval)] if radius_interval else []
    threshold_step_params = []

    for p_idx, param in enumerate(params):
        sym = param["symbol"]
        eff_cond = (
            pred_overrides[sym]
            if sym in pred_overrides
            else param.get("edge_condition", "unconditional")
        )
        pred_prog = edge_condition_predicate(eff_cond)
        symbolic_pred_prog = pred_prog
        if (
            active_threshold_probing
            and pred_prog != TRUE
            and threshold_interval is not None
            and len(threshold_interval) == 2
            and float(threshold_interval[1]) > float(threshold_interval[0])
        ):
            from .curved_hypersurface_probing import describe_linear_in_params_hypersurface

            desc = (
                None
                if describe_linear_in_params_hypersurface(pred_prog) is not None
                else describe_comparator_threshold_parameter(pred_prog)
            )
            if desc is not None:
                c_sym = f"c_{p_idx}"
                symbolic_pred_prog = _replace_node_at_path(
                    pred_prog, tuple(desc["path"]), Program("arg", name=c_sym)
                )
                symbolic_pred_overrides[sym] = symbolic_pred_prog
                t_iv = [[float(threshold_interval[0]), float(threshold_interval[1])]]
                threshold_step_params.append(
                    {
                        "symbol": c_sym,
                        "parameter_kind": "predicate_threshold",
                        "radius": float(desc["initial_value"]),
                        "radius_intervals": deepcopy(t_iv),
                        "prior_radius_intervals": deepcopy(t_iv),
                        "comparator_lhs": deepcopy(desc["comparator_lhs"]),
                        "comparator_monotonicity": desc["comparator_monotonicity"],
                        "relation": overrides.get(sym, param["direction"]),
                        "source_concept_id": None,
                    }
                )
        step_param = {
            "symbol": sym,
            "parameter_kind": "radius",
            "radius": float(fitted_radius),
            "radius_intervals": deepcopy(intervals),
            "prior_radius_intervals": deepcopy(intervals),
            "relation": overrides.get(sym, param["direction"]),
            "source_concept_id": commitment["source_concept_ids"][-1]
            if commitment.get("source_concept_ids")
            else None,
        }
        if pred_prog != TRUE:
            step_param["predicate"] = pred_prog.to_dict()
        step_params.append(step_param)

    step_params.extend(threshold_step_params)
    symbolic_terms = instantiate_commitment_terms(
        commitment,
        schema,
        direction_overrides=direction_overrides,
        predicate_overrides=symbolic_pred_overrides,
        role_terms_override=role_terms_override,
        symbolic_radii=True,
    )
    if symbolic_terms is None:
        return entry
    symbolic_program = _sum_terms(
        list(symbolic_terms), coefficients[1:], coefficients[0]
    )
    plan = entry["research_program"]
    plan.setdefault("selection_steps", []).append(
        {
            "operation": "resolve_relation_intervals",
            "question": "Instantiate arity-parametric world commitment parameters on current object view",
            "parameter_template": symbolic_program.to_dict(),
            "parameter_terms": [term.to_dict() for term in symbolic_terms],
            "parameters": step_params,
            "evidence_contexts": [],
            "compact_replacement": False,
            "program": entry["candidate"].program.to_dict(),
            "loss": entry["training_loss"],
        }
    )
    return entry


def _optimize_1d_latent_pole(
    eval_pole_fn,
    *,
    seed_poles: tuple[float, ...] = (),
    tolerance: float = 1e-8,
):
    """Coarse-to-fine 1D continuous decay pole search (reusing continuous_state bracket logic)."""
    grid = list(dict.fromkeys([*seed_poles, 0.3, 0.5, 0.65, 0.8, 0.92, -0.4]))
    scored: dict[float, tuple[float, object, object]] = {}
    for p in grid:
        if not (-0.98 <= p <= 0.995):
            continue
        res = eval_pole_fn(float(p))
        if res is None:
            break
        scored[float(p)] = res
        if res[0] <= tolerance * 0.1:
            return float(p), res
    if not scored:
        return None
    ordered = sorted(scored)
    best_p = min(scored, key=lambda k: scored[k][0])
    best_res = scored[best_p]
    idx = ordered.index(best_p)
    lo = ordered[idx - 1] if idx > 0 else max(-0.95, best_p - 0.25)
    hi = ordered[idx + 1] if idx + 1 < len(ordered) else min(0.99, best_p + 0.2)
    gr = (sqrt(5.0) - 1.0) / 2.0
    c1 = hi - gr * (hi - lo)
    c2 = lo + gr * (hi - lo)
    r1 = eval_pole_fn(c1)
    r2 = eval_pole_fn(c2)
    if r1 is None or r2 is None:
        return best_p, best_res
    for r_cand, p_cand in ((r1, c1), (r2, c2)):
        if r_cand[0] < best_res[0]:
            best_p, best_res = p_cand, r_cand
    for _ in range(36):
        if best_res[0] <= min(1e-16, tolerance * 1e-7) or hi - lo < 1e-10:
            break
        if hi - lo < 0.025 and best_res[0] > 5e-3:
            break
        if r1[0] < r2[0]:
            hi, c2, r2 = c2, c1, r1
            c1 = hi - gr * (hi - lo)
            r_next = eval_pole_fn(c1)
            if r_next is None:
                break
            r1 = r_next
            if r1[0] < best_res[0]:
                best_p, best_res = c1, r1
        else:
            lo, c1, r1 = c1, c2, r2
            c2 = lo + gr * (hi - lo)
            r_next = eval_pole_fn(c2)
            if r_next is None:
                break
            r2 = r_next
            if r2[0] < best_res[0]:
                best_p, best_res = c2, r2
    return best_p, best_res


def _optimize_2d_ar2_latent_poles(
    eval_ar2_fn,
    *,
    seed_pairs: tuple[tuple[float, float], ...] | list[list[float]] = (),
    tolerance: float = 1e-8,
):
    """Search 2nd-order AR(2) parameters ``(a1, a2)`` in decoupled ``(rho, cos_w)`` + Cartesian coordinates."""
    gr = (5.0 ** 0.5 - 1.0) / 2.0
    coarse_polar = [
        (rho, cw)
        for rho in (0.50, 0.64, 0.76, 0.87)
        for cw in (-0.30, 0.15, 0.52, 0.82)
    ]
    best_a1, best_a2, best_res = None, None, None
    best_rho, best_cw = 0.68, 0.55

    for rho_c, cw_c in coarse_polar:
        a1_c, a2_c = 2.0 * rho_c * cw_c, -(rho_c * rho_c)
        res_c = eval_ar2_fn(a1_c, a2_c)
        if res_c is None:
            return ((best_a1, best_a2), best_res) if best_res is not None else None
        if best_res is None or res_c[0] < best_res[0]:
            best_a1, best_a2, best_res = a1_c, a2_c, res_c
            best_rho, best_cw = rho_c, cw_c
            if best_res[0] <= min(1e-18, tolerance * 1e-8):
                return (best_a1, best_a2), best_res

    for pair in seed_pairs:
        if len(pair) == 2:
            a1_s, a2_s = float(pair[0]), float(pair[1])
            res_s = eval_ar2_fn(a1_s, a2_s)
            if res_s is None:
                break
            if best_res is None or res_s[0] < best_res[0]:
                best_a1, best_a2, best_res = a1_s, a2_s, res_s
                if a2_s < -1e-6:
                    best_rho = min(0.985, max(0.05, (-a2_s) ** 0.5))
                    best_cw = max(-0.98, min(0.98, a1_s / (2.0 * best_rho)))

    if best_res is None:
        return None

    sp_rho, sp_cw = 0.16, 0.28
    for rnd in range(8):
        if best_res[0] <= min(1e-18, tolerance * 1e-8):
            break
        if rnd == 2 and best_res[0] > 5e-2:
            break
        lo_r, hi_r = max(0.08, best_rho - sp_rho), min(0.985, best_rho + sp_rho)
        c1 = hi_r - gr * (hi_r - lo_r)
        c2 = lo_r + gr * (hi_r - lo_r)
        r1 = eval_ar2_fn(2.0 * c1 * best_cw, -(c1 * c1))
        r2 = eval_ar2_fn(2.0 * c2 * best_cw, -(c2 * c2))
        if r1 is None or r2 is None:
            break
        if r1[0] < best_res[0]:
            best_rho, best_a1, best_a2, best_res = c1, 2.0 * c1 * best_cw, -(c1 * c1), r1
        if r2[0] < best_res[0]:
            best_rho, best_a1, best_a2, best_res = c2, 2.0 * c2 * best_cw, -(c2 * c2), r2
        for _ in range(12):
            if hi_r - lo_r < 1e-10:
                break
            if r1[0] < r2[0]:
                hi_r, c2, r2 = c2, c1, r1
                c1 = hi_r - gr * (hi_r - lo_r)
                rn = eval_ar2_fn(2.0 * c1 * best_cw, -(c1 * c1))
                if rn is None:
                    break
                r1 = rn
                if r1[0] < best_res[0]:
                    best_rho, best_a1, best_a2, best_res = c1, 2.0 * c1 * best_cw, -(c1 * c1), r1
            else:
                lo_r, c1, r1 = c1, c2, r2
                c2 = lo_r + gr * (hi_r - lo_r)
                rn = eval_ar2_fn(2.0 * c2 * best_cw, -(c2 * c2))
                if rn is None:
                    break
                r2 = rn
                if r2[0] < best_res[0]:
                    best_rho, best_a1, best_a2, best_res = c2, 2.0 * c2 * best_cw, -(c2 * c2), r2

        lo_w, hi_w = max(-0.985, best_cw - sp_cw), min(0.985, best_cw + sp_cw)
        d1 = hi_w - gr * (hi_w - lo_w)
        d2 = lo_w + gr * (hi_w - lo_w)
        a2_fix = -(best_rho * best_rho)
        s1 = eval_ar2_fn(2.0 * best_rho * d1, a2_fix)
        s2 = eval_ar2_fn(2.0 * best_rho * d2, a2_fix)
        if s1 is None or s2 is None:
            break
        if s1[0] < best_res[0]:
            best_cw, best_a1, best_a2, best_res = d1, 2.0 * best_rho * d1, a2_fix, s1
        if s2[0] < best_res[0]:
            best_cw, best_a1, best_a2, best_res = d2, 2.0 * best_rho * d2, a2_fix, s2
        for _ in range(12):
            if hi_w - lo_w < 1e-10:
                break
            if s1[0] < s2[0]:
                hi_w, d2, s2 = d2, d1, s1
                d1 = hi_w - gr * (hi_w - lo_w)
                sn = eval_ar2_fn(2.0 * best_rho * d1, a2_fix)
                if sn is None:
                    break
                s1 = sn
                if s1[0] < best_res[0]:
                    best_cw, best_a1, best_a2, best_res = d1, 2.0 * best_rho * d1, a2_fix, s1
            else:
                lo_w, d1, s1 = d1, d2, s2
                d2 = lo_w + gr * (hi_w - lo_w)
                sn = eval_ar2_fn(2.0 * best_rho * d2, a2_fix)
                if sn is None:
                    break
                s2 = sn
                if s2[0] < best_res[0]:
                    best_cw, best_a1, best_a2, best_res = d2, 2.0 * best_rho * d2, a2_fix, s2
        sp_rho *= 0.52
        sp_cw *= 0.52

    return (best_a1, best_a2), best_res


def _fast_solve_1d_focal_pole(
    rows: list[Observation],
    target: ObjectSchema,
    base_terms: tuple[Program, ...],
    budget: SearchBudget,
    *,
    seed_poles: tuple[float, ...] = (),
    tolerance: float = 1e-8,
) -> tuple[float, float] | None:
    """Fast Frisch-Waugh-Lovell projected 1D solver for per-object 1st-order focal readout."""
    m_count = len(rows)
    if m_count < 5 or not budget.spend(m_count * (len(base_terms) + 2)):
        return None
    base_cols = [[1.0] * m_count] + [[float(bt.predict(r)) for r in rows] for bt in base_terms]
    orth_vecs: list[list[float]] = []
    for col in base_cols:
        v = list(col)
        for qv in orth_vecs:
            dot = sum(a * b for a, b in zip(v, qv))
            v = [vi - dot * qvi for vi, qvi in zip(v, qv)]
        norm_sq = sum(vi * vi for vi in v)
        if norm_sq > 1e-24:
            inv_n = 1.0 / (norm_sq ** 0.5)
            orth_vecs.append([vi * inv_n for vi in v])

    y_perp = [float(r.target) for r in rows]
    for qv in orth_vecs:
        dot = sum(a * b for a, b in zip(y_perp, qv))
        y_perp = [yi - dot * qvi for yi, qvi in zip(y_perp, qv)]

    row_seqs: list[tuple[tuple[float, ...], tuple[float, ...]] | None] = []
    for r in rows:
        foc_g = next(
            (
                g
                for g in target.objects
                if abs(float(r.features.get(target.fields[g]["f"], 0.0)) - 1.0) <= 1e-6
            ),
            None,
        )
        if foc_g is None:
            row_seqs.append(None)
            continue
        qf = target.fields[foc_g]["q"]
        uf = target.fields[foc_g]["u"]
        qs = tuple(float(h.get(qf, 0.0)) for h in r.history) + (float(r.features[qf]),)
        us = tuple(float(h.get(uf, 0.0)) for h in r.history) + (float(r.features[uf]),)
        row_seqs.append((qs, us))

    def _eval_fast_1d(p_val: float):
        if not budget.spend(1):
            return None
        zq_col = []
        zu_col = []
        for item in row_seqs:
            if item is None:
                zq_col.append(0.0)
                zu_col.append(0.0)
            else:
                qs, us = item
                zc_q = 0.0
                for dv in qs:
                    zc_q = p_val * zc_q + dv
                zc_u = 0.0
                for dv in us:
                    zc_u = p_val * zc_u + dv
                zq_col.append(zc_q)
                zu_col.append(zc_u)
        for qv in orth_vecs:
            dq = sum(a * b for a, b in zip(zq_col, qv))
            du = sum(a * b for a, b in zip(zu_col, qv))
            zq_col = [zqi - dq * qvi for zqi, qvi in zip(zq_col, qv)]
            zu_col = [zui - du * qvi for zui, qvi in zip(zu_col, qv)]
        g00 = sum(v * v for v in zq_col) + 1e-18
        g01 = sum(a * b for a, b in zip(zq_col, zu_col))
        g11 = sum(v * v for v in zu_col) + 1e-18
        b0 = sum(a * y for a, y in zip(zq_col, y_perp))
        b1 = sum(a * y for a, y in zip(zu_col, y_perp))
        det = g00 * g11 - g01 * g01
        if abs(det) <= 1e-28:
            wq, wu = b0 / g00, 0.0
        else:
            wq = (g11 * b0 - g01 * b1) / det
            wu = (g00 * b1 - g01 * b0) / det
        mse = sum((y_perp[i] - wq * zq_col[i] - wu * zu_col[i]) ** 2 for i in range(m_count)) / m_count
        return (mse, None, None)

    solved = _optimize_1d_latent_pole(_eval_fast_1d, seed_poles=seed_poles, tolerance=tolerance)
    if solved is None:
        return None
    fpole, (fmse, _, _) = solved
    return float(fpole), float(fmse)


def _fast_solve_ar2_focal_poles(
    rows: list[Observation],
    target: ObjectSchema,
    base_terms: tuple[Program, ...],
    budget: SearchBudget,
    *,
    seed_pairs: tuple[tuple[float, float], ...] | list[list[float]] = (),
    tolerance: float = 1e-8,
) -> tuple[float, float, float] | None:
    """Fast Frisch-Waugh-Lovell projected 2D solver for per-object AR(2) focal readout."""
    m_count = len(rows)
    if m_count < 6 or not budget.spend(m_count * (len(base_terms) + 2)):
        return None
    base_cols = [[1.0] * m_count] + [[float(bt.predict(r)) for r in rows] for bt in base_terms]
    orth_vecs: list[list[float]] = []
    for col in base_cols:
        v = list(col)
        for qv in orth_vecs:
            dot = sum(a * b for a, b in zip(v, qv))
            v = [vi - dot * qvi for vi, qvi in zip(v, qv)]
        norm_sq = sum(vi * vi for vi in v)
        if norm_sq > 1e-24:
            inv_n = 1.0 / (norm_sq ** 0.5)
            orth_vecs.append([vi * inv_n for vi in v])

    y_perp = [float(r.target) for r in rows]
    for qv in orth_vecs:
        dot = sum(a * b for a, b in zip(y_perp, qv))
        y_perp = [yi - dot * qvi for yi, qvi in zip(y_perp, qv)]

    row_seqs: list[tuple[tuple[float, ...], tuple[float, ...]] | None] = []
    for r in rows:
        foc_g = next(
            (
                g
                for g in target.objects
                if abs(float(r.features.get(target.fields[g]["f"], 0.0)) - 1.0) <= 1e-6
            ),
            None,
        )
        if foc_g is None:
            row_seqs.append(None)
            continue
        qf = target.fields[foc_g]["q"]
        uf = target.fields[foc_g]["u"]
        qs = tuple(float(h.get(qf, 0.0)) for h in r.history) + (float(r.features[qf]),)
        us = tuple(float(h.get(uf, 0.0)) for h in r.history) + (float(r.features[uf]),)
        row_seqs.append((qs, us))

    def _eval_fast(a1_v: float, a2_v: float):
        if not budget.spend(2):
            return None
        zq_col = []
        zu_col = []
        for item in row_seqs:
            if item is None:
                zq_col.append(0.0)
                zu_col.append(0.0)
            else:
                qs, us = item
                zc_q, zp_q = 0.0, 0.0
                for dv in qs:
                    zn_q = a1_v * zc_q + a2_v * zp_q + dv
                    zp_q, zc_q = zc_q, zn_q
                zc_u, zp_u = 0.0, 0.0
                for dv in us:
                    zn_u = a1_v * zc_u + a2_v * zp_u + dv
                    zp_u, zc_u = zc_u, zn_u
                zq_col.append(zc_q)
                zu_col.append(zc_u)
        for qv in orth_vecs:
            dq = sum(a * b for a, b in zip(zq_col, qv))
            du = sum(a * b for a, b in zip(zu_col, qv))
            zq_col = [zqi - dq * qvi for zqi, qvi in zip(zq_col, qv)]
            zu_col = [zui - du * qvi for zui, qvi in zip(zu_col, qv)]
        g00 = sum(v * v for v in zq_col) + 1e-18
        g01 = sum(a * b for a, b in zip(zq_col, zu_col))
        g11 = sum(v * v for v in zu_col) + 1e-18
        b0 = sum(a * y for a, y in zip(zq_col, y_perp))
        b1 = sum(a * y for a, y in zip(zu_col, y_perp))
        det = g00 * g11 - g01 * g01
        if abs(det) <= 1e-28:
            wq, wu = b0 / g00, 0.0
        else:
            wq = (g11 * b0 - g01 * b1) / det
            wu = (g00 * b1 - g01 * b0) / det
        mse = sum((y_perp[i] - wq * zq_col[i] - wu * zu_col[i]) ** 2 for i in range(m_count)) / m_count
        return (mse,)

    solved = _optimize_2d_ar2_latent_poles(
        _eval_fast, seed_pairs=seed_pairs, tolerance=tolerance
    )
    if solved is None:
        return None
    (fa1, fa2), (fmse,) = solved
    return float(fa1), float(fa2), float(fmse)


def instantiate_commitment_on_schema(
    spec,
    training: list[Observation],
    schema: ObjectSchema | dict | str,
    commitment: dict,
    budget: SearchBudget,
    *,
    radius_bounds: tuple[float, float] | list[float] | None = None,
    threshold_interval: list[float] | None = None,
    direction_overrides: dict[str, str] | None = None,
    predicate_overrides: dict[str, dict | str | Program | None] | None = None,
    role_terms_override: list[dict] | None = None,
    nonconservation_mechanism: str | None = None,
    latent_state_mechanism: str | None = None,
    reorganization_type: str = "equivalence_reorganization",
    scientific_revision: dict | None = None,
    audit_report: dict | None = None,
    transitions: list[dict] | tuple[dict, ...] = (),
    active_threshold_probing: bool = True,
    allow_latent_state: bool = True,
    allow_higher_order_latent: bool = True,
    tolerance: float = 1e-8,
) -> list[dict]:
    """Instantiate and fit ``commitment`` onto ``schema`` under ``SearchBudget``."""
    target = _as_schema(schema)
    rows = list(training)
    if target is None or not rows or not set(target.feature_names).issubset(spec.feature_names):
        return []
    start = budget.spent
    fitter = _Fitter(rows, budget, spec.risk_quantile)
    params = commitment.get("radius_parameters", ())
    overrides = dict(direction_overrides or {})
    pred_overrides = dict(predicate_overrides or {})
    effective_noncons = (
        nonconservation_mechanism
        if nonconservation_mechanism is not None
        else commitment.get("nonconservation_mechanism")
    )
    effective_latent_mech = (
        latent_state_mechanism
        if latent_state_mechanism is not None
        else commitment.get("latent_state_mechanism")
    )
    active_role_terms = (
        deepcopy(role_terms_override)
        if role_terms_override is not None
        else deepcopy(commitment.get("role_terms", ()))
    )
    transition_pole_calibrated = False
    transition_ar2_calibrated = False
    if (
        effective_latent_mech in {"transition_memory", "coupled_transition_memory"}
        and audit_report
        and audit_report.get("latent_transition_state")
    ):
        audit_pole = float(audit_report["latent_transition_state"]["pole"])
        active_role_terms = _rebind_latent_pole_in_role_terms(active_role_terms, audit_pole)
        transition_pole_calibrated = True
    elif (
        effective_latent_mech == "oscillatory_transition_memory"
        and audit_report
        and (audit_report.get("latent_transition_state") or {}).get("ar2_params")
    ):
        ar2_p = audit_report["latent_transition_state"]["ar2_params"]
        active_role_terms = _rebind_ar2_params_in_role_terms(
            active_role_terms, float(ar2_p[0]), float(ar2_p[1])
        )
        transition_ar2_calibrated = True
    elif (
        effective_latent_mech == "van_der_pol_transition_memory"
        and audit_report
        and (audit_report.get("latent_transition_state") or {}).get("vdp_params")
    ):
        vdp_p = audit_report["latent_transition_state"]["vdp_params"]
        active_role_terms = _rebind_vdp_params_in_role_terms(
            active_role_terms, float(vdp_p[0]), float(vdp_p[1]), float(vdp_p[2])
        )
        transition_ar2_calibrated = True
    elif (
        effective_latent_mech == "custom_transition_memory"
        and audit_report
        and audit_report.get("status") == "unexplained_latent_transition_state"
    ):
        # Cross-scale transfer of an open-grammar custom transition-memory commitment:
        # extract the custom drive AST inside accumulate and recalibrate pole + deflated radius_bounds!
        from .premise_dependency_audit import screen_custom_latent_transition_drive

        custom_drive_prog = None
        for rt_item in active_role_terms:
            if rt_item.get("kind") == "all_object_sum":
                s_dict = rt_item.get("summand") or {}
                if s_dict.get("op") == "accumulate" and s_dict.get("name") is None:
                    custom_drive_prog = Program.from_dict(s_dict["children"][0])
                    break
        lat_pay = audit_report.get("transition_latent_payload") or {}
        if custom_drive_prog is not None and lat_pay.get("valid_rows"):
            sc_x_to_q = {target.fields[g]["x"]: target.fields[g]["q"] for g in target.objects}
            dyn_l = lat_pay["dynamic"]
            xf_l = lat_pay["x_fields"]
            x_i_map = {xn: idx_x for idx_x, xn in enumerate(xf_l)}
            if set(sc_x_to_q.values()) == set(dyn_l) and set(sc_x_to_q.keys()) == set(xf_l):
                bij_target = tuple(
                    x_i_map[next(xf for xf, qf in sc_x_to_q.items() if qf == qn)] for qn in dyn_l
                )
                scr_c = screen_custom_latent_transition_drive(
                    lat_pay["valid_rows"],
                    dyn_l,
                    lat_pay["drive_by_q"],
                    xf_l,
                    [bij_target],
                    custom_drive_prog,
                    (lat_pay.get("representative_direction") or "right",),
                    budget=budget,
                    eff_tol=float(lat_pay.get("eff_tol", max(1e-12, tolerance))),
                    observations=rows,
                )
                if scr_c.get("supported") and scr_c.get("verified_pole") is not None:
                    active_role_terms = _rebind_latent_pole_in_role_terms(
                        active_role_terms, float(scr_c["verified_pole"])
                    )
                    transition_pole_calibrated = True
                    compat_m = [
                        m for m in scr_c.get("matches", ()) if m.get("mapping") == sc_x_to_q
                    ]
                    if compat_m:
                        radius_bounds = [
                            float(min(m["radius_interval"][0] for m in compat_m)),
                            float(max(m["radius_interval"][1] for m in compat_m)),
                        ]

    eff_threshold_interval = list(threshold_interval) if threshold_interval else None
    if audit_report and eff_threshold_interval is None and audit_report.get("verified_threshold_interval"):
        eff_threshold_interval = list(audit_report["verified_threshold_interval"])

    if not params:
        terms = instantiate_commitment_terms(
            commitment, target, role_terms_override=active_role_terms
        )
        if terms is None:
            return []
        fit = fitter.fit(terms)
        if fit is None:
            return []
        return [
            _build_commitment_entry(
                spec=spec,
                rows=rows,
                schema=target,
                commitment=commitment,
                terms=terms,
                fit=fit,
                fitted_radius=None,
                radius_interval=None,
                threshold_interval=None,
                active_threshold_probing=active_threshold_probing,
                complete=True,
                compute_cost=budget.spent - start,
                reorganization_type=reorganization_type,
                scientific_revision=scientific_revision,
                direction_overrides=overrides,
                predicate_overrides=pred_overrides,
                role_terms_override=active_role_terms,
                nonconservation_mechanism=effective_noncons,
                latent_state_mechanism=effective_latent_mech,
                tolerance=tolerance,
            )
        ]

    # Single shared radius symbol is standard across structural discoveries; if
    # multiple symbols exist, keep secondary symbols at source values initially
    # and refine the primary symbol over target observed distances.
    primary = params[0]
    primary_sym = primary["symbol"]
    effective_direction = overrides.get(primary_sym, primary["direction"])
    if (
        primary_sym not in pred_overrides
        and audit_report
        and isinstance(audit_report.get("verified_edge_condition"), dict)
        and isinstance(primary.get("edge_condition"), dict)
    ):
        pred_overrides[primary_sym] = audit_report["verified_edge_condition"]
    effective_cond = (
        pred_overrides[primary_sym]
        if primary_sym in pred_overrides
        else primary.get("edge_condition", "unconditional")
    )
    # If schema transitions were supplied directly (e.g. UnifiedObjectLab without object induction),
    # calibrate 1D custom threshold & extract its exact uncertainty interval on transitions.
    if (
        isinstance(effective_cond, (dict, Program))
        and eff_threshold_interval is None
        and transitions
        and effective_direction in {"right", "left"}
    ):
        from .curved_hypersurface_probing import describe_linear_in_params_hypersurface

        custom_prog = edge_condition_predicate(effective_cond)
        if (
            describe_comparator_threshold_parameter(custom_prog) is not None
            or describe_linear_in_params_hypersurface(custom_prog) is not None
        ):
            q_fields = [target.fields[g]["q"] for g in target.objects]
            u_fields = [target.fields[g]["u"] for g in target.objects]
            x_fields = [target.fields[g]["x"] for g in target.objects]
            q_r, x_r, d_r = [], [], []
            for tr in transitions:
                bef, aft = tr.get("before", {}), tr.get("after", {})
                if all(f in bef for f in (*q_fields, *u_fields, *x_fields)) and all(
                    f in aft for f in q_fields
                ):
                    q_r.append([float(bef[f]) for f in q_fields])
                    x_r.append([float(bef[f]) for f in x_fields])
                    d_r.append(
                        [float(aft[qf]) - float(bef[qf]) - float(bef[uf]) for qf, uf in zip(q_fields, u_fields)]
                    )
            if q_r:
                id_bij = tuple(range(len(target.objects)))
                cal_res = screen_custom_edge_predicate_on_transitions(
                    q_r,
                    x_r,
                    d_r,
                    q_fields,
                    x_fields,
                    [id_bij],
                    custom_prog,
                    (effective_direction,),
                    budget=budget,
                    eff_tol=max(1e-12, tolerance),
                )
                if cal_res.get("supported"):
                    pred_overrides[primary_sym] = cal_res["calibrated_predicate"].to_dict()
                    effective_cond = pred_overrides[primary_sym]
                    if cal_res.get("threshold_interval"):
                        eff_threshold_interval = list(cal_res["threshold_interval"])

    eff_cond_desc = describe_edge_condition(effective_cond)
    distances, distance_complete = _directional_relation_distances(
        rows, target, effective_direction, budget, edge_condition=eff_cond_desc
    )
    boundaries = [(left + right) / 2.0 for left, right in zip(distances, distances[1:])]
    candidates = _candidate_radii_from_distances(
        distances,
        radius_bounds=radius_bounds,
        prior_values=primary.get("source_values", ()),
    )
    if not candidates:
        fallback_val = (primary.get("source_values") or [1.0])[-1]
        candidates = [float(fallback_val)]

    evaluated: list[tuple[float, tuple[Program, ...], dict]] = []
    tested_radii: set[float] = set()

    def try_radius(r_val: float, terms_override: list[dict] | None = None):
        use_rt = active_role_terms if terms_override is None else terms_override
        if terms_override is None and any(abs(r_val - prev) <= 1e-12 for prev in tested_radii):
            return None
        if terms_override is None:
            tested_radii.add(r_val)
        bindings = {
            p["symbol"]: (r_val if p["symbol"] == primary_sym else float(p["source_values"][-1]))
            for p in params
        }
        terms = instantiate_commitment_terms(
            commitment,
            target,
            radius_bindings=bindings,
            direction_overrides=overrides,
            predicate_overrides=pred_overrides,
            role_terms_override=use_rt,
        )
        if terms is None:
            return None
        fitted = fitter.fit(terms)
        if fitted is None:
            return None
        record = (r_val, terms, fitted, deepcopy(use_rt))
        if terms_override is None:
            evaluated.append(record)
        return record

    existing_poles = _extract_latent_poles_from_role_terms(active_role_terms)
    existing_ar2_params = _extract_ar2_params_from_role_terms(active_role_terms)
    max_hist = max((len(r.history) for r in rows), default=0)
    recalibrated_existing_poles = transition_pole_calibrated
    recalibrated_existing_ar2 = transition_ar2_calibrated
    best_record = None
    for idx, radius in enumerate(candidates):
        rec = try_radius(radius)
        if rec is None and budget.denied_requests:
            distance_complete = False
            break
        if rec is not None and (best_record is None or rec[2]["loss"] < best_record[2]["loss"]):
            best_record = rec
        # When radius_bounds is already verified from transitions and active_role_terms carries
        # an uncalibrated source decay pole or AR(2) pair, recalibrate at r_mid BEFORE scanning outer radii.
        if (
            idx == 0
            and radius_bounds is not None
            and existing_poles
            and not recalibrated_existing_poles
            and best_record is not None
            and best_record[2]["loss"] > tolerance
            and max_hist >= 2
        ):
            recalibrated_existing_poles = True
            seed_poles = tuple(commitment.get("source_latent_poles", ())) or tuple(existing_poles)

            def _eval_early_pole(p_val: float, _r=radius):
                cand_rt = _rebind_latent_pole_in_role_terms(active_role_terms, p_val)
                rec_p = try_radius(_r, terms_override=cand_rt)
                if rec_p is None:
                    return None
                return rec_p[2]["loss"], cand_rt, rec_p

            opt_early = _optimize_1d_latent_pole(
                _eval_early_pole, seed_poles=seed_poles, tolerance=tolerance
            )
            if opt_early is not None and opt_early[1][0] < best_record[2]["loss"]:
                _, (_, best_rt_early, rec_opt_early) = opt_early
                active_role_terms = best_rt_early
                best_record = rec_opt_early
        elif (
            idx == 0
            and radius_bounds is not None
            and existing_ar2_params
            and not recalibrated_existing_ar2
            and best_record is not None
            and best_record[2]["loss"] > tolerance
            and max_hist >= 2
        ):
            recalibrated_existing_ar2 = True
            seed_ar2 = tuple(tuple(p) for p in (commitment.get("source_ar2_params") or existing_ar2_params))
            if effective_latent_mech == "oscillatory_focal_memory":
                non_ar2_rt = [
                    rt_i
                    for rt_i in active_role_terms
                    if not (
                        rt_i.get("kind") == "all_object_sum"
                        and _collect_prog_ar2_params(rt_i.get("summand") or {})
                    )
                ]
                base_t_early = instantiate_commitment_terms(
                    commitment,
                    target,
                    radius_bindings={
                        p["symbol"]: (radius if p["symbol"] == primary_sym else float(p["source_values"][-1]))
                        for p in params
                    },
                    direction_overrides=overrides,
                    predicate_overrides=pred_overrides,
                    role_terms_override=non_ar2_rt,
                )
                if base_t_early is not None:
                    fast_res = _fast_solve_ar2_focal_poles(
                        rows, target, base_t_early, budget, seed_pairs=seed_ar2, tolerance=tolerance
                    )
                    if fast_res is not None:
                        fa1_e, fa2_e, _ = fast_res
                        cand_rt_e = _rebind_ar2_params_in_role_terms(active_role_terms, fa1_e, fa2_e)
                        rec_p_e = try_radius(radius, terms_override=cand_rt_e)
                        if rec_p_e is not None and rec_p_e[2]["loss"] < best_record[2]["loss"]:
                            active_role_terms = cand_rt_e
                            best_record = rec_p_e
            else:
                def _eval_early_ar2(a1_v: float, a2_v: float, _r=radius):
                    cand_rt = _rebind_ar2_params_in_role_terms(active_role_terms, a1_v, a2_v)
                    rec_p = try_radius(_r, terms_override=cand_rt)
                    if rec_p is None:
                        return None
                    return rec_p[2]["loss"], cand_rt, rec_p

                opt_ar2_early = _optimize_2d_ar2_latent_poles(
                    _eval_early_ar2, seed_pairs=seed_ar2, tolerance=tolerance
                )
                if opt_ar2_early is not None and opt_ar2_early[1][0] < best_record[2]["loss"]:
                    _, (_, best_rt_ar2, rec_opt_ar2) = opt_ar2_early
                    active_role_terms = best_rt_ar2
                    best_record = rec_opt_ar2
        # If radius_bounds came from verified transitions and already hit tolerance,
        # avoid wasting compute scanning unverified outer cells!
        if (
            radius_bounds is not None
            and best_record is not None
            and best_record[2]["loss"] <= tolerance
            and idx >= 0
        ):
            break

    # Local bisection refinement inside untested distance cells if needed (only when radius_bounds is unverified).
    if best_record is not None and best_record[2]["loss"] > tolerance and boundaries and radius_bounds is None:
        for _ in range(16):
            if best_record[2]["loss"] <= tolerance:
                break
            ordered_eval = sorted(evaluated, key=lambda item: item[2]["loss"])
            cell = None
            for center, _, _, _ in ordered_eval:
                low = max((val for val in tested_radii if val < center), default=0.0)
                high = min(
                    (val for val in tested_radii if val > center),
                    default=(distances[-1] + 1e-6 if distances else center + 1.0),
                )
                left_cell = (bisect_right(boundaries, low), bisect_left(boundaries, center))
                right_cell = (bisect_right(boundaries, center), bisect_left(boundaries, high))
                chosen = max((left_cell, right_cell), key=lambda pair: pair[1] - pair[0])
                if chosen[0] < chosen[1]:
                    cell = chosen
                    break
            if cell is None:
                break
            mid = boundaries[(cell[0] + cell[1]) // 2]
            rec = try_radius(mid)
            if rec is None:
                distance_complete = False
                break
            if rec[2]["loss"] < best_record[2]["loss"]:
                best_record = rec

    if best_record is None:
        return []

    best_radius, best_terms, best_fit, active_role_terms = best_record

    # POMDP Track 2A: If active_role_terms ALREADY contains per-object accumulate poles
    # (e.g. consolidated focal_memory commitment transferring across scale to a new decay pole),
    # recalibrate the 1D continuous pole when loss > tolerance.
    if (
        best_fit["loss"] > tolerance
        and existing_poles
        and not recalibrated_existing_poles
        and max_hist >= 2
    ):
        seed_poles = tuple(commitment.get("source_latent_poles", ())) or tuple(existing_poles)
        rebind_radius = (
            0.5 * (float(radius_bounds[0]) + float(radius_bounds[1]))
            if radius_bounds is not None
            else best_radius
        )

        def _eval_rebind_pole(p_val: float):
            cand_rt = _rebind_latent_pole_in_role_terms(active_role_terms, p_val)
            rec_p = try_radius(rebind_radius, terms_override=cand_rt)
            if rec_p is None:
                return None
            return rec_p[2]["loss"], cand_rt, rec_p

        opt = _optimize_1d_latent_pole(_eval_rebind_pole, seed_poles=seed_poles, tolerance=tolerance)
        if opt is not None and opt[1][0] < best_fit["loss"]:
            _, (_, best_rt, rec_opt) = opt
            best_radius, best_terms, best_fit, active_role_terms = rec_opt
    elif (
        best_fit["loss"] > tolerance
        and existing_ar2_params
        and not recalibrated_existing_ar2
        and max_hist >= 2
    ):
        seed_ar2 = tuple(tuple(p) for p in (commitment.get("source_ar2_params") or existing_ar2_params))
        rebind_radius = (
            0.5 * (float(radius_bounds[0]) + float(radius_bounds[1]))
            if radius_bounds is not None
            else best_radius
        )

        def _eval_rebind_ar2(a1_v: float, a2_v: float):
            cand_rt = _rebind_ar2_params_in_role_terms(active_role_terms, a1_v, a2_v)
            rec_p = try_radius(rebind_radius, terms_override=cand_rt)
            if rec_p is None:
                return None
            return rec_p[2]["loss"], cand_rt, rec_p

        opt2 = _optimize_2d_ar2_latent_poles(_eval_rebind_ar2, seed_pairs=seed_ar2, tolerance=tolerance)
        if opt2 is not None and opt2[1][0] < best_fit["loss"]:
            _, (_, best_rt2, rec_opt2) = opt2
            best_radius, best_terms, best_fit, active_role_terms = rec_opt2

    # POMDP Track 2B: If commitment is Markovian (no existing latent poles), transitions are
    # Markov-consistent, yet trajectory response fails tolerance with history_depth >= 2,
    # audit P_markov_state -> P_latent_coupling -> P_latent_order hierarchically!
    if (
        allow_latent_state
        and best_fit["loss"] > tolerance
        and not existing_poles
        and not existing_ar2_params
        and effective_latent_mech is None
        and max((len(r.history) for r in rows), default=0) >= 2
    ):
        probe_radius = (
            0.5 * (float(radius_bounds[0]) + float(radius_bounds[1]))
            if radius_bounds is not None
            else best_radius
        )

        probe_radii_foc = [probe_radius]
        if radius_bounds is not None:
            lo_b, hi_b = float(radius_bounds[0]), float(radius_bounds[1])
            for bd in boundaries:
                if lo_b <= bd <= hi_b and all(abs(bd - pr) > 1e-12 for pr in probe_radii_foc):
                    probe_radii_foc.append(bd)
                    if len(probe_radii_foc) >= 4:
                        break

        opt_1d_hit = None
        for pr_foc in probe_radii_foc:
            base_foc_rec = (
                best_record
                if abs(pr_foc - best_radius) <= 1e-12
                else try_radius(pr_foc, terms_override=active_role_terms)
            )
            if base_foc_rec is None:
                continue
            fast_1d = _fast_solve_1d_focal_pole(
                rows, target, base_foc_rec[1], budget, tolerance=tolerance
            )
            if fast_1d is not None and fast_1d[1] <= tolerance * 4.0:
                fp_cand, _ = fast_1d
                cand_rt_1d = expand_role_terms_for_latent_focal_readout(active_role_terms, fp_cand)
                rec_1d_cand = try_radius(pr_foc, terms_override=cand_rt_1d)
                if rec_1d_cand is not None and rec_1d_cand[2]["loss"] <= tolerance:
                    opt_1d_hit = (fp_cand, cand_rt_1d, rec_1d_cand)
                    break

        if opt_1d_hit is not None:
            found_pole, expanded_rt, rec_new = opt_1d_hit
            best_radius, best_terms, best_fit, active_role_terms = rec_new
            effective_latent_mech = "focal_memory"
            reorganization_type = "scientific_premise_revision"
            retracted_list = list((scientific_revision or {}).get("retracted_premises", ()))
            retracted_list.append(
                {
                    "premise_id": "markov_state_sufficiency",
                    "expected": "supported",
                    "reason": "trajectory_readout_requires_per_object_latent_reservoir",
                }
            )
            replacement_list = list((scientific_revision or {}).get("replacement_premises", ()))
            replacement_list.append(
                {
                    "premise_id": "markov_state_sufficiency",
                    "latent_state_mechanism": "focal_memory",
                    "latent_pole": float(found_pole),
                }
            )
            retracted_ids = {p["premise_id"] for p in retracted_list}
            scientific_revision = {
                **(scientific_revision or {}),
                "trigger_source": (scientific_revision or {}).get(
                    "trigger_source", "trajectory_readout_premise_audit"
                ),
                "revision_kind": "latent_focal_readout_revision",
                "parent_commitment_id": commitment["commitment_id"],
                "latent_state_mechanism": "focal_memory",
                "latent_pole": float(found_pole),
                "retracted_premises": retracted_list,
                "replacement_premises": replacement_list,
                "retained_premises": [
                    p["premise_id"]
                    for p in commitment.get("support_premises", ())
                    if p["premise_id"] not in retracted_ids
                ],
            }
        elif allow_higher_order_latent:
            # Test 2nd-order oscillatory AR(2) focal memory (P_latent_order) first via ultra-cheap FWL projection (~800 compute)
            opt_ar2_hit = None
            for pr_ar2 in probe_radii_foc:
                base_probe_rec = (
                    best_record
                    if abs(pr_ar2 - best_radius) <= 1e-12
                    else try_radius(pr_ar2, terms_override=active_role_terms)
                )
                if base_probe_rec is None:
                    continue
                fast_ar2 = _fast_solve_ar2_focal_poles(
                    rows, target, base_probe_rec[1], budget, tolerance=tolerance
                )
                if fast_ar2 is not None and fast_ar2[2] <= tolerance * 4.0:
                    fa1_cand, fa2_cand, _ = fast_ar2
                    cand_rt_ar2 = expand_role_terms_for_ar2_focal_readout(
                        active_role_terms, fa1_cand, fa2_cand
                    )
                    rec_ar2_cand = try_radius(pr_ar2, terms_override=cand_rt_ar2)
                    if rec_ar2_cand is not None and rec_ar2_cand[2]["loss"] <= tolerance:
                        opt_ar2_hit = (fa1_cand, fa2_cand, cand_rt_ar2, rec_ar2_cand)
                        break

            if opt_ar2_hit is not None:
                fa1, fa2, expanded_rt_ar2, rec_ar2 = opt_ar2_hit
                best_radius, best_terms, best_fit, active_role_terms = rec_ar2
                effective_latent_mech = "oscillatory_focal_memory"
                reorganization_type = "scientific_premise_revision"
                retracted_list = list((scientific_revision or {}).get("retracted_premises", ()))
                retracted_list.extend(
                    [
                        {
                            "premise_id": "markov_state_sufficiency",
                            "expected": "supported",
                            "reason": "trajectory_readout_requires_second_order_oscillatory_reservoir",
                        },
                        {
                            "premise_id": "P_latent_order",
                            "expected": 1,
                            "reason": "first_order_exponential_reservoir_refuted_by_oscillation",
                        },
                    ]
                )
                replacement_list = list((scientific_revision or {}).get("replacement_premises", ()))
                replacement_list.append(
                    {
                        "premise_id": "markov_state_sufficiency",
                        "latent_state_mechanism": "oscillatory_focal_memory",
                        "latent_order": 2,
                        "ar2_params": [float(fa1), float(fa2)],
                    }
                )
                retracted_ids = {p["premise_id"] for p in retracted_list}
                scientific_revision = {
                    **(scientific_revision or {}),
                    "trigger_source": (scientific_revision or {}).get(
                        "trigger_source", "trajectory_readout_premise_audit"
                    ),
                    "revision_kind": "ar2_latent_focal_readout_revision",
                    "parent_commitment_id": commitment["commitment_id"],
                    "latent_state_mechanism": "oscillatory_focal_memory",
                    "ar2_params": [float(fa1), float(fa2)],
                    "retracted_premises": retracted_list,
                    "replacement_premises": replacement_list,
                    "retained_premises": [
                        p["premise_id"]
                        for p in commitment.get("support_premises", ())
                        if p["premise_id"] not in retracted_ids
                    ],
                }
            else:
                # Test 1st-order spatially coupled focal memory (P_latent_coupling)
                def _eval_new_coupled_focal_pole(p_val: float):
                    cand_rt = expand_role_terms_for_coupled_focal_readout(
                        active_role_terms, p_val, primary_sym, effective_direction
                    )
                    rec_p = try_radius(probe_radius, terms_override=cand_rt)
                    if rec_p is None:
                        return None
                    return rec_p[2]["loss"], cand_rt, rec_p

                opt_coup = _optimize_1d_latent_pole(_eval_new_coupled_focal_pole, tolerance=tolerance)
                if opt_coup is not None and opt_coup[1][0] <= tolerance:
                    found_pole, (_, expanded_rt_c, rec_coup) = opt_coup
                    best_radius, best_terms, best_fit, active_role_terms = rec_coup
                    effective_latent_mech = "coupled_focal_memory"
                    reorganization_type = "scientific_premise_revision"
                    retracted_list = list((scientific_revision or {}).get("retracted_premises", ()))
                    retracted_list.extend(
                        [
                            {
                                "premise_id": "markov_state_sufficiency",
                                "expected": "supported",
                                "reason": "trajectory_readout_requires_spatially_coupled_latent_reservoir",
                            },
                            {
                                "premise_id": "P_latent_coupling",
                                "expected": "independent",
                                "reason": "independent_per_object_reservoir_refuted_by_neighbor_flux",
                            },
                        ]
                    )
                    replacement_list = list((scientific_revision or {}).get("replacement_premises", ()))
                    replacement_list.append(
                        {
                            "premise_id": "markov_state_sufficiency",
                            "latent_state_mechanism": "coupled_focal_memory",
                            "latent_coupling": "spatial_neighbor_coupled",
                            "latent_pole": float(found_pole),
                        }
                    )
                    retracted_ids = {p["premise_id"] for p in retracted_list}
                    scientific_revision = {
                        **(scientific_revision or {}),
                        "trigger_source": (scientific_revision or {}).get(
                            "trigger_source", "trajectory_readout_premise_audit"
                        ),
                        "revision_kind": "coupled_latent_focal_readout_revision",
                        "parent_commitment_id": commitment["commitment_id"],
                        "latent_state_mechanism": "coupled_focal_memory",
                        "latent_pole": float(found_pole),
                        "retracted_premises": retracted_list,
                        "replacement_premises": replacement_list,
                        "retained_premises": [
                            p["premise_id"]
                            for p in commitment.get("support_premises", ())
                            if p["premise_id"] not in retracted_ids
                        ],
                    }

    interval = None
    if best_fit["loss"] <= tolerance:
        if distances:
            low = max((d for d in distances if d < best_radius), default=0.0)
            high = min((d for d in distances if d >= best_radius), default=best_radius + 1.0)
            if radius_bounds is not None:
                low = max(low, float(radius_bounds[0]))
                if float(radius_bounds[1]) > low:
                    high = min(high, float(radius_bounds[1]))
            interval = [float(low), float(high)]
        elif radius_bounds is not None:
            interval = [float(radius_bounds[0]), float(radius_bounds[1])]

    return [
        _build_commitment_entry(
            spec=spec,
            rows=rows,
            schema=target,
            commitment=commitment,
            terms=best_terms,
            fit=best_fit,
            fitted_radius=best_radius,
            radius_interval=interval,
            threshold_interval=eff_threshold_interval if best_fit["loss"] <= tolerance else None,
            active_threshold_probing=active_threshold_probing,
            complete=distance_complete,
            compute_cost=budget.spent - start,
            reorganization_type=reorganization_type,
            scientific_revision=scientific_revision,
            direction_overrides=overrides,
            predicate_overrides=pred_overrides,
            role_terms_override=active_role_terms,
            nonconservation_mechanism=effective_noncons,
            latent_state_mechanism=effective_latent_mech,
            tolerance=tolerance,
        )
    ]


def _build_commitment_entry(
    *,
    spec,
    rows: list[Observation],
    schema: ObjectSchema,
    commitment: dict,
    terms: tuple[Program, ...],
    fit: dict,
    fitted_radius: float | None,
    radius_interval: list[float] | None,
    threshold_interval: list[float] | None = None,
    active_threshold_probing: bool = True,
    complete: bool,
    compute_cost: int,
    reorganization_type: str,
    scientific_revision: dict | None,
    direction_overrides: dict[str, str],
    predicate_overrides: dict[str, dict | str | Program | None] | None = None,
    role_terms_override: list[dict] | None = None,
    nonconservation_mechanism: str | None = None,
    latent_state_mechanism: str | None = None,
    tolerance: float,
) -> dict:
    sources = tuple(commitment.get("source_concept_ids", ()))
    family = commitment.get("family", "world_commitment")
    family_id = commitment.get("family_id") or _FAMILY_IDS.get(family, "T12")
    search_tag = (
        "world_commitment_scientific_revision"
        if reorganization_type == "scientific_premise_revision"
        else "world_commitment_instantiation"
    )
    candidate_id = f"{commitment['commitment_id']}:{search_tag}:{len(schema.objects)}"
    candidate = Candidate(
        candidate_id,
        fit["program"],
        search_tag,
        "Instantiate verified arity-parametric world commitment on target schema",
        parents=sources,
    )
    raw_coeffs = list(fit["coefficients"])
    fit_active_terms = tuple(fit.get("fit_terms") or terms)
    if fit_active_terms != tuple(terms):
        coef_map = dict(zip(fit_active_terms, raw_coeffs[1:]))
        seen_t: set[Program] = set()
        aligned = [raw_coeffs[0]]
        for t_item in terms:
            if t_item not in seen_t and t_item in coef_map:
                aligned.append(coef_map[t_item])
                seen_t.add(t_item)
            else:
                aligned.append(0.0)
        coefficients = aligned
    else:
        coefficients = raw_coeffs
    effective_dim = sum(abs(weight) > 1e-7 for weight in coefficients[1:])
    fit_record = {
        "candidate_id": candidate_id,
        "family": family if family in _FAMILY_IDS else "joint_window_event",
        "training_loss": float(fit["loss"]),
        "effective_dimension": effective_dim,
        "raw_dimension": len(schema.feature_names),
        "radius": fitted_radius,
        "radius_interval": list(radius_interval) if radius_interval else None,
        "window": None,
        "fit_terms": [term.to_dict() for term in terms],
        "coefficients": coefficients,
    }
    hypotheses = []
    if fitted_radius is not None:
        hypotheses.append(
            {
                "kind": "locality",
                "status": "proposed",
                "radius": fitted_radius,
                "counterexamples": [],
            }
        )
    if nonconservation_mechanism is not None:
        hypotheses.append(
            {
                "kind": "conservation",
                "status": "refuted",
                "nonconservation_mechanism": nonconservation_mechanism,
                "counterexamples": [],
            }
        )
    if latent_state_mechanism is not None:
        hypotheses.append(
            {
                "kind": "markov_state_sufficiency",
                "status": "refuted",
                "latent_state_mechanism": latent_state_mechanism,
                "counterexamples": [],
            }
        )
    discovery = StructuralDiscovery(
        [candidate],
        [fit_record],
        hypotheses,
        schema,
        compute_cost,
        tuple(radius_interval) if radius_interval else None,
        complete,
    )
    base_gen = int(commitment.get("generation", 0))
    next_gen = (
        base_gen + 1 if reorganization_type == "scientific_premise_revision" else base_gen
    )
    lineage = deepcopy(commitment.get("revision_lineage", []))
    edge_cond_overrides = {
        k: describe_edge_condition(v) for k, v in (predicate_overrides or {}).items()
    }
    if scientific_revision is not None:
        scientific_revision = {
            **deepcopy(scientific_revision),
            "generation": next_gen,
            "nonconservation_mechanism": nonconservation_mechanism,
            "latent_state_mechanism": latent_state_mechanism,
            "revision_lineage": lineage,
        }
    entry = {
        "candidate": candidate,
        "kind": f"world_commitment:{family}",
        "family_id": family_id,
        "training_loss": float(fit["loss"]),
        "structure": discovery.to_dict(),
        "structure_search": search_tag,
        "method_source_ids": list(sources),
        "world_commitment_application": {
            "commitment_id": commitment["commitment_id"],
            "reorganization_type": reorganization_type,
            "generation": next_gen,
            "source_concept_ids": list(sources),
            "source_arities": list(commitment.get("source_arities", ())),
            "target_arity": len(schema.objects),
            "fitted_radius": fitted_radius,
            "radius_interval": list(radius_interval) if radius_interval else None,
            "threshold_interval": list(threshold_interval) if threshold_interval else None,
            "direction_overrides": dict(direction_overrides),
            "edge_condition_overrides": edge_cond_overrides,
            "nonconservation_mechanism": nonconservation_mechanism,
            "latent_state_mechanism": latent_state_mechanism,
            "revision_lineage": lineage,
        },
    }
    if scientific_revision is not None:
        entry["scientific_premise_revision"] = deepcopy(scientific_revision)
    entry = attach_structural_research_program(entry, fit_rows=len(rows))
    if entry.get("research_program"):
        entry["research_program"]["source_ids"] = list(sources)
        entry["research_program"]["world_commitment_application"] = deepcopy(
            entry["world_commitment_application"]
        )
        if scientific_revision is not None:
            entry["research_program"]["scientific_premise_revision"] = deepcopy(scientific_revision)
        entry = _attach_symbolic_parameter_step(
            entry,
            commitment,
            schema,
            coefficients,
            fitted_radius,
            radius_interval,
            direction_overrides=direction_overrides,
            predicate_overrides=predicate_overrides,
            role_terms_override=role_terms_override,
            threshold_interval=threshold_interval,
            active_threshold_probing=active_threshold_probing,
        )
    return entry


_TRANSITION_LATENT_MECHANISMS = {
    "transition_memory",
    "oscillatory_transition_memory",
    "van_der_pol_transition_memory",
    "coupled_transition_memory",
    "custom_transition_memory",
}


def commitment_verified_radius_bounds(
    commitment: dict,
    audit_report: dict | None,
    schema_x_to_q: dict[str, str],
) -> list[float] | None:
    """Return verified radius bounds from ``audit_report`` matching ``commitment``."""
    if not audit_report or audit_report.get("status") not in (
        *VERIFIED_EXCHANGE_STATUSES,
        "unexplained_latent_transition_state",
    ):
        return None
    params = commitment.get("radius_parameters", ())
    committed_dir = params[0]["direction"] if params else None
    committed_cond = (
        describe_edge_condition(_normalize_edge_condition_for_grouping(params[0].get("edge_condition") or "unconditional"))
        if params
        else "unconditional"
    )
    committed_noncons = commitment.get("nonconservation_mechanism")
    committed_latent = commitment.get("latent_state_mechanism")
    if audit_report.get("status") == "unexplained_latent_transition_state":
        if committed_latent != "custom_transition_memory":
            return None
    else:
        audit_latent = (audit_report.get("latent_transition_state") or {}).get("mechanism")
        if committed_latent in _TRANSITION_LATENT_MECHANISMS and audit_latent != committed_latent:
            return None
        if committed_latent not in _TRANSITION_LATENT_MECHANISMS and audit_latent in _TRANSITION_LATENT_MECHANISMS:
            return None
    matches = []
    for d_name, items in audit_report.get("matches_by_direction", {}).items():
        if committed_dir is not None and d_name != committed_dir:
            continue
        for m in items:
            if m.get("mapping") != schema_x_to_q:
                continue
            if (
                "edge_condition" in m
                and describe_edge_condition(_normalize_edge_condition_for_grouping(m["edge_condition"] or "unconditional"))
                != committed_cond
            ):
                continue
            if "nonconservation_mechanism" in m and m["nonconservation_mechanism"] != committed_noncons:
                continue
            matches.append(m)
    if not matches:
        return None
    return [
        float(min(m["radius_interval"][0] for m in matches)),
        float(max(m["radius_interval"][1] for m in matches)),
    ]


def commitment_applicable_triggers(commitment: dict, audit_report: dict | None) -> list[dict]:
    """Filter ``audit_report['scientific_revision_triggers']`` applicable to ``commitment``."""
    params = commitment.get("radius_parameters", ())
    if not params or not audit_report:
        return []
    primary = params[0]
    committed_dir = primary["direction"]
    raw_committed_cond = primary.get("edge_condition") or "unconditional"
    norm_committed_cond = describe_edge_condition(_normalize_edge_condition_for_grouping(raw_committed_cond))
    committed_noncons = commitment.get("nonconservation_mechanism") or "conserved"
    committed_latent = commitment.get("latent_state_mechanism")

    def _norm_c_str(val: object) -> str:
        return describe_edge_condition(_normalize_edge_condition_for_grouping(val or "unconditional"))  # type: ignore[arg-type]

    applicable = []
    for trigger in audit_report.get("scientific_revision_triggers", ()):
        kind = trigger.get("kind")
        if kind == "interaction_direction_revision":
            trig_cond_ok = (
                "edge_condition" not in trigger
                or _norm_c_str(trigger.get("edge_condition")) == norm_committed_cond
            )
            if (
                committed_noncons == "conserved"
                and committed_latent not in _TRANSITION_LATENT_MECHANISMS
                and trig_cond_ok
                and trigger.get("retracted_direction") == committed_dir
                and trigger.get("replacement_direction") != committed_dir
            ):
                applicable.append(trigger)
        elif kind == "exchange_condition_revision":
            retracted_raw = trigger.get("retracted_condition_raw", trigger.get("retracted_condition"))
            replacement_raw = trigger.get("replacement_condition")
            if (
                committed_noncons == "conserved"
                and committed_latent not in _TRANSITION_LATENT_MECHANISMS
                and _norm_c_str(retracted_raw) == norm_committed_cond
                and (
                    _norm_c_str(replacement_raw) != norm_committed_cond
                    or trigger.get("replacement_direction") != committed_dir
                )
            ):
                applicable.append(trigger)
        elif kind == "nonconservation_mechanism_revision":
            replacement_mech = trigger.get("replacement_mechanism")
            replacement_dir = trigger.get("replacement_direction")
            replacement_raw = trigger.get("replacement_condition")
            if (
                committed_noncons != replacement_mech
                or committed_dir != replacement_dir
                or _norm_c_str(replacement_raw) != norm_committed_cond
            ):
                applicable.append(trigger)
        elif kind == "latent_transition_state_revision":
            trigger_mech = trigger.get("latent_mechanism", "transition_memory")
            replacement_dir = trigger.get("replacement_direction")
            replacement_raw = trigger.get("replacement_condition")
            if (
                committed_latent != trigger_mech
                or committed_dir != replacement_dir
                or _norm_c_str(replacement_raw) != norm_committed_cond
            ):
                applicable.append(trigger)
    return applicable


def is_commitment_empirically_consistent(
    commitment: dict,
    audit_report: dict | None,
    *,
    task_id: str | None = None,
) -> bool:
    """Return True iff ``commitment`` has no invalidated support on ``task_id`` and matches ``audit_report``.

    Prevents both:
    - Reusing a refuted commitment unconditionally on a task that refuted it (Acceptance #10), and
    - Allowing a newer-generation ``source_status='supported'`` commitment (e.g. ``left`` or ``sink``)
      to bypass empirical premise checks when an older condition (e.g. ``right`` or ``conserved``) returns
      (Acceptance #18 & C2).
    """
    if task_id is not None and task_id in set(commitment.get("refuted_task_ids") or ()):
        return False
    if not audit_report:
        return True
    if commitment_applicable_triggers(commitment, audit_report):
        return False

    committed_noncons = commitment.get("nonconservation_mechanism")
    committed_latent = commitment.get("latent_state_mechanism")
    cons_sup = audit_report.get("conservation_supported")
    audit_lat = (audit_report.get("latent_transition_state") or {}).get("mechanism")

    if cons_sup is True and committed_noncons not in (None, "conserved"):
        return False
    if (
        cons_sup is False
        and audit_lat is None
        and audit_report.get("status") != "unexplained_latent_transition_state"
        and committed_noncons in (None, "conserved")
    ):
        return False

    if committed_latent in _TRANSITION_LATENT_MECHANISMS:
        if committed_latent == "custom_transition_memory":
            if audit_report.get("status") != "unexplained_latent_transition_state":
                return False
        elif audit_lat != committed_latent:
            return False
    elif audit_lat in _TRANSITION_LATENT_MECHANISMS:
        return False

    params = commitment.get("radius_parameters") or ()
    if params:
        primary = params[0]
        v_dir = audit_report.get("verified_direction")
        s_dirs = audit_report.get("supported_directions")
        if v_dir is not None and primary["direction"] != v_dir:
            return False
        if s_dirs and primary["direction"] not in s_dirs:
            return False
        v_cond = audit_report.get("verified_edge_condition")
        if v_cond is not None:
            norm_comm = describe_edge_condition(
                _normalize_edge_condition_for_grouping(primary.get("edge_condition") or "unconditional")
            )
            norm_ver = describe_edge_condition(
                _normalize_edge_condition_for_grouping(v_cond or "unconditional")
            )
            if norm_comm != norm_ver:
                return False

    return True


def propose_scientific_premise_revision(
    commitment: dict,
    audit_report: dict | None,
    spec,
    training: list[Observation],
    schema: ObjectSchema | dict | str,
    budget: SearchBudget,
    *,
    active_threshold_probing: bool = True,
    allow_latent_state: bool = True,
    allow_higher_order_latent: bool = True,
    transitions: list[dict] | tuple[dict, ...] = (),
    tolerance: float = 1e-8,
) -> list[dict]:
    """Conservatively revise ONLY refuted support premises with empirical witnesses."""
    params = commitment.get("radius_parameters", ())
    if not params:
        return []
    primary = params[0]
    sym = primary["symbol"]
    committed_dir = primary["direction"]
    committed_cond = describe_edge_condition(primary.get("edge_condition"))
    committed_noncons = commitment.get("nonconservation_mechanism") or "conserved"
    committed_latent = commitment.get("latent_state_mechanism")

    # Each tuple:
    # (alt_dir, pred_override_or_sentinel, role_terms_override, noncons_mech, latent_mech, bounds, thresh_iv, revision_meta)
    candidate_revisions: list[
        tuple[
            str,
            object,
            list[dict] | None,
            str | None,
            str | None,
            tuple[float, float] | None,
            list[float] | None,
            dict,
        ]
    ] = []
    no_pred_override = object()

    for trigger in commitment_applicable_triggers(commitment, audit_report):
        kind = trigger.get("kind")
        bounds = trigger.get("radius_interval")
        bounds_tuple = tuple(bounds) if bounds else None
        t_iv = trigger.get("threshold_interval")
        if kind == "interaction_direction_revision":
            replacement = trigger["replacement_direction"]
            retracted_ids = {f"interaction_direction:{sym}"}
            dir_pred = trigger.get("predicate_program")
            candidate_revisions.append(
                (
                    replacement,
                    dir_pred if dir_pred is not None else no_pred_override,
                    None,
                    commitment.get("nonconservation_mechanism"),
                    committed_latent,
                    bounds_tuple,
                    list(t_iv) if t_iv else None,
                    {
                        "trigger_source": "transition_premise_dependency_audit",
                        "revision_kind": "interaction_direction_revision",
                        "parent_commitment_id": commitment["commitment_id"],
                        "retracted_premises": [
                            {
                                "premise_id": f"interaction_direction:{sym}",
                                "direction": committed_dir,
                                "counterexample": deepcopy(trigger.get("counterexample")),
                            }
                        ],
                        "replacement_premises": [
                            {
                                "premise_id": f"interaction_direction:{sym}",
                                "direction": replacement,
                                "supporting_transitions": trigger.get("supporting_transitions"),
                            }
                        ],
                        "retained_premises": [
                            p["premise_id"]
                            for p in commitment.get("support_premises", ())
                            if p["premise_id"] not in retracted_ids
                        ],
                    },
                )
            )
        elif kind == "exchange_condition_revision":
            replacement_dir = trigger["replacement_direction"]
            replacement_cond = trigger["replacement_condition"]
            pred_prog = trigger.get("predicate_program")
            retracted = [
                {
                    "premise_id": f"exchange_form:{sym}",
                    "edge_condition": committed_cond,
                    "counterexample": deepcopy(trigger.get("counterexample")),
                }
            ]
            replacements = [
                {
                    "premise_id": f"exchange_form:{sym}",
                    "edge_condition": replacement_cond,
                    "supporting_transitions": trigger.get("supporting_transitions"),
                }
            ]
            retracted_ids = {f"exchange_form:{sym}"}
            if replacement_dir != committed_dir:
                retracted.append(
                    {
                        "premise_id": f"interaction_direction:{sym}",
                        "direction": committed_dir,
                    }
                )
                replacements.append(
                    {
                        "premise_id": f"interaction_direction:{sym}",
                        "direction": replacement_dir,
                    }
                )
                retracted_ids.add(f"interaction_direction:{sym}")
            candidate_revisions.append(
                (
                    replacement_dir,
                    pred_prog,
                    None,
                    commitment.get("nonconservation_mechanism"),
                    committed_latent,
                    bounds_tuple,
                    list(t_iv) if t_iv else None,
                    {
                        "trigger_source": "transition_premise_dependency_audit",
                        "revision_kind": "exchange_condition_revision",
                        "parent_commitment_id": commitment["commitment_id"],
                        "retracted_premises": retracted,
                        "replacement_premises": replacements,
                        "retained_premises": [
                            p["premise_id"]
                            for p in commitment.get("support_premises", ())
                            if p["premise_id"] not in retracted_ids
                        ],
                    },
                )
            )
        elif kind == "nonconservation_mechanism_revision":
            mech_name = trigger["replacement_mechanism"]
            replacement_dir = trigger["replacement_direction"]
            replacement_cond = trigger["replacement_condition"]
            pred_prog = trigger.get("predicate_program")
            expanded_terms = expand_role_terms_for_nonconservation(
                commitment["role_terms"], mech_name
            )
            retracted = []
            replacements = []
            retracted_ids = set()
            if committed_noncons != mech_name:
                retracted.append(
                    {
                        "premise_id": "conservation",
                        "expected": "supported" if committed_noncons == "conserved" else committed_noncons,
                        "counterexample": deepcopy(trigger.get("counterexample")),
                    }
                )
                replacements.append(
                    {
                        "premise_id": "conservation",
                        "nonconservation_mechanism": mech_name,
                        "estimated_rate": trigger.get("estimated_rate"),
                        "supporting_transitions": trigger.get("supporting_transitions"),
                    }
                )
                retracted_ids.add("conservation")
            if describe_edge_condition(replacement_cond) != committed_cond:
                retracted.append(
                    {
                        "premise_id": f"exchange_form:{sym}",
                        "edge_condition": committed_cond,
                    }
                )
                replacements.append(
                    {
                        "premise_id": f"exchange_form:{sym}",
                        "edge_condition": replacement_cond,
                    }
                )
                retracted_ids.add(f"exchange_form:{sym}")
            if replacement_dir != committed_dir:
                retracted.append(
                    {
                        "premise_id": f"interaction_direction:{sym}",
                        "direction": committed_dir,
                    }
                )
                replacements.append(
                    {
                        "premise_id": f"interaction_direction:{sym}",
                        "direction": replacement_dir,
                    }
                )
                retracted_ids.add(f"interaction_direction:{sym}")
            candidate_revisions.append(
                (
                    replacement_dir,
                    pred_prog,
                    expanded_terms,
                    mech_name,
                    committed_latent,
                    bounds_tuple,
                    None,
                    {
                        "trigger_source": "transition_premise_dependency_audit",
                        "revision_kind": "nonconservation_mechanism_revision",
                        "parent_commitment_id": commitment["commitment_id"],
                        "retracted_premises": retracted,
                        "replacement_premises": replacements,
                        "retained_premises": [
                            p["premise_id"]
                            for p in commitment.get("support_premises", ())
                            if p["premise_id"] not in retracted_ids
                        ],
                    },
                )
            )
        elif kind == "latent_transition_state_revision" and allow_latent_state:
            lat_mech = trigger.get("latent_mechanism", "transition_memory")
            if lat_mech != "transition_memory" and not allow_higher_order_latent:
                continue
            pole = float(trigger["latent_pole"])
            ar2_p = trigger.get("ar2_params")
            vdp_p = trigger.get("vdp_params")
            drive_weights = trigger.get("latent_drive_weights") or {"u": 1.0, "q": 0.0}
            replacement_dir = trigger["replacement_direction"]
            replacement_cond = trigger["replacement_condition"]
            pred_prog = trigger.get("predicate_program")
            if committed_latent == lat_mech:
                if lat_mech == "van_der_pol_transition_memory" and vdp_p:
                    expanded_terms = _rebind_vdp_params_in_role_terms(
                        commitment["role_terms"], float(vdp_p[0]), float(vdp_p[1]), float(vdp_p[2])
                    )
                elif lat_mech == "oscillatory_transition_memory" and ar2_p:
                    expanded_terms = _rebind_ar2_params_in_role_terms(
                        commitment["role_terms"], float(ar2_p[0]), float(ar2_p[1])
                    )
                else:
                    expanded_terms = _rebind_latent_pole_in_role_terms(commitment["role_terms"], pole)
            elif lat_mech == "van_der_pol_transition_memory" and vdp_p:
                expanded_terms = expand_role_terms_for_vdp_latent_transition(
                    commitment["role_terms"],
                    float(vdp_p[0]),
                    float(vdp_p[1]),
                    float(vdp_p[2]),
                    drive_weights,
                )
            elif lat_mech == "oscillatory_transition_memory" and ar2_p:
                expanded_terms = expand_role_terms_for_ar2_latent_transition(
                    commitment["role_terms"], float(ar2_p[0]), float(ar2_p[1]), drive_weights
                )
            elif lat_mech == "coupled_transition_memory":
                expanded_terms = expand_role_terms_for_coupled_latent_transition(
                    commitment["role_terms"], pole, sym, replacement_dir, drive_weights
                )
            else:
                expanded_terms = expand_role_terms_for_latent_transition(
                    commitment["role_terms"], pole, drive_weights
                )
            retracted = []
            replacements = []
            retracted_ids = set()
            if committed_latent != lat_mech:
                retracted.append(
                    {
                        "premise_id": "markov_state_sufficiency",
                        "expected": committed_latent or "supported",
                        "counterexample": deepcopy(trigger.get("counterexample")),
                    }
                )
                replacements.append(
                    {
                        "premise_id": "markov_state_sufficiency",
                        "latent_state_mechanism": lat_mech,
                        "latent_pole": pole,
                        "ar2_params": deepcopy(ar2_p),
                        "latent_drive_weights": deepcopy(drive_weights),
                        "supporting_transitions": trigger.get("supporting_transitions"),
                    }
                )
                retracted_ids.add("markov_state_sufficiency")
            if describe_edge_condition(replacement_cond) != committed_cond:
                retracted.append(
                    {
                        "premise_id": f"exchange_form:{sym}",
                        "edge_condition": committed_cond,
                    }
                )
                replacements.append(
                    {
                        "premise_id": f"exchange_form:{sym}",
                        "edge_condition": replacement_cond,
                    }
                )
                retracted_ids.add(f"exchange_form:{sym}")
            if replacement_dir != committed_dir:
                retracted.append(
                    {
                        "premise_id": f"interaction_direction:{sym}",
                        "direction": committed_dir,
                    }
                )
                replacements.append(
                    {
                        "premise_id": f"interaction_direction:{sym}",
                        "direction": replacement_dir,
                    }
                )
                retracted_ids.add(f"interaction_direction:{sym}")
            candidate_revisions.append(
                (
                    replacement_dir,
                    pred_prog,
                    expanded_terms,
                    None,
                    lat_mech,
                    bounds_tuple,
                    None,
                    {
                        "trigger_source": "transition_premise_dependency_audit",
                        "revision_kind": "latent_transition_state_revision",
                        "parent_commitment_id": commitment["commitment_id"],
                        "latent_state_mechanism": lat_mech,
                        "latent_pole": pole,
                        "ar2_params": deepcopy(ar2_p),
                        "retracted_premises": retracted,
                        "replacement_premises": replacements,
                        "retained_premises": [
                            p["premise_id"]
                            for p in commitment.get("support_premises", ())
                            if p["premise_id"] not in retracted_ids
                        ],
                    },
                )
            )

    # Fallback when response-level instantiation failed without transition audit:
    # test sibling directions, atomic edge conditions, and non-conservative mechanisms.
    if not candidate_revisions and audit_report is None:
        for alt_dir in ("symmetric", "right", "left"):
            if alt_dir == committed_dir:
                continue
            candidate_revisions.append(
                (
                    alt_dir,
                    no_pred_override,
                    None,
                    commitment.get("nonconservation_mechanism"),
                    committed_latent,
                    None,
                    None,
                    {
                        "trigger_source": "response_instantiation_refutation",
                        "revision_kind": "interaction_direction_revision",
                        "parent_commitment_id": commitment["commitment_id"],
                        "retracted_premises": [
                            {
                                "premise_id": f"interaction_direction:{sym}",
                                "direction": committed_dir,
                            }
                        ],
                        "replacement_premises": [
                            {
                                "premise_id": f"interaction_direction:{sym}",
                                "direction": alt_dir,
                            }
                        ],
                        "retained_premises": [
                            p["premise_id"]
                            for p in commitment.get("support_premises", ())
                            if p["premise_id"] != f"interaction_direction:{sym}"
                        ],
                    },
                )
            )

    results = []
    for index, (
        alt_dir,
        pred_override,
        role_terms_override,
        noncons_mech,
        latent_mech,
        bounds,
        t_iv,
        revision_meta,
    ) in enumerate(candidate_revisions):
        remaining_slots = len(candidate_revisions) - index
        local = SearchBudget(budget.remaining // max(1, remaining_slots))
        pred_map = None if pred_override is no_pred_override else {sym: pred_override}
        entries = instantiate_commitment_on_schema(
            spec,
            training,
            schema,
            commitment,
            local,
            radius_bounds=bounds,
            threshold_interval=t_iv,
            direction_overrides={sym: alt_dir},
            predicate_overrides=pred_map,
            role_terms_override=role_terms_override,
            nonconservation_mechanism=noncons_mech,
            latent_state_mechanism=latent_mech,
            reorganization_type="scientific_premise_revision",
            scientific_revision=revision_meta,
            audit_report=audit_report,
            transitions=transitions,
            active_threshold_probing=active_threshold_probing,
            allow_latent_state=allow_latent_state,
            allow_higher_order_latent=allow_higher_order_latent,
            tolerance=tolerance,
        )
        budget.spend(local.spent)
        results.extend(entries)
        if any(entry["training_loss"] <= tolerance for entry in entries):
            break
    return results
