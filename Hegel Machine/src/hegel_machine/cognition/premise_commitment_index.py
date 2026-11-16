"""Persistent write-time world commitment store and sublinear premise-signature index.

Addresses the v10.2 -> v10.3 scaling question (F17 follow-up):
When a ``KnowledgeLibrary`` contains only 1-2 source priors, an unindexed raw archive
baseline (``archive_full_tools``) that consolidates commitments on demand at target time
nearly ties with pre-consolidated commitments. As long-horizon research accumulates
dozens of heterogeneous and conflicting episodes (multiple directions, custom open-grammar
predicates, non-conservative sinks/nonlocal couplings, latent POMDP drives, and temporal
readouts, each carrying empirical support and counterexample witnesses), two costs diverge:

  1. **Consolidation & equivalence verification cost**: ``archive_full_tools`` must parse
     all ``M`` raw priors and verify column/rejection equivalence on stored witness rows
     at target time ($O(M)$), whereas ``PremiseCommitmentIndex`` consolidates incrementally
     once at write time and caches unchanged equivalence groups ($O(1)$ on target tasks).
  2. **Premise-gated retrieval & predicate deduplication**: ``PremiseCommitmentIndex``
     maintains an inverted directory keyed by ``(conservation_class, direction_class,
     predicate_group_key, has_latent_state)`` plus a deduplicated custom-predicate registry
     so ``audit_transition_premises`` evaluates each distinct open-grammar predicate AST at
     most once regardless of how many readout/arity variants share it.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from .models import Observation, SearchBudget
from .world_structure_commitments import (
    _normalize_edge_condition_for_grouping,
    _rebind_ar2_params_in_role_terms,
    _rebind_latent_pole_in_role_terms,
    _rebind_vdp_params_in_role_terms,
    commitment_applicable_triggers,
    consolidate_world_structure_commitments,
    describe_edge_condition,
    extract_parametric_role_pattern,
    is_commitment_empirically_consistent,
)


def _deserialize_witness_rows(raw_rows: list[Any] | tuple[Any, ...]) -> list[Observation]:
    out: list[Observation] = []
    for item in raw_rows or ():
        if isinstance(item, Observation):
            out.append(item)
        elif isinstance(item, dict) and "features" in item:
            out.append(
                Observation(
                    {k: float(v) for k, v in item["features"].items()},
                    float(item.get("target", 0.0)),
                    tuple(
                        {hk: float(hv) for hk, hv in h.items()}
                        for h in item.get("history", ())
                        if isinstance(h, dict)
                    ),
                )
            )
    return out


def extract_archive_witness_maps(
    priors: list[dict],
) -> tuple[dict[str, list[Observation]], dict[str, list[Observation]]]:
    """Extract stored support and counterexample witness rows attached to ``meta_priors``."""
    support_by_source: dict[str, list[Observation]] = {}
    ce_by_source: dict[str, list[Observation]] = {}
    for prior in priors:
        cid = prior.get("source_concept_id")
        if not cid:
            continue
        sup = _deserialize_witness_rows(prior.get("support_witness_rows") or ())
        if sup:
            support_by_source[cid] = sup
        ce_rows: list[Observation] = []
        for ce in prior.get("counterexamples") or ():
            if isinstance(ce, dict):
                if "observation" in ce and isinstance(ce["observation"], dict):
                    ce_rows.extend(_deserialize_witness_rows([ce["observation"]]))
                elif "rows" in ce and isinstance(ce["rows"], list):
                    ce_rows.extend(_deserialize_witness_rows(ce["rows"]))
        if ce_rows:
            ce_by_source[cid] = ce_rows
    return support_by_source, ce_by_source


def _commitment_conservation_class(comm: dict) -> str:
    noncons = sorted(set(comm.get("nonconservation_mechanisms") or ()))
    if not noncons:
        return "conservative"
    return "+".join(noncons)


def _commitment_direction_class(comm: dict) -> str:
    r_params = comm.get("radius_parameters") or ()
    if not r_params:
        return "position_free"
    dirs = sorted({str(p.get("direction") or "symmetric") for p in r_params})
    return "+".join(dirs)


def _canonical_cond_str(raw: Any) -> str:
    norm = describe_edge_condition(_normalize_edge_condition_for_grouping(raw))
    if isinstance(norm, str):
        return norm
    return json.dumps(norm, sort_keys=True, separators=(",", ":"), default=str)


def _commitment_predicate_key(comm: dict) -> str:
    r_params = comm.get("radius_parameters") or ()
    if not r_params:
        return "none"
    raw = r_params[0].get("edge_condition", "unconditional")
    return _canonical_cond_str(raw)


def _commitment_has_latent(comm: dict) -> bool:
    return bool(
        comm.get("latent_poles")
        or comm.get("ar2_params")
        or comm.get("vdp_params")
        or any(
            "accumulate" in json.dumps(rt.get("summand", {}))
            for rt in (comm.get("role_terms") or ())
        )
    )


class PremiseCommitmentIndex:
    """Inverted premise-signature index and deduplicated custom-predicate directory."""

    def __init__(self, commitments: list[dict] | None = None) -> None:
        self.commitments: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.by_direction: dict[str, list[str]] = {}
        self.by_conservation: dict[str, list[str]] = {}
        self.by_predicate_key: dict[str, list[str]] = {}
        self.by_signature: dict[tuple[str, str, str, bool], list[str]] = {}
        self.unique_custom_predicate_commitments: list[dict] = []
        if commitments:
            self.rebuild(commitments)

    def rebuild(self, commitments: list[dict], *, budget: SearchBudget | None = None) -> int:
        start = budget.spent if budget is not None else 0
        self.commitments = list(commitments)
        self.by_id = {}
        self.by_direction = {}
        self.by_conservation = {}
        self.by_predicate_key = {}
        self.by_signature = {}
        seen_pred_keys: set[str] = set()
        self.unique_custom_predicate_commitments = []

        for comm in self.commitments:
            if budget is not None and not budget.spend(2):
                break
            cid = comm["commitment_id"]
            self.by_id[cid] = comm
            c_cls = _commitment_conservation_class(comm)
            d_cls = _commitment_direction_class(comm)
            p_key = _commitment_predicate_key(comm)
            h_lat = _commitment_has_latent(comm)

            self.by_direction.setdefault(d_cls, []).append(cid)
            self.by_conservation.setdefault(c_cls, []).append(cid)
            self.by_predicate_key.setdefault(p_key, []).append(cid)
            self.by_signature.setdefault((c_cls, d_cls, p_key, h_lat), []).append(cid)

            r_params = comm.get("radius_parameters") or ()
            if r_params:
                raw_ec = r_params[0].get("edge_condition")
                if isinstance(raw_ec, dict) and p_key not in seen_pred_keys:
                    seen_pred_keys.add(p_key)
                    self.unique_custom_predicate_commitments.append(comm)

        return (budget.spent - start) if budget is not None else 0

    def deduplicated_audit_commitments(self, budget: SearchBudget | None = None) -> list[dict]:
        """Return minimal representative commitments covering all distinct custom predicates & directions.

        Prevents ``audit_transition_premises`` from screening ``M`` duplicate copies of the
        same edge predicate AST across readout/arity variants.
        """
        if budget is not None:
            budget.spend(1 + len(self.unique_custom_predicate_commitments))
        reps: dict[str, dict] = {}
        for comm in self.unique_custom_predicate_commitments:
            reps[comm["commitment_id"]] = comm
        for d_cls, cids in self.by_direction.items():
            if cids and cids[0] not in reps:
                reps[cids[0]] = self.by_id[cids[0]]
        return list(reps.values())

    def query_consistent_commitments(
        self,
        audit: dict | None,
        *,
        task_id: str | None = None,
        budget: SearchBudget | None = None,
    ) -> tuple[list[dict], dict[str, Any]]:
        """Retrieve empirically consistent commitments in sublinear bucket lookup time."""
        start = budget.spent if budget is not None else 0
        if not self.commitments:
            return [], {"candidates_inspected": 0, "index_compute_cost": 0, "matched_ids": []}

        if audit is None or audit.get("status") == "awaiting_role_identification":
            # Without transition audit, fall back to full list check
            matched = []
            inspected = 0
            for comm in self.commitments:
                if budget is not None and not budget.spend(1):
                    break
                inspected += 1
                if is_commitment_empirically_consistent(comm, audit, task_id=task_id):
                    matched.append(comm)
            return matched, {
                "candidates_inspected": inspected,
                "total_indexed_commitments": len(self.commitments),
                "index_compute_cost": (budget.spent - start) if budget is not None else 0,
                "matched_ids": [c["commitment_id"] for c in matched],
            }

        # Narrow candidate bucket by verified conservation + direction + predicate keys!
        if audit.get("conservation_supported", True) and not audit.get("nonconservation_mechanisms"):
            cons_Pool = set(self.by_conservation.get("conservative", ()))
        else:
            mech_key = "+".join(sorted(set(audit.get("nonconservation_mechanisms") or ())))
            cons_Pool = set(self.by_conservation.get(mech_key, ())) | set(
                self.by_conservation.get("conservative", ())
            )

        v_dir = audit.get("verified_direction")
        sup_dirs = list(audit.get("supported_directions") or ([v_dir] if v_dir else []))
        if sup_dirs:
            dir_Pool: set[str] = set(self.by_direction.get("position_free", ()))
            for d in sup_dirs:
                dir_Pool |= set(self.by_direction.get(str(d), ()))
        else:
            dir_Pool = set(self.by_id.keys())

        v_ec = audit.get("verified_edge_condition")
        if v_ec is not None:
            ec_key = _canonical_cond_str(v_ec)
            pred_Pool = set(self.by_predicate_key.get(ec_key, ())) | set(
                self.by_predicate_key.get("none", ())
            )
        else:
            pred_Pool = set(self.by_id.keys())

        if budget is not None:
            budget.spend(3)  # hash-bucket intersection cost

        shortlist_ids = [
            cid for cid in self.by_id if cid in cons_Pool and cid in dir_Pool and cid in pred_Pool
        ]
        matched: list[dict] = []
        inspected = 0
        for cid in shortlist_ids:
            if budget is not None and not budget.spend(2):
                break
            inspected += 1
            comm = self.by_id[cid]
            if is_commitment_empirically_consistent(comm, audit, task_id=task_id):
                matched.append(comm)

        return matched, {
            "candidates_inspected": inspected,
            "total_indexed_commitments": len(self.commitments),
            "index_compute_cost": (budget.spent - start) if budget is not None else 0,
            "matched_ids": [c["commitment_id"] for c in matched],
            "rejected_inconsistent_ids": [
                cid for cid in self.by_id if cid not in {c["commitment_id"] for c in matched}
            ],
        }

    def query_revision_parent_commitments(
        self,
        audit: dict | None,
        *,
        budget: SearchBudget | None = None,
    ) -> list[dict]:
        """Retrieve only commitments whose violated premise matches an active trigger in ``audit``."""
        if not self.commitments or audit is None:
            return list(self.commitments[:4])
        triggers = audit.get("scientific_revision_triggers") or ()
        if not triggers:
            return []
        candidate_ids: set[str] = set()
        for trig in triggers:
            kind = trig.get("kind")
            if kind == "interaction_direction_revision":
                ret_dir = str(trig.get("retracted_direction") or "")
                candidate_ids |= set(self.by_direction.get(ret_dir, ()))
            elif kind in {"nonconservation_revision", "latent_state_revision"}:
                candidate_ids |= set(self.by_conservation.get("conservative", ()))
            elif kind == "edge_condition_revision":
                ret_cond = trig.get("retracted_condition", "unconditional")
                rk = _canonical_cond_str(ret_cond)
                candidate_ids |= set(self.by_predicate_key.get(rk, ()))
            else:
                candidate_ids |= set(self.by_id.keys())
        if budget is not None:
            budget.spend(1 + len(candidate_ids))
        out = []
        for cid in self.by_id:
            if cid not in candidate_ids:
                continue
            comm = self.by_id[cid]
            if commitment_applicable_triggers(comm, audit):
                out.append(comm)
        return out


def _prior_state_fingerprint(prior: dict) -> str:
    payload = {
        "id": prior.get("source_concept_id"),
        "status": prior.get("status", "supported"),
        "ce_len": len(prior.get("counterexamples") or ()),
        "sup_len": len(prior.get("support_witness_rows") or ()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def ensure_library_commitment_index(
    library,
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> tuple[list[dict], PremiseCommitmentIndex, int]:
    """Incrementally maintain consolidated commitments + ``PremiseCommitmentIndex`` on ``library``.

    Unchanged ``meta_priors`` hit the persistent cache in O(1) (1 compute unit).
    Newly added or modified priors consolidate ONLY their affected equivalence group!
    """
    start = budget.spent if budget is not None else 0
    priors = list(getattr(library, "meta_priors", ()) or ())
    if not priors:
        empty_idx = PremiseCommitmentIndex([])
        return [], empty_idx, 0

    current_fps = {
        p.get("source_concept_id", f"anon_{i}"): _prior_state_fingerprint(p)
        for i, p in enumerate(priors)
    }
    cached_fps = getattr(library, "_indexed_prior_fingerprints", None)
    cached_index = getattr(library, "_premise_commitment_index", None)
    if cached_fps == current_fps and isinstance(cached_index, PremiseCommitmentIndex):
        if budget is not None:
            budget.spend(1)
        return list(cached_index.commitments), cached_index, (budget.spent - start) if budget is not None else 0

    # Incremental group-level update: only re-consolidate groups whose member priors changed!
    old_group_cache: dict[str, tuple[tuple[str, ...], dict]] = dict(
        getattr(library, "_indexed_group_commitments", None) or {}
    )
    sup_map, ce_map = extract_archive_witness_maps(priors)

    grouped_priors: dict[str, list[dict]] = {}
    group_fps: dict[str, tuple[str, ...]] = {}
    for i, p in enumerate(priors):
        if budget is not None and not budget.spend(2):
            break
        cid = p.get("source_concept_id", f"anon_{i}")
        # Cache extracted group key on the prior dict itself so unchanged priors never re-parse ASTs!
        g_key = p.get("_cached_commitment_group_key")
        if not g_key:
            pat = extract_parametric_role_pattern(p)
            if pat is None:
                continue
            norm_rt = (
                _rebind_latent_pole_in_role_terms(pat["role_terms"], "__LATENT_POLE__")
                if len(pat.get("latent_poles") or ()) == 1
                else (
                    _rebind_vdp_params_in_role_terms(
                        pat["role_terms"], "__VDP_A1__", "__VDP_A2__", "__VDP_BETA__"
                    )
                    if pat.get("vdp_params")
                    else (
                        _rebind_ar2_params_in_role_terms(pat["role_terms"], "__AR2_A1__", "__AR2_A2__")
                        if pat.get("ar2_params")
                        else pat["role_terms"]
                    )
                )
            )
            norm_rt_clean = deepcopy(norm_rt)
            for rt in norm_rt_clean:
                def _norm_node(n_dict: dict) -> None:
                    if n_dict.get("kind") == "all_relation_sum" and isinstance(n_dict.get("predicate"), dict):
                        n_dict["predicate"] = _normalize_edge_condition_for_grouping(n_dict["predicate"])
                    for ch in n_dict.get("children", ()):
                        _norm_node(ch)
                _norm_node(rt)
            g_key = hashlib.sha256(
                json.dumps(
                    {
                        "role_terms": norm_rt_clean,
                        "nonconservation_mechanism": pat.get("nonconservation_mechanism"),
                        "latent_state_mechanism": pat.get("latent_state_mechanism"),
                        "params": [
                            {
                                "symbol": rp["symbol"],
                                "direction": rp["direction"],
                                "edge_condition": _normalize_edge_condition_for_grouping(
                                    rp.get("edge_condition", "unconditional")
                                ),
                            }
                            for rp in pat["radius_parameters"]
                        ],
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()[:20]
            p["_cached_commitment_group_key"] = g_key
        grouped_priors.setdefault(g_key, []).append(p)
        group_fps.setdefault(g_key, ())
        group_fps[g_key] = (*group_fps[g_key], f"{cid}:{current_fps[cid]}")

    updated_group_cache: dict[str, tuple[tuple[str, ...], dict]] = {}
    final_commitments: list[dict] = []
    for idx_g, (g_key, g_list) in enumerate(grouped_priors.items()):
        fp_tuple = tuple(sorted(group_fps[g_key]))
        if g_key in old_group_cache and old_group_cache[g_key][0] == fp_tuple:
            comm = deepcopy(old_group_cache[g_key][1])
            comm["commitment_id"] = f"world_commitment:{idx_g:04d}"
            updated_group_cache[g_key] = (fp_tuple, comm)
            final_commitments.append(comm)
            continue
        g_sup = {k: v for k, v in sup_map.items() if any(p.get("source_concept_id") == k for p in g_list)}
        g_ce = {k: v for k, v in ce_map.items() if any(p.get("source_concept_id") == k for p in g_list)}
        built = consolidate_world_structure_commitments(
            g_list,
            budget=budget,
            support_rows_by_source=g_sup or None,
            counterexample_rows_by_source=g_ce or None,
            tolerance=tolerance,
        )
        if built:
            comm = built[0]
            comm["commitment_id"] = f"world_commitment:{idx_g:04d}"
            updated_group_cache[g_key] = (fp_tuple, deepcopy(comm))
            final_commitments.append(comm)

    new_index = PremiseCommitmentIndex()
    new_index.rebuild(final_commitments, budget=budget)

    library._indexed_prior_fingerprints = current_fps
    library._indexed_group_commitments = updated_group_cache
    library._premise_commitment_index = new_index
    return list(final_commitments), new_index, (budget.spent - start) if budget is not None else 0
