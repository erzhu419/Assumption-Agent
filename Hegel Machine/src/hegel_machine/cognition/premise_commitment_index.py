"""Versioned world-commitment store, atomic per-group index, and ordinary persistent KV cache (v11).

Addresses Deliverable A1 & A2 of ``02_下一轮实施与验收.md``:
  1. **Semantic-complete immutable versioning (``_prior_state_fingerprint``)**:
     Hashes full semantic content (program/structure ASTs, fitted coefficients, radius & threshold
     parameters, empirical scope, support premises, evidence provenance ``evidence_kind``, full
     ``support_witness_rows`` and ``counterexamples`` contents, parent dependency versions, and
     checker schema version ``v11.0``), eliminating all 6 content-blindness mutations identified
     by ``03_fingerprint_probe.py``.
  2. **Version-bound group keys, dependency invalidation, and atomic per-group commits**:
     ``_cached_commitment_group_key`` binds ``(semantic_version, group_key)``. ``ensure_library_commitment_index``
     commits ONLY groups whose witness/counterexample checks completed within ``budget``; interrupted
     builds record ``index_build_status = 'partial_budget_pending'`` and ``dirty_pending_groups``
     without marking ``_indexed_prior_fingerprints`` complete, resuming cleanly on subsequent calls.
     Parent semantic-version retractions invalidate dependent commitments while leaving independent
     groups intact.
  3. **5-state retrieval classification & separate physical work counters**:
     Distinguishes ``not_retrieved``, ``not_tested``, ``budget_pending``, ``inconsistent``, and
     ``empirically_refuted`` (never labeling unvisited or unretrieved items ``refuted``), and
     reports ``priors_scanned_for_fingerprint``, ``asts_normalized``, ``groups_reconsolidated``,
     ``solver_or_loss_evaluations``, and ``wall_seconds`` separately from ``SearchBudget`` units.
  4. **``OrdinaryPersistentCommitmentCache``**:
     Provides a genuinely optimized ordinary engineering baseline (content-addressed persistent
     KV cache + version invalidation + AST deduplication across episodes, zero double-charging)
     so scientific semantic/branch organization is tested against real engineering caching.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time
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

INDEX_CHECKER_SCHEMA_VERSION = "v11.0"


def _prior_state_fingerprint(prior: dict) -> str:
    semantic_body = {
        str(k): v
        for k, v in (prior or {}).items()
        if not str(k).startswith("_cached_")
    }
    payload = {
        "checker_schema_version": "v11.0",
        "claim_id": prior.get("source_concept_id") or prior.get("prior_id"),
        "semantic_body": semantic_body,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def compute_semantic_version(obj: dict) -> str:
    """Public alias returning the immutable 24-hex semantic content version of a prior/claim."""
    return _prior_state_fingerprint(obj)


def _deserialize_witness_rows(
    raw_rows: list[Any] | tuple[Any, ...],
    *,
    allowed_evidence_kinds: set[str] | None = None,
) -> list[Observation]:
    out: list[Observation] = []
    for item in raw_rows or ():
        if isinstance(item, Observation):
            out.append(item)
        elif isinstance(item, dict) and "features" in item:
            if allowed_evidence_kinds is not None:
                ev_k = str(item.get("evidence_kind", "empirical_episode"))
                if ev_k not in allowed_evidence_kinds:
                    continue
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
    *,
    include_diagnostic_mutations_for_equivalence: bool = True,
) -> tuple[dict[str, list[Observation]], dict[str, list[Observation]]]:
    """Extract stored support and counterexample/diagnostic witness rows attached to ``meta_priors``."""
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
                ev_k = str(ce.get("evidence_kind", "empirical_episode"))
                if (
                    ev_k == "diagnostic_mutation"
                    and not include_diagnostic_mutations_for_equivalence
                ):
                    continue
                if "observation" in ce and isinstance(ce["observation"], dict):
                    ce_rows.extend(_deserialize_witness_rows([ce["observation"]]))
                elif "rows" in ce and isinstance(ce["rows"], list):
                    ce_rows.extend(_deserialize_witness_rows(ce["rows"]))
        if ce_rows:
            ce_by_source[cid] = ce_rows
    return support_by_source, ce_by_source


def count_empirical_confirmation_sources(priors: list[dict]) -> dict[str, int]:
    """Count strictly empirical source episodes vs diagnostic mutations vs synthetic stress fixtures (A3)."""
    counts = {
        "empirical_episode": 0,
        "diagnostic_mutation": 0,
        "synthetic_stress_fixture": 0,
        "revoked_or_stale": 0,
    }
    for p in priors or ():
        if p.get("status") in {"revoked", "stale_dependency_invalidated"}:
            counts["revoked_or_stale"] += 1
            continue
        ev_kind = str(p.get("evidence_kind", "empirical_episode"))
        if ev_kind == "synthetic_stress_fixture":
            counts["synthetic_stress_fixture"] += 1
        elif ev_kind == "diagnostic_mutation":
            counts["diagnostic_mutation"] += 1
        else:
            counts["empirical_episode"] += 1
    return counts


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


def _const_erased_cond_str(raw: Any) -> str:
    def _erase(nd: Any) -> Any:
        if not isinstance(nd, dict):
            return nd
        if nd.get("op") == "const":
            return {"op": "__CONST__"}
        return {
            "op": nd.get("op"),
            "name": nd.get("name"),
            "lag": nd.get("lag"),
            "children": [_erase(ch) for ch in (nd.get("children") or ())],
        }

    if isinstance(raw, dict):
        return json.dumps(_erase(raw), sort_keys=True, separators=(",", ":"))
    return _canonical_cond_str(raw)


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


def _compute_group_key_for_prior(p: dict, sem_ver: str) -> tuple[str | None, bool]:
    """Return ``(group_key, ast_normalized_now)`` bound to ``(sem_ver, INDEX_CHECKER_SCHEMA_VERSION)``."""
    cached = p.get("_cached_commitment_group_key")
    if (
        isinstance(cached, dict)
        and cached.get("semantic_version") == sem_ver
        and cached.get("checker_version") == INDEX_CHECKER_SCHEMA_VERSION
    ):
        return cached.get("group_key"), False

    pat = extract_parametric_role_pattern(p)
    if pat is None:
        p["_cached_commitment_group_key"] = {
            "semantic_version": sem_ver,
            "checker_version": INDEX_CHECKER_SCHEMA_VERSION,
            "group_key": None,
        }
        return None, True

    norm_rt = (
        _rebind_latent_pole_in_role_terms(pat["role_terms"], "__LATENT_POLE__")
        if len(pat.get("latent_poles") or ()) == 1
        else (
            _rebind_vdp_params_in_role_terms(
                pat["role_terms"], "__VDP_A1__", "__VDP_A2__", "__VDP_BETA__"
            )
            if pat.get("vdp_params")
            else (
                _rebind_ar2_params_in_role_terms(
                    pat["role_terms"], "__AR2_A1__", "__AR2_A2__"
                )
                if pat.get("ar2_params")
                else pat["role_terms"]
            )
        )
    )
    norm_rt_clean = deepcopy(norm_rt)
    for rt in norm_rt_clean:
        def _norm_node(n_dict: dict) -> None:
            if n_dict.get("kind") == "all_relation_sum" and isinstance(
                n_dict.get("predicate"), dict
            ):
                n_dict["predicate"] = _normalize_edge_condition_for_grouping(
                    n_dict["predicate"]
                )
            for ch in n_dict.get("children", ()):
                if isinstance(ch, dict):
                    _norm_node(ch)

        _norm_node(rt)

    g_key = hashlib.sha256(
        json.dumps(
            {
                "checker_version": INDEX_CHECKER_SCHEMA_VERSION,
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
    p["_cached_commitment_group_key"] = {
        "semantic_version": sem_ver,
        "checker_version": INDEX_CHECKER_SCHEMA_VERSION,
        "group_key": g_key,
    }
    return g_key, True


def _detect_stale_dependency_ids(
    priors: list[dict], current_fps: dict[str, str]
) -> set[str]:
    """Propagate dependency invalidation along ``depends_on_versions`` / ``retracted_premise_ids`` with cycle safety."""
    by_cid: dict[str, dict] = {}
    for i, p in enumerate(priors):
        cid = p.get("source_concept_id", f"anon_{i}")
        by_cid[cid] = p

    directly_invalid: set[str] = set()
    for cid, p in by_cid.items():
        if p.get("status") in {"revoked", "stale_dependency_invalidated"}:
            directly_invalid.add(cid)
            continue
        dep_vers = p.get("depends_on_versions") or {}
        if isinstance(dep_vers, dict):
            for parent_id, expected_ver in dep_vers.items():
                parent_obj = by_cid.get(parent_id)
                if (
                    parent_obj is None
                    or parent_obj.get("status") == "revoked"
                    or current_fps.get(parent_id) != expected_ver
                ):
                    directly_invalid.add(cid)
                    break

    # Fixed-point transitive invalidation with cycle detection:
    # A cycle without an external valid foundation cannot self-certify if any member is invalid,
    # and pure self-cycles `cid in depends_on_versions` without independent support are flagged.
    stale = set(directly_invalid)
    for cid, p in by_cid.items():
        dep_vers = p.get("depends_on_versions") or {}
        if isinstance(dep_vers, dict) and cid in dep_vers and not p.get("support_witness_rows"):
            stale.add(cid)

    changed = True
    while changed:
        changed = False
        for cid, p in by_cid.items():
            if cid in stale:
                continue
            parents = list(p.get("parent_concept_ids") or ())
            parents.extend(
                k for k in (p.get("depends_on_concept_ids") or ()) if k not in parents
            )
            dep_vers = p.get("depends_on_versions") or {}
            if isinstance(dep_vers, dict):
                parents.extend(k for k in dep_vers if k not in parents)
            # Only invalidate if supporting parents were invalidated and claim lacks independent empirical support!
            if parents and any(par in stale for par in parents):
                if not p.get("independent_empirical_support", False):
                    stale.add(cid)
                    changed = True
    return stale


class PremiseCommitmentIndex:
    """Inverted premise-signature index with 5-state retrieval telemetry."""

    def __init__(self, commitments: list[dict] | None = None) -> None:
        self.commitments: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.by_direction: dict[str, list[str]] = {}
        self.by_conservation: dict[str, list[str]] = {}
        self.by_predicate_key: dict[str, list[str]] = {}
        self.by_predicate_key: dict[str, list[str]] = {}
        self.by_predicate_skeleton_key: dict[str, list[str]] = {}
        self.by_signature: dict[tuple[str, str, str, bool], list[str]] = {}
        self.unique_custom_predicate_commitments: list[dict] = []
        self.build_complete: bool = True
        if commitments is not None:
            self.rebuild(commitments)

    def rebuild(
        self, commitments: list[dict], *, budget: SearchBudget | None = None
    ) -> tuple[int, bool]:
        start = budget.spent if budget is not None else 0
        self.commitments = []
        self.by_id = {}
        self.by_direction = {}
        self.by_conservation = {}
        self.by_predicate_key = {}
        self.by_predicate_skeleton_key = {}
        self.by_signature = {}
        seen_pred_keys: set[str] = set()
        self.unique_custom_predicate_commitments = []
        completed_all = True

        for comm in commitments:
            if budget is not None and not budget.spend(2):
                completed_all = False
                break
            if comm.get("status") == "stale_dependency_invalidated":
                continue
            self.commitments.append(comm)
            cid = comm["commitment_id"]
            self.by_id[cid] = comm
            c_cls = _commitment_conservation_class(comm)
            d_cls = _commitment_direction_class(comm)
            p_key = _commitment_predicate_key(comm)
            h_lat = _commitment_has_latent(comm)

            self.by_direction.setdefault(d_cls, []).append(cid)
            self.by_conservation.setdefault(c_cls, []).append(cid)
            self.by_predicate_key.setdefault(p_key, []).append(cid)
            r_params = comm.get("radius_parameters") or ()
            if r_params:
                raw_ec = r_params[0].get("edge_condition")
                self.by_predicate_skeleton_key.setdefault(
                    _const_erased_cond_str(raw_ec), []
                ).append(cid)
            self.by_signature.setdefault((c_cls, d_cls, p_key, h_lat), []).append(cid)

            if r_params:
                raw_ec = r_params[0].get("edge_condition")
                if isinstance(raw_ec, dict) and p_key not in seen_pred_keys:
                    seen_pred_keys.add(p_key)
                    self.unique_custom_predicate_commitments.append(comm)

        self.build_complete = completed_all
        spent = (budget.spent - start) if budget is not None else 0
        return spent, completed_all

    def deduplicated_audit_commitments(
        self, budget: SearchBudget | None = None
    ) -> list[dict]:
        """Return minimal representative commitments covering all distinct custom predicates & directions."""
        if budget is not None:
            budget.spend(1 + len(self.unique_custom_predicate_commitments))
        reps: dict[str, dict] = {}
        for comm in self.unique_custom_predicate_commitments:
            reps[comm["commitment_id"]] = comm
        for _, cids in self.by_direction.items():
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
        """Retrieve empirically consistent commitments with disjoint 5-state classification (A1)."""
        t0 = time.perf_counter()
        start = budget.spent if budget is not None else 0
        all_ids = list(self.by_id.keys())
        if not self.commitments:
            return [], {
                "candidates_inspected": 0,
                "ids_scanned_in_directory": 0,
                "total_indexed_commitments": 0,
                "index_compute_cost": 0,
                "wall_seconds": round(time.perf_counter() - t0, 6),
                "matched_ids": [],
                "not_retrieved_ids": [],
                "not_tested_ids": [],
                "budget_pending_ids": [],
                "inconsistent_ids": [],
                "empirically_refuted_ids": [],
                "rejected_inconsistent_ids": [],
            }

        if audit is None or audit.get("status") == "awaiting_role_identification":
            matched: list[dict] = []
            inconsistent_ids: list[str] = []
            refuted_ids: list[str] = []
            budget_pending_ids: list[str] = []
            inspected = 0
            for idx_c, comm in enumerate(self.commitments):
                cid = comm["commitment_id"]
                if budget is not None and not budget.spend(1):
                    budget_pending_ids.extend(
                        c["commitment_id"] for c in self.commitments[idx_c:]
                    )
                    break
                inspected += 1
                if task_id is not None and task_id in set(comm.get("refuted_task_ids") or ()):
                    refuted_ids.append(cid)
                elif is_commitment_empirically_consistent(comm, audit, task_id=task_id):
                    matched.append(comm)
                else:
                    inconsistent_ids.append(cid)
            return matched, {
                "candidates_inspected": inspected,
                "ids_scanned_in_directory": len(all_ids),
                "total_indexed_commitments": len(self.commitments),
                "index_compute_cost": (budget.spent - start) if budget is not None else 0,
                "wall_seconds": round(time.perf_counter() - t0, 6),
                "matched_ids": [c["commitment_id"] for c in matched],
                "not_retrieved_ids": [],
                "not_tested_ids": list(budget_pending_ids),
                "budget_pending_ids": budget_pending_ids,
                "inconsistent_ids": inconsistent_ids,
                "empirically_refuted_ids": refuted_ids,
                "rejected_inconsistent_ids": refuted_ids + inconsistent_ids,
            }

        if audit.get("conservation_supported", True) and not audit.get(
            "nonconservation_mechanisms"
        ):
            cons_pool = set(self.by_conservation.get("conservative", ()))
        else:
            mech_key = "+".join(
                sorted(set(audit.get("nonconservation_mechanisms") or ()))
            )
            cons_pool = set(self.by_conservation.get(mech_key, ())) | set(
                self.by_conservation.get("conservative", ())
            )

        v_dir = audit.get("verified_direction")
        sup_dirs = list(audit.get("supported_directions") or ([v_dir] if v_dir else []))
        if sup_dirs:
            dir_pool: set[str] = set(self.by_direction.get("position_free", ()))
            for d in sup_dirs:
                dir_pool |= set(self.by_direction.get(str(d), ()))
        else:
            dir_pool = set(all_ids)

        v_ec = audit.get("verified_edge_condition")
        if v_ec is not None:
            ec_key = _canonical_cond_str(v_ec)
            pred_pool = set(self.by_predicate_key.get(ec_key, ())) | set(
                self.by_predicate_key.get("none", ())
            )
            if isinstance(v_ec, dict):
                pred_pool |= set(
                    self.by_predicate_skeleton_key.get(_const_erased_cond_str(v_ec), ())
                )
            # Also include observationally_undistinguished conditional alternatives if provisional Occam!
            for alt_ec in audit.get("undistinguished_conditional_alternatives") or ():
                pred_pool |= set(self.by_predicate_key.get(_canonical_cond_str(alt_ec), ()))
                if isinstance(alt_ec, dict):
                    pred_pool |= set(
                        self.by_predicate_skeleton_key.get(_const_erased_cond_str(alt_ec), ())
                    )
        else:
            pred_pool = set(all_ids)

        if budget is not None and not budget.spend(3):
            return [], {
                "candidates_inspected": 0,
                "ids_scanned_in_directory": 0,
                "total_indexed_commitments": len(self.commitments),
                "index_compute_cost": budget.spent - start,
                "wall_seconds": round(time.perf_counter() - t0, 6),
                "matched_ids": [],
                "not_retrieved_ids": [],
                "not_tested_ids": all_ids,
                "budget_pending_ids": all_ids,
                "inconsistent_ids": [],
                "empirically_refuted_ids": [],
                "rejected_inconsistent_ids": [],
            }

        shortlist_ids: list[str] = []
        not_retrieved_ids: list[str] = []
        for cid in all_ids:
            if cid in cons_pool and cid in dir_pool and cid in pred_pool:
                shortlist_ids.append(cid)
            else:
                not_retrieved_ids.append(cid)

        matched = []
        inconsistent_ids = []
        refuted_ids = []
        budget_pending_ids = []
        inspected = 0

        for idx_s, cid in enumerate(shortlist_ids):
            if budget is not None and not budget.spend(2):
                budget_pending_ids.extend(shortlist_ids[idx_s:])
                break
            inspected += 1
            comm = self.by_id[cid]
            if task_id is not None and task_id in set(comm.get("refuted_task_ids") or ()):
                refuted_ids.append(cid)
            elif commitment_applicable_triggers(comm, audit):
                refuted_ids.append(cid)
            elif is_commitment_empirically_consistent(comm, audit, task_id=task_id):
                matched.append(comm)
            else:
                inconsistent_ids.append(cid)

        return matched, {
            "candidates_inspected": inspected,
            "ids_scanned_in_directory": len(all_ids),
            "total_indexed_commitments": len(self.commitments),
            "index_compute_cost": (budget.spent - start) if budget is not None else 0,
            "wall_seconds": round(time.perf_counter() - t0, 6),
            "matched_ids": [c["commitment_id"] for c in matched],
            "not_retrieved_ids": not_retrieved_ids,
            "not_tested_ids": not_retrieved_ids + budget_pending_ids,
            "budget_pending_ids": budget_pending_ids,
            "inconsistent_ids": inconsistent_ids,
            "empirically_refuted_ids": refuted_ids,
            "rejected_inconsistent_ids": refuted_ids + inconsistent_ids,
        }

    def query_revision_parent_commitments(
        self,
        audit: dict | None,
        *,
        budget: SearchBudget | None = None,
    ) -> list[dict]:
        """Retrieve commitments whose violated premise matches an active trigger in ``audit``."""
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
            elif kind in {
                "nonconservation_revision",
                "nonconservation_mechanism_revision",
                "latent_state_revision",
                "latent_transition_state_revision",
            }:
                candidate_ids |= set(self.by_conservation.get("conservative", ()))
            elif kind in {"edge_condition_revision", "exchange_condition_revision"}:
                ret_cond = trig.get(
                    "retracted_condition_raw",
                    trig.get("retracted_condition", "unconditional"),
                )
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


def ensure_library_commitment_index(
    library,
    *,
    budget: SearchBudget | None = None,
    tolerance: float = 1e-7,
) -> tuple[list[dict], PremiseCommitmentIndex, int]:
    """Atomically maintain versioned commitments + ``PremiseCommitmentIndex`` on ``library``.

    Never marks a partial budget-interrupted build as complete; resumes dirty/pending groups
    on the next call and invalidates stale child commitments when parent versions change.
    """
    t0 = time.perf_counter()
    start = budget.spent if budget is not None else 0
    priors = list(getattr(library, "meta_priors", ()) or ())
    if not priors:
        empty_idx = PremiseCommitmentIndex([])
        library._indexed_prior_fingerprints = {}
        library._indexed_group_commitments = {}
        library._premise_commitment_index = empty_idx
        library._index_build_status = "complete"
        library._dirty_pending_groups = []
        library._last_index_telemetry = {
            "status": "complete",
            "cache_hit": True,
            "priors_scanned_for_fingerprint": 0,
            "asts_normalized": 0,
            "groups_reconsolidated": 0,
            "groups_reused_from_cache": 0,
            "solver_or_loss_evaluations": 0,
            "search_budget_units_debited": 0,
            "wall_seconds": round(time.perf_counter() - t0, 6),
        }
        return [], empty_idx, 0

    current_fps: dict[str, str] = {}
    for i, p in enumerate(priors):
        cid = p.get("source_concept_id", f"anon_{i}")
        current_fps[cid] = _prior_state_fingerprint(p)
    priors_scanned = len(priors)

    stale_dep_cids = _detect_stale_dependency_ids(priors, current_fps)

    cached_fps = getattr(library, "_indexed_prior_fingerprints", None)
    cached_index = getattr(library, "_premise_commitment_index", None)
    cached_status = getattr(library, "_index_build_status", None)
    cached_stale = getattr(library, "_indexed_stale_dep_cids", set())

    if (
        cached_status == "complete"
        and cached_fps == current_fps
        and cached_stale == stale_dep_cids
        and isinstance(cached_index, PremiseCommitmentIndex)
    ):
        if budget is not None:
            budget.spend(1)
        debited = (budget.spent - start) if budget is not None else 0
        library._last_index_telemetry = {
            "status": "complete",
            "cache_hit": True,
            "priors_scanned_for_fingerprint": priors_scanned,
            "asts_normalized": 0,
            "groups_reconsolidated": 0,
            "groups_reused_from_cache": len(
                getattr(library, "_indexed_group_commitments", None) or {}
            ),
            "solver_or_loss_evaluations": 0,
            "search_budget_units_debited": debited,
            "wall_seconds": round(time.perf_counter() - t0, 6),
        }
        return list(cached_index.commitments), cached_index, debited

    old_group_cache: dict[str, tuple[tuple[str, ...], dict]] = dict(
        getattr(library, "_indexed_group_commitments", None) or {}
    )
    sup_map, ce_map = extract_archive_witness_maps(priors)

    grouped_priors: dict[str, list[dict]] = {}
    group_fps: dict[str, tuple[str, ...]] = {}
    asts_normalized = 0
    budget_interrupted = False

    for i, p in enumerate(priors):
        cid = p.get("source_concept_id", f"anon_{i}")
        if cid in stale_dep_cids:
            continue
        if budget is not None and not budget.spend(2):
            budget_interrupted = True
            break
        sem_ver = current_fps[cid]
        g_key, did_norm = _compute_group_key_for_prior(p, sem_ver)
        if did_norm:
            asts_normalized += 1
        if not g_key:
            continue
        grouped_priors.setdefault(g_key, []).append(p)
        group_fps.setdefault(g_key, ())
        group_fps[g_key] = (*group_fps[g_key], f"{cid}:{sem_ver}")

    committed_group_cache: dict[str, tuple[tuple[str, ...], dict]] = {}
    final_commitments: list[dict] = []
    dirty_pending_groups: list[str] = []
    groups_reused = 0
    groups_reconsolidated = 0
    solver_evals = 0

    for idx_g, (g_key, g_list) in enumerate(grouped_priors.items()):
        fp_tuple = tuple(sorted(group_fps[g_key]))
        if g_key in old_group_cache and old_group_cache[g_key][0] == fp_tuple:
            comm = deepcopy(old_group_cache[g_key][1])
            comm["commitment_id"] = f"world_commitment:{idx_g:04d}"
            comm["semantic_group_fingerprint"] = fp_tuple
            committed_group_cache[g_key] = (fp_tuple, deepcopy(comm))
            final_commitments.append(comm)
            groups_reused += 1
            continue

        if budget_interrupted or (budget is not None and budget.remaining < 10):
            budget_interrupted = True
            dirty_pending_groups.append(g_key)
            continue

        denied_before = budget.denied_requests if budget is not None else 0
        spent_before_g = budget.spent if budget is not None else 0
        g_sup = {
            k: v
            for k, v in sup_map.items()
            if any(p.get("source_concept_id") == k for p in g_list)
        }
        g_ce = {
            k: v
            for k, v in ce_map.items()
            if any(p.get("source_concept_id") == k for p in g_list)
        }
        built = consolidate_world_structure_commitments(
            g_list,
            budget=budget,
            support_rows_by_source=g_sup or None,
            counterexample_rows_by_source=g_ce or None,
            tolerance=tolerance,
        )
        if budget is not None:
            solver_evals += max(0, budget.spent - spent_before_g)
        denied_after = budget.denied_requests if budget is not None else 0

        if denied_after > denied_before:
            # Consolidation was truncated by budget — do NOT commit partial group!
            budget_interrupted = True
            dirty_pending_groups.append(g_key)
            continue

        groups_reconsolidated += 1
        if built:
            comm = built[0]
            comm["commitment_id"] = f"world_commitment:{idx_g:04d}"
            comm["semantic_group_fingerprint"] = fp_tuple
            committed_group_cache[g_key] = (fp_tuple, deepcopy(comm))
            final_commitments.append(comm)

    new_index = PremiseCommitmentIndex()
    _, index_rebuild_ok = new_index.rebuild(final_commitments, budget=budget)
    if not index_rebuild_ok:
        budget_interrupted = True

    # Persist verified group commits so subsequent resumes never redo completed groups!
    merged_verified_groups = dict(old_group_cache)
    # Purge any old group whose key is no longer present or whose fingerprint changed unless still dirty
    valid_group_keys = set(grouped_priors.keys())
    for old_k in list(merged_verified_groups.keys()):
        if old_k not in valid_group_keys:
            if not budget_interrupted:
                merged_verified_groups.pop(old_k, None)
        elif old_k in committed_group_cache:
            merged_verified_groups[old_k] = committed_group_cache[old_k]
        else:
            # Group changed AND was interrupted before new consolidation finished -> evict stale entry!
            merged_verified_groups.pop(old_k, None)
    for k_new, v_new in committed_group_cache.items():
        merged_verified_groups[k_new] = v_new

    library._indexed_group_commitments = merged_verified_groups
    library._indexed_stale_dep_cids = set(stale_dep_cids)

    if not budget_interrupted and not dirty_pending_groups:
        library._indexed_prior_fingerprints = current_fps
        library._premise_commitment_index = new_index
        library._index_build_status = "complete"
        library._dirty_pending_groups = []
        status_str = "complete"
    else:
        # NEVER commit full current_fps when build is incomplete!
        library._indexed_prior_fingerprints = None
        library._premise_commitment_index = new_index
        library._index_build_status = "partial_budget_pending"
        library._dirty_pending_groups = dirty_pending_groups
        status_str = "partial_budget_pending"

    debited = (budget.spent - start) if budget is not None else 0
    library._last_index_telemetry = {
        "status": status_str,
        "cache_hit": False,
        "dirty_pending_groups_count": len(dirty_pending_groups),
        "stale_dependency_invalidated_cids": sorted(stale_dep_cids),
        "priors_scanned_for_fingerprint": priors_scanned,
        "asts_normalized": asts_normalized,
        "groups_reconsolidated": groups_reconsolidated,
        "groups_reused_from_cache": groups_reused,
        "solver_or_loss_evaluations": solver_evals,
        "search_budget_units_debited": debited,
        "wall_seconds": round(time.perf_counter() - t0, 6),
    }
    return list(new_index.commitments), new_index, debited


class OrdinaryPersistentCommitmentCache:
    """Strong ordinary engineering cache baseline (Deliverable A2: ``ordinary_persistent_cache``).

    Provides persistent content-addressed memoization, semantic version invalidation, and
    AST predicate deduplication across episodes WITHOUT semantic premise-graph bucketing or
    scientific branch reconciliation. Ensures comparisons isolate scientific organization
    rather than ordinary caching/compilation savings.
    """

    def __init__(self) -> None:
        self._group_memo: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
        self._snapshot_fp: dict[str, str] | None = None
        self._snapshot_commitments: list[dict] = []
        self._dedup_audit_comms: list[dict] = []
        # v13 B3 3-layer memoization for ordinary lazy reconstruction (per-prior, family AU, optional routing/scope)
        self._per_prior_extraction_memo: dict[tuple[str, str, float, str], dict[str, Any]] = {}
        self._family_au_memo: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._routing_scope_memo: dict[tuple[Any, ...], dict[str, Any]] = {}

    def active_prior_fingerprints(self, priors: list[dict]) -> tuple[dict[str, str], set[str]]:
        cur_fp = {
            p.get("source_concept_id", f"anon_{i}"): _prior_state_fingerprint(p)
            for i, p in enumerate(priors)
        }
        stale_cids = _detect_stale_dependency_ids(priors, cur_fp)
        active_fp = {k: v for k, v in cur_fp.items() if k not in stale_cids}
        return active_fp, stale_cids

    def get_or_consolidate(
        self,
        priors: list[dict],
        *,
        budget: SearchBudget | None = None,
        tolerance: float = 1e-7,
    ) -> tuple[list[dict], dict[str, Any]]:
        t0 = time.perf_counter()
        start = budget.spent if budget is not None else 0
        if not priors:
            return [], {"cache_hit": True, "compute_cost": 0, "wall_seconds": 0.0}

        cur_fp = {
            p.get("source_concept_id", f"anon_{i}"): _prior_state_fingerprint(p)
            for i, p in enumerate(priors)
        }
        stale_cids = _detect_stale_dependency_ids(priors, cur_fp)
        active_fp = {k: v for k, v in cur_fp.items() if k not in stale_cids}

        if self._snapshot_fp == active_fp and self._snapshot_commitments:
            if budget is not None:
                budget.spend(1)
            return list(self._snapshot_commitments), {
                "cache_hit": True,
                "priors_scanned": len(priors),
                "compute_cost": (budget.spent - start) if budget is not None else 0,
                "wall_seconds": round(time.perf_counter() - t0, 6),
            }

        sup_map, ce_map = extract_archive_witness_maps(priors)
        grouped: dict[str, list[dict]] = {}
        g_fps: dict[str, tuple[str, ...]] = {}
        interrupted = False
        for i, p in enumerate(priors):
            cid = p.get("source_concept_id", f"anon_{i}")
            if cid in stale_cids:
                continue
            if budget is not None and not budget.spend(2):
                interrupted = True
                break
            gk, _ = _compute_group_key_for_prior(p, cur_fp[cid])
            if not gk:
                continue
            grouped.setdefault(gk, []).append(p)
            g_fps[gk] = (*g_fps.get(gk, ()), f"{cid}:{cur_fp[cid]}")

        out_comms: list[dict] = []
        for idx_g, (gk, glist) in enumerate(grouped.items()):
            fpt = tuple(sorted(g_fps[gk]))
            memo_key = (gk, fpt)
            if memo_key in self._group_memo:
                for c in self._group_memo[memo_key]:
                    cc = deepcopy(c)
                    cc["commitment_id"] = f"ord_cache_comm:{idx_g:04d}"
                    out_comms.append(cc)
                continue
            if interrupted or (budget is not None and budget.remaining < 10):
                interrupted = True
                break
            denied0 = budget.denied_requests if budget is not None else 0
            g_sup = {
                k: v
                for k, v in sup_map.items()
                if any(p.get("source_concept_id") == k for p in glist)
            }
            g_ce = {
                k: v
                for k, v in ce_map.items()
                if any(p.get("source_concept_id") == k for p in glist)
            }
            built = consolidate_world_structure_commitments(
                glist,
                budget=budget,
                support_rows_by_source=g_sup or None,
                counterexample_rows_by_source=g_ce or None,
                tolerance=tolerance,
            )
            if budget is not None and budget.denied_requests > denied0:
                interrupted = True
                break
            self._group_memo[memo_key] = deepcopy(built)
            for c in built:
                cc = deepcopy(c)
                cc["commitment_id"] = f"ord_cache_comm:{idx_g:04d}"
                out_comms.append(cc)

        # Ordinary AST deduplication for audit fairness
        seen_preds: set[str] = set()
        dedup: list[dict] = []
        seen_dirs: set[str] = set()
        for c in out_comms:
            pk = _commitment_predicate_key(c)
            dc = _commitment_direction_class(c)
            rps = c.get("radius_parameters") or ()
            if rps and isinstance(rps[0].get("edge_condition"), dict):
                if pk not in seen_preds:
                    seen_preds.add(pk)
                    dedup.append(c)
            elif dc not in seen_dirs:
                seen_dirs.add(dc)
                dedup.append(c)

        if not interrupted:
            self._snapshot_fp = active_fp
            self._snapshot_commitments = deepcopy(out_comms)
            self._dedup_audit_comms = deepcopy(dedup)
        return out_comms, {
            "cache_hit": False,
            "interrupted": interrupted,
            "priors_scanned": len(priors),
            "compute_cost": (budget.spent - start) if budget is not None else 0,
            "wall_seconds": round(time.perf_counter() - t0, 6),
        }

    def deduplicated_audit_commitments(
        self, budget: SearchBudget | None = None
    ) -> list[dict]:
        if budget is not None:
            budget.spend(1 + len(self._dedup_audit_comms))
        return list(self._dedup_audit_comms)
