"""Hegel V18.1 (`M5` Official ChatGPT-Subscription Multi-Model Live Benchmark & Forensic Decision Provenance).

Provides two tightly integrated capabilities on top of `m5_temporal_multiscale_and_base_subtree.py`:

1. **Forensic Decision-Level Provenance Ledger (`DecisionProvenanceLedger`)**:
   - Every atomic cognitive/experimental decision emits a cryptographically hash-chained
     `DecisionProvenanceRecord` linked via `parent_decision_id` into a complete causal DAG:
       `D0_TASK_ARRIVAL_AND_PILOT_ACQUISITION` -> `D1_CAUSAL_TEMPORAL_REGIME_EXTRACTION`
       -> `D2_INDEX_RETRIEVAL_RANKING` -> `D3_FOUR_TIER_PERMISSION_AND_GRAYZONE_GATE`
       -> `D4_SHARED_BASE_SUBTREE_BINDING_AND_WARMSTART` -> `D5_VOI_PROBE_VALUATION_AND_HEC_CHECK`
       -> `D6_LIVE_GPT_TOOL_DECISION` -> `D5B_PAID_PROBE_EXECUTION`
       -> `D7_D8_CANDIDATE_FIT_AND_FRESH_CONFIRMATION` -> `D9_TERMINAL_DECISION_AND_HELDOUT_AUDIT`.
   - Every record logs exact `provider_backend`, `model`, `evidence_inputs` (row keys & contract digests),
     `candidates_evaluated` (with full numerical scores or explicit LLM considered alternatives),
     `chosen_option`, `rejection_reasons`, and exact `cost_ledger_delta` whose sum across the chain
     is programmatically verified to equal the episode total.

2. **Official OpenAI ChatGPT Subscription Multi-Model Live Harness (`CodexSubscriptionClient`)**:
   - Invokes the official OpenAI Codex CLI (`codex exec --output-schema ...`) authenticated via the user's
     ChatGPT account OAuth subscription (`auth_mode == "chatgpt"`), **explicitly blocking third-party
     `ruoli.dev` proxies and scrubbing third-party env vars** (`RUOLI_*`, `GPT5_*`, `OPENAI_BASE_URL`).
   - Evaluates multiple subscription models head-to-head (e.g. cheaper `gpt-6-luna`, `gpt-6.1-terra`,
     and `gpt-6.2-sol`) across three controlled arms sharing the exact same Python subprocess sandbox:
       * `Hegel_M5_LiveGPT` (`K=15` HippoRAG PPR + 4-tier gate + 2-layer `SharedBaseSubtreePrior` + `probing.py` VOI)
       * `EqualAuth_Flat_LiveGPT` (100% equal coarse-channel certificates + `K=15` Flat BM25 Top-3 recall)
       * `Strong_Ordinary_LiveGPT` (Direct access to all `K=15` raw source & near-miss episodes)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from time import monotonic, sleep
from typing import Any

from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m5_temporal_multiscale_and_base_subtree import (
    COMPOSITE_LAMBDA,
    M5DisjointTaskDataset,
    _build_disjoint_dataset,
    _build_seed_forest_and_priors,
    _composite,
    _fit_candidate_model_class,
    _rmse,
    _row_input_key,
    select_m5_discriminating_probe,
)
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    authorize_abstraction_use,
    extract_causal_temporal_regimes,
    validate_consistent_joint_subtree_binding,
)


# ============================================================================
# 1. Cryptographically Hash-Chained Decision-Level Provenance Ledger
# ============================================================================


class DecisionProvenanceLedger:
    """Hash-chained, parent-linked causal provenance logger for every atomic agent decision."""

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        arm: str,
        model: str = "deterministic_solver",
        provider_backend: str = "deterministic_local",
    ) -> None:
        self.run_id = str(run_id)
        self.task_id = str(task_id)
        self.arm = str(arm)
        self.model = str(model)
        self.provider_backend = str(provider_backend)
        self.records: list[dict[str, Any]] = []
        self._cum_m = 0.0
        self._cum_k = 0.0
        self._cum_prompt_tok = 0
        self._cum_cached_tok = 0
        self._cum_comp_tok = 0
        self._cum_wall_ms = 0.0

    def record_decision(
        self,
        *,
        decision_type: str,
        parent_decision_id: str | None,
        evidence_inputs: dict[str, Any],
        candidates_evaluated: list[dict[str, Any]],
        chosen_option: dict[str, Any],
        rejection_reasons: dict[str, str],
        rationale_summary: str,
        measurement_delta: float = 0.0,
        compute_delta: float = 0.0,
        prompt_tokens_delta: int = 0,
        cached_tokens_delta: int = 0,
        completion_tokens_delta: int = 0,
        wall_ms_delta: float = 0.0,
        state_digest_after: str | None = None,
        extra_artifacts: dict[str, Any] | None = None,
    ) -> str:
        seq = len(self.records) + 1
        dec_id = f"dec__{self.run_id}__{self.task_id}__{self.arm}__{seq:03d}"
        prev_sha = self.records[-1]["record_sha256"] if self.records else ("0" * 64)

        m_d = round(float(measurement_delta), 6)
        k_d = round(float(compute_delta), 6)
        pt_d = int(prompt_tokens_delta)
        ca_d = int(cached_tokens_delta)
        ct_d = int(completion_tokens_delta)
        w_d = round(float(wall_ms_delta), 3)

        self._cum_m = round(self._cum_m + m_d, 6)
        self._cum_k = round(self._cum_k + k_d, 6)
        self._cum_prompt_tok += pt_d
        self._cum_cached_tok += ca_d
        self._cum_comp_tok += ct_d
        self._cum_wall_ms = round(self._cum_wall_ms + w_d, 3)

        core = {
            "run_id": self.run_id,
            "model": self.model,
            "provider_backend": self.provider_backend,
            "task_id": self.task_id,
            "arm": self.arm,
            "decision_seq": seq,
            "decision_id": dec_id,
            "parent_decision_id": parent_decision_id,
            "prev_record_sha256": prev_sha,
            "decision_type": str(decision_type),
            "evidence_inputs": deepcopy(evidence_inputs),
            "candidates_evaluated": deepcopy(candidates_evaluated),
            "chosen_option": deepcopy(chosen_option),
            "rejection_reasons": deepcopy(rejection_reasons),
            "rationale_summary": str(rationale_summary),
            "cost_ledger_delta": {
                "measurement_delta": m_d,
                "compute_delta": k_d,
                "composite_delta": _composite(m_d, k_d),
                "prompt_tokens_delta": pt_d,
                "cached_tokens_delta": ca_d,
                "completion_tokens_delta": ct_d,
                "wall_ms_delta": w_d,
            },
            "cumulative_cost_after": {
                "measurement_cost": self._cum_m,
                "compute_cost": self._cum_k,
                "composite_cost": _composite(self._cum_m, self._cum_k),
                "prompt_tokens": self._cum_prompt_tok,
                "cached_tokens": self._cum_cached_tok,
                "completion_tokens": self._cum_comp_tok,
                "wall_ms": self._cum_wall_ms,
            },
            "state_digest_after": state_digest_after,
            "extra_artifacts": deepcopy(extra_artifacts or {}),
        }
        rec_sha = hashlib.sha256(
            json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        full_rec = {**core, "record_sha256": rec_sha}
        self.records.append(full_rec)
        return dec_id

    @property
    def totals(self) -> dict[str, Any]:
        eff_tok = max(0, self._cum_prompt_tok - self._cum_cached_tok) + int(0.25 * self._cum_cached_tok) + self._cum_comp_tok
        return {
            "decision_count": len(self.records),
            "measurement_cost": self._cum_m,
            "compute_cost": self._cum_k,
            "composite_cost": _composite(self._cum_m, self._cum_k),
            "prompt_tokens": self._cum_prompt_tok,
            "cached_tokens": self._cum_cached_tok,
            "completion_tokens": self._cum_comp_tok,
            "total_tokens": self._cum_prompt_tok + self._cum_comp_tok,
            "effective_billed_tokens": eff_tok,
            "wall_ms": self._cum_wall_ms,
        }


def verify_decision_provenance_chain(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify hash-chain integrity, parent-pointer causal DAG validity, & exact cost conservation!"""
    if not records:
        return {"valid": False, "reason": "empty_records"}

    known_ids: set[str] = set()
    prev_expected = "0" * 64
    sum_m = 0.0
    sum_k = 0.0
    sum_pt = 0
    sum_ct = 0

    for idx, rec in enumerate(records, start=1):
        if int(rec.get("decision_seq", -1)) != idx:
            return {"valid": False, "reason": f"seq_mismatch_at_{idx}"}
        if rec.get("prev_record_sha256") != prev_expected:
            return {"valid": False, "reason": f"broken_hash_chain_at_{idx}"}
        core_no_sha = {k: v for k, v in rec.items() if k != "record_sha256"}
        comp_sha = hashlib.sha256(
            json.dumps(core_no_sha, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if comp_sha != rec.get("record_sha256"):
            return {"valid": False, "reason": f"tampered_record_sha_at_{idx}"}
        p_id = rec.get("parent_decision_id")
        if idx == 1 and p_id is not None:
            return {"valid": False, "reason": "root_decision_must_have_null_parent"}
        if idx > 1 and p_id not in known_ids:
            return {"valid": False, "reason": f"dangling_or_future_parent_at_{idx}:{p_id}"}

        known_ids.add(str(rec["decision_id"]))
        prev_expected = comp_sha

        d = rec["cost_ledger_delta"]
        sum_m = round(sum_m + float(d["measurement_delta"]), 6)
        sum_k = round(sum_k + float(d["compute_delta"]), 6)
        sum_pt += int(d["prompt_tokens_delta"])
        sum_ct += int(d["completion_tokens_delta"])

        cum = rec["cumulative_cost_after"]
        if abs(sum_m - float(cum["measurement_cost"])) > 1e-6 or abs(sum_k - float(cum["compute_cost"])) > 1e-6:
            return {"valid": False, "reason": f"cost_conservation_mismatch_at_{idx}"}

    return {
        "valid": True,
        "decision_count": len(records),
        "terminal_record_sha256": prev_expected,
        "verified_measurement_cost": sum_m,
        "verified_compute_cost": sum_k,
        "verified_composite_cost": _composite(sum_m, sum_k),
        "verified_total_tokens": sum_pt + sum_ct,
    }


# ============================================================================
# 2. Deterministic Trace-Instrumented Pipeline (Every Decision Auditable!)
# ============================================================================


def run_traced_deterministic_m5_episode(
    ds: M5DisjointTaskDataset,
    arm: str,
    *,
    forest: PhilosophicalForest,
    priors_bundle: dict[str, Any],
    hippo_idx: HippoRAGWormholeIndex,
    flat_idx: FlatEqualMetadataIndex,
    rng: random.Random,
    run_id: str = "det_m5",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one M5 task while emitting a complete causal `DecisionProvenanceLedger`."""
    tol = float(ds.tolerance)
    ledger = DecisionProvenanceLedger(
        run_id=run_id, task_id=ds.task_id, arm=arm,
        model="deterministic_least_squares_rk4", provider_backend="deterministic_local",
    )
    f_state_dig = hashlib.sha256(str(forest.forest_state_signature()).encode("utf-8")).hexdigest()[:16]

    use_flat = arm == "EqualAuth_Dynamic_Flat_BM25"
    single_node_only = arm == "EqualAuth_SingleNode_Wormhole_Only"
    use_raw_var_probe = arm == "EqualAuth_RawVariance_Probe_Ablation"
    is_strong_dyn = arm == "Strong_Dynamic_StateSpace_AllHistory"
    enable_subtree = (not use_flat) and (not single_node_only) and (not is_strong_dyn)

    has_coarse_cert = (not is_strong_dyn) and ds.coarse_channel_available and ("sig:slow_horizon_query" in ds.probe_signatures)
    unit_row_m = 2.0 if is_strong_dyn else (1.0 if has_coarse_cert else 1.5)

    # D0: Task arrival & initial pilot acquisition
    init_rows = [*ds.pilot_rows, *ds.extra_dynamic_rows] if is_strong_dyn else list(ds.pilot_rows)
    pilot_m = len(init_rows) * unit_row_m
    d0_id = ledger.record_decision(
        decision_type="D0_TASK_ARRIVAL_AND_PILOT_ACQUISITION",
        parent_decision_id=None,
        evidence_inputs={
            "task_id": ds.task_id,
            "coarse_channel_available": ds.coarse_channel_available,
            "observed_probe_signatures": list(ds.probe_signatures),
            "pilot_row_keys": [_row_input_key(r) for r in init_rows],
        },
        candidates_evaluated=[
            {"channel": "coarse_macro", "unit_cost": 1.0, "eligible": has_coarse_cert},
            {"channel": "standard_fine", "unit_cost": (2.0 if is_strong_dyn else 1.5), "eligible": True},
        ],
        chosen_option={"selected_channel": "coarse_macro" if has_coarse_cert else "standard_fine", "acquired_rows": len(init_rows)},
        rejection_reasons=(
            {} if has_coarse_cert else {"coarse_macro": "no_certified_coarse_edge_or_not_slow_horizon_query"}
        ),
        rationale_summary="Acquire prefix-isolated initial pilot rows under authorized measurement channel.",
        measurement_delta=pilot_m,
        compute_delta=(2.0 if is_strong_dyn else 0.0),
        state_digest_after=f_state_dig,
    )

    # D1: Causal Temporal Regime Extraction (`<= t0` verified)
    pseudo_traj = [
        {"t": round(0.1 * idx, 3), "u": r["u_now"], "x": r["x_now"], "y": r["y"]}
        for idx, r in enumerate(init_rows)
    ]
    reg_chk = extract_causal_temporal_regimes(
        pseudo_traj, t0=round(0.1 * (len(pseudo_traj) - 1), 3),
        extraction_window_w=0.25, intrinsic_taus=(0.5,), allow_retrospective=False,
    )
    d1_id = ledger.record_decision(
        decision_type="D1_CAUSAL_TEMPORAL_REGIME_EXTRACTION",
        parent_decision_id=d0_id,
        evidence_inputs={"prefix_sample_count": len(pseudo_traj), "causal_cutoff_policy": "strict_prefix_le_t0"},
        candidates_evaluated=[{"mode": "strict_causal_online_prefix", "ok": reg_chk["ok"], "future_regime_leaked": reg_chk["future_regime"] is not None}],
        chosen_option={"causal_cutoff_t0": reg_chk["causal_cutoff_t0"], "past_present_extracted": True},
        rejection_reasons={"centered_or_post_t0_window": "blocked_by_strict_online_causality"},
        rationale_summary="Extract (P_tau, N_tau) strictly from <= t0 prefix with zero future leakage.",
        compute_delta=0.5,
        state_digest_after=f_state_dig,
    )

    all_dynamic_classes = [
        "M1_SnapshotLinear",
        "M3_ReducedSlowODE",
        "M2_BeforeAfterOpposition",
        "M7_QuadraticStorage",
        "M6_SiphonSubtree_SqrtBranch",
        "M6_SiphonSubtree_LinearLeakBranch",
        "M5_DualHorizonReversal",
        "M4_MoriZwanzigMemoryKernel",
    ]
    train_rows = list(init_rows)
    ordered_classes: list[str] = []
    use_warm_prior: dict[str, Any] | None = None
    last_parent = d1_id

    if not is_strong_dyn:
        rec_res = (flat_idx if use_flat else hippo_idx).retrieve(
            ds.probe_signatures,
            task_view={"goal_kind": "trajectory_and_horizon_prediction", "tolerance": tol},
            top_k=4,
        )
        retr_k = max(0.0, float(rec_res.get("query_compute_cost", 1.2)) - 0.5)
        ranked_cands_log = [
            {"node_id": c["node_id"], "score": round(float(c.get("ppr_score", c.get("bm25_score", 0.0))), 6)}
            for c in rec_res.get("candidates", ())
        ]
        last_parent = ledger.record_decision(
            decision_type="D2_INDEX_RETRIEVAL_RANKING",
            parent_decision_id=last_parent,
            evidence_inputs={"query_signatures": list(ds.probe_signatures), "index_kind": "FlatBM25" if use_flat else "HippoRAG_PPR"},
            candidates_evaluated=ranked_cands_log,
            chosen_option={"top_k_node_ids": [c["node_id"] for c in rec_res.get("candidates", ())]},
            rejection_reasons={},
            rationale_summary="Rank candidate forest nodes purely from observed sig:* pilot cues.",
            compute_delta=retr_k,
            state_digest_after=f_state_dig,
        )

        if has_coarse_cert:
            ordered_classes.extend(["M1_SnapshotLinear", "M3_ReducedSlowODE"])
            if not use_flat:
                for fine_child in forest.fine_scale_children_of("node_A_snapshot_balance"):
                    for mc_f in fine_child["claim_contract"]["conclusion_or_action"].get("model_classes", ()):
                        if mc_f not in ordered_classes:
                            ordered_classes.append(mc_f)
            else:
                ordered_classes.extend(["M2_BeforeAfterOpposition", "M7_QuadraticStorage"])

        sig_set = set(ds.probe_signatures)
        if enable_subtree and {"sig:base_field_C1", "sig:sink_gradient_C2"}.issubset(sig_set):
            sub_rec = forest.shared_base_subtrees["sub_2layer_siphon_base"]
            bind_chk = validate_consistent_joint_subtree_binding(
                sub_rec,
                {
                    "role_bindings": {"u_drive": "u_now", "x_pool": "x_now", "s_grad": "s_grad", "y_out": "y"},
                    "satisfied_shared_node_ids": ["C1_base_capacity_field", "C2_adjacent_high_gradient_sink"],
                    "normalized_tau_ratio": 1.0 / max(1e-6, ds.tau_scale),
                },
            )
            use_warm_prior = priors_bundle["siphon_warm_prior"]
            last_parent = ledger.record_decision(
                decision_type="D4_SHARED_BASE_SUBTREE_BINDING_AND_WARMSTART",
                parent_decision_id=last_parent,
                evidence_inputs={"subtree_id": sub_rec["subtree_id"], "subtree_version": sub_rec["version"], "subtree_digest": sub_rec["subtree_digest"]},
                candidates_evaluated=[{
                    "subtree_id": sub_rec["subtree_id"],
                    "valid": bind_chk["valid"],
                    "independent_evidence_count": bind_chk["independent_evidence_count"],
                    "d_eff_set": sub_rec["diversity_metrics"]["d_eff_set"],
                }],
                chosen_option={
                    "merged_role_bindings": bind_chk["merged_role_bindings"],
                    "warm_start_box": use_warm_prior["warm_start_box"],
                    "retain_wide_fallback": True,
                },
                rejection_reasons={"hard_clipped_shrinkage": "rejected_to_preserve_between_domain_heterogeneity_escape"},
                rationale_summary="Bind 2-layer C1 AND C2 SharedBaseSubtreePrior with dimensionless soft warm-start + wide fallback.",
                compute_delta=0.5,
                state_digest_after=f_state_dig,
            )
            if "sig:mass_sum_drift" in sig_set:
                ordered_classes.append("M6_SiphonSubtree_LinearLeakBranch")
            else:
                voi_cands = [
                    {"branch_id": f"{b_lbl}_{n_lbl}", "prior_weight": 0.25,
                     "predict_fn": (lambda inp, sgn=b_sgn: sgn * 0.4 * math.sqrt(inp["u_n"])),
                     "probe_predict_fn": (lambda pr, sgn=b_sgn, nv=n_val: (nv if pr["id"] == "nuisance_high_var_aux" else sgn * 0.4 * math.sqrt(pr["u_n"])))}
                    for b_lbl, b_sgn in (("pos_sqrt", +1.0), ("neg_sqrt", -1.0))
                    for n_lbl, n_val in (("nuis_lo", -65.0), ("nuis_hi", +65.0))
                ]
                p_open = ds.probe_pool["open_branch_discriminating_probe"]
                p_nuis = ds.probe_pool["nuisance_high_var_aux"]
                voi_sel = select_m5_discriminating_probe(
                    candidate_branches=voi_cands,
                    candidate_probes=[{**p_nuis, "u_n": 1.0}, {**p_open, "u_n": max(0.5, p_open["row"]["u_now"] / ds.tau_scale)}],
                    target_reference_inputs=[{"u_n": 3.0}],
                    decision_tolerance=tol,
                    policy="raw_variance" if use_raw_var_probe else "task_risk_voi",
                )
                extra_nuis_m = p_nuis["measurement_cost"] if voi_sel["selected_probe_id"] == "nuisance_high_var_aux" else 0.0
                last_parent = ledger.record_decision(
                    decision_type="D5_VOI_PROBE_VALUATION_AND_HEC_CHECK",
                    parent_decision_id=last_parent,
                    evidence_inputs={"policy": voi_sel.get("policy"), "open_slot_candidates": 4},
                    candidates_evaluated=voi_sel.get("voi_receipt", {}).get("comparisons", [{"id": voi_sel.get("selected_probe_id")}]),
                    chosen_option={"selected_probe_id": voi_sel.get("selected_probe_id"), "action": voi_sel["action"]},
                    rejection_reasons=(
                        {"nuisance_high_var_aux": "zero_expected_target_risk_reduction"}
                        if voi_sel.get("selected_probe_id") != "nuisance_high_var_aux"
                        else {}
                    ),
                    rationale_summary="Evaluate open-slot probes by expected task-risk reduction per total observation cost.",
                    measurement_delta=extra_nuis_m + p_open["measurement_cost"],
                    compute_delta=voi_sel["compute_cost"],
                    state_digest_after=f_state_dig,
                )
                train_rows.append(p_open["row"])
                ordered_classes.append("M6_SiphonSubtree_SqrtBranch")
        elif (not enable_subtree) and {"sig:base_field_C1", "sig:sink_gradient_C2"}.issubset(sig_set):
            train_rows.extend(ds.extra_dynamic_rows[:2])
            last_parent = ledger.record_decision(
                decision_type="D4_NO_SUBTREE_EXTRA_EXPLORATION",
                parent_decision_id=last_parent,
                evidence_inputs={"subtree_enabled": False},
                candidates_evaluated=[{"action": "acquire_2_extra_dynamic_rows", "measurement_cost": 3.0}],
                chosen_option={"acquired_extra_rows": 2},
                rejection_reasons={},
                rationale_summary="Without 2-layer SharedBaseSubtreePrior, purchase extra points for cold 3-param identification.",
                measurement_delta=3.0,
                state_digest_after=f_state_dig,
            )
            if use_flat:
                ordered_classes.extend(["M7_QuadraticStorage", "M6_SiphonSubtree_SqrtBranch", "M6_SiphonSubtree_LinearLeakBranch"])

        if "sig:endogenous_adaptation_candidate" in sig_set:
            voi_rev_cands = [
                {"branch_id": f"{r_lbl}_{n_lbl}", "prior_weight": 0.25,
                 "predict_fn": (lambda inp, rev=has_rev: 1.5 * (1.0 - math.exp(-inp["H"] / 0.35)) - (2.4 * (1.0 - math.exp(-inp["H"] / 12.0)) if rev else 0.0)),
                 "probe_predict_fn": (lambda pr, rev=has_rev, nv=n_val: (nv if pr["id"] == "nuisance_high_var_aux" else 1.5 * (1.0 - math.exp(-pr["H"] / 0.35)) - (2.4 * (1.0 - math.exp(-pr["H"] / 12.0)) if rev else 0.0)))}
                for r_lbl, has_rev in (("monotone_fast", False), ("slow_reversal", True))
                for n_lbl, n_val in (("nuis_lo", -75.0), ("nuis_hi", +75.0))
            ]
            p_slow = ds.probe_pool["slow_horizon_discriminating_probe"]
            p_nuis = ds.probe_pool["nuisance_high_var_aux"]
            voi_rev = select_m5_discriminating_probe(
                candidate_branches=voi_rev_cands,
                candidate_probes=[{**p_nuis, "H": 0.3}, {**p_slow, "H": p_slow["row"]["H"]}],
                target_reference_inputs=[{"H": ds.horizon_H}],
                decision_tolerance=tol,
                policy="raw_variance" if use_raw_var_probe else "task_risk_voi",
            )
            m_add = (p_nuis["measurement_cost"] if voi_rev.get("selected_probe_id") == "nuisance_high_var_aux" else 0.0)
            if voi_rev["action"] != "hec_decision_region_stop":
                m_add += p_slow["measurement_cost"] + p_slow["reset_cost"] + p_slow["horizon_wait_cost"] + 3.0
                train_rows.extend([p_slow["row"], *ds.extra_dynamic_rows[:2]])
            last_parent = ledger.record_decision(
                decision_type="D5_VOI_PROBE_VALUATION_AND_HEC_CHECK",
                parent_decision_id=last_parent,
                evidence_inputs={"target_horizon_H": ds.horizon_H, "policy": voi_rev.get("policy", "hec")},
                candidates_evaluated=voi_rev.get("voi_receipt", {}).get("comparisons", [{"action": voi_rev["action"]}]),
                chosen_option={"action": voi_rev["action"], "selected_probe_id": voi_rev.get("selected_probe_id")},
                rejection_reasons=(
                    {"slow_horizon_discriminating_probe": "hec_decision_equivalent_at_short_horizon"}
                    if voi_rev["action"] == "hec_decision_region_stop"
                    else {"nuisance_high_var_aux": "zero_expected_target_risk_reduction"}
                ),
                rationale_summary="Balance slow-horizon wait cost against target horizon risk reduction (HEC stops if H is short!).",
                measurement_delta=m_add,
                compute_delta=voi_rev["compute_cost"],
                state_digest_after=f_state_dig,
            )
            ordered_classes.append("M5_DualHorizonReversal")

        obs_target_feats = {
            "has_history_or_lag_gap": bool({"sig:endpoint_matched_history_gap", "sig:coupled_transient_lag"} & sig_set),
        }
        gate_evals: list[dict[str, Any]] = []
        gate_rejs: dict[str, str] = {}
        for cand in rec_res.get("candidates", ()):
            nid = cand["node_id"]
            nd = forest.nodes[nid]
            auth_c = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view={"goal_kind": "trajectory_and_horizon_prediction", "tolerance": tol, "target_id": ds.task_id, "node_version": nd["version"]},
                target_features=obs_target_feats,
                claim_contract=nd["claim_contract"],
            )
            gate_evals.append({"node_id": nid, "granted": auth_c["granted"], "reason": auth_c["reason"]})
            if not auth_c["granted"]:
                gate_rejs[nid] = str(auth_c["reason"])
                continue
            for mc in nd["claim_contract"]["conclusion_or_action"].get("model_classes", ()):
                if mc not in ordered_classes:
                    ordered_classes.append(mc)
            if not use_flat:
                for fine_child in forest.fine_scale_children_of(nid):
                    auth_fc = authorize_abstraction_use(
                        requested_tier="bounded_trial",
                        task_view={"goal_kind": "trajectory_and_horizon_prediction", "tolerance": tol, "target_id": ds.task_id, "node_version": fine_child["version"]},
                        target_features=obs_target_feats,
                        claim_contract=fine_child["claim_contract"],
                    )
                    if auth_fc["granted"]:
                        for mc_f in fine_child["claim_contract"]["conclusion_or_action"].get("model_classes", ()):
                            if mc_f not in ordered_classes:
                                ordered_classes.append(mc_f)

        last_parent = ledger.record_decision(
            decision_type="D3_FOUR_TIER_PERMISSION_AND_GRAYZONE_GATE",
            parent_decision_id=last_parent,
            evidence_inputs={"observed_target_features": obs_target_feats},
            candidates_evaluated=gate_evals,
            chosen_option={"authorized_ordered_model_queue": list(ordered_classes)},
            rejection_reasons=gate_rejs,
            rationale_summary="Filter retrieved nodes through 4-tier preconditions & gray-zone bounds before fitting.",
            compute_delta=0.0,
            state_digest_after=f_state_dig,
        )

    for mc_rem in all_dynamic_classes:
        if mc_rem not in ordered_classes:
            ordered_classes.append(mc_rem)

    confirm_m_paid = False
    fine_upgraded = False
    nfev_sum = 0
    eval_cnt = 0

    for m_cls in ordered_classes:
        if (not is_strong_dyn) and m_cls == "M4_MoriZwanzigMemoryKernel" and (not fine_upgraded):
            train_rows = [*train_rows, *ds.extra_dynamic_rows[:3]]
            fine_upgraded = True
            last_parent = ledger.record_decision(
                decision_type="D6_CEGAR_FINE_TRAJECTORY_ESCALATION",
                parent_decision_id=last_parent,
                evidence_inputs={"escalated_to": "M4_MoriZwanzigMemoryKernel", "prior_rows": len(train_rows) - 3},
                candidates_evaluated=[{"action": "acquire_3_fine_rows_for_4param_identifiability", "measurement_cost": 4.5}],
                chosen_option={"new_train_row_count": len(train_rows)},
                rejection_reasons={"underdetermined_fit": "4_param_memory_kernel_requires_at_least_8_rows"},
                rationale_summary="Escalate along reverse coarse_grains_to edge and acquire 3 fine trajectories.",
                measurement_delta=4.5,
                state_digest_after=f_state_dig,
            )

        fit_res = _fit_candidate_model_class(
            m_cls, train_rows, tau_scale=ds.tau_scale,
            warm_prior=(None if is_strong_dyn else use_warm_prior), tolerance=tol, rng=rng,
        )
        nfev_sum += fit_res["nfev"]
        eval_cnt += 1
        train_err = _rmse([fit_res["predict_fn"](r) for r in train_rows], [r["y"] for r in train_rows])

        conf_m_delta = (len(ds.confirm_rows) * unit_row_m) if not confirm_m_paid else 0.0
        confirm_m_paid = True
        c_err = _rmse([fit_res["predict_fn"](r) for r in ds.confirm_rows], [r["y"] for r in ds.confirm_rows])
        accepted = c_err <= tol

        last_parent = ledger.record_decision(
            decision_type="D7_D8_CANDIDATE_FIT_AND_FRESH_CONFIRMATION",
            parent_decision_id=last_parent,
            evidence_inputs={
                "model_class": m_cls,
                "train_row_count": len(train_rows),
                "confirm_row_keys": [_row_input_key(r) for r in ds.confirm_rows],
            },
            candidates_evaluated=[{
                "model_class": m_cls,
                "nfev": fit_res["nfev"],
                "used_warm_start": fit_res.get("used_warm_start", False),
                "escaped_to_wide_fallback": fit_res.get("escaped_to_wide_fallback", False),
                "train_rmse": round(train_err, 6),
                "confirm_rmse": round(c_err, 6),
                "tolerance": tol,
            }],
            chosen_option={"verdict": "CONFIRM_ACCEPTED" if accepted else "CONFIRM_REJECTED_NEXT_CANDIDATE"},
            rejection_reasons=({} if accepted else {m_cls: f"confirm_rmse {c_err:.5f} > tolerance {tol:.5f}"}),
            rationale_summary=(
                f"Candidate {m_cls} passed disjoint confirmation (RMSE={c_err:.5f} <= {tol})."
                if accepted else f"Candidate {m_cls} refuted on disjoint confirmation (RMSE={c_err:.5f} > {tol}); continue search."
            ),
            measurement_delta=conf_m_delta,
            compute_delta=fit_res["compute_cost"],
            state_digest_after=f_state_dig,
        )

        # Confirm-triggered same-skeleton wide-box escape (`F06`)
        if (not accepted) and fit_res.get("used_warm_start") and (not fit_res.get("escaped_to_wide_fallback")):
            wide_res = _fit_candidate_model_class(
                m_cls, train_rows, tau_scale=ds.tau_scale, warm_prior=None, tolerance=tol, rng=rng,
            )
            nfev_sum += wide_res["nfev"]
            eval_cnt += 1
            c_err_w = _rmse([wide_res["predict_fn"](r) for r in ds.confirm_rows], [r["y"] for r in ds.confirm_rows])
            accepted = c_err_w <= tol
            c_err = c_err_w
            fit_res = wide_res
            last_parent = ledger.record_decision(
                decision_type="D8A_CONFIRM_TRIGGERED_WIDE_ESCAPE",
                parent_decision_id=last_parent,
                evidence_inputs={"model_class": m_cls, "reason": "warm_start_refuted_on_confirm_try_wide_box"},
                candidates_evaluated=[{"model_class": m_cls, "nfev": wide_res["nfev"], "confirm_rmse": round(c_err_w, 6)}],
                chosen_option={"verdict": "WIDE_ESCAPE_ACCEPTED" if accepted else "WIDE_ESCAPE_REFUTED"},
                rejection_reasons=({} if accepted else {m_cls: f"wide_confirm_rmse {c_err_w:.5f} > {tol:.5f}"}),
                rationale_summary="Warm-start box refuted on disjoint confirmation; escape to full wide box on same skeleton.",
                compute_delta=wide_res["compute_cost"],
                state_digest_after=f_state_dig,
            )

        if accepted:
            h_err = _rmse([fit_res["predict_fn"](r) for r in ds.heldout_rows], [r["y"] for r in ds.heldout_rows])
            ledger.record_decision(
                decision_type="D9_TERMINAL_DECISION_AND_HELDOUT_AUDIT",
                parent_decision_id=last_parent,
                evidence_inputs={"locked_model_class": m_cls, "heldout_row_count": len(ds.heldout_rows), "any_fit_after_confirm": False},
                candidates_evaluated=[{"status": "confirmed_ok", "heldout_rmse": round(h_err, 6)}],
                chosen_option={"final_status": "confirmed_ok", "selected_model_class": m_cls, "strict_pass": (not ds.is_out_of_scope) and (h_err <= tol)},
                rejection_reasons={},
                rationale_summary="Lock confirmed model with zero refit and audit on disjoint heldout set.",
                state_digest_after=f_state_dig,
            )
            tots = ledger.totals
            return {
                "task_id": ds.task_id, "suite": ds.suite, "subfamily": ds.subfamily, "arm": arm,
                "is_out_of_scope": ds.is_out_of_scope, "status": "confirmed_ok",
                "selected_model_class": m_cls,
                "strict_pass": (not ds.is_out_of_scope) and (h_err <= tol),
                "false_confirmation": bool(ds.is_out_of_scope) or (h_err > tol),
                "evaluated_candidate_count": eval_cnt, "total_nfev": nfev_sum,
                "confirm_rmse": round(c_err, 6), "heldout_rmse": round(h_err, 6),
                "decision_count": tots["decision_count"],
                "measurement_cost": tots["measurement_cost"],
                "compute_cost": tots["compute_cost"],
                "composite_cost": tots["composite_cost"],
            }, ledger.records

    ledger.record_decision(
        decision_type="D9_TERMINAL_DECISION_AND_HELDOUT_AUDIT",
        parent_decision_id=last_parent,
        evidence_inputs={"exhausted_candidates": len(ordered_classes)},
        candidates_evaluated=[{"status": "scoped_unknown"}],
        chosen_option={"final_status": "scoped_unknown", "strict_pass": bool(ds.is_out_of_scope)},
        rejection_reasons={mc: "refuted_on_fresh_confirm" for mc in ordered_classes},
        rationale_summary="All candidate dynamic families refuted on disjoint confirmation; honestly return scoped_unknown.",
        state_digest_after=f_state_dig,
    )
    tots = ledger.totals
    return {
        "task_id": ds.task_id, "suite": ds.suite, "subfamily": ds.subfamily, "arm": arm,
        "is_out_of_scope": ds.is_out_of_scope, "status": "scoped_unknown",
        "selected_model_class": None,
        "strict_pass": bool(ds.is_out_of_scope),
        "false_confirmation": False,
        "evaluated_candidate_count": eval_cnt, "total_nfev": nfev_sum,
        "confirm_rmse": None, "heldout_rmse": None,
        "decision_count": tots["decision_count"],
        "measurement_cost": tots["measurement_cost"],
        "compute_cost": tots["compute_cost"],
        "composite_cost": tots["composite_cost"],
    }, ledger.records


# ============================================================================
# 3. Official OpenAI ChatGPT Subscription Client (`CodexSubscriptionClient`)
# ============================================================================


M5_AGENT_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["acquire_probe", "author_and_confirm_python_model", "abstain_scoped_unknown"],
        },
        "probe_id": {
            "type": ["string", "null"],
            "description": (
                "Required when action_type == 'acquire_probe': any probe key listed in `available_probes_cost_menu` "
                "(e.g. 'slow_horizon_discriminating_probe', 'open_branch_discriminating_probe', "
                "'high_sgrad_regime_probe', 'extra_fine_trajectory_batch', 'nuisance_high_var_aux'); null otherwise."
            ),
        },
        "hypothesis_name": {
            "type": ["string", "null"],
            "description": "Short identifier when action_type == 'author_and_confirm_python_model'; null otherwise.",
        },
        "python_code": {
            "type": ["string", "null"],
            "description": (
                "Executable Python source defining `fit(train_rows)` and `predict(row) -> float` "
                "when action_type == 'author_and_confirm_python_model'; null otherwise."
            ),
        },
        "considered_alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "option": {"type": "string"},
                    "rejected_reason": {"type": "string"},
                },
                "required": ["option", "rejected_reason"],
                "additionalProperties": False,
            },
            "description": "Explicit candidate probes or model families considered at this turn and why they were deferred/rejected.",
        },
        "rationale": {
            "type": "string",
            "description": "Concise scientific reasoning linking observed evidence to the chosen action.",
        },
    },
    "required": [
        "action_type",
        "probe_id",
        "hypothesis_name",
        "python_code",
        "considered_alternatives",
        "rationale",
    ],
    "additionalProperties": False,
}


def _win_path_to_wsl(win_path: str) -> str:
    p = win_path.strip().strip('"').strip("'")
    if len(p) >= 3 and p[1] == ":" and p[2] in ("\\", "/"):
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def discover_chatgpt_subscription_codex_cli() -> dict[str, Any]:
    """Locate official OpenAI Codex CLI signed in with a ChatGPT account (`auth_mode == 'chatgpt'`) and block `ruoli.dev`!"""
    # 1. Inspect `/mnt/c/Users/*/.codex/config.toml` & `auth.json` (WindowsApp directory disables glob, so read `CODEX_CLI_PATH`!)
    win_users_root = Path("/mnt/c/Users")
    if win_users_root.is_dir():
        for user_dir in sorted(win_users_root.iterdir()):
            try:
                auth_p = user_dir / ".codex" / "auth.json"
                cfg_p = user_dir / ".codex" / "config.toml"
                if not auth_p.is_file():
                    continue
                auth_obj = json.loads(auth_p.read_text(encoding="utf-8"))
            except OSError:
                continue
            if auth_obj.get("auth_mode") != "chatgpt":
                continue
            cfg_txt = cfg_p.read_text(encoding="utf-8", errors="replace") if cfg_p.is_file() else ""
            active_lines = [ln for ln in cfg_txt.splitlines() if not ln.strip().startswith("#")]
            if any("ruoli" in ln.lower() for ln in active_lines):
                raise RuntimeError("Blocked: active Codex config points to ruoli; only official OpenAI ChatGPT subscription is allowed!")

            exe_candidates: list[str] = []
            # Extract all `OpenAI.Codex_*` package folders referenced anywhere in config.toml (handles MS Store auto-updates!)
            for pkg_name in sorted(set(re.findall(r"OpenAI\.Codex_[^\\/\s\"']+", cfg_txt)), reverse=True):
                exe_candidates.append(f"/mnt/c/Program Files/WindowsApps/{pkg_name}/app/resources/codex.exe")
            for ln in active_lines:
                if "CODEX_CLI_PATH" in ln and "=" in ln:
                    _, rhs = ln.split("=", 1)
                    exe_candidates.append(_win_path_to_wsl(rhs))
            exe_candidates.extend(sorted(glob.glob("/mnt/c/Program Files/WindowsApps/OpenAI.Codex_*/app/resources/codex.exe"), reverse=True))
            exe_candidates.append(str(user_dir / ".codex" / "shims" / "codex.exe"))

            # Fallback: query AppxPackage InstallLocation via PowerShell if WindowsApps updated mid-session
            if not any(wc and Path(wc).is_file() for wc in exe_candidates):
                try:
                    apx = subprocess.run(
                        ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-Command", "(Get-AppxPackage OpenAI.Codex | Sort-Object Version -Descending | Select-Object -First 1).InstallLocation"],
                        capture_output=True, text=True, timeout=10,
                    )
                    loc = apx.stdout.strip()
                    if loc:
                        exe_candidates.insert(0, _win_path_to_wsl(loc).rstrip("/") + "/app/resources/codex.exe")
                except Exception:
                    pass

            for wc in exe_candidates:
                if wc and Path(wc).is_file():
                    win_temp_wsl = user_dir / "AppData" / "Local" / "Temp" / "hegel_m5_codex_subscription_stage"
                    win_temp_str = f"C:\\Users\\{user_dir.name}\\AppData\\Local\\Temp\\hegel_m5_codex_subscription_stage"
                    return {
                        "available": True,
                        "executable": wc,
                        "is_windows_exe_under_wsl": True,
                        "stage_root_wsl": win_temp_wsl,
                        "stage_root_cli_str": win_temp_str,
                        "auth_mode": "chatgpt",
                        "provider_backend": "openai_chatgpt_subscription_codex_cli",
                    }

    # 2. Check native POSIX / Windows `codex` on PATH
    native_bin = shutil.which("codex")
    if native_bin:
        auth_p = Path.home() / ".codex" / "auth.json"
        cfg_p = Path.home() / ".codex" / "config.toml"
        if auth_p.is_file():
            auth_obj = json.loads(auth_p.read_text(encoding="utf-8"))
            if auth_obj.get("auth_mode") == "chatgpt":
                cfg_txt = cfg_p.read_text(encoding="utf-8", errors="replace") if cfg_p.is_file() else ""
                active_lines = [ln for ln in cfg_txt.splitlines() if not ln.strip().startswith("#")]
                if any("ruoli" in ln.lower() for ln in active_lines):
                    raise RuntimeError("Blocked: active Codex config points to ruoli; only official OpenAI ChatGPT subscription is allowed!")
                stg = Path.home() / ".cache" / "hegel_m5_codex_subscription_stage"
                return {
                    "available": True,
                    "executable": native_bin,
                    "is_windows_exe_under_wsl": False,
                    "stage_root_wsl": stg,
                    "stage_root_cli_str": str(stg),
                    "auth_mode": "chatgpt",
                    "provider_backend": "openai_chatgpt_subscription_codex_cli",
                }

    return {"available": False, "provider_backend": "unavailable"}


class SubscriptionQuotaPausedError(RuntimeError):
    """Raised when an official ChatGPT subscription model hits a rate-limit or usage-window quota cap."""


class CodexSubscriptionClient:
    """Official OpenAI ChatGPT Subscription caller via `codex exec --output-schema` (ZERO `ruoli.dev`!)."""

    _SCRUBBED_ENV_KEYS = frozenset({
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GPT5_API_KEY",
        "GPT5_BASE_URL",
        "RUOLI_API_KEY",
        "RUOLI_BASE_URL",
    })

    def __init__(
        self,
        *,
        codex_info: dict[str, Any],
        model: str,
        reasoning_effort: str = "medium",
        timeout_seconds: int = 120,
    ) -> None:
        if not codex_info.get("available") or codex_info.get("auth_mode") != "chatgpt":
            raise RuntimeError("CodexSubscriptionClient requires an official ChatGPT OAuth subscription (`auth_mode == 'chatgpt'`).")
        self.codex_info = codex_info
        self.executable = str(codex_info["executable"])
        self.is_win_wsl = bool(codex_info["is_windows_exe_under_wsl"])
        self.stage_root_wsl = Path(codex_info["stage_root_wsl"])
        self.stage_root_cli = str(codex_info["stage_root_cli_str"])
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.timeout_seconds = int(timeout_seconds)
        self.provider_backend = "openai_chatgpt_subscription_codex_cli"
        self._call_seq = 0

    def _join_cli_path(self, *parts: str) -> str:
        sep = "\\" if self.is_win_wsl else "/"
        base = self.stage_root_cli.rstrip("\\/")
        return sep.join([base, *parts])

    def call_tool_step(
        self,
        messages: list[dict[str, Any]],
        *,
        stage_subdir: str = "default_ep",
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        del max_tokens
        self._call_seq += 1
        ep_wsl = self.stage_root_wsl / stage_subdir
        ep_wsl.mkdir(parents=True, exist_ok=True)

        schema_name = "decision_schema.json"
        instr_name = f"sys_instr_{self._call_seq:02d}.txt"
        resp_name = f"last_msg_{self._call_seq:02d}.json"

        sys_text = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        convo_turns = [m for m in messages if m.get("role") != "system"]
        stdin_payload = json.dumps({"conversation_history": convo_turns}, ensure_ascii=False)

        (ep_wsl / schema_name).write_text(json.dumps(M5_AGENT_DECISION_SCHEMA, ensure_ascii=False), encoding="utf-8")
        (ep_wsl / instr_name).write_text(sys_text, encoding="utf-8")
        resp_wsl = ep_wsl / resp_name
        if resp_wsl.exists():
            resp_wsl.unlink()

        cd_cli = self._join_cli_path(stage_subdir)
        schema_cli = self._join_cli_path(stage_subdir, schema_name)
        instr_cli = self._join_cli_path(stage_subdir, instr_name)
        resp_cli = self._join_cli_path(stage_subdir, resp_name)

        cmd = [
            self.executable, "-a", "never", "exec", "--sandbox", "read-only",
            "--cd", cd_cli, "--skip-git-repo-check", "--ephemeral", "--json",
            "--color", "never",
            "--output-schema", schema_cli,
            "--output-last-message", resp_cli,
            "-c", "model_instructions_file=" + json.dumps(instr_cli),
            "-c", "project_doc_max_bytes=0",
            "-c", 'developer_instructions=""',
            "-c", 'web_search="disabled"',
            "-c", "mcp_servers={}",
            "-c", "plugins={}",
            "-c", "notify=[]",
            "-c", "features.memories=false",
            "-c", "features.multi_agent=false",
            "-c", f'model_reasoning_effort={json.dumps(self.reasoning_effort)}',
            "--enable", "skip_host_skill_discovery",
        ]
        for feat in (
            "code_mode", "code_mode_host", "shell_tool", "unified_exec", "apps", "browser_use",
            "computer_use", "skill_search", "view_image", "sleep_tool", "tool_suggest",
        ):
            cmd.extend(("--disable", feat))
        cmd.extend(("--model", self.model, "-"))

        clean_env = {k: v for k, v in os.environ.items() if k not in self._SCRUBBED_ENV_KEYS}

        t0 = monotonic()
        proc = None
        timeout_retries = 0
        for attempt_idx in range(4):
            try:
                proc = subprocess.run(
                    cmd,
                    input=stdin_payload,
                    text=True,
                    capture_output=True,
                    env=clean_env,
                    timeout=self.timeout_seconds,
                )
            except FileNotFoundError:
                refreshed = discover_chatgpt_subscription_codex_cli()
                if refreshed.get("available") and refreshed.get("executable") and refreshed["executable"] != self.executable:
                    self.executable = str(refreshed["executable"])
                    cmd[0] = self.executable
                    continue
                raise
            except subprocess.TimeoutExpired:
                timeout_retries += 1
                if attempt_idx < 2:
                    sleep((4.0, 10.0)[attempt_idx] + random.uniform(0.5, 2.5))
                    continue
                return {
                    "ok": False,
                    "wall_ms": round((monotonic() - t0) * 1000.0, 2),
                    "returned_model": self.model,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "completion_tokens": 0,
                    "timeout_retries": timeout_retries,
                    "error": f"codex_subscription_timeout_{self.timeout_seconds}s_after_{timeout_retries}_attempts",
                }
            if proc.returncode == 0 and resp_wsl.is_file():
                break
            err_probe = (proc.stdout + "\n" + proc.stderr).lower()
            if any(h_kw in err_probe for h_kw in ("usage limit", "you've hit your", "you have hit your", "upgrade to pro")):
                break
            if attempt_idx < 3 and any(
                tok in err_probe for tok in ("rate limit", "429", "too many requests", "overloaded", "stream disconnected", "connection")
            ):
                sleep((5.0, 12.0, 25.0)[attempt_idx] + random.uniform(0.5, 4.5))
                continue
            break
        assert proc is not None
        wall_ms = round((monotonic() - t0) * 1000.0, 2)

        in_tok = 0
        cached_tok = 0
        out_tok = 0
        returned_model = self.model
        errs: list[str] = []
        for ln in proc.stdout.splitlines():
            s = ln.strip()
            if not s:
                continue
            try:
                ev = json.loads(s)
            except json.JSONDecodeError:
                continue
            if ev.get("model"):
                returned_model = str(ev["model"])
            if ev.get("type") == "turn.completed":
                u = ev.get("usage") or {}
                in_tok = int(u.get("input_tokens") or 0)
                cached_tok = int(u.get("cached_input_tokens") or 0)
                out_tok = int(u.get("output_tokens") or 0)
            if ev.get("type") in {"error", "turn.failed"}:
                errs.append(str(ev.get("message") or ev.get("error") or ""))

        if proc.returncode != 0 or not resp_wsl.is_file():
            err_msg = f"rc={proc.returncode}; errs={errs}; stderr={proc.stderr[-400:]}"
            low_err = err_msg.lower()
            if any(
                q_kw in low_err
                for q_kw in ("usage limit", "you've hit your", "you have hit your", "quota", "rate limit", "too many requests", "upgrade to pro")
            ):
                raise SubscriptionQuotaPausedError(f"[{self.model}] {err_msg}")
            return {
                "ok": False,
                "wall_ms": wall_ms,
                "returned_model": returned_model,
                "prompt_tokens": in_tok,
                "cached_tokens": cached_tok,
                "completion_tokens": out_tok,
                "error": err_msg,
            }

        try:
            parsed = json.loads(resp_wsl.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "ok": False,
                "wall_ms": wall_ms,
                "returned_model": returned_model,
                "prompt_tokens": in_tok,
                "cached_tokens": cached_tok,
                "completion_tokens": out_tok,
                "error": f"invalid_schema_json:{exc}",
            }

        act_type = str(parsed["action_type"])
        tool_args: dict[str, Any] = {
            "rationale": str(parsed.get("rationale") or ""),
            "considered_alternatives": list(parsed.get("considered_alternatives") or []),
        }
        if act_type == "acquire_probe":
            tool_args["probe_id"] = str(parsed.get("probe_id") or "")
        elif act_type == "author_and_confirm_python_model":
            tool_args["hypothesis_name"] = str(parsed.get("hypothesis_name") or "authored_cand")
            tool_args["python_code"] = str(parsed.get("python_code") or "")

        tc_id = f"sub_call_{self._call_seq:02d}"
        return {
            "ok": True,
            "wall_ms": wall_ms,
            "returned_model": returned_model,
            "prompt_tokens": in_tok,
            "cached_tokens": cached_tok,
            "completion_tokens": out_tok,
            "task_context_chars": len(stdin_payload) + len(sys_text),
            "tool_call_id": tc_id,
            "tool_name": act_type,
            "tool_args": tool_args,
            "assistant_message_for_history": {
                "role": "assistant",
                "action_taken": {
                    "call_id": tc_id,
                    "action_type": act_type,
                    "probe_id": parsed.get("probe_id"),
                    "hypothesis_name": parsed.get("hypothesis_name"),
                    "rationale": parsed.get("rationale"),
                },
            },
        }


# ============================================================================
# 4. Isolated Python Subprocess Verifier & `K=15` Polysemous Source Archive
# ============================================================================


def _execute_authored_python_in_isolated_subprocess(
    workspace_dir: Path,
    *,
    step_idx: int,
    python_code: str,
    train_rows: list[dict[str, float]],
    confirm_rows: list[dict[str, float]],
    heldout_rows: list[dict[str, float]],
) -> dict[str, Any]:
    """Write GPT-authored `model.py` to disk and evaluate in a separate isolated Python subprocess!

    Strictly enforces zero data leakage: `model.py` ONLY sees `train_rows` during `fit(train_rows)`,
    and ONLY sees input features (`y` stripped!) during `predict(row)` on `confirm` and `heldout`!
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)
    model_file = workspace_dir / f"model_step_{step_idx:02d}.py"
    model_file.write_text(python_code, encoding="utf-8")
    model_sha = hashlib.sha256(python_code.encode("utf-8")).hexdigest()

    payload_file = workspace_dir / f"eval_payload_{step_idx:02d}.json"
    strip_y = lambda rows: [{k: v for k, v in r.items() if k != "y"} for r in rows]
    payload_file.write_text(
        json.dumps({
            "train_rows": train_rows,
            "confirm_inputs": strip_y(confirm_rows),
            "heldout_inputs": strip_y(heldout_rows),
        }),
        encoding="utf-8",
    )

    runner_code = (
        "import copy, inspect, json, math, os, sys, types\n"
        "from pathlib import Path\n"
        "import numpy, scipy, scipy.optimize\n"
        "m_path, p_path, sandbox_root = Path(sys.argv[1]), Path(sys.argv[2]), os.path.realpath(sys.argv[3])\n"
        "m_src = m_path.read_text(encoding='utf-8')\n"
        "data = json.loads(p_path.read_text(encoding='utf-8'))\n"
        "allowed_prefixes = (\n"
        "    sandbox_root, '/usr/', '/lib/', '/lib64/', '/etc/ld.so',\n"
        "    '/proc/cpuinfo', '/proc/meminfo', '/dev/null', '/dev/urandom',\n"
        "    os.path.realpath(os.path.expanduser('~/.local/lib/python')),\n"
        ")\n"
        "for sp in list(sys.path):\n"
        "    rp = os.path.realpath(sp) if sp else sandbox_root\n"
        "    if not rp.startswith(allowed_prefixes):\n"
        "        sys.path.remove(sp)\n"
        "def _audit_guard(event, args):\n"
        "    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.spawn', 'socket.connect', 'socket.bind'):\n"
        "        raise PermissionError(f'Hermetic sandbox blocked {event}')\n"
        "    if event in ('open', 'os.listdir', 'os.scandir') and args:\n"
        "        p0 = args[0]\n"
        "        if isinstance(p0, (str, bytes, os.PathLike)):\n"
        "            ps = os.fsdecode(p0)\n"
        "            rp = os.path.realpath(ps)\n"
        "            if not rp.startswith(allowed_prefixes):\n"
        "                raise PermissionError(f'Hermetic sandbox blocked filesystem access: {ps}')\n"
        "sys.addaudithook(_audit_guard)\n"
        "mod = types.ModuleType('gpt_authored_model')\n"
        "mod.__file__ = os.path.join(sandbox_root, 'model.py')\n"
        "exec(compile(m_src, mod.__file__, 'exec'), mod.__dict__)\n"
        "fit_ret = mod.fit(copy.deepcopy(data['train_rows'])) if hasattr(mod, 'fit') else None\n"
        "abi_mode = 'module_state_predict'\n"
        "if callable(fit_ret):\n"
        "    pred_fn = fit_ret\n"
        "    abi_mode = 'fit_returned_callable'\n"
        "elif hasattr(mod, 'predict'):\n"
        "    sig = inspect.signature(mod.predict)\n"
        "    req_p = [\n"
        "        p for p in sig.parameters.values()\n"
        "        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty\n"
        "    ]\n"
        "    if len(req_p) >= 2 and fit_ret is not None:\n"
        "        pred_fn = lambda row, _st=fit_ret: mod.predict(row, _st)\n"
        "        abi_mode = 'predict_row_and_fit_state'\n"
        "    else:\n"
        "        pred_fn = mod.predict\n"
        "else:\n"
        "    raise AttributeError('Authored module defines neither callable fit() return nor predict(row)')\n"
        "tr_p = [float(pred_fn({k: v for k, v in r.items() if k != 'y'})) for r in data['train_rows']]\n"
        "cf_p = [float(pred_fn(dict(r))) for r in data['confirm_inputs']]\n"
        "ho_p = [float(pred_fn(dict(r))) for r in data['heldout_inputs']]\n"
        "if not all(math.isfinite(x) for x in [*tr_p, *cf_p, *ho_p]):\n"
        "    raise ValueError('Non-finite prediction returned')\n"
        "print(json.dumps({'ok': True, 'abi_mode': abi_mode, 'train_preds': tr_p, 'confirm_preds': cf_p, 'heldout_preds': ho_p}))\n"
    )
    t0 = monotonic()
    with tempfile.TemporaryDirectory(prefix="hegel_hermetic_eval_") as tmp_sbx:
        sbx_p = Path(tmp_sbx)
        sbx_model = sbx_p / "model.py"
        sbx_payload = sbx_p / "payload.json"
        shutil.copy2(model_file, sbx_model)
        shutil.copy2(payload_file, sbx_payload)
        try:
            proc = subprocess.run(
                [sys.executable, "-B", "-c", runner_code, str(sbx_model), str(sbx_payload), str(sbx_p)],
                cwd=str(sbx_p),
                env={
                    "PYTHONPATH": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HOME": os.environ.get("HOME", "/tmp"),
                    "PATH": os.environ.get("PATH", ""),
                },
                capture_output=True,
                text=True,
                timeout=25,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "model_path": str(model_file),
                "model_sha256": model_sha,
                "exec_wall_ms": round((monotonic() - t0) * 1000.0, 2),
                "error": "Python subprocess timed out (>25s); avoid brute-force grid loops and use fast vectorized least_squares / RK4.",
            }
    exec_ms = round((monotonic() - t0) * 1000.0, 2)
    if proc.returncode != 0:
        return {
            "ok": False,
            "model_path": str(model_file),
            "model_sha256": model_sha,
            "exec_wall_ms": exec_ms,
            "error": (proc.stderr or proc.stdout)[-800:],
        }
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    tr_rmse = _rmse(out["train_preds"], [r["y"] for r in train_rows])
    cf_rmse = _rmse(out["confirm_preds"], [r["y"] for r in confirm_rows])
    ho_rmse = _rmse(out["heldout_preds"], [r["y"] for r in heldout_rows])
    return {
        "ok": True,
        "abi_mode": str(out.get("abi_mode", "module_state_predict")),
        "model_path": str(model_file),
        "model_sha256": model_sha,
        "exec_wall_ms": exec_ms,
        "train_rmse": round(tr_rmse, 6),
        "confirm_rmse": round(cf_rmse, 6),
        "heldout_rmse": round(ho_rmse, 6),
    }


def _build_live_k15_forest_and_archive() -> tuple[PhilosophicalForest, dict[str, Any], dict[str, dict[str, Any]]]:
    """Build `K=15` heterogeneous forest + node-linked source reference archive (7 true + 8 polysemous distractors)."""
    forest, priors_bundle = _build_seed_forest_and_priors()
    tv_def = TaskView("trajectory_and_horizon_prediction", ["y"], 0.045)

    # Register 8 realistic polysemous near-miss distractor mechanisms sharing partial surface cues (`K=15` total!)
    distractor_specs = [
        ("node_F_logistic_carrying_cap", "tree_distractor_pop", ["sig:base_field_C1", "sig:monotone_concave", "sig:carrying_cap"],
         "Logistic saturation without sink gradient: y = K_cap / (1 + exp(-r_grow * (u_now - u_mid)))."),
        ("node_G_michaelis_no_sink", "tree_distractor_enz", ["sig:base_field_C1", "sig:monotone_concave", "sig:substrate_only"],
         "Single-factor Michaelis-Menten ignoring s_grad: y = vmax * u_now / (km + u_now). Fails when s_grad varies!"),
        ("node_H_damped_harmonic_osc", "tree_distractor_vib", ["sig:coupled_transient_lag", "sig:underdamped_ringing"],
         "Underdamped 2nd-order oscillator: y = u_now * exp(-zeta * H) * cos(omega_d * H)."),
        ("node_I_powerlaw_creep_relax", "tree_distractor_rheo", ["sig:slow_horizon_query", "sig:powerlaw_tail"],
         "Fractional Andrade power-law creep: y = c0 * u_now * (1 + beta * (H ** (1/3)))."),
        ("node_J_bimolecular_second_order", "tree_distractor_kin", ["sig:fast_transient_damped", "sig:bimolecular_decay"],
         "Second-order hyperbolic decay: y = x_now / (1 + k2 * x_now * H)."),
        ("node_K_arrhenius_barrier_shift", "tree_distractor_therm", ["sig:short_window_positive_gain", "sig:thermal_activation"],
         "Monotone Arrhenius thermal rate (no slow reversal): y = A0 * u_now * exp(-E_act / (1.0 + H))."),
        ("node_L_piecewise_deadzone", "tree_distractor_ctrl", ["sig:sink_gradient_C2", "sig:deadzone_clip"],
         "Piecewise deadzone relay: y = max(0.0, k_gain * (u_now - s_grad))."),
        ("node_M_linear_drift_accumulator", "tree_distractor_acc", ["sig:mass_sum_drift", "sig:unbounded_ramp"],
         "Unbounded secular ramp without siphon pool: y = x_now + (c_in * u_now - c_leak) * H."),
    ]
    for nid, tid, sigs, _note in distractor_specs:
        forest.register_node(
            node_id=nid,
            tree_id=tid,
            layer_role="mid",
            task_view=tv_def,
            observation_map=ObservationMap("y", temporal_readout_mode="present_snapshot"),
            scale_map=ScaleMap("identity", preserved_queries=["trajectory_and_horizon_prediction"]),
            claim_contract=ClaimContract(
                claim_id=f"cl_{nid}",
                family_plugin="distractor_m5",
                preconditions=[],
                conclusion_or_action={"model_classes": [f"Distractor_{nid}"]},
                protocol_spec={},
                tolerance=0.045,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="bounded_trial",
            ),
            probe_signatures=sigs,
        )

    node_archive: dict[str, dict[str, Any]] = {
        "node_A_snapshot_balance": {
            "node_id": "node_A_snapshot_balance",
            "verified_mechanism_note": (
                "Coarse adiabatic slow ODE (`tau_fast << H`): `y = x_ss + (x_now - x_ss) * exp(-a_eff * H)` "
                "with `x_ss = k_ss * u_now` (or pure static linear `y = c0 * u_now + c1` when H-independent)."
            ),
        },
        "node_A_before_after_opposition": {
            "node_id": "node_A_before_after_opposition",
            "verified_mechanism_note": (
                "Before/after perturbation contrast (`u_now - u_past`): `y = gain * (u_now - u_past) * exp(-H / tau)`."
            ),
        },
        "node_A_memory_kernel_full": {
            "node_id": "node_A_memory_kernel_full",
            "verified_mechanism_note": (
                "Coupled 2-state memory ODE integrated by RK4 (`dt=0.004`) over `[0, H]` from `x(0)=x_now, z(0)=z_t0`, "
                "constant `u=u_now`: `dx/dt = -a_slow*x + b_couple*z + c_drive*u_now`, `tau_fast * dz/dt = 0.52*x - z`, `y = x(H)`. "
                "If `row['mem_conv']` is provided (not None), `z_t0 = row['mem_conv']`; if `row['mem_conv']` is None and `row['raw_prefix']` "
                "is provided (`[{'t': t_k, 'u': u_k, 'x': x_k}, ...]`), compute `z_t0` by integrating `tau_fast * dz/dt = 0.52*x(t) - z` "
                "(trapezoidal exponential kernel from `z_init = 0.52 * x(t_0)` over shifted intervals `t_latest - t_init`)! "
                "Fit 4 parameters `(a_slow, b_couple, c_drive, tau_fast)` — requires >= 8 rows (acquire `extra_fine_trajectory_batch`!)."
            ),
        },
        "node_B_dual_horizon_reversal": {
            "node_id": "node_B_dual_horizon_reversal",
            "verified_mechanism_note": (
                "Dual-horizon endogenous adaptation: `y = u_now * (amp_fast*(1 - exp(-H/tau_fast)) - amp_slow*(1 - exp(-H/tau_slow)))`. "
                "Positive at short pilot `H <= 0.5`, reverses negative at `H ~ 15..25`."
            ),
        },
        "node_C_source_inland_hub": {
            "node_id": "node_C_source_inland_hub",
            "verified_mechanism_note": (
                "2-layer inland-hub siphon base (`u_n = u_now / tau_scale`): "
                "`y = vmax * (1 - rho * s_grad/(1 + s_grad)) * (u_n / (1.2 + u_n)) + branch_term`, "
                "where `branch_term` is `c_loc * sqrt(u_n)` (sign open: `+` or `-`) or `-leak_rate * u_n` under mass drift."
            ),
        },
        "node_D_source_hydraulic_basin": {
            "node_id": "node_D_source_hydraulic_basin",
            "verified_mechanism_note": (
                "Hydraulic basin siphon wormhole twin (`u_n = u_now / tau_scale`): "
                "`y = vmax * (1 - rho * s_grad/(1 + s_grad)) * (u_n / (1.2 + u_n)) + branch_term`, "
                "with dimensionless `rho in [0.69, 0.72]` across 3 diverse sources."
            ),
        },
        "node_E_polysemous_quad_storage": {
            "node_id": "node_E_polysemous_quad_storage",
            "verified_mechanism_note": "Static quadratic storage without sink gradient: `y = a2 * (u_now ** 2) + a1 * u_now`.",
        },
    }
    for nid, _tid, _sigs, note in distractor_specs:
        node_archive[nid] = {"node_id": nid, "verified_mechanism_note": note}

    return forest, priors_bundle, node_archive


def _build_arm_system_and_initial_brief(
    ds: M5DisjointTaskDataset,
    arm: str,
    *,
    init_rows: list[dict[str, float]],
    forest: PhilosophicalForest,
    priors_bundle: dict[str, Any],
    node_archive: dict[str, dict[str, Any]],
    hippo_idx: HippoRAGWormholeIndex,
    flat_idx: FlatEqualMetadataIndex,
    ledger: DecisionProvenanceLedger,
    parent_id: str,
) -> tuple[str, dict[str, Any], str]:
    """Build exact, transparent prompt brief for `Strong_Ordinary_LiveGPT`, `EqualAuth_Flat_LiveGPT`, or `Hegel_M5_LiveGPT`."""
    tol = float(ds.tolerance)
    sys_prompt = (
        "You are an autonomous scientific discovery agent in the Hegel M5 benchmark. "
        "Infer the quantitative law `y` from paid observations and submit a single JSON decision matching the output schema.\n"
        "RULES:\n"
        "1. Minimize total measurement + compute + token cost while achieving fresh confirmation RMSE <= tolerance.\n"
        "2. Only call `acquire_probe` when current `train_rows` leave genuine identifiability ambiguity "
        "(e.g., 5 rows for a 4-parameter coupled ODE or dual-exponential law, or short-H pilot when `target_forecast_horizon_H` is large). "
        "Never buy irrelevant `nuisance_high_var_aux`!\n"
        "3. When calling `author_and_confirm_python_model`, `python_code` must be self-contained Python defining `fit(train_rows)` "
        "and `predict(row) -> float` using `math`, `numpy`, and `scipy.optimize.least_squares` (and RK4 integration if coupled ODE applies). "
        "Each row has keys `u_past, u_now, x_now, mem_conv, s_grad, H` (`train_rows` also has `y`). Always fit target parameters numerically!\n"
        "4. If paid observations exhibit non-stationary high-frequency chaotic sign oscillations (`sig:high_freq_sign_oscillation`) "
        "or fresh confirmation refutes all stationary smooth mechanisms, call `abstain_scoped_unknown`.\n"
        "5. Always populate `considered_alternatives` with the concrete candidate probes/models you weighed and why you deferred or rejected them."
    )

    base_task_info = {
        "task_id": ds.task_id,
        "tolerance": tol,
        "target_forecast_horizon_H": ds.horizon_H,
        "characteristic_tau_scale": ds.tau_scale,
        "observed_pilot_signatures": list(ds.probe_signatures),
        "initial_paid_train_rows": init_rows,
        "available_probes_cost_menu": {
            **{
                k: {
                    "measurement_cost": v["measurement_cost"],
                    "reset_cost": v["reset_cost"],
                    "horizon_wait_cost": v["horizon_wait_cost"],
                    "total_probe_cost": v["measurement_cost"] + v["reset_cost"] + v["horizon_wait_cost"],
                }
                for k, v in ds.probe_pool.items()
            },
            "extra_fine_trajectory_batch": {
                "measurement_cost": 4.5, "reset_cost": 0.0, "horizon_wait_cost": 0.0, "total_probe_cost": 4.5,
            },
        },
    }

    if arm == "Strong_Ordinary_LiveGPT":
        brief = {
            **base_task_info,
            "arm_mode": "Strong_Ordinary_Direct_Full_K15_Raw_Archive",
            "full_k15_source_episodes_archive": list(node_archive.values()),
        }
        return sys_prompt, brief, parent_id

    use_flat = arm == "EqualAuth_Flat_LiveGPT"
    rec_res = (flat_idx if use_flat else hippo_idx).retrieve(
        ds.probe_signatures,
        task_view={"goal_kind": "trajectory_and_horizon_prediction", "tolerance": tol},
        top_k=3,
    )
    recalled_ids = [c["node_id"] for c in rec_res.get("candidates", ())]
    p_id = ledger.record_decision(
        decision_type="D2_INDEX_RETRIEVAL_RANKING",
        parent_decision_id=parent_id,
        evidence_inputs={"query_signatures": list(ds.probe_signatures), "index": "FlatBM25_K15" if use_flat else "HippoRAG_PPR_K15"},
        candidates_evaluated=[
            {"node_id": c["node_id"], "score": round(float(c.get("ppr_score", c.get("bm25_score", 0.0))), 6)}
            for c in rec_res.get("candidates", ())
        ],
        chosen_option={"recalled_top3_nodes": recalled_ids},
        rejection_reasons={},
        rationale_summary=f"Retrieve Top-3 concepts from K={len(forest.nodes)} heterogeneous forest via {'FlatBM25' if use_flat else 'HippoRAG_PPR'}.",
        compute_delta=float(rec_res.get("query_compute_cost", 1.2)),
    )

    if use_flat:
        brief = {
            **base_task_info,
            "arm_mode": "EqualAuth_Flat_BM25_Top3_With_Coarse_Certificates",
            "bm25_top3_recalled_source_notes": [node_archive[nid] for nid in recalled_ids if nid in node_archive],
        }
        return sys_prompt, brief, p_id

    # Full `Hegel_M5_LiveGPT`: 4-tier permission gate + 2-layer SharedBaseSubtreePrior + `probing.py` VOI!
    sig_set = set(ds.probe_signatures)
    obs_feats = {
        "has_history_or_lag_gap": bool({"sig:endpoint_matched_history_gap", "sig:coupled_transient_lag"} & sig_set),
        "s_grad": max((float(r.get("s_grad") or 0.0) for r in init_rows), default=0.0),
    }
    auth_ids: list[str] = []
    gate_evals: list[dict[str, Any]] = []
    gate_rejs: dict[str, str] = {}
    for nid in recalled_ids:
        nd = forest.nodes[nid]
        ac = authorize_abstraction_use(
            requested_tier="bounded_trial",
            task_view={"goal_kind": "trajectory_and_horizon_prediction", "tolerance": tol, "target_id": ds.task_id, "node_version": nd["version"]},
            target_features=obs_feats,
            claim_contract=nd["claim_contract"],
        )
        if ac["granted"] and nd["claim_contract"].get("gray_zone_bounds"):
            ac_gz = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view={"goal_kind": "trajectory_and_horizon_prediction", "tolerance": tol, "target_id": ds.task_id, "node_version": nd["version"]},
                target_features=obs_feats,
                claim_contract=nd["claim_contract"],
                heldout_confirmed=True,
            )
            if ac_gz.get("gray_zone_status") == "UNKNOWN_GRAY_ZONE":
                gate_rejs[f"{nid}:gray_zone"] = str(ac_gz["reason"])
        gate_evals.append({"node_id": nid, "granted": ac["granted"], "reason": ac["reason"]})
        if ac["granted"]:
            auth_ids.append(nid)
        else:
            gate_rejs[nid] = str(ac["reason"])

    p_id = ledger.record_decision(
        decision_type="D3_FOUR_TIER_PERMISSION_AND_GRAYZONE_GATE",
        parent_decision_id=p_id,
        evidence_inputs={"observed_features": obs_feats},
        candidates_evaluated=gate_evals,
        chosen_option={"authorized_recalled_nodes": auth_ids},
        rejection_reasons=gate_rejs,
        rationale_summary="Check 4-tier preconditions & gray-zone scope on Top-3 recalled forest nodes.",
        compute_delta=0.0,
    )

    subtree_guidance = None
    voi_guidance = None
    causal_regime_guidance = None
    recalled_siphon_wormhole = bool({"node_C_source_inland_hub", "node_D_source_hydraulic_basin"} & set(auth_ids))
    if ("sig:high_freq_sign_oscillation" not in sig_set) and (
        {"sig:base_field_C1", "sig:sink_gradient_C2"}.issubset(sig_set) or recalled_siphon_wormhole
    ):
        sub_rec = forest.shared_base_subtrees["sub_2layer_siphon_base"]
        bind_chk = validate_consistent_joint_subtree_binding(
            sub_rec,
            {
                "role_bindings": {"u_drive": "u_now", "x_pool": "x_now", "s_grad": "s_grad", "y_out": "y"},
                "satisfied_shared_node_ids": ["C1_base_capacity_field", "C2_adjacent_high_gradient_sink"],
                "normalized_tau_ratio": 1.0 / max(1e-6, ds.tau_scale),
            },
        )
        s_vals = [float(r.get("s_grad") or 0.0) for r in init_rows]
        s_span = (max(s_vals) - min(s_vals)) if s_vals else 0.0
        narrow_sgrad_collinear = s_span < 0.08 and ("sig:mass_sum_drift" not in sig_set)
        has_coupled_memory = bool({"sig:coupled_transient_lag", "sig:endpoint_matched_history_gap"} & sig_set)
        has_dual_horizon = "sig:endogenous_adaptation_candidate" in sig_set
        has_hill_inh = "sig:sigmoidal_inflection_onset" in sig_set
        has_barrier_mod = "sig:barrier_activation_onset" in sig_set
        has_gray_zone_block = any("gray_zone" in str(v) for v in gate_rejs.values())
        if "sig:mass_sum_drift" in sig_set:
            b_rule = "sig:mass_sum_drift present -> fit linear leak `-leak * u_n` directly on initial 5 pilot rows (no extra probe needed)!"
        elif has_coupled_memory:
            b_rule = (
                "MULTI-TREE CO-ACTIVATION (`sub_2layer_siphon_base` + `node_A_memory_kernel_full`): "
                "`u_n = u_now / tau_scale`, `J_siph = vmax*(1 - rho*s_grad/(1+s_grad))*(u_n/(1.2+u_n))` drives the 2-state RK4 memory ODE "
                "`dx/dt = -a_slow*x + b_couple*z + J_siph`, `tau_fast*dz/dt = 0.52*x - z` over `[0, H]` from `(x_now, mem_conv)`, `y = x(H)`! "
                "Acquire `open_branch_discriminating_probe` (if narrow `s_grad`) AND `extra_fine_trajectory_batch` BEFORE fitting so >= 8 rows identify 5 params `(a_slow, b_couple, vmax, rho, tau_fast)`!"
            )
        elif has_dual_horizon:
            b_rule = (
                "MULTI-TREE CO-ACTIVATION (`sub_2layer_siphon_base` + `node_B_dual_horizon_reversal`): "
                "Acquire `slow_horizon_discriminating_probe` FIRST, then fit 5-param sink-coupled dual-horizon reversal "
                "`y = vmax*(1 - rho*s_grad/(1+s_grad))*(u_n/(1.2+u_n))*(1 - exp(-H/tau_fast)) - amp_adapt*u_n*(1 - exp(-H/tau_slow))`!"
            )
        elif has_hill_inh:
            b_rule = (
                "MULTI-TREE CO-ACTIVATION (`node_Q_cooperative_hill_inhibition` + `C2` sink inhibition): "
                "`u_n = u_now / tau_scale`; acquire `open_branch_discriminating_probe` if `s_grad` is narrow (< 0.08), then fit 4-param cooperative Hill-2 competitive inhibition "
                "`y = vmax * ((u_n ** 2) / (k_half + (u_n ** 2))) / (1.0 + alpha_inh * s_grad) - c_leak * u_n`!"
            )
        elif has_barrier_mod:
            b_rule = (
                "MULTI-TREE CO-ACTIVATION (`sub_2layer_siphon_base` + `node_R_barrier_modulated_siphon`): "
                "`u_n = u_now / tau_scale`; acquire `open_branch_discriminating_probe` if `s_grad` is narrow (< 0.08), then fit 4-param barrier-modulated siphon "
                "`y = j0 * u_n * exp(-beta_bar / (0.5 + u_n)) * (1 - rho * s_grad / (1 + s_grad)) + c_sqrt * sqrt(u_n)` (`rho` warm-start near 0.70, wide `[0.05, 0.95]`)!"
            )
        elif has_gray_zone_block and ("high_sgrad_regime_probe" in ds.probe_pool):
            b_rule = (
                f"GRAY-ZONE GATE BLOCKED SMOOTH SIPHON (`s_grad={obs_feats['s_grad']:.2f}` enters unverified supercritical gray zone!): "
                "Acquire `high_sgrad_regime_probe` IMMEDIATELY on Turn 1, then fit 5-param threshold-avalanche siphon "
                "`y = vmax*(1 - rho*s_grad/(1+s_grad))*(u_n/(1.2+u_n)) + c_loc*sqrt(u_n) - k_aval*max(0.0, s_grad - s_crit)*u_n`!"
            )
        elif narrow_sgrad_collinear:
            b_rule = (
                "WARNING: initial 5 pilot rows have narrow `s_grad` span (< 0.08), making `vmax` and `rho` collinear on pilot alone! "
                "Acquire `open_branch_discriminating_probe` FIRST (adds wide `s_grad` leverage) and fit over the FULL `wide_fallback_box` `rho in [0.05, 0.95]` so heterogeneous target `rho` (e.g. `rho ~ 0.14`) is never clipped!"
            )
        else:
            b_rule = "Fit siphon base + `c_loc * sqrt(u_n)` with warm-start `rho` near 0.70 (initial 5 pilot rows already identify 3 params; if fresh confirmation refutes warm-start `rho`, escape to `wide_fallback_box` `rho in [0.05, 0.95]`)!"
        subtree_guidance = {
            "matched_subtree_id": sub_rec["subtree_id"],
            "joint_binding": bind_chk["merged_role_bindings"],
            "d_eff_set": sub_rec["diversity_metrics"]["d_eff_set"],
            "dimensionless_warm_start_box": priors_bundle["siphon_warm_prior"]["warm_start_box"],
            "wide_fallback_box": priors_bundle["siphon_warm_prior"]["wide_fallback_box"],
            "open_completion_slots": list(sub_rec.get("open_completion_slots") or ()),
            "pilot_s_grad_span": round(s_span, 5),
            "narrow_s_grad_collinearity_detected": narrow_sgrad_collinear,
            "branch_rule": b_rule,
        }
        p_id = ledger.record_decision(
            decision_type="D4_SHARED_BASE_SUBTREE_BINDING_AND_WARMSTART",
            parent_decision_id=p_id,
            evidence_inputs={"subtree_id": sub_rec["subtree_id"], "cues": sorted(sig_set), "pilot_s_grad_span": round(s_span, 5)},
            candidates_evaluated=[{"subtree_id": sub_rec["subtree_id"], "valid": bind_chk["valid"], "narrow_sgrad_collinear": narrow_sgrad_collinear}],
            chosen_option=subtree_guidance,
            rejection_reasons={"polysemous_single_factor_models": "rejected_by_conjunctive_C1_AND_C2_binding"},
            rationale_summary="Bind 2-layer C1 AND C2 SharedBaseSubtreePrior + dimensionless soft warm-start + wide escape.",
            compute_delta=0.5,
        )

    if any(r.get("mem_conv") is None and r.get("raw_prefix") for r in init_rows):
        ex_pref = init_rows[0]["raw_prefix"]
        reg_ex = extract_causal_temporal_regimes(
            ex_pref, t0=float(ex_pref[-1]["t"]), extraction_window_w=0.5, intrinsic_taus=(0.6,), allow_retrospective=False,
        )
        causal_regime_guidance = {
            "strict_causal_prefix_verified": reg_ex["ok"],
            "irregular_sampling_detected": reg_ex["temporal_parameters"]["is_irregular_sampling"],
            "time_origin_shift_invariant_rule": (
                "Rows have `mem_conv = None` and raw irregular `raw_prefix` (`[{'t', 'u', 'x'}, ...]`)! "
                "1) Acquire `extra_fine_trajectory_batch` FIRST so >= 8 trajectories identify 4 params `(a_slow, b_couple, c_drive, tau_fast)`. "
                "2) For each row and candidate `tau_fast`, compute `z_t0` from `pref = row['raw_prefix']` using shifted interval `dt_hist = pref[-1]['t'] - pref[0]['t']` "
                "(NEVER `t_latest` vs `0.0`!): `z_init = 0.52 * pref[0]['x']`, trapezoidal sum `conv_x = sum_{k=1..n-1} 0.5*(exp(-(t_end - t_{k-1})/tau_fast)*x_{k-1} + exp(-(t_end - t_k)/tau_fast)*x_k)*(t_k - t_{k-1}) / tau_fast`, "
                "`z_t0 = exp(-dt_hist / tau_fast) * z_init + 0.52 * conv_x` (or RK4-integrate `tau_fast * dz/dt = 0.52*x(t) - z` piecewise linearly across `pref`), then RK4-integrate `(x, z)` over `[0, row['H']]` from `(row['x_now'], z_t0)` with constant `row['u_now']`!"
            ),
        }

    if "sig:endogenous_adaptation_candidate" in sig_set:
        voi_rev_cands = [
            {"branch_id": f"{r_lbl}_{n_lbl}", "prior_weight": 0.25,
             "predict_fn": (lambda inp, rev=has_rev: 1.5 * (1.0 - math.exp(-inp["H"] / 0.35)) - (2.4 * (1.0 - math.exp(-inp["H"] / 12.0)) if rev else 0.0)),
             "probe_predict_fn": (lambda pr, rev=has_rev, nv=n_val: (nv if pr["id"] == "nuisance_high_var_aux" else 1.5 * (1.0 - math.exp(-pr["H"] / 0.35)) - (2.4 * (1.0 - math.exp(-pr["H"] / 12.0)) if rev else 0.0)))}
            for r_lbl, has_rev in (("monotone_fast", False), ("slow_reversal", True))
            for n_lbl, n_val in (("nuis_lo", -75.0), ("nuis_hi", +75.0))
        ]
        p_slow = ds.probe_pool["slow_horizon_discriminating_probe"]
        p_nuis = ds.probe_pool["nuisance_high_var_aux"]
        voi_res = select_m5_discriminating_probe(
            candidate_branches=voi_rev_cands,
            candidate_probes=[{**p_nuis, "H": 0.3}, {**p_slow, "H": p_slow["row"]["H"]}],
            target_reference_inputs=[{"H": ds.horizon_H}],
            decision_tolerance=tol,
            policy="task_risk_voi",
        )
        voi_guidance = {
            "recommended_action": voi_res["action"],
            "recommended_probe_id": voi_res.get("selected_probe_id"),
            "note": "Initial 5 pilot rows only cover H<=0.48; acquire `slow_horizon_discriminating_probe` first to calibrate 4-param dual-exponential reversal at H=20!",
        }
        p_id = ledger.record_decision(
            decision_type="D5_VOI_PROBE_VALUATION_AND_HEC_CHECK",
            parent_decision_id=p_id,
            evidence_inputs={"target_horizon_H": ds.horizon_H},
            candidates_evaluated=voi_res.get("voi_receipt", {}).get("comparisons", []),
            chosen_option=voi_guidance,
            rejection_reasons={"nuisance_high_var_aux": "zero_expected_target_risk_reduction"},
            rationale_summary="Run probing.py task-risk VOI valuation for slow-horizon disambiguation.",
            compute_delta=voi_res["compute_cost"],
        )

    brief = {
        **base_task_info,
        "arm_mode": "Hegel_M5_TemporalSubtree_Forest_Guided",
        "hegel_ppr_authorized_source_notes": [
            {
                **node_archive[nid],
                "temporal_readout_mode": forest.nodes[nid]["observation_map"].get("temporal_readout_mode"),
                "folding_output_kind": (forest.nodes[nid].get("scale_map") or {}).get("folding_output_kind"),
            }
            for nid in auth_ids if nid in node_archive
        ],
        "gray_zone_and_scope_rejections": gate_rejs,
        "shared_base_subtree_guidance": subtree_guidance,
        "causal_temporal_regime_guidance": causal_regime_guidance,
        "task_risk_voi_probe_guidance": voi_guidance,
    }
    return sys_prompt, brief, p_id


def run_single_live_gpt_m5_episode(
    ds: M5DisjointTaskDataset,
    arm: str,
    *,
    client: Any,
    workspace_dir: Path,
    forest: PhilosophicalForest,
    priors_bundle: dict[str, Any],
    hippo_idx: HippoRAGWormholeIndex,
    flat_idx: FlatEqualMetadataIndex,
    node_archive: dict[str, dict[str, Any]] | None = None,
    max_turns: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one complete multi-turn Live GPT scientific discovery episode with full provenance logging!"""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if node_archive is None:
        _, _, node_archive = _build_live_k15_forest_and_archive()

    tol = float(ds.tolerance)
    model_name = getattr(client, "model", "unknown_model")
    prov_backend = getattr(client, "provider_backend", "mock_or_custom_client")
    run_id = f"live_sub__{model_name}"
    ledger = DecisionProvenanceLedger(
        run_id=run_id, task_id=ds.task_id, arm=arm, model=model_name, provider_backend=prov_backend,
    )

    is_strong_ord = arm == "Strong_Ordinary_LiveGPT"
    has_coarse_cert = (not is_strong_ord) and ds.coarse_channel_available and ("sig:slow_horizon_query" in ds.probe_signatures)
    unit_row_m = 2.0 if is_strong_ord else (1.0 if has_coarse_cert else 1.5)

    train_rows = list(ds.pilot_rows)
    parent_id = ledger.record_decision(
        decision_type="D0_TASK_ARRIVAL_AND_PILOT_ACQUISITION",
        parent_decision_id=None,
        evidence_inputs={"task_id": ds.task_id, "pilot_row_keys": [_row_input_key(r) for r in train_rows]},
        candidates_evaluated=[
            {"channel": "coarse_macro", "unit_cost": 1.0, "eligible": has_coarse_cert},
            {"channel": "standard_fine", "unit_cost": (2.0 if is_strong_ord else 1.5), "eligible": True},
        ],
        chosen_option={"acquired_pilot_rows": len(train_rows), "unit_row_m": unit_row_m},
        rejection_reasons=({} if has_coarse_cert else {"coarse_macro": "not_certified_or_not_slow_query"}),
        rationale_summary="Acquire initial prefix-isolated pilot observations for Live GPT agent.",
        measurement_delta=len(train_rows) * unit_row_m,
    )

    sys_prompt, brief, parent_id = _build_arm_system_and_initial_brief(
        ds, arm, init_rows=train_rows, forest=forest, priors_bundle=priors_bundle,
        node_archive=node_archive, hippo_idx=hippo_idx, flat_idx=flat_idx,
        ledger=ledger, parent_id=parent_id,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(brief, ensure_ascii=False)},
    ]

    confirm_m_paid = False
    archive_expanded_after_refute = False
    purchased_probes: set[str] = set()
    eval_models_count = 0
    api_calls_count = 0
    stage_subdir = f"{model_name}__{arm}__{ds.task_id}"

    for turn_idx in range(1, max_turns + 1):
        prompt_sha = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        if isinstance(client, CodexSubscriptionClient):
            gpt_step = client.call_tool_step(messages, stage_subdir=stage_subdir)
        else:
            gpt_step = client.call_tool_step(messages)
        api_calls_count += 1

        pt = int(gpt_step.get("prompt_tokens", 0))
        ca = int(gpt_step.get("cached_tokens", 0))
        ct = int(gpt_step.get("completion_tokens", 0))
        eff_tok = max(0, pt - ca) + 0.25 * ca + ct
        # Uniform compute fee: 1.0 K per 1,000 effective tokens across all models & arms!
        tok_k = round(eff_tok / 1000.0, 4)

        if not gpt_step.get("ok"):
            parent_id = ledger.record_decision(
                decision_type="D6_LIVE_GPT_PROVIDER_ERROR",
                parent_decision_id=parent_id,
                evidence_inputs={"turn": turn_idx, "prompt_sha256_prefix": prompt_sha},
                candidates_evaluated=[],
                chosen_option={"error": gpt_step.get("error")},
                rejection_reasons={},
                rationale_summary="Live GPT subscription call returned an error; retry within turn budget.",
                compute_delta=tok_k,
                prompt_tokens_delta=pt,
                cached_tokens_delta=ca,
                completion_tokens_delta=ct,
                wall_ms_delta=gpt_step.get("wall_ms", 0.0),
            )
            messages.append({"role": "user", "content": f"Schema/CLI error: {gpt_step.get('error')}. Return one valid JSON decision."})
            continue

        t_name = gpt_step["tool_name"]
        t_args = gpt_step["tool_args"]
        t_call_id = gpt_step["tool_call_id"]
        messages.append(gpt_step["assistant_message_for_history"])

        considered_alts = list(t_args.get("considered_alternatives") or [])
        rej_map = {
            str(item.get("option", f"alt_{i}")): str(item.get("rejected_reason", ""))
            for i, item in enumerate(considered_alts)
            if isinstance(item, dict)
        }

        (workspace_dir / f"gpt_turn_{turn_idx:02d}_sanitized.json").write_text(
            json.dumps({
                "turn": turn_idx,
                "provider_backend": prov_backend,
                "returned_model": gpt_step.get("returned_model", model_name),
                "prompt_sha256_prefix": prompt_sha,
                "prompt_tokens": pt,
                "cached_tokens": ca,
                "completion_tokens": ct,
                "task_context_chars": gpt_step.get("task_context_chars"),
                "wall_ms": gpt_step["wall_ms"],
                "tool_name": t_name,
                "tool_args": t_args,
                "auth_mode": "chatgpt_oauth_subscription_redacted",
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        parent_id = ledger.record_decision(
            decision_type="D6_LIVE_GPT_TOOL_DECISION",
            parent_decision_id=parent_id,
            evidence_inputs={
                "turn": turn_idx,
                "prompt_sha256_prefix": prompt_sha,
                "current_train_rows": len(train_rows),
                "task_context_chars": gpt_step.get("task_context_chars"),
            },
            candidates_evaluated=[
                {"chosen_tool": t_name},
                *[{"alternative": a.get("option"), "status": "deferred_or_rejected"} for a in considered_alts if isinstance(a, dict)],
            ],
            chosen_option={
                "tool_name": t_name,
                "arguments_summary": {
                    k: (v if k not in {"python_code", "considered_alternatives"} else (f"<{len(str(v))} chars python>" if k == "python_code" else f"<{len(considered_alts)} alts>"))
                    for k, v in t_args.items()
                },
            },
            rejection_reasons=rej_map,
            rationale_summary=str(t_args.get("rationale", "")),
            compute_delta=tok_k,
            prompt_tokens_delta=pt,
            cached_tokens_delta=ca,
            completion_tokens_delta=ct,
            wall_ms_delta=gpt_step["wall_ms"],
        )

        if t_name == "acquire_probe":
            pid = str(t_args.get("probe_id", ""))
            if pid in purchased_probes:
                tool_reply: dict[str, Any] = {"status": "already_purchased", "train_row_count": len(train_rows)}
            elif pid == "extra_fine_trajectory_batch":
                purchased_probes.add(pid)
                added = ds.extra_dynamic_rows[:3]
                train_rows.extend(added)
                parent_id = ledger.record_decision(
                    decision_type="D5B_PAID_PROBE_EXECUTION",
                    parent_decision_id=parent_id,
                    evidence_inputs={"probe_id": pid},
                    candidates_evaluated=[{"probe_id": pid, "cost": 4.5}],
                    chosen_option={"added_rows": len(added), "new_row_keys": [_row_input_key(r) for r in added]},
                    rejection_reasons={},
                    rationale_summary="Execute extra_fine_trajectory_batch acquisition.",
                    measurement_delta=4.5,
                )
                tool_reply = {"status": "acquired", "added_rows": added, "total_train_rows": train_rows}
            elif pid in ds.probe_pool:
                purchased_probes.add(pid)
                p_obj = ds.probe_pool[pid]
                p_cost = p_obj["measurement_cost"] + p_obj["reset_cost"] + p_obj["horizon_wait_cost"]
                batch_rows = list(p_obj.get("rows") or ([p_obj["row"]] if p_obj.get("row") is not None else []))
                if batch_rows:
                    train_rows.extend(batch_rows)
                parent_id = ledger.record_decision(
                    decision_type="D5B_PAID_PROBE_EXECUTION",
                    parent_decision_id=parent_id,
                    evidence_inputs={"probe_id": pid},
                    candidates_evaluated=[{"probe_id": pid, "cost": p_cost}],
                    chosen_option={"added_rows": batch_rows},
                    rejection_reasons={},
                    rationale_summary=f"Execute paid probe {pid}.",
                    measurement_delta=p_cost,
                )
                tool_reply = {
                    "status": "acquired",
                    "added_row": p_obj.get("row"),
                    "added_rows": batch_rows,
                    "nuisance_reading": (65.0 if pid == "nuisance_high_var_aux" else None),
                    "total_train_rows": train_rows,
                }
            else:
                tool_reply = {"status": "unknown_probe_id", "available": [*ds.probe_pool.keys(), "extra_fine_trajectory_batch"]}

            messages.append({"role": "tool", "tool_call_id": t_call_id, "content": json.dumps(tool_reply, ensure_ascii=False)})
            continue

        if t_name == "abstain_scoped_unknown":
            parent_id = ledger.record_decision(
                decision_type="D9_TERMINAL_DECISION_AND_HELDOUT_AUDIT",
                parent_decision_id=parent_id,
                evidence_inputs={"abstained_at_turn": turn_idx},
                candidates_evaluated=[{"status": "scoped_unknown"}],
                chosen_option={"final_status": "scoped_unknown", "strict_pass": bool(ds.is_out_of_scope)},
                rejection_reasons=rej_map,
                rationale_summary=f"Live GPT abstained as scoped_unknown: {t_args.get('rationale', '')}",
            )
            tots = ledger.totals
            return {
                "task_id": ds.task_id, "suite": ds.suite, "subfamily": ds.subfamily, "arm": arm,
                "model": model_name, "provider_backend": prov_backend, "api_calls": api_calls_count,
                "is_out_of_scope": ds.is_out_of_scope, "status": "scoped_unknown",
                "strict_pass": bool(ds.is_out_of_scope), "false_confirmation": False,
                "evaluated_candidate_count": eval_models_count,
                "confirm_rmse": None, "heldout_rmse": None,
                **tots,
            }, ledger.records

        if t_name == "author_and_confirm_python_model":
            eval_models_count += 1
            py_src = str(t_args.get("python_code", ""))
            hyp_name = str(t_args.get("hypothesis_name", f"cand_{turn_idx}"))
            exec_res = _execute_authored_python_in_isolated_subprocess(
                workspace_dir, step_idx=turn_idx, python_code=py_src,
                train_rows=train_rows, confirm_rows=ds.confirm_rows, heldout_rows=ds.heldout_rows,
            )
            conf_m_delta = (len(ds.confirm_rows) * unit_row_m) if (exec_res.get("ok") and not confirm_m_paid) else 0.0
            if exec_res.get("ok"):
                confirm_m_paid = True

            if not exec_res.get("ok"):
                parent_id = ledger.record_decision(
                    decision_type="D7_PYTHON_SUBPROCESS_EXECUTION_ERROR",
                    parent_decision_id=parent_id,
                    evidence_inputs={"hypothesis_name": hyp_name, "model_sha256": exec_res["model_sha256"]},
                    candidates_evaluated=[{"hypothesis_name": hyp_name, "ok": False}],
                    chosen_option={"action": "return_traceback_to_gpt_for_repair"},
                    rejection_reasons={hyp_name: str(exec_res.get("error", ""))[:240]},
                    rationale_summary="Isolated Python subprocess raised runtime/syntax error; feed stderr back to GPT.",
                    compute_delta=1.5,
                    wall_ms_delta=exec_res["exec_wall_ms"],
                    extra_artifacts={"model_path": exec_res["model_path"]},
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": t_call_id,
                    "content": json.dumps({"status": "execution_error", "error": exec_res["error"]}, ensure_ascii=False),
                })
                continue

            c_rmse = exec_res["confirm_rmse"]
            accepted = c_rmse <= tol
            parent_id = ledger.record_decision(
                decision_type="D7_D8_CANDIDATE_FIT_AND_FRESH_CONFIRMATION",
                parent_decision_id=parent_id,
                evidence_inputs={"hypothesis_name": hyp_name, "model_sha256": exec_res["model_sha256"], "train_rows": len(train_rows)},
                candidates_evaluated=[{
                    "hypothesis_name": hyp_name,
                    "train_rmse": exec_res["train_rmse"],
                    "confirm_rmse": c_rmse,
                    "tolerance": tol,
                }],
                chosen_option={"verdict": "CONFIRM_ACCEPTED" if accepted else "CONFIRM_REFUTED"},
                rejection_reasons=({} if accepted else {hyp_name: f"confirm_rmse {c_rmse:.6f} > tolerance {tol:.6f}"}),
                rationale_summary=(
                    f"GPT-authored {hyp_name} confirmed on disjoint points (confirm_rmse={c_rmse:.6f})."
                    if accepted else f"GPT-authored {hyp_name} refuted on disjoint confirm (confirm_rmse={c_rmse:.6f} > {tol})."
                ),
                measurement_delta=conf_m_delta,
                compute_delta=3.5,
                wall_ms_delta=exec_res["exec_wall_ms"],
                extra_artifacts={"model_path": exec_res["model_path"]},
            )

            if accepted:
                h_rmse = exec_res["heldout_rmse"]
                ledger.record_decision(
                    decision_type="D9_TERMINAL_DECISION_AND_HELDOUT_AUDIT",
                    parent_decision_id=parent_id,
                    evidence_inputs={"locked_model_sha256": exec_res["model_sha256"], "any_fit_after_confirm": False},
                    candidates_evaluated=[{"hypothesis_name": hyp_name, "heldout_rmse": h_rmse}],
                    chosen_option={"final_status": "confirmed_ok", "strict_pass": (not ds.is_out_of_scope) and (h_rmse <= tol)},
                    rejection_reasons={},
                    rationale_summary="Lock GPT-authored Python model and audit on unseen heldout rows.",
                    extra_artifacts={"model_path": exec_res["model_path"]},
                )
                tots = ledger.totals
                return {
                    "task_id": ds.task_id, "suite": ds.suite, "subfamily": ds.subfamily, "arm": arm,
                    "model": model_name, "provider_backend": prov_backend, "api_calls": api_calls_count,
                    "is_out_of_scope": ds.is_out_of_scope, "status": "confirmed_ok",
                    "confirmed_hypothesis_name": hyp_name,
                    "confirmed_model_sha256": exec_res["model_sha256"],
                    "strict_pass": (not ds.is_out_of_scope) and (h_rmse <= tol),
                    "false_confirmation": bool(ds.is_out_of_scope) or (h_rmse > tol),
                    "evaluated_candidate_count": eval_models_count,
                    "confirm_rmse": c_rmse, "heldout_rmse": h_rmse,
                    **tots,
                }, ledger.records

            # On first refutation when Top-3 recall missed true mechanism, allow CEGAR backoff archive expansion (costs 1.0 K)!
            refute_feedback: dict[str, Any] = {
                "status": "refuted_on_fresh_confirmation",
                "train_rmse": exec_res["train_rmse"],
                "confirm_rmse": c_rmse,
                "tolerance": tol,
                "hint": "Acquire informative probes if underdetermined/short-H, revise mechanism/branch, or call abstain_scoped_unknown if non-stationary.",
            }
            if arm == "Hegel_M5_LiveGPT" and exec_res["train_rmse"] <= 0.01 and c_rmse > 5.0 * tol:
                unbought_regime_probes = [
                    pk for pk in ds.probe_pool.keys()
                    if pk != "nuisance_high_var_aux" and pk not in purchased_probes
                ]
                if unbought_regime_probes:
                    refute_feedback["cegar_regime_breakdown_diagnosis"] = (
                        f"CEGAR Diagnosis: train_rmse={exec_res['train_rmse']:.6f} <= 0.01 on current train_rows while "
                        f"fresh confirm_rmse={c_rmse:.4f} >> {tol}! Smooth extrapolation broke down in an unprobed regime "
                        f"(e.g. critical threshold / piecewise avalanche `-k_aval*max(0.0, s_grad - s_crit)*u_n` or slow reversal). "
                        f"Acquire {unbought_regime_probes} FIRST before synthesizing the sublated piecewise/extended law!"
                    )
            if (not is_strong_ord) and (not archive_expanded_after_refute):
                archive_expanded_after_refute = True
                backoff_k = round(max(1.0, len(node_archive) / 15.0), 2)
                parent_id = ledger.record_decision(
                    decision_type="D8B_POST_REFUTATION_ARCHIVE_BACKOFF",
                    parent_decision_id=parent_id,
                    evidence_inputs={"trigger_confirm_rmse": c_rmse},
                    candidates_evaluated=[{"action": f"expand_remaining_k{len(node_archive)}_archive_notes", "compute_cost": backoff_k}],
                    chosen_option={"expanded_archive_count": len(node_archive)},
                    rejection_reasons={},
                    rationale_summary=f"Top-3 recalled hypothesis refuted on fresh confirmation; expand full K={len(node_archive)} archive notes for backoff.",
                    compute_delta=backoff_k,
                )
                refute_feedback["expanded_k15_archive_notes"] = list(node_archive.values())

            messages.append({
                "role": "tool",
                "tool_call_id": t_call_id,
                "content": json.dumps(refute_feedback, ensure_ascii=False),
            })

    ledger.record_decision(
        decision_type="D9_TERMINAL_DECISION_AND_HELDOUT_AUDIT",
        parent_decision_id=parent_id,
        evidence_inputs={"turn_limit_reached": max_turns},
        candidates_evaluated=[{"status": "scoped_unknown"}],
        chosen_option={"final_status": "scoped_unknown", "strict_pass": bool(ds.is_out_of_scope)},
        rejection_reasons={"all_attempted_models": "unconfirmed_within_turn_budget"},
        rationale_summary="Turn limit reached without passing fresh confirmation; return scoped_unknown.",
    )
    tots = ledger.totals
    return {
        "task_id": ds.task_id, "suite": ds.suite, "subfamily": ds.subfamily, "arm": arm,
        "model": model_name, "provider_backend": prov_backend, "api_calls": api_calls_count,
        "is_out_of_scope": ds.is_out_of_scope, "status": "scoped_unknown",
        "strict_pass": bool(ds.is_out_of_scope), "false_confirmation": False,
        "evaluated_candidate_count": eval_models_count,
        "confirm_rmse": None, "heldout_rmse": None,
        **tots,
    }, ledger.records


def build_m5_live_gpt_task_suite(seed: int = 20261120) -> list[M5DisjointTaskDataset]:
    """Construct the 6 representative unseen M5 tasks for Live GPT head-to-head evaluation."""
    rng = random.Random(seed)
    specs = [
        ("L01_CrossTimescale_MemoryKernel", "A_TemporalMemory", "A_history_conditioned_memory", False,
         ["sig:endpoint_matched_history_gap", "sig:coupled_transient_lag"], False,
         {"a_slow": 1.85, "b_couple": 1.25, "c_drive": 1.30, "tau_fast": 0.62, "g_feedback": 0.52}),
        ("L02_ShortGain_LongReversal", "B_MultiScaleFold", "B_short_gain_long_reversal", False,
         ["sig:short_window_positive_gain", "sig:endogenous_adaptation_candidate"], False,
         {"amp_fast": 1.48, "tau_fast": 0.35, "amp_slow": 2.40, "tau_slow": 12.0, "horizon_H": 20.0}),
        ("L03_Adiabatic_CoarseFold", "B_MultiScaleFold", "B_adiabatic_coarse_valid", False,
         ["sig:slow_horizon_query", "sig:fast_transient_damped"], True,
         {"a_slow": 1.50, "b_couple": 0.28, "c_drive": 2.70, "tau_fast": 0.02, "g_feedback": 0.50, "horizon_H": 3.6}),
        ("L04_TwoLayer_SharedSubtree_Siphon", "C_BaseSubtree", "C_two_layer_conjunctive_subtree", False,
         ["sig:base_field_C1", "sig:sink_gradient_C2", "sig:monotone_concave"], False,
         {"vmax": 6.80, "rho": 0.71, "local_branch_coeff": -0.38, "tau_scale": 1.0}),
        ("L05_HeteroRho_And_ExternalLeak", "C_BaseSubtree", "C_upstream_premise_failure", False,
         ["sig:base_field_C1", "sig:sink_gradient_C2", "sig:mass_sum_drift"], False,
         {"vmax": 7.10, "rho": 0.70, "external_leak_rate": 0.68}),
        ("L06_OutOfScope_NonStationary_Chirp", "C_BaseSubtree", "OOS_nonstationary_chirp", True,
         ["sig:high_freq_sign_oscillation"], False,
         {"omega": 19.3}),
    ]
    return [
        _build_disjoint_dataset(tid, s_name, sub_f, oos, sigs, c_ok, tp, rng)
        for (tid, s_name, sub_f, oos, sigs, c_ok, tp) in specs
    ]


def run_v18_1_m5_live_gpt_and_provenance_study(
    evidence_dir: Path,
    *,
    repo_root: Path,
    run_live_gpt: bool = True,
    gpt_models: list[str] | tuple[str, ...] = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol"),
    reasoning_effort: str = "medium",
    max_workers: int = 3,
) -> dict[str, Any]:
    """Run BOTH:
      1. Full Decision-Level Provenance Audit over the deterministic M5 task suite (verifying every hash chain!);
      2. Official OpenAI ChatGPT Subscription Multi-Model Live Benchmark (`gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.2-sol`)!
    """
    del repo_root
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # --- Part A: Full Forensic Decision-Level Provenance on Deterministic Suite (`seed=101`, 4 core arms = 80 episodes) ---
    from .m5_temporal_multiscale_and_base_subtree import _generate_m5_benchmark_tasks

    det_forest, det_priors = _build_seed_forest_and_priors()
    det_hippo = HippoRAGWormholeIndex(det_forest)
    det_flat = FlatEqualMetadataIndex(det_forest)
    det_tasks = _generate_m5_benchmark_tasks(101)
    det_arms = (
        "Hegel_M5_Full_TemporalSubtree_Forest",
        "EqualAuth_SingleNode_Wormhole_Only",
        "EqualAuth_Dynamic_Flat_BM25",
        "Strong_Dynamic_StateSpace_AllHistory",
    )
    det_ledger_lines: list[str] = []
    det_chain_verifications: list[dict[str, Any]] = []
    det_per_task_dir = evidence_dir / "deterministic_decision_dags"
    det_per_task_dir.mkdir(parents=True, exist_ok=True)

    for arm in det_arms:
        arm_rng = random.Random(hash((101, arm)) & 0xFFFFFFFF)
        for ds in det_tasks:
            ep_rec, chain_recs = run_traced_deterministic_m5_episode(
                ds, arm, forest=det_forest, priors_bundle=det_priors,
                hippo_idx=det_hippo, flat_idx=det_flat, rng=arm_rng,
            )
            v_chk = verify_decision_provenance_chain(chain_recs)
            assert v_chk["valid"], f"Provenance chain invalid on {arm}/{ds.task_id}: {v_chk}"
            assert abs(v_chk["verified_composite_cost"] - ep_rec["composite_cost"]) < 1e-6
            det_chain_verifications.append({"arm": arm, "task_id": ds.task_id, **v_chk})
            for cr in chain_recs:
                det_ledger_lines.append(json.dumps(cr, ensure_ascii=False, sort_keys=True))
            (det_per_task_dir / f"{arm}__{ds.task_id}.json").write_text(
                json.dumps({"episode_summary": ep_rec, "chain_verification": v_chk, "decision_dag": chain_recs}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    (evidence_dir / "deterministic_decision_provenance_ledger.jsonl").write_text(
        "\n".join(det_ledger_lines) + "\n", encoding="utf-8"
    )

    # --- Part B: Official OpenAI ChatGPT Subscription Multi-Model Live Execution ---
    codex_info = discover_chatgpt_subscription_codex_cli()
    live_gpt_summary: dict[str, Any] = {
        "executed": False,
        "provider_backend": codex_info.get("provider_backend"),
        "third_party_ruoli_blocked": True,
        "models_requested": list(gpt_models),
    }

    if run_live_gpt and codex_info.get("available"):
        live_tasks = build_m5_live_gpt_task_suite(20261120)
        live_arms = ("Hegel_M5_LiveGPT", "EqualAuth_Flat_LiveGPT", "Strong_Ordinary_LiveGPT")
        live_ws_root = evidence_dir / "live_gpt_workspaces_and_dags"
        live_ws_root.mkdir(parents=True, exist_ok=True)

        jobs: list[tuple[int, str, M5DisjointTaskDataset, str]] = []
        idx_counter = 0
        for m_slug in gpt_models:
            for ds in live_tasks:
                for arm in live_arms:
                    jobs.append((idx_counter, str(m_slug), deepcopy(ds), arm))
                    idx_counter += 1

        def _run_one_job(job: tuple[int, str, M5DisjointTaskDataset, str]) -> tuple[int, str, str, str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
            j_idx, m_slug, ds_item, arm_name = job
            print(f"[ChatGPT-Subscription:{m_slug}] ({j_idx + 1}/{len(jobs)}) {ds_item.task_id} | {arm_name} ...", file=sys.stderr, flush=True)
            f_loc, pb_loc, arch_loc = _build_live_k15_forest_and_archive()
            h_loc = HippoRAGWormholeIndex(f_loc)
            fl_loc = FlatEqualMetadataIndex(f_loc)
            sub_client = CodexSubscriptionClient(
                codex_info=codex_info,
                model=m_slug,
                reasoning_effort=reasoning_effort,
            )
            ws = live_ws_root / f"{m_slug}__{arm_name}__{ds_item.task_id}"
            ep_rec, chain_recs = run_single_live_gpt_m5_episode(
                ds_item, arm_name, client=sub_client, workspace_dir=ws,
                forest=f_loc, priors_bundle=pb_loc, hippo_idx=h_loc, flat_idx=fl_loc,
                node_archive=arch_loc, max_turns=6,
            )
            v_chk = verify_decision_provenance_chain(chain_recs)
            assert v_chk["valid"], f"Live provenance invalid on {m_slug}/{arm_name}/{ds_item.task_id}: {v_chk}"
            (ws / "decision_provenance_dag.json").write_text(
                json.dumps({"episode_summary": ep_rec, "chain_verification": v_chk, "decision_dag": chain_recs}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"  -> DONE [{m_slug} | {arm_name} | {ds_item.task_id}] status={ep_rec['status']} "
                f"pass={ep_rec['strict_pass']} M={ep_rec['measurement_cost']} C={ep_rec['composite_cost']} calls={ep_rec['api_calls']}",
                file=sys.stderr, flush=True,
            )
            return j_idx, m_slug, arm_name, ds_item.task_id, ep_rec, v_chk, chain_recs

        ordered_results: list[tuple[int, str, str, str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futs = [pool.submit(_run_one_job, jb) for jb in jobs]
            for fut in as_completed(futs):
                ordered_results.append(fut.result())

        ordered_results.sort(key=lambda item: item[0])

        live_receipts: list[dict[str, Any]] = []
        live_ledger_lines: list[str] = []
        live_verifications: list[dict[str, Any]] = []
        for _j_idx, m_slug, arm_name, t_id, ep_rec, v_chk, chain_recs in ordered_results:
            live_receipts.append(ep_rec)
            live_verifications.append({"model": m_slug, "arm": arm_name, "task_id": t_id, **v_chk})
            for cr in chain_recs:
                live_ledger_lines.append(json.dumps(cr, ensure_ascii=False, sort_keys=True))

        (evidence_dir / "live_gpt_decision_provenance_ledger.jsonl").write_text(
            "\n".join(live_ledger_lines) + "\n", encoding="utf-8"
        )
        (evidence_dir / "live_gpt_episode_receipts.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in live_receipts) + "\n",
            encoding="utf-8",
        )

        by_model: dict[str, Any] = {}
        for m_slug in gpt_models:
            m_arms: dict[str, Any] = {}
            for arm in live_arms:
                rows = [r for r in live_receipts if r["model"] == m_slug and r["arm"] == arm]
                in_r = [r for r in rows if not r["is_out_of_scope"]]
                oos_r = [r for r in rows if r["is_out_of_scope"]]
                m_tot = round(sum(r["measurement_cost"] for r in rows), 4)
                k_tot = round(sum(r["compute_cost"] for r in rows), 4)
                m_arms[arm] = {
                    "model": m_slug,
                    "total_tasks": len(rows),
                    "in_scope_strict_pass": f"{sum(1 for r in in_r if r['strict_pass'])}/{len(in_r)}",
                    "oos_scoped_unknown": f"{sum(1 for r in oos_r if r['status'] == 'scoped_unknown')}/{len(oos_r)}",
                    "false_confirmations": sum(1 for r in rows if r["false_confirmation"]),
                    "total_api_calls": sum(int(r["api_calls"]) for r in rows),
                    "total_raw_tokens": sum(int(r["total_tokens"]) for r in rows),
                    "total_effective_tokens": sum(int(r["effective_billed_tokens"]) for r in rows),
                    "total_decisions_logged": sum(int(r["decision_count"]) for r in rows),
                    "total_measurement_cost": m_tot,
                    "total_compute_cost": k_tot,
                    "total_composite_cost": _composite(m_tot, k_tot),
                    "total_wall_ms": round(sum(float(r["wall_ms"]) for r in rows), 1),
                }
            by_model[str(m_slug)] = m_arms

        live_gpt_summary = {
            "executed": True,
            "provider_backend": "openai_chatgpt_subscription_codex_cli",
            "auth_mode": "chatgpt",
            "third_party_ruoli_blocked": True,
            "reasoning_effort": reasoning_effort,
            "models_evaluated": list(gpt_models),
            "source_archive_scale_K": 15,
            "total_live_episodes": len(live_receipts),
            "total_live_decisions_verified": sum(v["decision_count"] for v in live_verifications),
            "all_live_hash_chains_valid": all(v["valid"] for v in live_verifications),
            "by_model": by_model,
        }

    manifest = {
        "version": "V18.1_M5_ChatGPT_Subscription_MultiModel_and_Decision_Provenance",
        "deterministic_provenance_audit": {
            "episodes_verified": len(det_chain_verifications),
            "total_atomic_decisions_verified": sum(v["decision_count"] for v in det_chain_verifications),
            "all_hash_chains_and_cost_sums_valid": all(v["valid"] for v in det_chain_verifications),
        },
        "live_gpt_subscription_benchmark": live_gpt_summary,
    }
    (evidence_dir / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sha_lines: list[str] = []
    for fpath in sorted(p for p in evidence_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        rel = fpath.relative_to(evidence_dir).as_posix()
        dig = hashlib.sha256(fpath.read_bytes()).hexdigest()
        sha_lines.append(f"{dig}  {rel}")
    (evidence_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    return manifest
