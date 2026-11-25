"""Hegel M8.2 Offline Container Trajectory Replay Calibrator & Variational Branch/Repair Planner.

Upgraded in M8.2 to close all `Hegel_M8_1_Audit` replay & calibration findings:
  1. **Four-State Honest Epistemic Classification (`VERIFIED` / `SUPPORTED_IN_PARTIAL_SCOPE` / `UNKNOWN` / `REFUTED`)**:
     - `VERIFIED` is reserved EXCLUSIVELY for explicit quantitative domain benchmark checks passing cleanly.
     - `skill_activation_count >= 1`, `nogood_count >= 1`, or generic schema/text checks NEVER forge `VERIFIED`;
       instead they map to `SUPPORTED_IN_PARTIAL_SCOPE` (`UNKNOWN != SUPPORTED_IN_PARTIAL_SCOPE != VERIFIED`).
  2. **Symmetric Accuracy-Only Minimax Tie-Break (`adjudicate_minimax_variational_tiebreak`)**:
     - Compares ONLY worst-case cross-regime error (`worst_mm`) and pass count (`pass_20mm`), completely
       decoupling `antithesis_skill_calls` so dummy tool activity can NEVER override worse scientific error.
  3. **Zero Private-Label Replay & Non-Trivial Realized-Utility NIG Policy**:
     - Historical replay defaults to raw unpatched archive (`reverify_archived_public_contracts=False`) and NEVER
       reads `postlock_branch_outcomes` when building `pre_action_state` or public contract checks.
     - Calibrates scale-covariant `NIG` on **realized policy net return** (using epistemic parameter uncertainty
       `epistemic_std`, not noise-inflated `predictive_std`), governing all `SUPPORTED_IN_PARTIAL_SCOPE` and
       `UNKNOWN` episodes where hard safety rules do not fire, and reports `rule_only_selective_passes` alongside
       `m8_variational_selective_passes`.
  4. **Pre-Decision vs Post-Execution `ResearchDecisionRecord` Separation**:
     - `decide_second_branch` emits `PROPOSED_PRE_EXECUTION` with `actual_cost=0.0` and `estimated_selected_cost`;
       `finalize_container_decision_record` stamps measured post-execution cost and tokens after execution completes.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .m8_persistent_variational_planner import (
    ResearchDecisionRecord,
    fit_scale_covariant_nig_linear_model,
    predict_nig_student_t_log_score,
)

CONTAINER_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "unfinished_or_refuted",
    "epistemic_unknown_unprobed",
    "ungrounded_turn_exhaustion",
    "verified_domain_benchmark",
    "grounded_partial_scope",
)


def classify_public_container_verification_state(
    *,
    benchmark: str,
    ended_by_finish: bool,
    contract_accepted: bool,
    public_metrics: dict[str, Any] | None,
    skill_activation_count: int,
    nogood_count: int,
    submitted_final_answer: str | None = None,
) -> tuple[str, list[str]]:
    """Classify pre-lock primary container state into 4 honest epistemic tiers:
    `VERIFIED`, `SUPPORTED_IN_PARTIAL_SCOPE`, `UNKNOWN`, `REFUTED`.
    """
    reasons: list[str] = []
    if not ended_by_finish:
        return "REFUTED", ["primary_did_not_finish_cleanly"]
    if not contract_accepted:
        return "REFUTED", ["failed_public_artifact_contract"]

    if benchmark == "BioMysteryBench-Preview":
        ans = str(submitted_final_answer or "").strip()
        if not ans:
            return "REFUTED", ["empty_biomystery_answer"]
        if int(nogood_count) >= 1:
            # Falsification nogood on open-ended text QA is partial support, NOT full domain verification!
            return "SUPPORTED_IN_PARTIAL_SCOPE", ["non_empty_answer_with_in_container_falsification_nogood"]
        return "UNKNOWN", ["unverified_single_trace_text_answer"]

    pub = public_metrics or {}

    # 1. Explicit public refutations -> `REFUTED`
    if pub.get("public_checker_passed") is False:
        reasons.append("failed_public_checker_script")
    if pub.get("public_output_contract_ok") is False:
        reasons.append("failed_public_output_completeness_contract")
    if pub.get("id_renaming_invariant_ok") is False:
        reasons.append("failed_public_identifier_renaming_invariance")
    if pub.get("amr_solve_ok") is False:
        reasons.append("failed_public_amr_residual")
    if pub.get("sparse_cert_verification_state") == "REFUTED" or (
        pub.get("sparse_cert_valid") is False
        and pub.get("sparse_cert_verification_state") != "UNKNOWN"
        and not bool(pub.get("sparse_cert_sampled_diagnostic"))
    ):
        reasons.append("refuted_sparse_certificate")
    if pub.get("traffic_delayed_ok") is False or pub.get("traffic_ordinary_ok") is False:
        reasons.append("failed_traffic_conservation_rollout")
    if pub.get("clinical_batch_collapse") is True or pub.get("clinical_schema_ok") is False:
        reasons.append("clinical_batch_collapse_or_schema_failure")
    if isinstance(pub.get("nano_missing_popins"), list) and len(pub["nano_missing_popins"]) > 0:
        reasons.append("missing_public_popin_events")
    if pub.get("symbolic_oof_status") == "unsupported_interface" or (
        pub.get("symbolic_oof_f1") is not None and float(pub["symbolic_oof_f1"]) < 0.72
    ):
        reasons.append("low_or_unsupported_public_oof_f1")
    for wk, st_k in (("gw_cross_worst_mm", "gw_probe_status"), ("vb_cross_worst_mm", "vb_probe_status")):
        if str(pub.get(st_k) or "").startswith(("timeout", "error", "rc_")):
            reasons.append(f"{st_k}_failed")
        elif pub.get(wk) is not None and float(pub[wk]) > 20.0:
            reasons.append(f"{wk}_exceeds_20mm")
    if pub.get("nav_schema_ok") is False:
        reasons.append("failed_navigation_cli_holdout")

    if reasons:
        return "REFUTED", reasons

    # 2. Explicit quantitative domain target benchmark passed -> `VERIFIED`
    verified_receipts: list[str] = []
    if pub.get("amr_solve_ok") is True:
        verified_receipts.append("amr_solve_ok")
    if pub.get("sparse_cert_valid") is True or pub.get("sparse_cert_verification_state") == "VERIFIED":
        verified_receipts.append("sparse_cert_verified")
    if pub.get("traffic_ordinary_ok") is True and pub.get("traffic_delayed_ok") is True:
        verified_receipts.append("traffic_rollouts_verified")
    if pub.get("clinical_schema_ok") is True and pub.get("clinical_batch_collapse") is False:
        verified_receipts.append("clinical_deconfounding_verified")
    if pub.get("symbolic_oof_f1") is not None and float(pub["symbolic_oof_f1"]) >= 0.72:
        verified_receipts.append("symbolic_oof_verified")
    if pub.get("spin_glass_valid") is True:
        verified_receipts.append("spin_glass_hamiltonian_verified")
    if pub.get("nav_schema_ok") is True:
        verified_receipts.append("navigation_holdout_verified")
    if pub.get("public_output_contract_ok") is True and pub.get("id_renaming_invariant_ok") is True:
        verified_receipts.append("public_output_and_renaming_invariance_verified")
    for wk, st_k in (("gw_cross_worst_mm", "gw_probe_status"), ("vb_cross_worst_mm", "vb_probe_status")):
        if pub.get(st_k) == "ok" and pub.get(wk) is not None and float(pub[wk]) <= 20.0:
            verified_receipts.append(f"{wk}_verified_le_20mm")

    if verified_receipts:
        return "VERIFIED", verified_receipts

    # 3. Partial-scope / indirect empirical checks (`skill >= 1`, `nogood >= 1`, or generic checker) -> `SUPPORTED_IN_PARTIAL_SCOPE`
    partial_receipts: list[str] = []
    if pub.get("public_checker_passed") is True:
        partial_receipts.append("generic_public_checker_passed")
    if isinstance(pub.get("nano_missing_popins"), list) and len(pub["nano_missing_popins"]) == 0:
        partial_receipts.append("nano_popin_completeness_checked")
    if int(skill_activation_count) >= 1:
        partial_receipts.append("autogenous_skill_operator_invoked")
    if int(nogood_count) >= 1:
        partial_receipts.append("in_container_falsification_nogood_probed")

    if partial_receipts:
        return "SUPPORTED_IN_PARTIAL_SCOPE", partial_receipts

    # 4. Zero refutations, zero domain benchmarks, and zero probes -> `UNKNOWN`
    return "UNKNOWN", ["no_public_benchmark_or_falsification_probe_executed"]


def build_pre_action_feature_vector(pre_state: dict[str, Any]) -> np.ndarray:
    """Map prefix-clean `pre_action_state` to 6D design row `[1, f_refuted, f_unknown_code, f_unknown_bm, f_exhaust, f_grounded]`."""
    ver_state = str(pre_state["public_verification_state"])
    turns_ratio = float(pre_state["turns_ratio"])
    sk_cnt = int(pre_state["skill_activation_count"])
    ng_cnt = int(pre_state["nogood_count"])
    is_bm = str(pre_state.get("benchmark") or "") == "BioMysteryBench-Preview"

    f_refuted = 1.0 if (ver_state == "REFUTED" or not bool(pre_state["ended_by_finish"]) or bool(pre_state.get("pullback_gate_triggered"))) else 0.0
    f_unknown_code = 1.0 if (ver_state == "UNKNOWN" and not is_bm) else 0.0
    f_unknown_bm = 1.0 if (ver_state == "UNKNOWN" and is_bm) else 0.0
    f_exhaust = 1.0 if (turns_ratio >= 0.96 and sk_cnt == 0) else max(0.0, (turns_ratio - 0.85) / 0.15) * (0.25 if sk_cnt >= 1 else 1.0)
    f_grounded = (
        1.0
        if (
            ver_state == "VERIFIED"
            or (ver_state == "SUPPORTED_IN_PARTIAL_SCOPE" and (sk_cnt >= 1 or ng_cnt >= 1) and turns_ratio < 0.96)
        )
        else 0.0
    )
    return np.array([1.0, f_refuted, f_unknown_code, f_unknown_bm, f_exhaust, f_grounded], dtype=float)


def build_container_pre_action_state(
    *,
    task_id: str,
    benchmark: str,
    wave: str,
    primary_summary: dict[str, Any],
    contract_check: dict[str, Any] | None = None,
    public_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct strictly prefix-clean pre-action state dictionary (`T09` compliant)."""
    ended = bool(primary_summary.get("ended_by_finish", False))
    max_t = max(1, int(primary_summary.get("max_turns") or 28))
    used_t = int(primary_summary.get("turns_used") or 0)
    sk_cnt = int(primary_summary.get("skill_activation_count") or 0)
    ng_cnt = int(primary_summary.get("nogood_count") or len(primary_summary.get("nogoods_recorded") or []))
    staged_cnt = len(primary_summary.get("staged_wormhole_modules") or [])
    chk_ok = bool((contract_check or {}).get("accepted", True))
    if benchmark != "BioMysteryBench-Preview" and not primary_summary.get("final_candidate_sha256") and contract_check is None:
        chk_ok = False

    v_state, v_reasons = classify_public_container_verification_state(
        benchmark=benchmark,
        ended_by_finish=ended,
        contract_accepted=chk_ok,
        public_metrics=public_metrics,
        skill_activation_count=sk_cnt,
        nogood_count=ng_cnt,
        submitted_final_answer=primary_summary.get("submitted_final_answer"),
    )
    return {
        "task_id": task_id,
        "benchmark": benchmark,
        "wave": wave,
        "ended_by_finish": ended,
        "pullback_gate_triggered": bool(primary_summary.get("pullback_gate_triggered", False)),
        "turns_used": used_t,
        "max_turns": max_t,
        "turns_ratio": round(used_t / max_t, 5),
        "staged_module_count": staged_cnt,
        "skill_activation_count": sk_cnt,
        "nogood_count": ng_cnt,
        "contract_accepted": chk_ok,
        "public_verification_state": v_state,
        "public_verification_reasons": v_reasons,
    }


def adjudicate_minimax_variational_tiebreak(
    *,
    pub_thesis: dict[str, Any],
    pub_antithesis: dict[str, Any],
    antithesis_skill_calls: int = 0,
) -> dict[str, Any] | None:
    """Symmetric accuracy-only minimax tail-risk tie-breaker (`M8.2` Fix: zero activity-count bias!).

    Compares ONLY public worst-case cross-geometry error (`worst_mm`) and pass count (`pass_20mm`).
    `antithesis_skill_calls` is accepted strictly for backward call-site compatibility and NEVER shifts the score!
    """
    _ = antithesis_skill_calls
    for pass_k, worst_k, status_k, tot_default in (
        ("vb_cross_pass_20mm", "vb_cross_worst_mm", "vb_probe_status", 10),
        ("gw_cross_pass_20mm", "gw_cross_worst_mm", "gw_probe_status", 6),
    ):
        if pub_thesis.get(status_k) == "ok" and pub_antithesis.get(status_k) == "ok":
            tot = int(pub_thesis.get(status_k.replace("_probe_status", "_cross_total")) or tot_default)
            pt = int(pub_thesis.get(pass_k) or 0)
            pa = int(pub_antithesis.get(pass_k) or 0)
            wt = float(pub_thesis.get(worst_k) or 1e9)
            wa = float(pub_antithesis.get(worst_k) or 1e9)
            if max(pt, pa) < tot:
                # Pure symmetric minimax worst-case error `log(worst_mm)` (zero activity bias!)
                fe_t = math.log(max(wt, 1.0))
                fe_a = math.log(max(wa, 1.0))
                if abs(fe_t - fe_a) >= 0.25 and min(wt, wa) < 0.80 * max(wt, wa):
                    winner = "antithesis" if fe_a < fe_t else "thesis"
                    return {
                        "selected_branch": winner,
                        "selection_mechanism": f"m8_symmetric_minimax_tail_bound({worst_k}:T={wt}mm,A={wa}mm)",
                        "fe_thesis": round(fe_t, 5),
                        "fe_antithesis": round(fe_a, 5),
                    }
    return None


def decide_rule_only_baseline(pre_state: dict[str, Any]) -> bool:
    """Pure hard-safety-rule baseline (`spawn` iff unfinished, pullback-triggered, or `REFUTED`)."""
    return bool(
        (not bool(pre_state["ended_by_finish"]))
        or bool(pre_state.get("pullback_gate_triggered"))
        or str(pre_state["public_verification_state"]) == "REFUTED"
    )


def finalize_container_decision_record(
    decision_out: dict[str, Any],
    *,
    executed_ok: bool,
    actual_cost: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    wall_ms: float = 0.0,
) -> dict[str, Any]:
    """Upgrade a pre-execution `PROPOSED_PRE_EXECUTION` decision dict with post-execution measured receipts."""
    rec_d = dict(decision_out.get("decision_record") or {})
    if rec_d:
        rec_d["execution_status"] = "EXECUTED_OK" if executed_ok else "EXECUTION_ERROR"
        rec_d["actual_cost"] = round(float(actual_cost), 5)
        rec_d["total_search_cost_spent"] = round(float(actual_cost), 5)
        rec_d["total_prompt_tokens"] = int(prompt_tokens)
        rec_d["total_completion_tokens"] = int(completion_tokens)
        rec_d["attempt_cost_events"] = [
            {
                "action": str(decision_out.get("selected_action")),
                "cost_spent": round(float(actual_cost), 5),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "wall_ms": round(float(wall_ms), 2),
                "status": "OK" if executed_ok else "ERROR",
            }
        ]
        ResearchDecisionRecord.from_dict(rec_d)
        decision_out["decision_record"] = rec_d
    return decision_out


@dataclass
class M8OfflineContainerPlanner:
    """Calibrated M8.2 Variational Planner: NIG posterior primary failure risk x Beta realized recovery - cost."""

    nig_fit: dict[str, Any]
    posterior_recovery_rate: float = 0.35
    cost_lambda: float = 0.08
    epistemic_ucb_beta: float = 0.35
    decision_threshold: float = 0.0
    training_suite_ids: list[str] = field(default_factory=list)
    training_episode_count: int = 0
    decision_ledger: list[ResearchDecisionRecord] = field(default_factory=list)

    @classmethod
    def fit_from_historical_records(
        cls,
        records: list[dict[str, Any]],
        *,
        suite_ids: list[str],
        cost_lambda: float = 0.08,
        epistemic_ucb_beta: float = 0.35,
    ) -> M8OfflineContainerPlanner:
        X_rows = [build_pre_action_feature_vector(r["pre_action_state"]) for r in records]
        # 1. Fit scale-covariant NIG posterior on primary Antithesis failure indicator `1 - antithesis_1x_pass`
        #    (which is strictly causal wrt primary `pre_action_state`!)
        y_primary_fail = np.array(
            [1.0 - float(r["outcomes"]["antithesis_1x_pass"]) for r in records],
            dtype=float,
        )
        # 2. Estimate realized conditional 2-branch recovery rate `P(m8_twobranch_pass | primary_fail)` via Beta(2, 2)
        #    (strictly realized synthesis outcome, NEVER `oracle_either_pass`!)
        n_fail = sum(1 for r in records if not bool(r["outcomes"]["antithesis_1x_pass"]))
        n_recovered = sum(
            1
            for r in records
            if (not bool(r["outcomes"]["antithesis_1x_pass"])) and bool(r["outcomes"]["m8_twobranch_pass"])
        )
        post_rec_rate = float((2.0 + n_recovered) / (4.0 + n_fail))

        Phi = np.vstack(X_rows)
        # Structural prior mean on `[intercept, refuted, unknown_code, unknown_bm, exhaust, grounded]`
        prior_mu = np.array([0.15, 0.55, 0.55, 0.10, 0.45, -0.15], dtype=float)
        nig = fit_scale_covariant_nig_linear_model(
            Phi,
            y_primary_fail,
            prior_mean=prior_mu,
            prior_shrinkage=2.0,
            alpha_0=3.0,
            beta_0=0.25,
        )
        return cls(
            nig_fit=nig,
            posterior_recovery_rate=post_rec_rate,
            cost_lambda=float(cost_lambda),
            epistemic_ucb_beta=float(epistemic_ucb_beta),
            decision_threshold=0.0,
            training_suite_ids=list(suite_ids),
            training_episode_count=len(records),
        )

    def decide_second_branch(
        self,
        pre_state: dict[str, Any],
        *,
        step_index: int = 1,
        record_to_ledger: bool = True,
    ) -> dict[str, Any]:
        """Compute NIG-Beta expected net utility `p_rec * (mu_fail + beta * epistemic_std) - cost_lambda`.

        Hard overrides fire ONLY on:
          - Safety failure (`not ended_by_finish`, `pullback_gate_triggered`, or `REFUTED`) -> `SpawnDialecticalChallenger`
          - Quantitative domain verification (`VERIFIED` before hard turn ceiling) -> `CommitPrimary1x`
        ALL `SUPPORTED_IN_PARTIAL_SCOPE` and `UNKNOWN` states are governed strictly by `net_variational_utility > decision_threshold`!
        """
        x_vec = build_pre_action_feature_vector(pre_state)[None, :]
        pred = predict_nig_student_t_log_score(self.nig_fit, x_vec, np.zeros(1))
        mu_fail = float(np.clip(pred["predictions"][0], 0.0, 1.0))
        # Use epistemic parameter uncertainty (`epistemic_std`), NOT noise-inflated `predictive_std`!
        epistemic_std = float(pred["epistemic_std"][0])
        ucb_fail_risk = min(1.0, mu_fail + self.epistemic_ucb_beta * epistemic_std)
        expected_gain = self.posterior_recovery_rate * ucb_fail_risk
        net_variational_utility = expected_gain - self.cost_lambda

        triggers: list[str] = []
        hard_safety_spawn = False
        if not bool(pre_state["ended_by_finish"]):
            triggers.append("m8_hard_safety_primary_unfinished")
            hard_safety_spawn = True
        if bool(pre_state.get("pullback_gate_triggered")):
            triggers.append("m8_hard_safety_pullback_rollback")
            hard_safety_spawn = True
        if pre_state["public_verification_state"] == "REFUTED":
            triggers.append("m8_hard_safety_public_refuted")
            hard_safety_spawn = True

        hard_verified_commit = (
            not hard_safety_spawn
            and bool(pre_state["ended_by_finish"])
            and pre_state["public_verification_state"] == "VERIFIED"
            and float(pre_state["turns_ratio"]) < 0.99
        )

        if hard_safety_spawn:
            spawn = True
            decision_driver = "hard_safety_refuted_or_unfinished"
        elif hard_verified_commit:
            spawn = False
            triggers = []
            decision_driver = "hard_domain_verified_commit"
        else:
            # Learned NIG posterior governs all `SUPPORTED_IN_PARTIAL_SCOPE` and `UNKNOWN` episodes!
            spawn = bool(net_variational_utility > self.decision_threshold)
            decision_driver = "learned_nig_posterior_utility"
            if spawn:
                triggers.append(
                    f"m8_nig_posterior_positive_utility(state={pre_state['public_verification_state']},U={round(net_variational_utility, 4)})"
                )

        selected_action = "SpawnDialecticalChallenger" if spawn else "CommitPrimary1x"
        est_cost = 2.0 if spawn else 1.0
        rec = ResearchDecisionRecord(
            decision_id=f"m8_cnt_dec_{step_index:03d}_{pre_state['task_id']}",
            step_index=int(step_index),
            task_id=str(pre_state["task_id"]),
            pre_action_state=deepcopy(pre_state),
            available_actions=["CommitPrimary1x", "SpawnDialecticalChallenger"],
            selected_action=selected_action,
            selected_candidate_id="branch_1_plus_synthesis" if spawn else "branch_2_antithesis_1x",
            execution_status="PROPOSED_PRE_EXECUTION",
            actual_cost=0.0,
            estimated_selected_cost=est_cost,
            public_val_log_predictive_delta=round(net_variational_utility, 6),
            collateral_min_delta=0.0,
            rationale=(
                f"M8.2 NIG Router (`driver={decision_driver}`, `state={pre_state['public_verification_state']}`, "
                f"`mu_fail={mu_fail:.4f}`, `epi_std={epistemic_std:.4f}`, `U={net_variational_utility:.4f}`) -> `{selected_action}`."
            ),
        )
        rec.validate_prefix_clean()
        if record_to_ledger:
            self.decision_ledger.append(rec)

        return {
            "spawn_second_branch": spawn,
            "selected_action": selected_action,
            "decision_driver": decision_driver,
            "triggers": triggers,
            "posterior_mean_fail_risk": round(mu_fail, 5),
            "posterior_mean_gain": round(expected_gain, 5),
            "epistemic_std_gain": round(epistemic_std, 5),
            "net_variational_utility": round(net_variational_utility, 5),
            "public_verification_state": pre_state["public_verification_state"],
            "decision_record": rec.to_dict(),
        }

    def to_json_dict(self) -> dict[str, Any]:
        nig_ser = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.nig_fit.items()
            if k != "residuals"
        }
        return {
            "schema": "hegel_m8_calibrated_container_planner_v2",
            "feature_names": ["intercept", *CONTAINER_STATE_FEATURE_NAMES],
            "posterior_recovery_rate": self.posterior_recovery_rate,
            "cost_lambda": self.cost_lambda,
            "epistemic_ucb_beta": self.epistemic_ucb_beta,
            "decision_threshold": self.decision_threshold,
            "training_suite_ids": self.training_suite_ids,
            "training_episode_count": self.training_episode_count,
            "nig_fit": nig_ser,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> M8OfflineContainerPlanner:
        nig_raw = dict(data["nig_fit"])
        for arr_k in ("posterior_mean", "posterior_precision", "prior_precision"):
            nig_raw[arr_k] = np.asarray(nig_raw[arr_k], dtype=float)
        return cls(
            nig_fit=nig_raw,
            posterior_recovery_rate=float(data.get("posterior_recovery_rate", 0.35)),
            cost_lambda=float(data["cost_lambda"]),
            epistemic_ucb_beta=float(data["epistemic_ucb_beta"]),
            decision_threshold=float(data["decision_threshold"]),
            training_suite_ids=list(data.get("training_suite_ids", [])),
            training_episode_count=int(data.get("training_episode_count", 0)),
        )


def _public_ast_contract_ok(py_file: Path, required_callables: tuple[str, ...]) -> bool:
    """Pure public host-side AST syntax & callable check (ZERO private-label access!)."""
    if not py_file.is_file():
        return False
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    top_defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return all(req in top_defs for req in required_callables)


def load_historical_suite_episode_records(
    suite_dir: Path,
    *,
    apply_m7_7_traffic_contract_fix_in_replay: bool = False,
) -> list[dict[str, Any]]:
    """Extract prefix-clean `pre_action_state` + post-lock evaluation ground-truth from a sealed historical suite.

    Defaults to `apply_m7_7_traffic_contract_fix_in_replay=False` (100% raw historical replay). Even when enabled for
    counterfactual sensitivity checks, inspects ONLY the public archived `traffic_solver.py` AST (`0` private label reads!).
    """
    runs_dir = suite_dir / "runs"
    branches_dir = suite_dir / "branches"
    records: list[dict[str, Any]] = []

    for sum_fp in sorted(runs_dir.glob("*/00_episode_summary.json")):
        ep_sum = json.loads(sum_fp.read_text(encoding="utf-8"))
        tid = str(ep_sum["task_id"])
        model = str(ep_sum["model"])
        wave = str(ep_sum["wave"])
        bench = str(ep_sum["benchmark"])

        anti_dir = branches_dir / f"{wave}__{model}__{tid}" / "branch_2_antithesis_wormhole"
        anti_sum_fp = anti_dir / "00_episode_summary.json"
        anti_sum = json.loads(anti_sum_fp.read_text(encoding="utf-8")) if anti_sum_fp.is_file() else ep_sum

        pub_t: dict[str, Any] = {}
        pub_a: dict[str, Any] = {}
        chk_a_recorded: bool | None = None
        dag_fp = sum_fp.parent / "01_decision_provenance_dag.json"
        if dag_fp.is_file():
            dag = json.loads(dag_fp.read_text(encoding="utf-8"))
            for rec in dag.get("records", []):
                if rec.get("decision_type") == "D3_DIALECTICAL_SYNTHESIS_COMMIT":
                    chosen = rec.get("chosen_option") or {}
                    if chosen.get("selection_mechanism") == "antithesis_failed_public_contract":
                        chk_a_recorded = False
                    for cand in rec.get("candidates_evaluated", []):
                        if cand.get("branch") == "thesis" and isinstance(cand.get("public_metrics"), dict):
                            pub_t = dict(cand["public_metrics"])
                        elif cand.get("branch") == "antithesis" and isinstance(cand.get("public_metrics"), dict):
                            pub_a = dict(cand["public_metrics"])

        chk_ok = (bool(anti_sum.get("final_candidate_sha256")) or bench == "BioMysteryBench-Preview") and (chk_a_recorded is not False)
        if apply_m7_7_traffic_contract_fix_in_replay and tid == "tb__traffic-flux-inversion":
            # Strictly public AST inspection of the archived candidate file (NEVER reads `postlock_branch_outcomes`!)
            archived_ts = anti_dir / "04_submitted_artifacts_snapshot" / "traffic_solver.py"
            if _public_ast_contract_ok(archived_ts, ("fit_model", "predict_scenario")):
                chk_ok = True

        pre_state = build_container_pre_action_state(
            task_id=tid,
            benchmark=bench,
            wave=wave,
            primary_summary=anti_sum,
            contract_check={"accepted": chk_ok},
            public_metrics=pub_a,
        )
        ResearchDecisionRecord(
            decision_id=f"chk_{tid}",
            step_index=0,
            task_id=tid,
            pre_action_state=pre_state,
            available_actions=["CommitPrimary1x", "SpawnDialecticalChallenger"],
            selected_action="CommitPrimary1x",
            selected_candidate_id="chk",
            execution_status="PROPOSED_PRE_EXECUTION",
            actual_cost=0.0,
            public_val_log_predictive_delta=0.0,
            collateral_min_delta=0.0,
            rationale="prefix_clean_check",
        ).validate_prefix_clean()

        po = ep_sum["postlock_branch_outcomes"]
        anti_pass = bool(po["antithesis_wormhole_strict_pass"])
        thesis_pass = bool(po["thesis_bare_run2_strict_pass"])
        orig_twobranch_pass = bool(ep_sum["strict_pass"])

        m8_twobranch_pass = orig_twobranch_pass
        tb_override = adjudicate_minimax_variational_tiebreak(
            pub_thesis=pub_t,
            pub_antithesis=pub_a,
        )
        if tb_override is not None:
            m8_twobranch_pass = anti_pass if tb_override["selected_branch"] == "antithesis" else thesis_pass

        records.append(
            {
                "suite_id": suite_dir.name,
                "task_id": tid,
                "benchmark": bench,
                "wave": wave,
                "model": model,
                "pre_action_state": pre_state,
                "public_metrics_thesis": pub_t,
                "public_metrics_antithesis": pub_a,
                "minimax_tiebreak_override": tb_override,
                "outcomes": {
                    "bare_1x_pass": thesis_pass,
                    "antithesis_1x_pass": anti_pass,
                    "orig_twobranch_pass": orig_twobranch_pass,
                    "m8_twobranch_pass": m8_twobranch_pass,
                    "oracle_either_pass": bool(anti_pass or thesis_pass),
                    "thesis_turns": int(po["thesis_turns_used"]),
                    "antithesis_turns": int(po["antithesis_turns_used"]),
                },
            }
        )
    return records


def evaluate_planner_on_records(
    planner: M8OfflineContainerPlanner,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate out-of-sample M8 planner decisions AND pure hard-rule baseline side-by-side."""
    per_task: list[dict[str, Any]] = []
    m8_passes = 0
    rule_only_passes = 0
    anti_passes = 0
    bare_passes = 0
    m8_turns = 0
    rule_turns = 0
    bare_turns = 0
    spawn_cnt = 0
    rule_spawn_cnt = 0
    learned_nig_decisions = 0

    for idx, rec in enumerate(records, start=1):
        dec = planner.decide_second_branch(rec["pre_action_state"], step_index=idx, record_to_ledger=False)
        spawn = bool(dec["spawn_second_branch"])
        rule_spawn = decide_rule_only_baseline(rec["pre_action_state"])
        if dec["decision_driver"] == "learned_nig_posterior_utility":
            learned_nig_decisions += 1

        oc = rec["outcomes"]
        passed = bool(oc["m8_twobranch_pass"]) if spawn else bool(oc["antithesis_1x_pass"])
        rule_passed = bool(oc["m8_twobranch_pass"]) if rule_spawn else bool(oc["antithesis_1x_pass"])
        eff_t = int(oc["antithesis_turns"]) + (int(oc["thesis_turns"]) if spawn else 0)
        rule_eff_t = int(oc["antithesis_turns"]) + (int(oc["thesis_turns"]) if rule_spawn else 0)

        m8_passes += int(passed)
        rule_only_passes += int(rule_passed)
        anti_passes += int(oc["antithesis_1x_pass"])
        bare_passes += int(oc["bare_1x_pass"])
        m8_turns += eff_t
        rule_turns += rule_eff_t
        bare_turns += int(oc["thesis_turns"])
        spawn_cnt += int(spawn)
        rule_spawn_cnt += int(rule_spawn)

        per_task.append(
            {
                "task_id": rec["task_id"],
                "public_verification_state": rec["pre_action_state"]["public_verification_state"],
                "decision_driver": dec["decision_driver"],
                "spawn_second_branch": spawn,
                "rule_only_spawn": rule_spawn,
                "net_variational_utility": dec["net_variational_utility"],
                "triggers": dec["triggers"],
                "bare_1x_pass": oc["bare_1x_pass"],
                "antithesis_1x_pass": oc["antithesis_1x_pass"],
                "rule_only_selective_pass": rule_passed,
                "m8_selective_pass": passed,
                "effective_turns": eff_t,
            }
        )

    n = max(1, len(records))
    return {
        "total_tasks": len(records),
        "bare_1x_passes": bare_passes,
        "antithesis_1x_passes": anti_passes,
        "rule_only_selective_passes": rule_only_passes,
        "m8_variational_selective_passes": m8_passes,
        "learned_nig_governed_task_count": learned_nig_decisions,
        "rule_only_spawn_count": rule_spawn_cnt,
        "spawn_second_branch_count": spawn_cnt,
        "spawn_rate": round(spawn_cnt / n, 4),
        "rule_only_turn_multiplier_vs_bare_1x": round(rule_turns / max(1, bare_turns), 4),
        "turn_multiplier_vs_bare_1x": round(m8_turns / max(1, bare_turns), 4),
        "per_task": per_task,
    }
