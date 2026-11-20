"""Hegel V17.7 (`M4.5` Audit Closure & `M5.0` Counterexample-Driven Continual Lifecycle):

1. **Part 1 (`run_m4_4_audit_closure_verification_suite`)**: Closes all 5 `Hegel_M44_Audit` P2 packages:
   - Shared side-effect-free `node / OR-branch / edge / certificate` instance identity guard across
     `add_coarse_graining_edge`, `revalidate_vertical_edges_with_certificate`, `fine_scale_children_of`,
     and `verify_forest_snapshot_digests` (including `node_history` semantic digests);
   - Unified `selected_action` consumption across `R3` AND refit-recall (`M4.1` / `M4.2` / `M4.3`) so an
     unavailable unselected OR branch never blocks an intact applicable OR branch;
   - Unified public `validate_canonical_role_map_contract` enforcing full & partial canonical-axis identity
     locks across `R3` explicit `role_map`, channel precheck, candidate fitting, fresh confirm, and query guard;
   - Confirmation-gated empirical scope revision (`confirmed_ok=False` never increments version on protocol
     updates; empirical counterexample residuals synthesize an executable coordinate precondition `b <= theta_emp`
     that changes `authorize_abstraction_use` and `R3` behavior on in-scope vs out-of-scope inputs);
   - Three-tier cross-domain representation (`SUPPORTED_IN_SCOPE` source instance vs `STRUCTURAL_HYPOTHESIS_ONLY`
     bounded-trial bridge vs genuinely confirmed second-domain physical instance with distinct `confirmed_instance_digest`,
     empirical `hop_error_budget`, and two-domain `source_episode_refs`).

2. **Part 3 (`run_m5_0_counterexample_sublation_and_persistent_stream` + `resume_m5_0_continual_checkpoint_cold_continuation`)**:
   - Complete 24-step outcome-driven continual lifecycle (`S01..S24`, zero step-index hardcoding) comparing 5 arms:
     `Hegel_M5_LiveSublation_PPR_PoolOn`, `Hegel_M5_LiveSublation_PPR_PoolOff`, `Compiled_M5_LiveSublation_CueTable`,
     `Dynamic_Flat_StructureMemory_LiveWriteBack`, and `Frozen_PostStage2_OpenRediscovery`;
   - Explicitly meters initial index build cost (`initial_index_build`), refresh cost, and exact node/edge write-back cost;
   - Demonstrates end-to-end paid-task `PoolOn < PoolOff < Compiled < Dynamic_Flat < Frozen`, empirical counterexample
     scope sublation on `W4`, genuine two-domain wormhole promotion on `W6` & `W7`, negative cross-domain mapping
     rejection (`Domain C`), full persisted `online_writebacks/` + `post_m5_online_stream_forest_snapshot.json` +
     `continual_session_checkpoint.json`, and post-deletion cold-reload continuation (`S25..S27`)!
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from random import Random
import shutil
from statistics import mean
from typing import Any
from unittest.mock import patch

from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m4_1_unified_discovery_and_renormalization import _load_node_verified_operator_spec
from .m4_2_unified_closure_and_pi_lifecycle import (
    ForestScopeDispatcher,
    _eval_pi_s_fine_scalar,
    _make_verified_instance_action,
    _run_metered_atomic_calibration,
    run_interleaved_zero_fit_dispatcher_stream,
)
from .m4_3_channel_bound_lifecycle_and_audit_closure import (
    CompiledRelationalCueTable,
    M43TargetSpec,
    build_m4_3_continual_forest,
    build_m4_3_target_suite,
    evaluate_m4_3_target_under_arm,
)
from .m4_4_instance_role_closure_and_online_writeback import (
    load_location_independent_forest,
    run_m4_4_online_writeback_and_multicue_stream,
    serialize_location_independent_forest,
    verify_forest_snapshot_digests,
)
from .m4_compositional_discovery import (
    _VERIFIED_AUTHORED_CACHE,
    canonicalize_fitted_params_snapshot,
    compute_confirmed_instance_digest,
    load_verified_authored_method,
    predict_with_frozen_instance,
)
from .m4_multiscale_renormalization import MultiScaleWorkspaceTaskLab
from .open_workspace_wormhole import canonicalize_python_model_ast
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    TaskView,
    authorize_abstraction_use,
)


def _all_forest_state_sha256(forest: PhilosophicalForest) -> str:
    return hashlib.sha256(json.dumps(forest.__dict__, sort_keys=True).encode("utf-8")).hexdigest()


def _persist_confirmed_recall_bundle_as_artifact(
    recall_bundle: dict[str, Any],
    *,
    artifact_abs_dir: Path,
) -> dict[str, Any]:
    """Materialize a target-refitted & freshly confirmed recall model into a full 3-file verified instance bundle."""
    src = str(recall_bundle["model_source"])
    role_names = [str(r) for r in recall_bundle["role_names"]]
    p_rows = deepcopy(recall_bundle["training_prefix_rows"])
    c_rows = deepcopy(recall_bundle["confirm_rows"])
    ast_info = canonicalize_python_model_ast(src)

    ns_fit: dict[str, Any] = {}
    exec(compile(src, "<recall_domain_instance>", "exec"), ns_fit)
    ns_fit["fit"](deepcopy(p_rows))
    fitted_params, fitted_params_sha = canonicalize_fitted_params_snapshot(
        {str(k): float(v) for k, v in ns_fit["_PARAMS"].items()}
    )
    inst_digest = compute_confirmed_instance_digest(
        model_sha256=ast_info["model_sha256"],
        ast_skeleton_digest=ast_info["ast_skeleton_digest"],
        fitted_params_sha256=fitted_params_sha,
        role_names=role_names,
    )
    pref_sha = hashlib.sha256(json.dumps(p_rows, sort_keys=True).encode("utf-8")).hexdigest()
    conf_sha = hashlib.sha256(json.dumps(c_rows, sort_keys=True).encode("utf-8")).hexdigest()
    obs_rmse = round(float(recall_bundle["observed_confirm_rmse"]), 8)
    tol = float(recall_bundle["tolerance"])

    host_bundle = {
        "task_id": str(recall_bundle["target_id"]),
        "model_sha256": ast_info["model_sha256"],
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "fitted_params_sha256": fitted_params_sha,
        "confirmed_instance_digest": inst_digest,
        "training_prefix_rows": p_rows,
        "training_prefix_sha256": pref_sha,
        "confirm_rows": c_rows,
        "confirm_rows_sha256": conf_sha,
        "observed_confirm_rmse": obs_rmse,
        "tolerance": tol,
        "confirmed_ok": True,
    }
    art_dict = {
        "task_id": str(recall_bundle["target_id"]),
        "surface_domain": str(recall_bundle["surface_domain"]),
        "operator_spec": deepcopy(recall_bundle["operator_spec"]),
        "role_names": role_names,
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "skeleton_status": ast_info["skeleton_status"],
        "model_sha256": ast_info["model_sha256"],
        "authored_model_source": src,
        "fitted_params": fitted_params,
        "fitted_params_sha256": fitted_params_sha,
        "confirmed_instance_digest": inst_digest,
        "confirmation_digest": conf_sha[:16],
        "confirmed_ok": True,
        "host_confirmation_bundle": host_bundle,
    }
    artifact_abs_dir.mkdir(parents=True, exist_ok=True)
    (artifact_abs_dir / "authored_method.json").write_text(
        json.dumps(art_dict, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifact_abs_dir / "confirmed_fitted_params.json").write_text(
        json.dumps(fitted_params, indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifact_abs_dir / "confirmed_manifest.json").write_text(
        json.dumps(
            {k: v for k, v in host_bundle.items() if k not in {"training_prefix_rows", "confirm_rows"}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return art_dict


# ============================================================================
# Outcome-Driven Forest Consolidator (`M5.0` Continual Lifecycle Engine)
# ============================================================================

DEFAULT_CROSS_DOMAIN_HYPOTHESES: dict[str, list[dict[str, Any]]] = {
    "cue_tubule_saturation_a": [
        {
            "candidate_cue": "cue_reuptake_saturation_b",
            "candidate_node_id": "online_W6_reuptake_domain_b",
            "candidate_tree_id": "tree_online_reuptake_cap",
            "wormhole_id": "wh_online_W6_tubule_reuptake",
        },
        {
            "candidate_cue": "cue_piecewise_incompatible_c",
            "candidate_node_id": "online_W6_piecewise_domain_c",
            "candidate_tree_id": "tree_online_incompatible_c",
            "wormhole_id": "wh_online_W6_tubule_incompatible_c",
        },
    ],
    "cue_photobioreactor_cap2d_a": [
        {
            "candidate_cue": "cue_allosteric_cap2d_b",
            "candidate_node_id": "online_W7_allosteric_domain_b",
            "candidate_tree_id": "tree_online_allosteric",
            "wormhole_id": "wh_online_W7_photobio_allosteric",
        },
    ],
}

DOMAIN_CUE_TO_TREE: dict[str, tuple[str, str]] = {
    "cue_tubule_saturation_a": ("online_W6_tubule_domain_a", "tree_online_tubule_cap"),
    "cue_reuptake_saturation_b": ("online_W6_reuptake_domain_b", "tree_online_reuptake_cap"),
    "cue_piecewise_incompatible_c": ("online_W6_piecewise_domain_c", "tree_online_incompatible_c"),
    "cue_photobioreactor_cap2d_a": ("online_W7_photobio_domain_a", "tree_online_photobio"),
    "cue_allosteric_cap2d_b": ("online_W7_allosteric_domain_b", "tree_online_allosteric"),
}


class OutcomeDrivenForestConsolidator:
    """Consolidates runtime episode outcomes into ``PhilosophicalForest`` without step-index hardcoding:

    1. **Empirical Counterexample Scope Sublation**: When a ``SUPPORTED_IN_SCOPE`` node fails confirmation
       on a paid target episode and pointwise residuals separate a clean low-coordinate region from a
       high-residual counterexample region, synthesizes an empirical coordinate precondition (`role <= theta_emp`)
       and commits ``empirical_revise_node`` + vertical edge rebinding.
    2. **Novel Confirmed Structure Write-Back**: When open ``CodeAgent`` confirms a new AST law in Domain A,
       persists Domain A's confirmed instance (`SUPPORTED_IN_SCOPE`, `local_solver_guidance`) and registers
       hypothesized cross-domain slots strictly as ``STRUCTURAL_HYPOTHESIS_ONLY`` (`bounded_trial`, no instance digest!).
    3. **Genuine Two-Domain Wormhole Promotion / Refutation**: When a ``STRUCTURAL_HYPOTHESIS_ONLY`` slot passes
       paid refit + disjoint fresh confirmation in Domain B, persists Domain B's own confirmed instance bundle,
       promotes Domain B to ``SUPPORTED_IN_SCOPE`` (`local_solver_guidance`), and upgrades the wormhole with
       two-domain `source_episode_refs` and empirical `hop_error_budget = max(err_A, err_B)`. When confirmation
       fails (e.g. incompatible Domain C), records negative witness and marks ``REFUTED_IN_SCOPE``!
    """

    def __init__(
        self,
        forest: PhilosophicalForest,
        *,
        bundle_root: Path,
        enable_flat_only: bool = False,
        cross_domain_hypotheses: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.forest = forest
        self.bundle_root = bundle_root.resolve()
        self.enable_flat_only = bool(enable_flat_only)
        self.cross_domain_hypotheses = deepcopy(
            cross_domain_hypotheses if cross_domain_hypotheses is not None else DEFAULT_CROSS_DOMAIN_HYPOTHESES
        )
        # Track historical clean coordinate values per node & canonical role for empirical boundary estimation
        self.supported_coord_samples: dict[tuple[str, str], list[float]] = {}
        self.failure_witness_ledger: list[dict[str, Any]] = []
        self.writeback_event_log: list[dict[str, Any]] = []
        self.paid_observation_history: list[dict[str, Any]] = []

    def _seed_historical_clean_samples(self, node_id: str) -> None:
        if node_id not in self.forest.nodes:
            return
        nd = self.forest.nodes[node_id]
        act = (nd.get("claim_contract") or {}).get("conclusion_or_action") or {}
        art_loc = act.get("authored_artifact_path")
        if not art_loc:
            return
        art_p = self.forest.resolve_artifact_path(art_loc)
        if not art_p.is_file():
            return
        art_d = json.loads(art_p.read_text(encoding="utf-8"))
        hb = art_d.get("host_confirmation_bundle") or {}
        rows = [*(hb.get("training_prefix_rows") or ()), *(hb.get("confirm_rows") or ())]
        for r in rows:
            for k, v in (r.get("features") or {}).items():
                self.supported_coord_samples.setdefault((node_id, str(k)), []).append(float(v))

    def consolidate_episode(
        self,
        *,
        step_num: int,
        target: M43TargetSpec,
        episode_receipt: dict[str, Any],
        step_work_dir: Path,
    ) -> dict[str, Any]:
        """Inspect runtime files & receipt from ``step_work_dir`` and apply outcome-driven forest updates."""
        nodes_written = 0
        edges_written = 0
        revisions_committed = 0
        events: list[dict[str, Any]] = []

        # Record paid episode summary in checkpoint ledger
        self.paid_observation_history.append({
            "step": step_num,
            "target_id": target.target_id,
            "domain_cue": target.domain_cue,
            "observation_cost": int(episode_receipt["cost"]["observation_cost"]),
            "measurement_cost": float(episode_receipt["cost"]["measurement_cost"]),
            "status": episode_receipt["status"],
            "confirmed_ok": bool(episode_receipt["confirmed_ok"]),
            "selected_node_id": episode_receipt["selected_node_id"],
        })

        # 1. Record clean coordinate samples when an existing node confirms via recall
        conf_recall_p = step_work_dir / "confirmed_recall_bundle.json"
        conf_recall = json.loads(conf_recall_p.read_text(encoding="utf-8")) if conf_recall_p.is_file() else None
        if episode_receipt["confirmed_ok"] and conf_recall is not None:
            sel_nid = str(conf_recall["selected_node_id"])
            rmap = conf_recall.get("canonical_role_map") or {}
            paid_rows = [*conf_recall["training_prefix_rows"], *conf_recall["confirm_rows"]]
            for r in paid_rows:
                for tf_k, cr_k in rmap.items():
                    if tf_k in r["features"]:
                        self.supported_coord_samples.setdefault((sel_nid, str(cr_k)), []).append(float(r["features"][tf_k]))

            # 1b. Promote `STRUCTURAL_HYPOTHESIS_ONLY` node to `SUPPORTED_IN_SCOPE` upon genuine Domain B confirmation!
            if (
                sel_nid in self.forest.nodes
                and self.forest.nodes[sel_nid]["claim_contract"].get("evidence_status") == "STRUCTURAL_HYPOTHESIS_ONLY"
            ):
                wb_dir = self.bundle_root / "persisted_authored_workspaces" / "online_writebacks" / sel_nid
                dom_b_art = _persist_confirmed_recall_bundle_as_artifact(conf_recall, artifact_abs_dir=wb_dir)
                rel_loc = str((wb_dir / "authored_method.json").resolve().relative_to(self.bundle_root).as_posix())
                old_nd = self.forest.nodes[sel_nid]
                old_act = old_nd["claim_contract"]["conclusion_or_action"]
                src_node_id = old_act.get("structural_alias_of_source_node")
                op_tag = str(old_act["operator_tag"])
                promoted_act = _make_verified_instance_action(
                    operator_tag=op_tag,
                    operator_spec=dom_b_art["operator_spec"],
                    artifact_dict=dom_b_art,
                    artifact_relpath=rel_loc,
                )
                upd_cc = deepcopy(old_nd["claim_contract"])
                upd_cc["conclusion_or_action"] = promoted_act
                upd_cc["evidence_status"] = "SUPPORTED_IN_SCOPE"
                upd_cc["permission_tier"] = "local_solver_guidance"
                upd_cc["protocol_spec"] = {"isolated_confirm": True, "second_domain_confirmed": True}
                self.forest.empirical_revise_node(
                    sel_nid,
                    witness_episode_id=target.target_id,
                    cause="protocol_or_scope_mismatch",
                    updated_claim_contract=upd_cc,
                )
                revisions_committed += 1

                # Upgrade incident wormhole with BOTH domains' `source_episode_refs` and empirical `hop_error_budget`!
                if not self.enable_flat_only and src_node_id and src_node_id in self.forest.nodes:
                    src_nd = self.forest.nodes[src_node_id]
                    src_loc = src_nd["claim_contract"]["conclusion_or_action"].get("authored_artifact_path")
                    src_err = 0.005
                    src_ep = src_node_id
                    if src_loc:
                        src_p = self.forest.resolve_artifact_path(src_loc)
                        if src_p.is_file():
                            sd = json.loads(src_p.read_text(encoding="utf-8"))
                            src_ep = str(sd.get("task_id") or src_node_id)
                            src_err = float((sd.get("host_confirmation_bundle") or {}).get("observed_confirm_rmse", 0.005))
                    emp_hop_err = round(max(src_err, float(conf_recall["observed_confirm_rmse"]), 0.005), 6)
                    for wid, wh in list(self.forest.horizontal_wormhole_edges.items()):
                        if {wh["node_a_id"], wh["node_b_id"]} == {src_node_id, sel_nid}:
                            self.forest.add_horizontal_wormhole(
                                wormhole_id=wid,
                                node_a_id=wh["node_a_id"],
                                node_b_id=wh["node_b_id"],
                                probe_signature_refs=list(wh.get("probe_signature_refs", ())),
                                role_and_observation_map={
                                    "role_map": dict(conf_recall.get("canonical_role_map") or {}),
                                    "status": "two_domain_empirically_confirmed",
                                },
                                hop_error_budget=emp_hop_err,
                                permission_tier="local_solver_guidance",
                                source_episode_refs=[src_ep, target.target_id],
                            )
                            edges_written += 1
                ev_obj = {
                    "step": step_num,
                    "event": "promote_second_domain_confirmed_instance",
                    "node_id": sel_nid,
                    "new_version": self.forest.nodes[sel_nid]["version"],
                    "confirmed_instance_digest": promoted_act["confirmed_instance_digest"],
                }
                events.append(ev_obj)
                self.writeback_event_log.append(ev_obj)

        # 2. Check refuted recall traces for (a) unconfirmed structural bridge refutation or (b) empirical scope sublation!
        ref_traces_p = step_work_dir / "refuted_recall_traces.json"
        ref_traces = json.loads(ref_traces_p.read_text(encoding="utf-8")) if ref_traces_p.is_file() else []
        for rt in ref_traces:
            r_nid = str(rt["node_id"])
            if r_nid not in self.forest.nodes:
                continue
            r_nd = self.forest.nodes[r_nid]
            r_cc = r_nd["claim_contract"]
            self.failure_witness_ledger.append({
                "step": step_num,
                "target_id": target.target_id,
                "refuted_node_id": r_nid,
                "stage": rt["stage"],
                "tolerance": rt["tolerance"],
            })

            # 2a. If a `STRUCTURAL_HYPOTHESIS_ONLY` cross-domain slot failed confirmation on its target domain, refute it!
            if r_cc.get("evidence_status") == "STRUCTURAL_HYPOTHESIS_ONLY":
                primary_cue = str(target.domain_cue).split("|")[0]
                if f"sig:{primary_cue}" in set(r_nd.get("probe_signatures", ())):
                    upd_cc = deepcopy(r_cc)
                    upd_cc["evidence_status"] = "REFUTED_IN_SCOPE"
                    upd_cc["permission_tier"] = "retrieve_candidate"
                    self.forest.empirical_revise_node(
                        r_nid,
                        witness_episode_id=target.target_id,
                        cause="claim_refuted_or_unprofitable",
                        updated_claim_contract=upd_cc,
                    )
                    revisions_committed += 1
                    ev_ref = {
                        "step": step_num,
                        "event": "refute_incompatible_cross_domain_hypothesis",
                        "node_id": r_nid,
                        "new_version": self.forest.nodes[r_nid]["version"],
                    }
                    events.append(ev_ref)
                    self.writeback_event_log.append(ev_ref)
                continue

            # 2b. Empirical Counterexample Scope Sublation on a `SUPPORTED_IN_SCOPE` node!
            self._seed_historical_clean_samples(r_nid)
            ev_rows = rt.get("evaluated_rows") or []
            preds = list(rt.get("predictions") or ())
            if not preds and rt.get("model_source") and rt.get("training_prefix_rows") and ev_rows:
                ns_rt: dict[str, Any] = {}
                exec(compile(str(rt["model_source"]), "<refuted_trace_eval>", "exec"), ns_rt)
                ns_rt["fit"](deepcopy(rt["training_prefix_rows"]))
                preds = [float(ns_rt["predict"](row)) for row in ev_rows]
            role_order = rt.get("role_order") or []
            canon_roles = rt.get("canon_roles") or []
            tol_r = float(rt["tolerance"])
            if len(ev_rows) == len(preds) and len(role_order) == len(canon_roles) and len(ev_rows) >= 4:
                existing_fields = {
                    str(pr.get("field"))
                    for pr in r_cc.get("preconditions", ())
                    if isinstance(pr, dict) and pr.get("op") in {"<=", "<"}
                }
                for cr_name, tf_name in zip(canon_roles, role_order):
                    if cr_name in existing_fields:
                        continue
                    hist_clean = list(self.supported_coord_samples.get((r_nid, cr_name), ()))
                    ep_clean = [
                        float(row["features"][tf_name])
                        for row, pr_y in zip(ev_rows, preds)
                        if abs(float(row["target"]) - float(pr_y)) <= tol_r
                    ]
                    ep_broken = [
                        float(row["features"][tf_name])
                        for row, pr_y in zip(ev_rows, preds)
                        if abs(float(row["target"]) - float(pr_y)) > 2.2 * tol_r
                    ]
                    all_clean = [*hist_clean, *ep_clean]
                    if all_clean and len(ep_broken) >= 1 and max(all_clean) < min(ep_broken) - 0.05:
                        theta_emp = round(0.5 * (max(all_clean) + min(ep_broken)), 4)
                        upd_cc = deepcopy(r_cc)
                        upd_cc["preconditions"] = [
                            *list(upd_cc.get("preconditions", ())),
                            {"field": str(cr_name), "op": "<=", "value": theta_emp},
                        ]
                        self.forest.empirical_revise_node(
                            r_nid,
                            witness_episode_id=target.target_id,
                            cause="protocol_or_scope_mismatch",
                            updated_claim_contract=upd_cc,
                        )
                        self.forest.revalidate_vertical_edges_after_revision(
                            r_nid, witness_episode_id=target.target_id
                        )
                        revisions_committed += 1
                        ev_sub = {
                            "step": step_num,
                            "event": "empirical_counterexample_scope_sublation",
                            "node_id": r_nid,
                            "canonical_axis": str(cr_name),
                            "empirical_upper_bound": theta_emp,
                            "max_supported_coord": round(max(all_clean), 4),
                            "min_refuted_coord": round(min(ep_broken), 4),
                            "new_version": self.forest.nodes[r_nid]["version"],
                        }
                        events.append(ev_sub)
                        self.writeback_event_log.append(ev_sub)
                        break

        # 3. Novel Confirmed `CodeAgent` Discovery Write-Back!
        if episode_receipt["confirmed_ok"] and str(episode_receipt["selected_node_id"] or "").startswith("open_code_agent:"):
            ep_dirs = sorted(step_work_dir.glob("open_code_agent_ep_*"))
            chosen_art = None
            for ed in ep_dirs:
                am_p = ed / "authored_method.json"
                if am_p.is_file():
                    ca = json.loads(am_p.read_text(encoding="utf-8"))
                    if ca.get("confirmed_ok") is True:
                        chosen_art = ca
                        break
            primary_cue = str(target.domain_cue).split("|")[0]
            if chosen_art is not None and primary_cue in DOMAIN_CUE_TO_TREE:
                p_nid, p_tid = DOMAIN_CUE_TO_TREE[primary_cue]
                if p_nid not in self.forest.nodes:
                    arity = len(target.feature_names)
                    op_tag = f"op_m5_ast_{str(chosen_art['ast_skeleton_digest'])[:10]}"
                    wb_sub = self.bundle_root / "persisted_authored_workspaces" / "online_writebacks" / p_nid
                    wb_sub.mkdir(parents=True, exist_ok=True)
                    (wb_sub / "authored_method.json").write_text(
                        json.dumps(chosen_art, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    (wb_sub / "confirmed_fitted_params.json").write_text(
                        json.dumps(chosen_art["fitted_params"], indent=2, sort_keys=True), encoding="utf-8"
                    )
                    hb_src = chosen_art["host_confirmation_bundle"]
                    (wb_sub / "confirmed_manifest.json").write_text(
                        json.dumps(
                            {k: v for k, v in hb_src.items() if k not in {"training_prefix_rows", "confirm_rows"}},
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    rel_loc = str((wb_sub / "authored_method.json").resolve().relative_to(self.bundle_root).as_posix())
                    act_a = _make_verified_instance_action(
                        operator_tag=op_tag,
                        operator_spec=chosen_art["operator_spec"],
                        artifact_dict=chosen_art,
                        artifact_relpath=rel_loc,
                    )
                    prog_steps = [
                        {"step": "calibrate_observation_lens", "active_arity": arity},
                        {"step": "compile_and_fit_composed_ast", "operator_tag": op_tag},
                        {"step": "subprocess_isolated_confirm", "fresh_points": 1},
                    ]
                    tv = TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", target.tolerance)
                    self.forest.register_node(
                        node_id=p_nid,
                        tree_id=p_tid,
                        layer_role="mid",
                        task_view=tv,
                        observation_map=ObservationMap("online_discovered_flux", parameter_semantics={"active_arity": arity}),
                        claim_contract=ClaimContract(
                            f"claim_{p_nid}",
                            "m4_open_compositional_ast",
                            preconditions=[{"field": "active_arity", "op": "==", "value": arity}],
                            conclusion_or_action=act_a,
                            protocol_spec={"isolated_confirm": True, "source_episode_id": target.target_id},
                            tolerance=target.tolerance,
                            evidence_status="SUPPORTED_IN_SCOPE",
                            permission_tier="local_solver_guidance",
                        ),
                        probe_signatures=[f"sig:{primary_cue}"],
                        program_steps=prog_steps,
                        source_cluster_ids=[f"online_{p_tid}"],
                    )
                    nodes_written += 1

                    # Seed clean coordinate samples from Domain A discovery!
                    for r in [*(hb_src.get("training_prefix_rows") or ()), *(hb_src.get("confirm_rows") or ())]:
                        for k, v in (r.get("features") or {}).items():
                            self.supported_coord_samples.setdefault((p_nid, str(k)), []).append(float(v))

                    # Register cross-domain structural hypotheses ONLY when graph wormholes are enabled (`not enable_flat_only`)!
                    if not self.enable_flat_only:
                        src_err = float(hb_src.get("observed_confirm_rmse", 0.005))
                        for hyp in self.cross_domain_hypotheses.get(primary_cue, ()):
                            c_nid = str(hyp["candidate_node_id"])
                            c_tid = str(hyp["candidate_tree_id"])
                            c_cue = str(hyp["candidate_cue"])
                            wid = str(hyp["wormhole_id"])
                            c_op_tag = op_tag
                            c_prog = [
                                {"step": "calibrate_observation_lens", "active_arity": arity},
                                {"step": "compile_and_fit_composed_ast", "operator_tag": c_op_tag},
                                {"step": "subprocess_isolated_confirm", "fresh_points": 1},
                            ]
                            struct_act = {
                                "operator_tag": c_op_tag,
                                "operator_spec": deepcopy(chosen_art["operator_spec"]),
                                "role_names": [str(r) for r in chosen_art.get("role_names", ())],
                                "ast_skeleton_digest": str(chosen_art["ast_skeleton_digest"]),
                                "model_source_sha256": str(chosen_art["model_sha256"]),
                                "structural_alias_of_source_node": p_nid,
                            }
                            self.forest.register_node(
                                node_id=c_nid,
                                tree_id=c_tid,
                                layer_role="mid",
                                task_view=tv,
                                observation_map=ObservationMap("online_hypothesis_flux", parameter_semantics={"active_arity": arity}),
                                claim_contract=ClaimContract(
                                    f"claim_{c_nid}",
                                    "m4_open_compositional_ast",
                                    preconditions=[{"field": "active_arity", "op": "==", "value": arity}],
                                    conclusion_or_action=struct_act,
                                    protocol_spec={"isolated_confirm": True, "requires_second_domain_confirmation": True},
                                    tolerance=target.tolerance,
                                    evidence_status="STRUCTURAL_HYPOTHESIS_ONLY",
                                    permission_tier="bounded_trial",
                                ),
                                probe_signatures=[f"sig:{c_cue}"],
                                program_steps=c_prog,
                                source_cluster_ids=[f"online_{c_tid}"],
                            )
                            nodes_written += 1
                            self.forest.add_horizontal_wormhole(
                                wormhole_id=wid,
                                node_a_id=p_nid,
                                node_b_id=c_nid,
                                probe_signature_refs=[f"sig:{primary_cue}", f"sig:{c_cue}"],
                                role_and_observation_map={
                                    "role_map": "identity_canonical",
                                    "status": "structural_candidate_bridge",
                                },
                                hop_error_budget=round(max(src_err, 0.005), 6),
                                permission_tier="bounded_trial",
                                source_episode_refs=[target.target_id],
                            )
                            edges_written += 1
                    else:
                        # In Dynamic Flat Structure Memory (non-graph baseline), store newly discovered structure
                        # directly with its own surface tag + generic arity cue so same-domain & generic recall works!
                        nd_flat = self.forest.nodes[p_nid]
                        self.forest.register_node(
                            node_id=p_nid,
                            tree_id=p_tid,
                            layer_role="mid",
                            task_view=tv,
                            observation_map=nd_flat["observation_map"],
                            claim_contract=nd_flat["claim_contract"],
                            probe_signatures=[f"sig:{primary_cue}", f"sig:pos_arity_{arity}", f"sig:neg_arity_{arity}"],
                            program_steps=prog_steps,
                            source_cluster_ids=[f"online_{p_tid}"],
                        )

                    ev_disc = {
                        "step": step_num,
                        "event": "register_confirmed_source_discovery",
                        "primary_node_id": p_nid,
                        "confirmed_instance_digest": act_a["confirmed_instance_digest"],
                        "ast_skeleton_digest": act_a["ast_skeleton_digest"],
                    }
                    events.append(ev_disc)
                    self.writeback_event_log.append(ev_disc)

        writeback_compute = round(1.5 * nodes_written + 0.8 * edges_written + 2.0 * revisions_committed, 4)
        return {
            "writeback_compute_cost": writeback_compute,
            "nodes_written": nodes_written,
            "edges_written": edges_written,
            "revisions_committed": revisions_committed,
            "events": events,
        }


# ============================================================================
# Part 1. Direct Programmatic Closure of ALL 5 `Hegel_M44_Audit` P2 Packages
# ============================================================================

def run_m4_4_audit_closure_verification_suite(work_root: Path) -> dict[str, Any]:
    """Reproduce and lock in 100% closure across all 5 P2 packages from `Hegel_M44_Audit`."""
    work_root.mkdir(parents=True, exist_ok=True)
    built = build_m4_3_continual_forest(work_root / "shared_lifecycle")
    true_s0: PhilosophicalForest = deepcopy(built["forest_after_stage0"])
    post_s2: PhilosophicalForest = deepcopy(built["forest"])

    # -------------------------------------------------------------------------
    # Package 1: `add_coarse_graining_edge` + consumer + archive verifier relational identity!
    # -------------------------------------------------------------------------
    coarse_id = "ode_meso_schur_pi_thermal"
    fine_id = "ode_micro_2mode_node"
    cert_s0 = deepcopy(
        next(e["empirical_closure_certificate"] for e in true_s0.coarse_graining_edges if e["coarse_node_id"] == coarse_id)
    )
    f_no_edge = deepcopy(post_s2)
    f_no_edge.coarse_graining_edges = [e for e in f_no_edge.coarse_graining_edges if e["coarse_node_id"] != coarse_id]
    f_try_add = deepcopy(f_no_edge)
    before_add_sha = _all_forest_state_sha256(f_try_add)
    add_raised = False
    add_err_msg = ""
    try:
        f_try_add.add_coarse_graining_edge(
            fine_node_id=fine_id,
            coarse_node_id=coarse_id,
            scale_map=deepcopy(f_try_add.nodes[coarse_id]["scale_map"]),
            empirical_closure_certificate=cert_s0,
        )
    except ValueError as exc:
        add_raised = True
        add_err_msg = str(exc)
    after_add_sha = _all_forest_state_sha256(f_try_add)

    # Also test in-memory tampering of an existing edge: both `fine_scale_children_of` AND `verify_forest_snapshot_digests` catch it!
    f_tampered_edge = deepcopy(post_s2)
    for e_mut in f_tampered_edge.coarse_graining_edges:
        if e_mut["coarse_node_id"] == coarse_id:
            e_mut["empirical_closure_certificate"] = deepcopy(cert_s0)
            e_mut["closure_certificate_digest"] = hashlib.sha256(
                json.dumps(cert_s0, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
    tampered_live_children = [n["node_id"] for n in f_tampered_edge.fine_scale_children_of(coarse_id)]
    tampered_digest_chk = verify_forest_snapshot_digests(f_tampered_edge)

    t21 = build_m4_3_target_suite(seed=20_301_120)[0][20]
    rec_no_edge = evaluate_m4_3_target_under_arm(
        t21,
        arm_name="audit_no_coarse_edge",
        seed=20_301_120,
        work_dir=work_root / "pkg1_no_edge",
        forest=f_no_edge,
        index=HippoRAGWormholeIndex(f_no_edge, damping_alpha=0.65),
        ordinary_history_specs=None,
        fallback_pool_specs=[],
        open_code_agent_fallback=False,
        allow_pool_fallback=False,
    )
    rec_tampered_edge = evaluate_m4_3_target_under_arm(
        t21,
        arm_name="audit_tampered_cross_instance_edge",
        seed=20_301_120,
        work_dir=work_root / "pkg1_tampered_edge",
        forest=f_tampered_edge,
        index=HippoRAGWormholeIndex(f_tampered_edge, damping_alpha=0.65),
        ordinary_history_specs=None,
        fallback_pool_specs=[],
        open_code_agent_fallback=False,
        allow_pool_fallback=False,
    )

    # -------------------------------------------------------------------------
    # Package 2: Ordinary 2-Instance Monod OR (`selected_action` independence) + R3/Partial Canonical Role Lock!
    # -------------------------------------------------------------------------
    monod_nd = post_s2.nodes["stage0_monod_2d"]
    act_a = deepcopy(monod_nd["claim_contract"]["conclusion_or_action"])
    art_a_path = Path(str(post_s2.resolve_artifact_path(act_a["authored_artifact_path"])))
    art_a_raw = json.loads(art_a_path.read_text(encoding="utf-8"))
    art_a_loaded = load_verified_authored_method(
        art_a_path,
        expected_ast_digest=act_a["ast_skeleton_digest"],
        expected_instance_digest=act_a["confirmed_instance_digest"],
        require_external_manifest=True,
        verify_frozen_only=True,
    )
    assert art_a_loaded is not None
    orig_hb = art_a_raw["host_confirmation_bundle"]
    cal_b = _run_metered_atomic_calibration(
        work_root / "pkg2_monod_gain12_instance_b",
        task_id="pkg2_monod_gain12_instance_b",
        surface_domain="enzymatic_kinetics_high_gain",
        spec=art_a_raw["operator_spec"],
        role_names=art_a_raw["role_names"],
        pilot_points=[r["features"] for r in orig_hb["training_prefix_rows"]],
        confirm_points=[r["features"] for r in orig_hb["confirm_rows"]],
        oracle_fn=lambda pt: 2.0 * float(predict_with_frozen_instance(art_a_loaded, [pt])["predictions"][0]),
        tolerance=float(orig_hb["tolerance"]),
    )
    art_b = cal_b["artifact"]
    act_b = _make_verified_instance_action(
        operator_tag=act_a["operator_tag"],
        operator_spec=art_b["operator_spec"],
        artifact_dict=art_b,
        artifact_relpath=str(Path(cal_b["artifact_path"]).resolve()),
    )
    cc_or = deepcopy(monod_nd["claim_contract"])
    cc_or["preconditions"] = [{"field": "active_arity", "op": "==", "value": 2}]
    cc_or["disjunctive_branches"] = [
        {
            "preconditions": [{"field": "secondary_bleed_active", "op": "==", "value": False}],
            "conclusion_or_action": deepcopy(act_a),
            "permission_tier": "local_solver_guidance",
            "evidence_status": "SUPPORTED_IN_SCOPE",
        },
        {
            "preconditions": [{"field": "secondary_bleed_active", "op": "==", "value": True}],
            "conclusion_or_action": deepcopy(act_b),
            "permission_tier": "local_solver_guidance",
            "evidence_status": "SUPPORTED_IN_SCOPE",
        },
    ]

    def _mk_or_forest(claim_d: dict[str, Any]) -> PhilosophicalForest:
        f_or = PhilosophicalForest(artifact_root=post_s2.artifact_root)
        f_or.register_node(
            node_id=monod_nd["node_id"],
            tree_id=monod_nd["tree_id"],
            layer_role=monod_nd["layer_role"],
            task_view=monod_nd["task_view"],
            observation_map=monod_nd["observation_map"],
            scale_map=monod_nd["scale_map"],
            claim_contract=claim_d,
            probe_signatures=[*monod_nd["probe_signatures"], "sig:cue_or_instance_control"],
            program_steps=monod_nd["program_steps"],
        )
        return f_or

    art_b_loaded = load_verified_authored_method(
        Path(cal_b["artifact_path"]),
        expected_ast_digest=act_b["ast_skeleton_digest"],
        expected_instance_digest=act_b["confirmed_instance_digest"],
        require_external_manifest=True,
        verify_frozen_only=True,
    )
    assert art_b_loaded is not None
    oracle_b = lambda pt: float(predict_with_frozen_instance(art_b_loaded, [pt])["predictions"][0])
    target_or_b = M43TargetSpec(
        target_id="generic_OR_selected_B",
        mechanism_family="Audit_Same_AST_Different_Instance",
        feature_names=("s", "b"),
        feature_bounds={"s": (0.5, 6.0), "b": (0.0, 1.0)},
        pilot_points=[{"s": s, "b": b} for s, b in [(0.5, 0.1), (1.0, 0.5), (2.0, 0.3), (3.0, 0.7), (4.0, 0.2), (5.5, 0.9)]],
        confirm_points=[{"s": 2.5, "b": 0.45}],
        heldout_points=[{"s": 1.6, "b": 0.6}, {"s": 3.7, "b": 0.25}, {"s": 4.5, "b": 0.85}],
        tolerance=0.05,
        oracle_fn=oracle_b,
        coarse_oracle_fn=None,
        domain_cue="cue_or_instance_control",
        scale_regime="macro",
    )
    f_or_intact = _mk_or_forest(cc_or)
    loaded_inst_intact: list[str | None] = []
    orig_loader_fn = _load_node_verified_operator_spec

    def _spy_loader(node, **kw):
        act_used = kw.get("action_override") or node["claim_contract"]["conclusion_or_action"]
        loaded_inst_intact.append(act_used.get("confirmed_instance_digest"))
        return orig_loader_fn(node, **kw)

    with patch("hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure._load_node_verified_operator_spec", _spy_loader):
        m43_or_intact = evaluate_m4_3_target_under_arm(
            target_or_b,
            arm_name="or_intact",
            seed=20_301_120,
            work_dir=work_root / "pkg2_or_intact",
            forest=f_or_intact,
            index=HippoRAGWormholeIndex(f_or_intact, damping_alpha=0.65),
            ordinary_history_specs=None,
            fallback_pool_specs=[],
            allow_pool_fallback=False,
            open_code_agent_fallback=False,
        )

    cc_absent_a = deepcopy(cc_or)
    cc_absent_a["conclusion_or_action"]["authored_artifact_path"] = str(work_root / "intentionally_absent_A.json")
    cc_absent_a["disjunctive_branches"][0]["conclusion_or_action"]["authored_artifact_path"] = str(
        work_root / "intentionally_absent_A.json"
    )
    f_or_absent_a = _mk_or_forest(cc_absent_a)
    r3_absent_a = ForestScopeDispatcher(f_or_absent_a).dispatch_predict(
        query_id="generic_or_B_absent_A",
        candidate_tree_id=monod_nd["tree_id"],
        features={"s": 1.6, "b": 0.6},
        target_metadata={"active_arity": 2, "secondary_bleed_active": True},
        ground_truth=oracle_b({"s": 1.6, "b": 0.6}),
    )
    m43_absent_a = evaluate_m4_3_target_under_arm(
        target_or_b,
        arm_name="or_absent_A",
        seed=20_301_120,
        work_dir=work_root / "pkg2_or_absent_A",
        forest=f_or_absent_a,
        index=HippoRAGWormholeIndex(f_or_absent_a, damping_alpha=0.65),
        ordinary_history_specs=None,
        fallback_pool_specs=[],
        allow_pool_fallback=False,
        open_code_agent_fallback=False,
    )

    # R3 explicit contradictory `role_map` vs valid anonymous `role_map` + partial canonical axis lock!
    pt_short = {"u_drive": 2.7, "T_horizon": 0.47}
    truth_short = float(_eval_pi_s_fine_scalar(pt_short["u_drive"], pt_short["T_horizon"])["y_micro"])
    disp_s2 = ForestScopeDispatcher(post_s2)
    pi_tree = post_s2.nodes[coarse_id]["tree_id"]
    r3_norm = disp_s2.dispatch_predict(
        query_id="canonical_without_map",
        candidate_tree_id=pi_tree,
        features=pt_short,
        target_metadata={"active_arity": 2, "modality": "thermal"},
        ground_truth=truth_short,
    )
    r3_contra = disp_s2.dispatch_predict(
        query_id="canonical_with_contradictory_map",
        candidate_tree_id=pi_tree,
        features=pt_short,
        target_metadata={"active_arity": 2, "modality": "thermal"},
        role_map={"u_drive": "T_horizon", "T_horizon": "u_drive"},
        ground_truth=truth_short,
    )
    r3_anon_ok = disp_s2.dispatch_predict(
        query_id="anonymous_valid_map",
        candidate_tree_id=pi_tree,
        features={"x0": 2.7, "x1": 0.47},
        target_metadata={"active_arity": 2, "modality": "thermal"},
        role_map={"x0": "u_drive", "x1": "T_horizon"},
        ground_truth=truth_short,
    )

    partial_acqs: list[dict[str, Any]] = []
    orig_ms_acq = MultiScaleWorkspaceTaskLab.acquire_channel

    def _spy_partial_acq(self, points, *, channel="coarse_macro", phase="pilot"):
        partial_acqs.append({"phase": phase, "channel": channel})
        return orig_ms_acq(self, points, channel=channel, phase=phase)

    partial_names = ("u_drive", "x1")
    partial_target = replace(
        t21,
        target_id="partial_canonical_short_time",
        feature_names=partial_names,
        feature_bounds={"u_drive": (2.4, 3.0), "x1": (0.4, 1.3)},
        pilot_points=[{"u_drive": x, "x1": y} for x, y in [(2.45, 0.40), (2.90, 0.52), (2.55, 0.67), (2.80, 0.85), (2.65, 1.05), (2.95, 1.20)]],
        confirm_points=[{"u_drive": 2.70, "x1": 0.47}],
        heldout_points=[{"u_drive": 2.50, "x1": 0.42}, {"u_drive": 2.85, "x1": 0.55}],
        oracle_fn=lambda p: t21.oracle_fn({"u_drive": p["u_drive"], "T_horizon": p["x1"]}),
        coarse_oracle_fn=lambda p: t21.coarse_oracle_fn({"u_drive": p["u_drive"], "T_horizon": p["x1"]}),
    )
    with patch.object(MultiScaleWorkspaceTaskLab, "acquire_channel", _spy_partial_acq):
        rec_partial = evaluate_m4_3_target_under_arm(
            partial_target,
            arm_name="partial_canonical",
            seed=20_301_120,
            work_dir=work_root / "pkg2_partial_canonical",
            forest=post_s2,
            index=HippoRAGWormholeIndex(post_s2, damping_alpha=0.65),
            ordinary_history_specs=None,
            fallback_pool_specs=[],
            allow_pool_fallback=False,
            open_code_agent_fallback=False,
        )

    # -------------------------------------------------------------------------
    # Package 3: Failed-Receipt Guard, Empirical Counterexample Scope Sublation, & 3-Tier Twin Contract!
    # -------------------------------------------------------------------------
    class _StopAfterS01(Exception):
        pass

    f_fail_s01 = deepcopy(post_s2)
    w4_before_fail = deepcopy(f_fail_s01.nodes["stage1_W4_monod_shunt"])
    w4_after_fail: dict[str, Any] = {}

    def _mock_fail_s01(target, **kwargs):
        nonlocal w4_after_fail
        if target.target_id == "M44_S01_W4_LiveRevisionTrigger":
            return {
                "confirmed_ok": False,
                "strict_pass": False,
                "false_confirmation": False,
                "status": "scoped_unknown",
                "cost": {"measurement_cost": 0.0, "compute_cost": 0.0, "composite_cost": 0.0, "phase_breakdown": {}},
            }
        w4_after_fail = deepcopy(kwargs["forest"].nodes["stage1_W4_monod_shunt"])
        raise _StopAfterS01

    with patch("hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback.evaluate_m4_3_target_under_arm", _mock_fail_s01):
        try:
            run_m4_4_online_writeback_and_multicue_stream(
                work_root / "pkg3_fail_s01",
                {"forest": f_fail_s01, "ordinary_history_forward": []},
            )
        except _StopAfterS01:
            pass

    # Empirical scope sublation on `W4` (`b <= theta_emp` learned from paid counterexample observations!)
    f_emp = deepcopy(post_s2)
    emp_bundle_dir = work_root / "pkg3_emp_bundle"
    emp_bundle_dir.mkdir(parents=True, exist_ok=True)
    for sub_n in ("stage0_metered_sources", "persisted_stage1_authored_workspaces", "stage2_lifecycle"):
        s_src = Path(str(post_s2.artifact_root)) / sub_n
        if s_src.exists():
            shutil.copytree(s_src, emp_bundle_dir / "persisted_authored_workspaces" / sub_n, dirs_exist_ok=True)
    f_emp.artifact_root = str(emp_bundle_dir.resolve())
    consolidator_emp = OutcomeDrivenForestConsolidator(f_emp, bundle_root=emp_bundle_dir)
    t_counterexample_w4 = M43TargetSpec(
        target_id="pkg3_W4_HighBleed_Counterexample",
        mechanism_family="Counterexample_NonlinearBleed",
        feature_names=("s", "b"),
        feature_bounds={"s": (0.4, 9.5), "b": (0.2, 3.0)},
        pilot_points=[
            {"s": 0.6, "b": 0.25},
            {"s": 1.6, "b": 0.55},
            {"s": 3.2, "b": 0.85},
            {"s": 5.0, "b": 1.15},
            {"s": 7.2, "b": 1.35},
            {"s": 9.0, "b": 1.45},
        ],
        confirm_points=[{"s": 5.5, "b": 2.25}],
        heldout_points=[{"s": 3.5, "b": 1.1}, {"s": 6.0, "b": 2.6}],
        tolerance=0.05,
        oracle_fn=lambda p: 6.0 * p["s"] / (2.0 + p["s"]) + 0.45 * p["b"] + 2.5 * (max(0.0, p["b"] - 1.55) ** 2),
        coarse_oracle_fn=None,
        domain_cue="internal_monod_shunt_specialization",
        scale_regime="macro",
    )
    ce_ws = work_root / "pkg3_ce_ws"
    rec_ce = evaluate_m4_3_target_under_arm(
        t_counterexample_w4,
        arm_name="pkg3_ce",
        seed=20_301_120,
        work_dir=ce_ws,
        forest=f_emp,
        index=HippoRAGWormholeIndex(f_emp, damping_alpha=0.65),
        ordinary_history_specs=None,
        fallback_pool_specs=[],
        allow_pool_fallback=False,
        open_code_agent_fallback=False,
    )
    cons_ce = consolidator_emp.consolidate_episode(
        step_num=1, target=t_counterexample_w4, episode_receipt=rec_ce, step_work_dir=ce_ws
    )
    w4_v2_nd = f_emp.nodes["stage1_W4_monod_shunt"]
    meta_w4 = {"active_arity": 2, "haldane_turnover_shunt_regime": False, "secondary_bleed_active": True}
    gate_in_1_45 = authorize_abstraction_use(
        requested_tier="bounded_trial",
        task_view=w4_v2_nd["task_view"],
        target_features={"s": 4.0, "b": 1.45, **meta_w4},
        claim_contract=w4_v2_nd["claim_contract"],
        scale_map=w4_v2_nd.get("scale_map"),
    )
    gate_out_2_10 = authorize_abstraction_use(
        requested_tier="bounded_trial",
        task_view=w4_v2_nd["task_view"],
        target_features={"s": 4.0, "b": 2.10, **meta_w4},
        claim_contract=w4_v2_nd["claim_contract"],
        scale_map=w4_v2_nd.get("scale_map"),
    )
    r3_in_1_45 = ForestScopeDispatcher(f_emp).dispatch_predict(
        query_id="w4_v2_b_1_45",
        candidate_tree_id="tree_monod_reactor",
        features={"s": 4.0, "b": 1.45},
        target_metadata=meta_w4,
    )
    r3_out_2_10 = ForestScopeDispatcher(f_emp).dispatch_predict(
        query_id="w4_v2_b_2_10",
        candidate_tree_id="tree_monod_reactor",
        features={"s": 4.0, "b": 2.10},
        target_metadata=meta_w4,
    )

    return {
        "p2_1_certificate_edge_node_relational_identity": {
            "add_cross_instance_edge_raised": add_raised,
            "add_cross_instance_error": add_err_msg,
            "entire_forest_state_unchanged_on_rejected_add": before_add_sha == after_add_sha,
            "tampered_edge_live_children": tampered_live_children,
            "tampered_edge_verifier_mismatch_count": tampered_digest_chk["edge_certificate_instance_mismatch_count"],
            "no_edge_T21_channel": rec_no_edge["selected_channel"],
            "tampered_edge_T21_channel": rec_tampered_edge["selected_channel"],
            "tampered_edge_T21_measurement_cost": rec_tampered_edge["cost"]["measurement_cost"],
        },
        "p2_2_selected_action_and_canonical_role_unification": {
            "or_intact_loaded_only_selected_branch_B": loaded_inst_intact == [act_b["confirmed_instance_digest"]],
            "or_intact_m43_strict_pass": m43_or_intact["strict_pass"],
            "or_absent_A_r3_status": r3_absent_a["status"],
            "or_absent_A_r3_within_tolerance": r3_absent_a["within_tolerance"],
            "or_absent_A_m43_status": m43_absent_a["status"],
            "or_absent_A_m43_confirmed_via_recall": m43_absent_a["confirmed_via_recall"],
            "or_absent_A_m43_strict_pass": m43_absent_a["strict_pass"],
            "or_absent_A_m43_heldout_rmse": m43_absent_a["heldout_rmse"],
            "r3_normal_canonical_status": r3_norm["status"],
            "r3_normal_canonical_abs_error": r3_norm["abs_error"],
            "r3_contradictory_canonical_map_status": r3_contra["status"],
            "r3_contradictory_canonical_map_prediction": r3_contra["prediction"],
            "r3_valid_anonymous_map_status": r3_anon_ok["status"],
            "r3_valid_anonymous_map_abs_error": r3_anon_ok["abs_error"],
            "partial_canonical_selected_channel": rec_partial["selected_channel"],
            "partial_canonical_wasted_coarse_acquisitions": sum(
                1 for a in partial_acqs if a["channel"] == "coarse_macro"
            ),
            "partial_canonical_measurement_cost": rec_partial["cost"]["measurement_cost"],
            "partial_canonical_strict_pass": rec_partial["strict_pass"],
            "partial_canonical_false_confirmation": rec_partial["false_confirmation"],
        },
        "p2_3_online_scope_semantics_and_failure_guard": {
            "version_incremented_on_failed_s01": w4_after_fail["version"] > w4_before_fail["version"],
            "w4_unchanged_on_failed_s01": w4_after_fail == w4_before_fail,
            "counterexample_w4_false_confirmation": rec_ce["false_confirmation"],
            "empirical_sublation_events": cons_ce["events"],
            "w4_version_after_counterexample": w4_v2_nd["version"],
            "w4_in_scope_b_1_45_gate_granted": gate_in_1_45["granted"],
            "w4_in_scope_b_1_45_r3_status": r3_in_1_45["status"],
            "w4_out_of_scope_b_2_10_gate_granted": gate_out_2_10["granted"],
            "w4_out_of_scope_b_2_10_r3_status": r3_out_2_10["status"],
        },
    }


# ============================================================================
# Part 3. Complete `M5.0` Outcome-Driven Continual Stream + Cold-Process Resume
# ============================================================================

def _build_m5_0_stream_tasks() -> list[tuple[str, M43TargetSpec]]:
    """Construct the pre-registered 24-step `M5.0` continual task sequence (`S01..S24`) + 3 continuation tasks (`S25..S27`)."""
    rng = Random(20_501_120)
    tasks: list[tuple[str, M43TargetSpec]] = []

    # --- Phase A (`S01..S04`): `W4` In-Scope -> Empirical Counterexample -> Post-Sublation In-Scope vs Out-of-Scope ---
    tasks.append((
        "phaseA_W4_in_scope_baseline",
        M43TargetSpec(
            target_id="M50_S01_W4_InScope_LowBleed",
            mechanism_family="M5_W4_ScopeSublation",
            feature_names=("s", "b"),
            feature_bounds={"s": (0.4, 9.5), "b": (0.1, 1.45)},
            pilot_points=[
                {"s": 0.6, "b": 0.2},
                {"s": 1.5, "b": 0.5},
                {"s": 3.0, "b": 0.8},
                {"s": 5.2, "b": 1.1},
                {"s": 7.4, "b": 1.35},
                {"s": 9.0, "b": 1.42},
            ],
            confirm_points=[{"s": 6.0, "b": 1.25}],
            heldout_points=[{"s": 2.2, "b": 0.6}, {"s": 6.8, "b": 1.30}, {"s": 8.4, "b": 0.95}],
            tolerance=0.05,
            oracle_fn=lambda p: 7.2 * p["s"] / (1.85 + p["s"]) + 0.58 * p["b"],
            coarse_oracle_fn=None,
            domain_cue="internal_monod_shunt_specialization",
            scale_regime="macro",
        ),
    ))
    tasks.append((
        "phaseA_W4_empirical_counterexample",
        M43TargetSpec(
            target_id="M50_S02_W4_HighBleed_Counterexample",
            mechanism_family="M5_W4_Counterexample_Surge",
            feature_names=("s", "b"),
            feature_bounds={"s": (0.4, 9.5), "b": (0.2, 3.0)},
            pilot_points=[
                {"s": 0.7, "b": 0.25},
                {"s": 1.8, "b": 0.55},
                {"s": 3.5, "b": 0.85},
                {"s": 5.2, "b": 1.15},
                {"s": 7.4, "b": 1.35},
                {"s": 9.0, "b": 1.45},
            ],
            confirm_points=[{"s": 5.5, "b": 2.25}],
            heldout_points=[{"s": 3.5, "b": 1.1}, {"s": 6.0, "b": 2.6}],
            tolerance=0.05,
            oracle_fn=lambda p: 7.2 * p["s"] / (1.85 + p["s"]) + 0.58 * p["b"] + 2.5 * (max(0.0, p["b"] - 1.55) ** 2),
            coarse_oracle_fn=None,
            domain_cue="internal_monod_shunt_specialization",
            scale_regime="macro",
            is_out_of_scope=True,
        ),
    ))
    tasks.append((
        "phaseA_W4_post_sublation_in_scope",
        M43TargetSpec(
            target_id="M50_S03_W4_PostSublation_SafeLowBleed",
            mechanism_family="M5_W4_ScopeSublation",
            feature_names=("s", "b"),
            feature_bounds={"s": (0.4, 9.5), "b": (0.1, 1.45)},
            pilot_points=[
                {"s": 0.7, "b": 0.25},
                {"s": 1.8, "b": 0.55},
                {"s": 3.4, "b": 0.85},
                {"s": 5.6, "b": 1.15},
                {"s": 7.6, "b": 1.38},
                {"s": 9.1, "b": 1.44},
            ],
            confirm_points=[{"s": 6.2, "b": 1.28}],
            heldout_points=[{"s": 2.4, "b": 0.7}, {"s": 7.0, "b": 1.32}, {"s": 8.6, "b": 1.05}],
            tolerance=0.05,
            oracle_fn=lambda p: 6.9 * p["s"] / (2.05 + p["s"]) + 0.54 * p["b"],
            coarse_oracle_fn=None,
            domain_cue="internal_monod_shunt_specialization",
            scale_regime="macro",
        ),
    ))
    tasks.append((
        "phaseA_W4_post_sublation_above_bound_blocked",
        M43TargetSpec(
            target_id="M50_S04_W4_PostSublation_HighBleedBlocked",
            mechanism_family="M5_W4_Counterexample_Surge",
            feature_names=("s", "b"),
            feature_bounds={"s": (0.4, 9.5), "b": (1.9, 3.1)},
            pilot_points=[
                {"s": 0.9, "b": 1.95},
                {"s": 2.1, "b": 2.15},
                {"s": 3.8, "b": 2.35},
                {"s": 5.4, "b": 2.55},
                {"s": 7.2, "b": 2.80},
                {"s": 8.9, "b": 3.02},
            ],
            confirm_points=[{"s": 6.1, "b": 2.65}],
            heldout_points=[{"s": 2.8, "b": 2.25}, {"s": 7.5, "b": 2.90}],
            tolerance=0.05,
            oracle_fn=lambda p: 7.0 * p["s"] / (1.95 + p["s"]) + 0.60 * p["b"] + 2.8 * math.sin(6.5 * p["b"]),
            coarse_oracle_fn=None,
            domain_cue="internal_monod_shunt_specialization",
            scale_regime="macro",
            is_out_of_scope=True,
        ),
    ))

    # --- Phase B (`S05..S11`): `W6` (`1D Haldane`) Domain A Discovery -> Domain B Promotion -> Domain C Negative -> Reuse ---
    def _mk_w6(vm: float, km: float, ki: float):
        return lambda p: vm * p["x_load"] / (km + p["x_load"] + ki * (p["x_load"] ** 2))

    tasks.append((
        "phaseB_discover_W6_domainA",
        M43TargetSpec(
            target_id="M50_S05_Discover_W6_TubuleDomainA",
            mechanism_family="Online_W6_Haldane_1D",
            feature_names=("x_load",),
            feature_bounds={"x_load": (0.2, 12.0)},
            pilot_points=[{"x_load": x} for x in (0.4, 1.2, 2.5, 5.0, 8.2, 11.2)],
            confirm_points=[{"x_load": 3.6}, {"x_load": 9.6}],
            heldout_points=[{"x_load": 0.8}, {"x_load": 3.1}, {"x_load": 7.0}, {"x_load": 10.6}],
            tolerance=0.045,
            oracle_fn=_mk_w6(14.5, 1.60, 0.18),
            coarse_oracle_fn=None,
            domain_cue="cue_tubule_saturation_a",
            scale_regime="macro",
        ),
    ))
    tasks.append((
        "phaseB_confirm_W6_domainB_wormhole",
        M43TargetSpec(
            target_id="M50_S06_Confirm_W6_ReuptakeDomainB",
            mechanism_family="Online_W6_Haldane_1D",
            feature_names=("x_load",),
            feature_bounds={"x_load": (0.2, 12.0)},
            pilot_points=[{"x_load": x} for x in (0.45, 1.25, 2.6, 5.2, 8.4, 11.4)],
            confirm_points=[{"x_load": 3.8}, {"x_load": 9.8}],
            heldout_points=[{"x_load": 0.9}, {"x_load": 3.2}, {"x_load": 7.2}, {"x_load": 10.8}],
            tolerance=0.045,
            oracle_fn=_mk_w6(11.8, 2.15, 0.145),
            coarse_oracle_fn=None,
            domain_cue="cue_reuptake_saturation_b",
            scale_regime="macro",
        ),
    ))
    tasks.append((
        "phaseB_negative_mapping_domainC_refuted",
        M43TargetSpec(
            target_id="M50_S07_NegativeMapping_IncompatibleDomainC",
            mechanism_family="Negative_IncompatiblePiecewise_1D",
            feature_names=("x_load",),
            feature_bounds={"x_load": (0.2, 12.0)},
            pilot_points=[{"x_load": x} for x in (0.4, 1.3, 2.7, 5.1, 8.3, 11.1)],
            confirm_points=[{"x_load": 3.9}, {"x_load": 9.4}],
            heldout_points=[{"x_load": 1.0}, {"x_load": 4.5}, {"x_load": 9.9}],
            tolerance=0.045,
            oracle_fn=lambda p: (2.8 * math.sin(1.9 * p["x_load"]) + (4.2 if p["x_load"] > 4.0 else -2.1)),
            coarse_oracle_fn=None,
            domain_cue="cue_piecewise_incompatible_c",
            scale_regime="macro",
            is_out_of_scope=True,
        ),
    ))
    for idx_w6 in range(4):
        vm6 = round(rng.uniform(13.0, 15.9), 3)
        km6 = round(rng.uniform(1.40, 1.95), 3)
        ki6 = round(rng.uniform(0.15, 0.22), 4)
        cue6 = (
            "cue_tubule_saturation_a"
            if idx_w6 == 0
            else ("cue_reuptake_saturation_b" if idx_w6 == 1 else "cue_tubule_saturation_a|cue_reuptake_saturation_b")
        )
        tasks.append((
            "downstream_W6_reuse",
            M43TargetSpec(
                target_id=f"M50_S{len(tasks)+1:02d}_Reuse_W6_Haldane1D",
                mechanism_family="Online_W6_Haldane_1D",
                feature_names=("x_load",),
                feature_bounds={"x_load": (0.2, 12.0)},
                pilot_points=[{"x_load": x} for x in (0.45, 1.25, 2.6, 5.2, 8.4, 11.4)],
                confirm_points=[{"x_load": 3.8}, {"x_load": 9.8}],
                heldout_points=[{"x_load": 0.9}, {"x_load": 3.2}, {"x_load": 7.2}, {"x_load": 10.8}],
                tolerance=0.045,
                oracle_fn=_mk_w6(vm6, km6, ki6),
                coarse_oracle_fn=None,
                domain_cue=cue6,
                scale_regime="macro",
            ),
        ))

    # --- Phase C (`S12..S19`): `W7` (`2D Stress-Relaxation`) Domain A -> Domain B -> End-to-End Multi-Cue Pooling Win! ---
    def _mk_w7(g_eq: float, tau_r: float, g_fast: float):
        return lambda p: g_eq * p["u_step"] + g_fast * p["u_step"] * math.exp(-p["t_soak"] / tau_r)

    tasks.append((
        "phaseC_discover_W7_domainA",
        M43TargetSpec(
            target_id="M50_S12_Discover_W7_PhotobioDomainA",
            mechanism_family="Online_W7_StressRelax_2D",
            feature_names=("u_step", "t_soak"),
            feature_bounds={"u_step": (0.4, 3.0), "t_soak": (0.05, 4.5)},
            pilot_points=[
                {"u_step": 1.2, "t_soak": 0.08},
                {"u_step": 2.4, "t_soak": 0.25},
                {"u_step": 1.5, "t_soak": 0.65},
                {"u_step": 2.8, "t_soak": 1.40},
                {"u_step": 1.8, "t_soak": 2.80},
                {"u_step": 2.6, "t_soak": 4.10},
            ],
            confirm_points=[{"u_step": 2.1, "t_soak": 0.14}, {"u_step": 1.9, "t_soak": 3.40}],
            heldout_points=[{"u_step": 1.0, "t_soak": 0.12}, {"u_step": 2.5, "t_soak": 0.90}, {"u_step": 2.2, "t_soak": 3.80}],
            tolerance=0.045,
            oracle_fn=_mk_w7(1.45, 0.48, 2.75),
            coarse_oracle_fn=None,
            domain_cue="cue_photobioreactor_cap2d_a",
            scale_regime="macro",
        ),
    ))
    tasks.append((
        "phaseC_confirm_W7_domainB_wormhole",
        M43TargetSpec(
            target_id="M50_S13_Confirm_W7_AllostericDomainB",
            mechanism_family="Online_W7_StressRelax_2D",
            feature_names=("u_step", "t_soak"),
            feature_bounds={"u_step": (0.4, 3.0), "t_soak": (0.05, 4.5)},
            pilot_points=[
                {"u_step": 1.15, "t_soak": 0.09},
                {"u_step": 2.45, "t_soak": 0.26},
                {"u_step": 1.55, "t_soak": 0.68},
                {"u_step": 2.75, "t_soak": 1.45},
                {"u_step": 1.75, "t_soak": 2.85},
                {"u_step": 2.65, "t_soak": 4.15},
            ],
            confirm_points=[{"u_step": 2.05, "t_soak": 0.15}, {"u_step": 1.85, "t_soak": 3.45}],
            heldout_points=[{"u_step": 1.05, "t_soak": 0.13}, {"u_step": 2.55, "t_soak": 0.95}, {"u_step": 2.25, "t_soak": 3.85}],
            tolerance=0.045,
            oracle_fn=_mk_w7(1.92, 0.56, 2.25),
            coarse_oracle_fn=None,
            domain_cue="cue_allosteric_cap2d_b",
            scale_regime="macro",
        ),
    ))
    for idx_w7 in range(10):
        ge7 = round(rng.uniform(1.25, 1.78), 3)
        tau7 = round(rng.uniform(0.40, 0.62), 3)
        gf7 = round(rng.uniform(2.35, 3.05), 3)
        cue7 = (
            "cue_photobioreactor_cap2d_a"
            if idx_w7 == 0
            else ("cue_allosteric_cap2d_b" if idx_w7 == 1 else "cue_photobioreactor_cap2d_a|cue_allosteric_cap2d_b")
        )
        tasks.append((
            "downstream_W7_reuse",
            M43TargetSpec(
                target_id=f"M50_S{len(tasks)+1:02d}_Reuse_W7_StressRelax2D",
                mechanism_family="Online_W7_StressRelax_2D",
                feature_names=("u_step", "t_soak"),
                feature_bounds={"u_step": (0.4, 3.0), "t_soak": (0.05, 4.5)},
                pilot_points=[
                    {"u_step": 1.15, "t_soak": 0.09},
                    {"u_step": 2.45, "t_soak": 0.26},
                    {"u_step": 1.55, "t_soak": 0.68},
                    {"u_step": 2.75, "t_soak": 1.45},
                    {"u_step": 1.75, "t_soak": 2.85},
                    {"u_step": 2.65, "t_soak": 4.15},
                ],
                confirm_points=[{"u_step": 2.05, "t_soak": 0.15}, {"u_step": 1.85, "t_soak": 3.45}],
                heldout_points=[{"u_step": 1.05, "t_soak": 0.13}, {"u_step": 2.55, "t_soak": 0.95}, {"u_step": 2.25, "t_soak": 3.85}],
                tolerance=0.045,
                oracle_fn=_mk_w7(ge7, tau7, gf7),
                coarse_oracle_fn=None,
                domain_cue=cue7,
                scale_regime="macro",
            ),
        ))

    # --- Phase D (`S20..S24`): 5 Historical Multi-Mechanism & Multi-Scale `Pi_s` Retention Tasks! ---
    hist_suite, _ = build_m4_3_target_suite(seed=20_591_120)
    for pick_idx in (0, 4, 8, 20, 22):
        ht = hist_suite[pick_idx]
        tasks.append((
            "historical_retention_check",
            replace(ht, target_id=f"M50_S{len(tasks)+1:02d}_Retain_{ht.target_id}"),
        ))

    assert len(tasks) == 28

    # --- Phase E (`S29..S31`): 3 Post-Archive Cold-Process Continuation Tasks (consumed ONLY after reload!) ---
    tasks.append((
        "cold_resume_W6_domainB_reuse",
        M43TargetSpec(
            target_id="M50_S29_ColdResume_W6_Reuptake",
            mechanism_family="Online_W6_Haldane_1D",
            feature_names=("x_load",),
            feature_bounds={"x_load": (0.2, 12.0)},
            pilot_points=[{"x_load": x} for x in (0.5, 1.3, 2.7, 5.3, 8.5, 11.3)],
            confirm_points=[{"x_load": 3.7}],
            heldout_points=[{"x_load": 1.1}, {"x_load": 3.4}, {"x_load": 7.4}],
            tolerance=0.045,
            oracle_fn=_mk_w6(13.9, 1.75, 0.175),
            coarse_oracle_fn=None,
            domain_cue="cue_reuptake_saturation_b",
            scale_regime="macro",
        ),
    ))
    tasks.append((
        "cold_resume_W7_multicue_pooling",
        M43TargetSpec(
            target_id="M50_S30_ColdResume_W7_MultiCue",
            mechanism_family="Online_W7_StressRelax_2D",
            feature_names=("u_step", "t_soak"),
            feature_bounds={"u_step": (0.4, 3.0), "t_soak": (0.05, 4.5)},
            pilot_points=[
                {"u_step": 1.2, "t_soak": 0.10},
                {"u_step": 2.5, "t_soak": 0.28},
                {"u_step": 1.6, "t_soak": 0.70},
                {"u_step": 2.7, "t_soak": 1.50},
                {"u_step": 1.8, "t_soak": 2.90},
                {"u_step": 2.6, "t_soak": 4.10},
            ],
            confirm_points=[{"u_step": 2.0, "t_soak": 0.16}],
            heldout_points=[{"u_step": 1.1, "t_soak": 0.14}, {"u_step": 2.4, "t_soak": 1.00}, {"u_step": 2.2, "t_soak": 3.75}],
            tolerance=0.045,
            oracle_fn=_mk_w7(1.58, 0.52, 2.60),
            coarse_oracle_fn=None,
            domain_cue="cue_photobioreactor_cap2d_a|cue_allosteric_cap2d_b",
            scale_regime="macro",
        ),
    ))
    tasks.append((
        "cold_resume_W4_in_scope_check",
        M43TargetSpec(
            target_id="M50_S31_ColdResume_W4_SafeBleed",
            mechanism_family="M5_W4_ScopeSublation",
            feature_names=("s", "b"),
            feature_bounds={"s": (0.4, 9.5), "b": (0.1, 1.40)},
            pilot_points=[
                {"s": 0.8, "b": 0.2},
                {"s": 1.9, "b": 0.5},
                {"s": 3.5, "b": 0.8},
                {"s": 5.7, "b": 1.1},
                {"s": 7.8, "b": 1.32},
                {"s": 9.2, "b": 1.38},
            ],
            confirm_points=[{"s": 6.4, "b": 1.22}],
            heldout_points=[{"s": 2.5, "b": 0.65}, {"s": 7.1, "b": 1.28}],
            tolerance=0.05,
            oracle_fn=lambda p: 7.1 * p["s"] / (1.92 + p["s"]) + 0.56 * p["b"],
            coarse_oracle_fn=None,
            domain_cue="internal_monod_shunt_specialization",
            scale_regime="macro",
        ),
    ))
    return tasks


def _evaluate_comprehensive_post_stream_r3(forest: PhilosophicalForest) -> dict[str, Any]:
    """Run full 10-query historical `R3` PLUS `W4` sublated boundary + `W6_A/B` + `W7_A/B` + negative `Domain C` checks!"""
    base_r3 = run_interleaved_zero_fit_dispatcher_stream(forest)
    disp = ForestScopeDispatcher(forest)

    meta_w4 = {"active_arity": 2, "haldane_turnover_shunt_regime": False, "secondary_bleed_active": True}
    w4_in = disp.dispatch_predict(
        query_id="post_stream_W4_in_scope_b_1_40",
        candidate_tree_id="tree_monod_reactor",
        features={"s": 4.0, "b": 1.40},
        target_metadata=meta_w4,
        ground_truth=9.0 * 4.0 / (1.6 + 4.0) + 0.72 * 1.40,
    )
    w4_out = disp.dispatch_predict(
        query_id="post_stream_W4_above_bound_b_2_10",
        candidate_tree_id="tree_monod_reactor",
        features={"s": 4.0, "b": 2.10},
        target_metadata=meta_w4,
    )
    w6_a = disp.dispatch_predict(
        query_id="post_stream_W6_domainA",
        candidate_tree_id="tree_online_tubule_cap",
        features={"x_load": 3.2},
        target_metadata={"active_arity": 1},
        ground_truth=14.5 * 3.2 / (1.60 + 3.2 + 0.18 * (3.2 ** 2)),
    )
    w6_b = disp.dispatch_predict(
        query_id="post_stream_W6_domainB",
        candidate_tree_id="tree_online_reuptake_cap",
        features={"x_load": 3.2},
        target_metadata={"active_arity": 1},
        ground_truth=11.8 * 3.2 / (2.15 + 3.2 + 0.145 * (3.2 ** 2)),
    )
    w6_c_neg = disp.dispatch_predict(
        query_id="post_stream_W6_negative_domainC",
        candidate_tree_id="tree_online_incompatible_c",
        features={"x_load": 3.2},
        target_metadata={"active_arity": 1},
    )
    w7_a = disp.dispatch_predict(
        query_id="post_stream_W7_domainA",
        candidate_tree_id="tree_online_photobio",
        features={"u_step": 2.0, "t_soak": 0.5},
        target_metadata={"active_arity": 2},
        ground_truth=1.45 * 2.0 + 2.75 * 2.0 * math.exp(-0.5 / 0.48),
    )
    w7_b = disp.dispatch_predict(
        query_id="post_stream_W7_domainB",
        candidate_tree_id="tree_online_allosteric",
        features={"u_step": 2.0, "t_soak": 0.5},
        target_metadata={"active_arity": 2},
        ground_truth=1.92 * 2.0 + 2.25 * 2.0 * math.exp(-0.5 / 0.56),
    )

    return {
        "historical_r3_full": base_r3,
        "w4_in_scope_r3": w4_in,
        "w4_above_empirical_bound_r3": w4_out,
        "w6_domain_a_r3": w6_a,
        "w6_domain_b_r3": w6_b,
        "w6_negative_domain_c_r3": w6_c_neg,
        "w7_domain_a_r3": w7_a,
        "w7_domain_b_r3": w7_b,
        "distinct_domain_instances_verified": bool(
            w6_a.get("status") == "dispatched_ok"
            and w6_b.get("status") == "dispatched_ok"
            and w6_a.get("confirmed_instance_digest") != w6_b.get("confirmed_instance_digest")
            and w7_a.get("status") == "dispatched_ok"
            and w7_b.get("status") == "dispatched_ok"
            and w7_a.get("confirmed_instance_digest") != w7_b.get("confirmed_instance_digest")
        ),
    }


def run_m5_0_counterexample_sublation_and_persistent_stream(
    work_root: Path,
    built_lifecycle: dict[str, Any],
) -> dict[str, Any]:
    """Execute the 24-step `M5.0` outcome-driven continual lifecycle across 5 controlled arms (`S01..S24`)
    and emit the full persistent online bundle & continual session checkpoint for `Hegel_M5_LiveSublation_PPR_PoolOn`!
    """
    work_root.mkdir(parents=True, exist_ok=True)
    base_forest: PhilosophicalForest = deepcopy(built_lifecycle["forest"])

    # Active polysemous surface hubs in 1D (`stage1_W3_murray_capped`) and 2D (`stage1_W5_murray_shunt` — ACTIVE in post-Stage2!)
    for hub_id, extra_cues in (
        ("stage1_W3_murray_capped", ("sig:cue_tubule_saturation_a", "sig:cue_reuptake_saturation_b")),
        ("stage1_W5_murray_shunt", ("sig:cue_photobioreactor_cap2d_a", "sig:cue_allosteric_cap2d_b")),
    ):
        h_nd = base_forest.nodes[hub_id]
        h_sigs = list(dict.fromkeys([*h_nd["probe_signatures"], *extra_cues]))
        base_forest.register_node(
            node_id=hub_id,
            tree_id=h_nd["tree_id"],
            layer_role=h_nd["layer_role"],
            task_view=h_nd["task_view"],
            observation_map=h_nd["observation_map"],
            scale_map=h_nd["scale_map"],
            claim_contract=h_nd["claim_contract"],
            probe_signatures=h_sigs,
            program_steps=h_nd["program_steps"],
            source_cluster_ids=h_nd["source_cluster_ids"],
        )
        base_forest.revalidate_vertical_edges_after_revision(hub_id, witness_episode_id=f"m50_init_{hub_id}")
        for wid, wh_e in list(base_forest.horizontal_wormhole_edges.items()):
            if hub_id in {wh_e["node_a_id"], wh_e["node_b_id"]}:
                base_forest.revalidate_wormhole_after_revision(
                    wid, witness_episode_id=f"m50_init_{wid}", reason="sync_hub_version"
                )

    all_tasks = _build_m5_0_stream_tasks()
    stream_tasks = all_tasks[:28]

    arm_defs = (
        ("Hegel_M5_LiveSublation_PPR_PoolOn", True, "ppr_pool_on"),
        ("Hegel_M5_LiveSublation_PPR_PoolOff", True, "ppr_pool_off"),
        ("Compiled_M5_LiveSublation_CueTable", True, "compiled_cuetable"),
        ("Dynamic_Flat_StructureMemory_LiveWriteBack", True, "dynamic_flat"),
        ("Frozen_PostStage2_OpenRediscovery", False, "ppr_pool_on"),
    )

    stream_receipts: list[dict[str, Any]] = []
    arm_reports: dict[str, Any] = {}
    primary_persisted_state: dict[str, Any] = {}

    for arm_name, enable_writeback, index_kind in arm_defs:
        arm_dir = work_root / "m5_stream" / arm_name
        wb_bundle_dir = arm_dir / "live_bundle"
        wb_bundle_dir.mkdir(parents=True, exist_ok=True)
        src_life = built_lifecycle["forest"].artifact_root
        assert src_life is not None
        for sub_n in ("stage0_metered_sources", "persisted_stage1_authored_workspaces", "stage2_lifecycle"):
            s_p = Path(src_life) / sub_n
            if s_p.exists():
                shutil.copytree(
                    s_p,
                    wb_bundle_dir / "persisted_authored_workspaces" / sub_n,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    dirs_exist_ok=True,
                )

        live_f = deepcopy(base_forest)
        live_f.artifact_root = str(wb_bundle_dir.resolve())
        if index_kind == "dynamic_flat":
            # Flat non-graph structure memory: retains identical live nodes & write-backs, zero wormhole/coarse edges!
            live_f.horizontal_wormhole_edges = {}
            live_f.coarse_graining_edges = []
            live_f.local_implication_edges = []
            live_ix: Any = FlatEqualMetadataIndex(live_f, mode="bm25", wormhole_equivalence_pooling=False)
            initial_index_build_compute = float(live_ix.index_build_compute_cost)
        elif index_kind == "ppr_pool_on":
            live_ix = HippoRAGWormholeIndex(
                live_f, damping_alpha=0.70, wormhole_equivalence_pooling=True, scale_aware=True
            )
            initial_index_build_compute = float(live_ix.index_build_compute_cost)
        elif index_kind == "ppr_pool_off":
            live_ix = HippoRAGWormholeIndex(
                live_f, damping_alpha=0.70, wormhole_equivalence_pooling=False, scale_aware=True
            )
            initial_index_build_compute = float(live_ix.index_build_compute_cost)
        else:
            live_ix = CompiledRelationalCueTable(live_f)
            initial_index_build_compute = float(live_ix.compile_compute_cost)

        consolidator = OutcomeDrivenForestConsolidator(
            live_f,
            bundle_root=wb_bundle_dir,
            enable_flat_only=(index_kind == "dynamic_flat"),
        )

        arm_rows: list[dict[str, Any]] = []
        total_wb_compute = 0.0

        for step_num, (phase_kind, t_spec) in enumerate(stream_tasks, 1):
            step_ws = arm_dir / f"step_{step_num:02d}_{t_spec.target_id}"
            rec = evaluate_m4_3_target_under_arm(
                t_spec,
                arm_name=arm_name,
                seed=20_501_120 + step_num,
                work_dir=step_ws,
                forest=live_f,
                index=live_ix,
                ordinary_history_specs=None,
                fallback_pool_specs=built_lifecycle["ordinary_history_forward"],
                open_code_agent_fallback=(not t_spec.is_out_of_scope),
                allow_pool_fallback=False,
            )

            # Meter `initial_index_build` explicitly on Step 1 so 100% of upfront index build is in the receipt ledger!
            extra_comp_step = initial_index_build_compute if step_num == 1 else 0.0
            if step_num == 1 and initial_index_build_compute > 0.0:
                rec["cost"]["phase_breakdown"]["initial_index_build"] = {
                    "obs": 0.0,
                    "meas": 0.0,
                    "compute": round(initial_index_build_compute, 4),
                }

            if enable_writeback:
                cons_out = consolidator.consolidate_episode(
                    step_num=step_num,
                    target=t_spec,
                    episode_receipt=rec,
                    step_work_dir=step_ws,
                )
                wb_c = float(cons_out["writeback_compute_cost"])
                if wb_c > 0.0:
                    total_wb_compute += wb_c
                    extra_comp_step += wb_c
                    rec["cost"]["phase_breakdown"]["online_forest_writeback"] = {
                        "obs": 0.0,
                        "meas": 0.0,
                        "compute": round(wb_c, 4),
                    }

            if extra_comp_step > 0.0:
                rec["cost"]["compute_cost"] = round(float(rec["cost"]["compute_cost"]) + extra_comp_step, 4)
                rec["cost"]["composite_cost"] = round(
                    float(rec["cost"]["measurement_cost"]) + 0.05 * float(rec["cost"]["compute_cost"]), 4
                )

            rec["stream_step"] = step_num
            rec["stream_phase"] = phase_kind
            arm_rows.append(rec)
            stream_receipts.append(rec)

        in_scope_rows = [r for r in arm_rows if not r["is_out_of_scope"]]
        oos_rows = [r for r in arm_rows if r["is_out_of_scope"]]
        r3_comp = _evaluate_comprehensive_post_stream_r3(live_f)
        dig_comp = verify_forest_snapshot_digests(live_f)
        s04_rec = next(r for r in arm_rows if r["target_id"] == "M50_S04_W4_PostSublation_HighBleedBlocked")
        w7_multicue_rows = [
            r for r in arm_rows
            if r["stream_phase"] == "downstream_W7_reuse" and "|" in str(stream_tasks[r["stream_step"] - 1][1].domain_cue)
        ]

        refresh_cnt = (
            int(getattr(live_ix, "refresh_count", 0))
            if hasattr(live_ix, "refresh_count")
            else int(getattr(live_ix, "recompile_count", 0))
        )

        arm_reports[arm_name] = {
            "enable_writeback": enable_writeback,
            "index_kind": index_kind,
            "stream_target_count": len(arm_rows),
            "in_scope_target_count": len(in_scope_rows),
            "out_of_scope_target_count": len(oos_rows),
            "in_scope_strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in in_scope_rows), 4),
            "in_scope_false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in in_scope_rows), 4),
            "out_of_scope_false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in oos_rows), 4),
            "out_of_scope_scoped_unknown_rate": round(
                mean(1.0 if r["status"] == "scoped_unknown" else 0.0 for r in oos_rows), 4
            ),
            "max_prefix_confirm_overlap": max(int(r["prefix_confirm_overlap_count"]) for r in arm_rows),
            "downstream_reuse_recall_count": sum(
                1 for r in arm_rows
                if (r["stream_phase"].startswith("downstream_") or "_confirm_" in r["stream_phase"])
                and r["confirmed_via_recall"]
            ),
            "w7_multicue_evaluated_candidates_total": sum(int(r["evaluated_candidate_count"]) for r in w7_multicue_rows),
            "s04_high_bleed_evaluated_candidates": int(s04_rec["evaluated_candidate_count"]),
            "total_evaluated_candidates": sum(int(r["evaluated_candidate_count"]) for r in arm_rows),
            "mean_evaluated_candidates": round(mean(r["evaluated_candidate_count"] for r in arm_rows), 4),
            "initial_index_build_compute_cost": round(initial_index_build_compute, 4),
            "writeback_compute_cost_total": round(total_wb_compute, 4),
            "index_auto_refresh_count": refresh_cnt,
            "total_measurement_cost": round(sum(float(r["cost"]["measurement_cost"]) for r in arm_rows), 4),
            "total_compute_cost": round(sum(float(r["cost"]["compute_cost"]) for r in arm_rows), 4),
            "total_stream_composite_cost": round(sum(float(r["cost"]["composite_cost"]) for r in arm_rows), 4),
            "writeback_events": deepcopy(consolidator.writeback_event_log),
            "failure_witness_count": len(consolidator.failure_witness_ledger),
            "post_stream_digest_verification": dig_comp,
            "post_stream_comprehensive_r3": r3_comp,
        }

        if arm_name == "Hegel_M5_LiveSublation_PPR_PoolOn":
            checkpoint_payload = {
                "schema_version": "hegel_m5_0_continual_checkpoint_v1",
                "arm_name": arm_name,
                "episode_cursor": len(arm_rows),
                "total_pre_registered_tasks": len(all_tasks),
                "remaining_task_ids": [t.target_id for _, t in all_tasks[len(arm_rows) :]],
                "cumulative_budget_ledger": {
                    "observation_cost": sum(int(r["cost"]["observation_cost"]) for r in arm_rows),
                    "measurement_cost": round(sum(float(r["cost"]["measurement_cost"]) for r in arm_rows), 4),
                    "compute_cost": round(sum(float(r["cost"]["compute_cost"]) for r in arm_rows), 4),
                    "composite_cost": round(sum(float(r["cost"]["composite_cost"]) for r in arm_rows), 4),
                },
                "supported_coord_samples": {
                    f"{nid}::{cr}": vals for (nid, cr), vals in sorted(consolidator.supported_coord_samples.items())
                },
                "paid_observation_history": deepcopy(consolidator.paid_observation_history),
                "failure_witness_ledger": deepcopy(consolidator.failure_witness_ledger),
                "writeback_event_log": deepcopy(consolidator.writeback_event_log),
                "ordinary_history_forward": deepcopy(built_lifecycle["ordinary_history_forward"]),
            }
            primary_persisted_state = {
                "live_forest": deepcopy(live_f),
                "live_bundle_dir": wb_bundle_dir,
                "checkpoint": checkpoint_payload,
            }

    ppr_on_c = arm_reports["Hegel_M5_LiveSublation_PPR_PoolOn"]["total_stream_composite_cost"]
    ppr_off_c = arm_reports["Hegel_M5_LiveSublation_PPR_PoolOff"]["total_stream_composite_cost"]
    cue_c = arm_reports["Compiled_M5_LiveSublation_CueTable"]["total_stream_composite_cost"]
    flat_c = arm_reports["Dynamic_Flat_StructureMemory_LiveWriteBack"]["total_stream_composite_cost"]
    froz_c = arm_reports["Frozen_PostStage2_OpenRediscovery"]["total_stream_composite_cost"]

    return {
        "stream_length": len(stream_tasks),
        "arms": arm_reports,
        "end_to_end_cost_deltas": {
            "net_savings_vs_frozen_post_stage2": round(froz_c - ppr_on_c, 4),
            "net_savings_vs_dynamic_flat_memory": round(flat_c - ppr_on_c, 4),
            "net_savings_vs_compiled_cuetable": round(cue_c - ppr_on_c, 4),
            "net_end_to_end_pooling_savings_vs_pool_off": round(ppr_off_c - ppr_on_c, 4),
        },
        "stream_receipts": stream_receipts,
        "primary_persisted_state": primary_persisted_state,
    }


def persist_m5_0_online_state_to_bundle(
    primary_persisted_state: dict[str, Any],
    *,
    out_bundle_dir: Path,
) -> dict[str, Any]:
    """Copy online write-back bundles (`W6_A/B`, `W7_A/B`), `post_m5_online_stream_forest_snapshot.json`,
    and `continual_session_checkpoint.json` into ``out_bundle_dir``!
    """
    src_wb = Path(primary_persisted_state["live_bundle_dir"]) / "persisted_authored_workspaces" / "online_writebacks"
    dst_wb = out_bundle_dir / "persisted_authored_workspaces" / "online_writebacks"
    if dst_wb.exists():
        shutil.rmtree(dst_wb)
    if src_wb.exists():
        shutil.copytree(src_wb, dst_wb, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    snaps_dir = out_bundle_dir / "relocatable_forest_snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snaps_dir / "post_m5_online_stream_forest_snapshot.json"
    snap_json = serialize_location_independent_forest(primary_persisted_state["live_forest"])
    snap_path.write_text(json.dumps(snap_json, indent=2, sort_keys=True), encoding="utf-8")

    ckpt_path = out_bundle_dir / "continual_session_checkpoint.json"
    ckpt_path.write_text(
        json.dumps(primary_persisted_state["checkpoint"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "snapshot_path": str(snap_path),
        "checkpoint_path": str(ckpt_path),
        "online_writeback_node_dirs": sorted(p.name for p in dst_wb.iterdir() if p.is_dir()) if dst_wb.exists() else [],
    }


def resume_m5_0_continual_checkpoint_cold_continuation(
    bundle_dir: Path,
    *,
    continuation_work_dir: Path,
) -> dict[str, Any]:
    """Cold-reload `post_m5_online_stream_forest_snapshot.json` + `continual_session_checkpoint.json`
    AFTER deleting temporary build directories (safe in a brand-new process!) and execute `S25..S27`!
    """
    _VERIFIED_AUTHORED_CACHE.clear()
    bundle_dir = bundle_dir.resolve()
    continuation_work_dir.mkdir(parents=True, exist_ok=True)

    snap_path = bundle_dir / "relocatable_forest_snapshots" / "post_m5_online_stream_forest_snapshot.json"
    ckpt_path = bundle_dir / "continual_session_checkpoint.json"
    restored_forest = load_location_independent_forest(snap_path, bundle_dir)
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))

    digest_chk = verify_forest_snapshot_digests(restored_forest)
    cold_r3 = _evaluate_comprehensive_post_stream_r3(restored_forest)

    live_ix = HippoRAGWormholeIndex(
        restored_forest, damping_alpha=0.70, wormhole_equivalence_pooling=True, scale_aware=True
    )
    consolidator = OutcomeDrivenForestConsolidator(restored_forest, bundle_root=bundle_dir)
    consolidator.paid_observation_history = deepcopy(ckpt["paid_observation_history"])
    consolidator.failure_witness_ledger = deepcopy(ckpt["failure_witness_ledger"])
    consolidator.writeback_event_log = deepcopy(ckpt["writeback_event_log"])
    for k_str, vals in ckpt.get("supported_coord_samples", {}).items():
        nid_s, cr_s = k_str.split("::", 1)
        consolidator.supported_coord_samples[(nid_s, cr_s)] = [float(x) for x in vals]

    all_tasks = _build_m5_0_stream_tasks()
    cursor_start = int(ckpt["episode_cursor"])
    rem_tasks = all_tasks[cursor_start:]
    cont_receipts: list[dict[str, Any]] = []

    cum_obs = int(ckpt["cumulative_budget_ledger"]["observation_cost"])
    cum_meas = float(ckpt["cumulative_budget_ledger"]["measurement_cost"])
    cum_comp = float(ckpt["cumulative_budget_ledger"]["compute_cost"])

    for offset_idx, (phase_kind, t_spec) in enumerate(rem_tasks, 1):
        step_num = cursor_start + offset_idx
        step_ws = continuation_work_dir / f"resumed_step_{step_num:02d}_{t_spec.target_id}"
        rec = evaluate_m4_3_target_under_arm(
            t_spec,
            arm_name="ColdResumed_Hegel_M5_PPR_PoolOn",
            seed=20_501_120 + step_num,
            work_dir=step_ws,
            forest=restored_forest,
            index=live_ix,
            ordinary_history_specs=None,
            fallback_pool_specs=ckpt["ordinary_history_forward"],
            open_code_agent_fallback=True,
            allow_pool_fallback=False,
        )
        rec["stream_step"] = step_num
        rec["stream_phase"] = phase_kind
        cum_obs += int(rec["cost"]["observation_cost"])
        cum_meas = round(cum_meas + float(rec["cost"]["measurement_cost"]), 4)
        cum_comp = round(cum_comp + float(rec["cost"]["compute_cost"]), 4)
        cont_receipts.append(rec)

    any_open_rediscovery = any(
        str(r.get("selected_node_id") or "").startswith("open_code_agent:") for r in cont_receipts
    )
    return {
        "resumed_from_cursor": cursor_start,
        "final_cursor": cursor_start + len(cont_receipts),
        "continuation_task_count": len(cont_receipts),
        "all_continuation_strict_pass": all(r["strict_pass"] for r in cont_receipts),
        "all_continuation_confirmed_via_recall": all(r["confirmed_via_recall"] for r in cont_receipts),
        "any_source_rediscovery_triggered": any_open_rediscovery,
        "max_prefix_confirm_overlap": max(int(r["prefix_confirm_overlap_count"]) for r in cont_receipts),
        "s30_w7_multicue_evaluated_candidates": next(
            int(r["evaluated_candidate_count"]) for r in cont_receipts if r["target_id"] == "M50_S30_ColdResume_W7_MultiCue"
        ),
        "restored_snapshot_digest_verification": digest_chk,
        "restored_cold_comprehensive_r3": cold_r3,
        "updated_cumulative_budget_ledger": {
            "observation_cost": cum_obs,
            "measurement_cost": cum_meas,
            "compute_cost": cum_comp,
            "composite_cost": round(cum_meas + 0.05 * cum_comp, 4),
        },
        "continuation_receipts": cont_receipts,
    }
