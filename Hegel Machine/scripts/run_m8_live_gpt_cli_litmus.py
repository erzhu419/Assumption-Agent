#!/usr/bin/env python3
"""Hegel M8.0 Live `GPT CLI` (`codex` `gpt-6.2-sol`) Continual Discovery & Transfer Litmus.

Executes a fast (<90s), traceable, SHA-256-locked continual learning & revision sequence on ONE
persistent `PersistentMechanismGraph` (`PhilosophicalForest` backed), using official OpenAI ChatGPT
CLI (`auth_mode == "chatgpt"`, `gpt-6.2-sol`) for anonymous nonlinear `ConceptLift` equation-conduit
synthesis and verifying:
  1. `AncestorReweight` on shared upstream shift + zero-shot forward transfer to unseen related task
     + exact `0.0` drift on unrelated tasks;
  2. `LocalSplit` on localized observable sub-manifold breakdown without corrupting the majority regime;
  3. Live `GPT CLI` (`gpt-6.2-sol`) `ConceptLift` on anonymous features `('f0', 'f1')` beating bilinear
     `f0 * f1` & parameter-only baselines on held-out data, PLUS zero-LLM-call structural reuse on a
     subsequent unseen sister task;
  4. `ChangeMemory` (`Pi_fold` exponential history kernel) on *same-present, different-past* dynamics;
  5. `NoChange` safety on heteroscedastic zero-mean noise (`r = sigma(x) * eps`) even with `GPT CLI` active!
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f
from hegel_machine.cognition.m4_4_instance_role_closure_and_online_writeback import (
    serialize_location_independent_forest,
)
from hegel_machine.cognition.m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TaskModel,
    TypedMechanism,
    fit_and_score_task_model,
)


def get_git_head_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def write_sha256_sums(root: Path) -> int:
    lines: list[str] = []
    for fp in sorted(root.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(root).as_posix()
            digest = hashlib.sha256(fp.read_bytes()).hexdigest()
            lines.append(f"{digest}  {rel}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def split_ds(
    x_dict: dict[str, np.ndarray],
    y_arr: np.ndarray,
    n_tr: int,
    *,
    extra_tr: dict[str, Any] | None = None,
    extra_va: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tr = {"x": {k: v[:n_tr] for k, v in x_dict.items()}, "y": y_arr[:n_tr], **(extra_tr or {})}
    va = {"x": {k: v[n_tr:] for k, v in x_dict.items()}, "y": y_arr[n_tr:], **(extra_va or {})}
    return tr, va


def run_litmus(*, out_dir: Path, model: str = "gpt-6.2-sol", reasoning_effort: str = "medium") -> dict[str, Any]:
    t_start = monotonic()
    codex_info = m6f.discover_chatgpt_subscription_codex_cli()
    if codex_info.get("auth_mode") != "chatgpt":
        raise RuntimeError(f"Expected official ChatGPT OAuth `codex` CLI (`auth_mode=='chatgpt'`), got: {codex_info}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cli_stage_dir = out_dir / "gpt_cli_traces"
    cli_stage_dir.mkdir(parents=True, exist_ok=True)

    client = m6f.FrontierCodexClient(
        codex_info=codex_info,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=120,
    )

    rng = np.random.default_rng(2026112508)
    graph = PersistentMechanismGraph()
    planner = M8VariationalPlanner(graph)

    # Register 2 disjoint shared upstream premises in the SAME persistent graph:
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_upstream_alpha",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("f0", "f1"),
            shared_weights={"f0": -1.55, "f1": 0.75},
        ),
        tree_id="tree_m8_alpha",
    )
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_upstream_sentinel_acoustics",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("freq_khz", "damping"),
            shared_weights={"freq_khz": 2.15, "damping": -0.85},
        ),
        tree_id="tree_m8_sentinel",
    )

    # Step 0: Unrelated sentinel task
    fq0 = rng.normal(0.0, 1.0, 140)
    dm0 = rng.normal(0.0, 1.0, 140)
    y0 = 2.15 * fq0 - 0.85 * dm0 + rng.normal(0.0, 0.11, 140)
    tr0, va0 = split_ds({"freq_khz": fq0, "damping": dm0}, y0, 90)
    planner.register_task(
        task_id="task_0_sentinel_acoustics",
        shared_upstream_id="shared_upstream_sentinel_acoustics",
        feature_names=("freq_khz", "damping"),
        train_data=tr0,
        val_data=va0,
    )
    sentinel_rmse_initial = planner.evaluate_task_zero_shot_from_shared("task_0_sentinel_acoustics")["val_rmse"]

    # Step 1: Shared upstream shift across related tasks -> `AncestorReweight` + unseen `task_1c` forward transfer
    for tid in ("task_1a_related_shift", "task_1b_related_shift"):
        f0 = rng.normal(0.0, 1.0, 140)
        f1 = rng.normal(0.0, 1.0, 140)
        y_sh = 1.80 * f0 + 0.75 * f1 + rng.normal(0.0, 0.11, 140)
        tr_i, va_i = split_ds({"f0": f0, "f1": f1}, y_sh, 90)
        planner.register_task(
            task_id=tid,
            shared_upstream_id="shared_upstream_alpha",
            feature_names=("f0", "f1"),
            train_data=tr_i,
            val_data=va_i,
        )

    f0_c = rng.normal(0.0, 1.0, 120)
    f1_c = rng.normal(0.0, 1.0, 120)
    yc = 1.80 * f0_c + 0.75 * f1_c + rng.normal(0.0, 0.11, 120)
    tr_c, va_c = split_ds({"f0": f0_c, "f1": f1_c}, yc, 70)
    w_old = dict(graph.mechanisms["shared_upstream_alpha"].shared_weights)
    unseen_1c_zero_shot_rmse_before = float(
        np.sqrt(np.mean((va_c["y"] - (w_old["f0"] * va_c["x"]["f0"] + w_old["f1"] * va_c["x"]["f1"])) ** 2))
    )

    step1_res = planner.evaluate_and_revise_task("task_1a_related_shift")
    planner.register_task(
        task_id="task_1c_unseen_forward_transfer",
        shared_upstream_id="shared_upstream_alpha",
        feature_names=("f0", "f1"),
        train_data=tr_c,
        val_data=va_c,
    )
    unseen_1c_zero_shot_rmse_after = planner.evaluate_task_zero_shot_from_shared("task_1c_unseen_forward_transfer")["val_rmse"]

    # Step 2: Localized sub-regime exception -> `LocalSplit`
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="shared_upstream_shear",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("load_axis", "regime_sensor"),
            shared_weights={"load_axis": -1.35, "regime_sensor": 0.25},
        ),
        tree_id="tree_m8_shear",
    )
    ld = rng.normal(0.0, 1.0, 220)
    rs = rng.normal(0.0, 1.0, 220)
    exc = (rs > 1.15).astype(float)
    y2 = -1.35 * ld + 0.25 * rs + exc * (3.5 + 1.7 * ld) + rng.normal(0.0, 0.11, 220)
    tr2, va2 = split_ds({"load_axis": ld, "regime_sensor": rs}, y2, 150)
    planner.register_task(
        task_id="task_2_localized_regime_exception",
        shared_upstream_id="shared_upstream_shear",
        feature_names=("load_axis", "regime_sensor"),
        train_data=tr2,
        val_data=va2,
    )
    step2_res = planner.evaluate_and_revise_task("task_2_localized_regime_exception")

    # Step 3: Anonymous nonlinear rational conduit discovery via LIVE `GPT CLI` (`gpt-6.2-sol`) -> `ConceptLift`
    # True underlying residual conduit: `3.6 * f0 / (1.0 + 1.8 * abs(f1))` (anonymous feature names `f0`, `f1`!)
    f0_3 = rng.uniform(-2.2, 2.2, 220)
    f1_3 = rng.uniform(-2.5, 2.5, 220)
    w_a = graph.mechanisms["shared_upstream_alpha"].shared_weights
    true_conduit_3 = 3.6 * f0_3 / (1.0 + 1.8 * np.abs(f1_3))
    y3 = w_a["f0"] * f0_3 + w_a["f1"] * f1_3 + true_conduit_3 + rng.normal(0.0, 0.10, 220)
    tr3, va3 = split_ds({"f0": f0_3, "f1": f1_3}, y3, 150)
    tm3 = planner.register_task(
        task_id="task_3a_anonymous_rational_conduit",
        shared_upstream_id="shared_upstream_alpha",
        feature_names=("f0", "f1"),
        train_data=tr3,
        val_data=va3,
    )
    rmse_3a_before = tm3.fit_summary["val_rmse"]
    print(f"[M8-GPT-CLI] Calling `{model}` (`auth_mode={codex_info['auth_mode']}`) for `task_3a_anonymous_rational_conduit`...", flush=True)
    step3_res = planner.evaluate_and_revise_task("task_3a_anonymous_rational_conduit", codex_client=client)
    rmse_3a_after = graph.active_task_models["task_3a_anonymous_rational_conduit"].fit_summary["val_rmse"]

    # Step 3b: Forward Structural Reuse of `gpt-6.2-sol`'s persisted `interaction_lift` on an unseen sister task (0 extra LLM calls!)
    f0_3b = rng.uniform(-2.2, 2.2, 90)
    f1_3b = rng.uniform(-2.5, 2.5, 90)
    y3b = w_a["f0"] * f0_3b + w_a["f1"] * f1_3b + 2.9 * f0_3b / (1.0 + 1.8 * np.abs(f1_3b)) + rng.normal(0.0, 0.10, 90)
    tr3b, va3b = split_ds({"f0": f0_3b, "f1": f1_3b}, y3b, 28)  # scarce 28-sample sister regime!

    lifted_ids_in_graph = sorted(
        mid for mid, m in graph.mechanisms.items()
        if m.parent_id == "shared_upstream_alpha" and m.role == "interaction_lift"
    )
    tm_3b_no_lift = planner.register_task(
        task_id="task_3b_unseen_sister_reactor",
        shared_upstream_id="shared_upstream_alpha",
        feature_names=("f0", "f1"),
        train_data=tr3b,
        val_data=va3b,
        initial_local_shrinkage=1e-2,
    )
    sister_rmse_without_lifted_conduit = tm_3b_no_lift.fit_summary["val_rmse"]

    tm_3b_with_lift = deepcopy(tm_3b_no_lift)
    tm_3b_with_lift.interaction_lift_ids = lifted_ids_in_graph
    sc_3b_lift = fit_and_score_task_model(tm_3b_with_lift, graph, tr3b, va3b, prior_shrinkage=1e-2)
    sister_rmse_with_persisted_gpt_conduit = sc_3b_lift["val_rmse"]

    # Step 4: Non-Markovian *same present, different past* -> `ChangeMemory`
    n4 = 180
    hist4 = rng.normal(0.0, 1.5, size=(n4, 8))
    dec4 = np.exp(-1.0 / 2.0)
    z4 = np.zeros(n4)
    for k in range(8):
        z4 = dec4 * z4 + (1.0 - dec4) * hist4[:, k]
    fq4 = rng.normal(0.0, 1.0, n4)
    dm4 = rng.normal(0.0, 1.0, n4)
    y4 = 2.15 * fq4 - 0.85 * dm4 + 2.55 * z4 + rng.normal(0.0, 0.11, n4)
    tr4, va4 = split_ds(
        {"freq_khz": fq4, "damping": dm4},
        y4,
        120,
        extra_tr={"past_driver_history": hist4[:120], "history_dt": 1.0},
        extra_va={"past_driver_history": hist4[120:], "history_dt": 1.0},
    )
    planner.register_task(
        task_id="task_4_memory_hysteresis",
        shared_upstream_id="shared_upstream_sentinel_acoustics",
        feature_names=("freq_khz", "damping"),
        train_data=tr4,
        val_data=va4,
    )
    step4_res = planner.evaluate_and_revise_task("task_4_memory_hysteresis")

    # Step 5: Heteroscedastic zero-mean noise WITH live `codex_client` passed -> must stay `NoChange`!
    fq5 = rng.normal(0.0, 1.0, 180)
    dm5 = rng.normal(0.0, 1.0, 180)
    y5 = 2.15 * fq5 - 0.85 * dm5 + (0.12 + 0.60 * np.abs(fq5)) * rng.normal(0.0, 1.0, 180)
    tr5, va5 = split_ds({"freq_khz": fq5, "damping": dm5}, y5, 120)
    planner.register_task(
        task_id="task_5_heteroscedastic_noise_guard",
        shared_upstream_id="shared_upstream_sentinel_acoustics",
        feature_names=("freq_khz", "damping"),
        train_data=tr5,
        val_data=va5,
    )
    step5_res = planner.evaluate_and_revise_task("task_5_heteroscedastic_noise_guard", codex_client=client)

    sentinel_rmse_final = planner.evaluate_task_zero_shot_from_shared("task_0_sentinel_acoustics")["val_rmse"]
    for rec in graph.decision_ledger:
        rec.validate_prefix_clean()

    summary = {
        "git_commit": get_git_head_sha(),
        "model": model,
        "auth_mode": codex_info["auth_mode"],
        "total_wall_ms": round((monotonic() - t_start) * 1000.0, 2),
        "persistent_forest_node_count": len(graph.forest.nodes),
        "persistent_mechanism_count": len(graph.mechanisms),
        "decision_ledger_count": len(graph.decision_ledger),
        "prefix_clean_verified": True,
        "unrelated_sentinel_rmse_initial": round(sentinel_rmse_initial, 6),
        "unrelated_sentinel_rmse_final": round(sentinel_rmse_final, 6),
        "unrelated_sentinel_abs_drift": round(abs(sentinel_rmse_final - sentinel_rmse_initial), 12),
        "scenarios": {
            "step1_shared_upstream_shift": {
                "selected_action": step1_res["selected_candidate"]["action_type"],
                "unseen_task_1c_zero_shot_rmse_before": round(unseen_1c_zero_shot_rmse_before, 5),
                "unseen_task_1c_zero_shot_rmse_after": round(unseen_1c_zero_shot_rmse_after, 5),
                "unrelated_drift": step1_res["selected_candidate"]["unrelated_max_abs_drift"],
            },
            "step2_localized_exception": {
                "selected_action": step2_res["selected_candidate"]["action_type"],
                "scope_guard": step2_res["selected_candidate"]["metadata"].get("scope_guard"),
                "val_log_predictive_gain": step2_res["selected_candidate"]["val_log_predictive_gain"],
            },
            "step3_live_gpt_cli_concept_lift": {
                "selected_action": step3_res["selected_candidate"]["action_type"],
                "source": step3_res["selected_candidate"]["metadata"].get("source"),
                "synthesized_expression": step3_res["selected_candidate"]["metadata"].get("expression"),
                "llm_rationale": step3_res["selected_candidate"]["metadata"].get("llm_rationale"),
                "empirical_lipschitz": step3_res["selected_candidate"]["metadata"].get("empirical_lipschitz"),
                "prompt_tokens": step3_res["selected_candidate"]["metadata"].get("prompt_tokens", 0),
                "completion_tokens": step3_res["selected_candidate"]["metadata"].get("completion_tokens", 0),
                "task_3a_val_rmse_before": round(rmse_3a_before, 5),
                "task_3a_val_rmse_after": round(rmse_3a_after, 5),
                "unseen_sister_3b_rmse_without_lift": round(sister_rmse_without_lifted_conduit, 5),
                "unseen_sister_3b_rmse_with_persisted_gpt_conduit": round(sister_rmse_with_persisted_gpt_conduit, 5),
            },
            "step4_nonmarkovian_memory": {
                "selected_action": step4_res["selected_candidate"]["action_type"],
                "temporal_tau": step4_res["selected_candidate"]["metadata"].get("temporal_tau"),
                "val_log_predictive_gain": step4_res["selected_candidate"]["val_log_predictive_gain"],
            },
            "step5_heteroscedastic_noise_guard": {
                "selected_action": step5_res["selected_candidate"]["action_type"],
                "residual_diagnostic_status": step5_res["decision_record"]["pre_action_state"]["residual_diagnostics"]["status"],
            },
        },
    }

    import shutil

    if hasattr(client, "stage_root_wsl") and Path(client.stage_root_wsl).is_dir():
        for sub_fp in sorted(Path(client.stage_root_wsl).rglob("*")):
            if sub_fp.is_file() and "m8_step_" in sub_fp.as_posix():
                dst_fp = cli_stage_dir / sub_fp.relative_to(Path(client.stage_root_wsl))
                dst_fp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sub_fp, dst_fp)

    (out_dir / "00_m8_gpt_cli_litmus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "01_decision_ledger.json").write_text(
        json.dumps([r.to_dict() for r in graph.decision_ledger], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "02_persistent_forest_snapshot.json").write_text(
        json.dumps(serialize_location_independent_forest(graph.forest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    num_hashed = write_sha256_sums(out_dir)
    print(json.dumps({"summary": summary, "sha256_locked_files": num_hashed}, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs/evidence/m8_0_variational_planner_gpt_cli_litmus_20261125",
    )
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    args = ap.parse_args()
    run_litmus(out_dir=args.out_dir, model=args.model, reasoning_effort=args.reasoning_effort)


if __name__ == "__main__":
    main()
