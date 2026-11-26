"""`M9.3.1` Deterministic Audit Closure Suite (`F01..F10`, AST Guard Lifter, 2-Process Determinism & `28/28` Regrade)."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any
from unittest.mock import patch

import numpy as np

from hegel_machine.cognition import (
    m6_4_executable_wormhole_library as m64,
    m6_8_b_prime_capability_bridge as m68,
    m6_frontier_real_benchmarks as m6f,
    m7_autogenous_forest_pretraining as m7a,
)
from hegel_machine.cognition.m9_3_online_lifecycle_benchmark import (
    M93_TASKS_BY_ID,
    evaluate_post_freeze_dev_seed_plugin_controls,
    m93_lifecycle_task_to_frontier_task as m93_task_to_frontier_task,
)
from hegel_machine.cognition.wormhole_forest import PhilosophicalForest

REPO_ROOT = Path(__file__).resolve().parents[1]
M93_EVIDENCE = REPO_ROOT / "docs/evidence/m9_3_online_source_freeze_lifecycle_20261126"
M931_EVIDENCE = REPO_ROOT / "docs/evidence/m9_3_1_audit_closure_and_regrade_20261126"


def test_f01_f02_component_bound_bifurcation_and_three_valued_guards() -> None:
    forest, archive, _, _ = m7a.load_autogenous_forest_bundle(
        M93_EVIDENCE / "m9_3_frozen_post_source_bundle.json"
    )
    snap_07 = deepcopy(archive["node_auto_07_control_panel_deconfounding"])
    comp = {
        "compiled_ok": True,
        "episode_id": "test_c1_explicit_parent",
        "module_stem": "autogenous_c1_viscoplastic_branch",
        "filename": "autogenous_c1_viscoplastic_branch.py",
        "function_names": ["overstress_rate"],
        "exported_apis": "overstress_rate(sigma, sigma_y)",
        "packaged_python_code": "import numpy as np\ndef overstress_rate(sigma, sigma_y):\n    return np.maximum(0.0, np.abs(sigma) - sigma_y) ** 2\n",
        "compiled_code_sha256": hashlib.sha256(b"c1_mod").hexdigest(),
        "synthetic_unit_test_code": "assert True\n",
        "synthetic_unit_test_sha256": hashlib.sha256(b"c1_ut").hexdigest(),
        "unit_test_receipt": {"passed": True},
        "sublated_methodology_principle": "Above yield stress, add Perzyna viscoplastic overstress.",
        "routed_signatures": list(archive["node_auto_10_hereditary_spectrum_identifi"]["signatures"]),
        "scope_preconditions": [],
        "regime_bifurcation": {
            "should_split_node": True,
            "regime_axis": "overstress_excess",
            "branch_guard_op": ">",
            "branch_guard_value": 0.0,
            "counter_regime_warning": "Linear Prony valid only when overstress_excess <= 0.",
        },
        "operator_pipeline_stages": [],
    }
    g_c1 = m7a.ObservablePreconditionGuard(
        guard_id="g_c1",
        observable_extractor="overstress_excess_max_history_and_hold",
        op=">",
        value=0.0,
        applies_to_component_id="linear_viscoelastic_memory",
    )
    res = m7a.consolidate_autogenous_episode_into_forest(
        forest=forest,
        archive=archive,
        compilation=comp,
        nogoods_recorded=[],
        allow_horizontal_wormholes=False,
        source_node_id="node_auto_10_hereditary_spectrum_identifi",
        source_version=1,
        violated_component_id="linear_viscoelastic_memory",
        preserved_component_ids=["reference_panel_calibration", "variable_step_gap_recursion"],
        guard_spec_override=g_c1,
    )
    child_id = res["target_node_id"]
    assert archive[child_id]["parent_node_id"] == "node_auto_10_hereditary_spectrum_identifi"
    assert archive["node_auto_07_control_panel_deconfounding"] == snap_07
    assert any(
        p.get("counterpart_child_id") == child_id
        for p in forest.nodes["node_auto_10_hereditary_spectrum_identifi"]["claim_contract"]["preconditions"]
    )

    # Guard must keep C1 active when current hold is unloaded (`0.3 <= 1.5`) but historical loading exceeded yield (`2.1 > 1.5`)!
    assert (
        m7a.evaluate_observable_precondition_guard(
            g_c1, {"yield_stress": 1.5, "test": {"past_stress": [[0.2, 2.1]], "stress_hold_0": [0.3]}}
        )["state"]
        == "TRUE"
    )
    # Missing stress arrays -> UNKNOWN (never silent FALSE!)
    assert m7a.evaluate_observable_precondition_guard(g_c1, {"empty": 1})["state"] == "UNKNOWN"

    # Additive 2D potential `V(x,y) = a(x) + b(y)` has `0.0` non-separable residual!
    g_c2 = m7a.ObservablePreconditionGuard(
        guard_id="g_c2",
        observable_extractor="nonseparable_2d_additive_residual_ratio",
        op=">",
        value=1e-6,
    )
    v_add = np.arange(6, dtype=float)[:, None] + 2.0 * np.arange(5, dtype=float)[None, :] + 3.0
    r_add = m7a.evaluate_observable_precondition_guard(g_c2, {"V_pocket_channel_b": v_add})
    assert r_add["state"] == "FALSE" and float(r_add["statistic"]) <= 1e-12


def test_f03_public_schema_receipt_never_promotes_scientific_nogoods() -> None:
    fp = REPO_ROOT / "src/hegel_machine/cognition/m6_frontier_real_benchmarks.py"
    tree = ast.parse(fp.read_text(encoding="utf-8"))
    block = next(
        n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "pending_nogood_resolutions and nogoods_recorded"
    )

    # 1. PublicSchemaReceipt (`schema_valid=True`) CANNOT promote `numerical_residual` or `physical_or_causal_mechanism`,
    #    ONLY `schema_contract`!
    ng_sci = {
        "nogood_id": "ng_sci",
        "risk_kind": "numerical_residual",
        "assumption_refuted": "active-set box-simplex solver cycled with duality gap 1e-2",
        "scope": "box_simplex_kkt",
        "resolved": False,
    }
    ng_schema = {
        "nogood_id": "ng_fmt",
        "risk_kind": "schema_contract",
        "assumption_refuted": "missing kkt_certificate key in output dict",
        "scope": "public_delivery_schema",
        "resolved": False,
    }
    ns_schema_only = {
        "deepcopy": deepcopy,
        "json": json,
        "_HOST_INJECTED_PROBE_RECEIPT": {
            "signed_by_host_probe_executor": True,
            "receipt_kind": "public_schema_receipt",
            "candidate_bundle_sha256": "sha_cand_1",
            "verified_scope": "public_delivery_schema",
            "schema_valid": True,
            "result": "supported_on_public_schema_validator",
        },
        "pending_nogood_resolutions": [{"proposed_turn": 3, "resolves_ids": ["ng_sci", "ng_fmt"], "proposed_by_nogood_id": None}],
        "nogoods_recorded": [ng_sci, ng_schema],
        "nogood_resolution_events": [],
        "turn_idx": 4,
        "current_candidate_sha256": "sha_cand_1",
        "cmd_str": "python3 -I /holdout_task/validate_public_schema.py",
        "ex_res": {"stdout": '{"schema_valid": true}'},
    }
    exec(compile(ast.Module(body=[deepcopy(block)], type_ignores=[]), str(fp), "exec"), ns_schema_only)
    assert ng_sci["resolved"] is False
    assert ng_sci["probe_support_status"] == "supported_on_public_schema_only_scientific_witness_pending"
    assert ng_schema["resolved"] is True

    # 2. ScientificProbeReceipt with EMPTY `verified_scope` is rejected (never back-filled from `ng_scope`!)
    ng_empty_scope = {
        "nogood_id": "ng_sci2",
        "risk_kind": "numerical_residual",
        "assumption_refuted": "high residual",
        "scope": "box_simplex_kkt",
        "resolved": False,
    }
    ns_missing_scope = {
        "deepcopy": deepcopy,
        "json": json,
        "_HOST_INJECTED_PROBE_RECEIPT": {
            "signed_by_host_probe_executor": True,
            "receipt_kind": "scientific_probe_receipt",
            "candidate_bundle_sha256": "sha_cand_2",
            "verified_scope": "",
            "scientific_invariants_verified": True,
        },
        "pending_nogood_resolutions": [{"proposed_turn": 2, "resolves_ids": ["ng_sci2"], "proposed_by_nogood_id": None}],
        "nogoods_recorded": [ng_empty_scope],
        "nogood_resolution_events": [],
        "turn_idx": 3,
        "current_candidate_sha256": "sha_cand_2",
        "cmd_str": "python3 test.py",
        "ex_res": {"stdout": "ok"},
    }
    exec(compile(ast.Module(body=[deepcopy(block)], type_ignores=[]), str(fp), "exec"), ns_missing_scope)
    assert ng_empty_scope["resolved"] is False
    assert ng_empty_scope["probe_support_status"] == "host_probe_missing_verified_scope"


def test_f04_f05_regrade_and_funnel_receipts_match_sealed_m93() -> None:
    regrade = json.loads((M931_EVIDENCE / "m9_3_28of28_tightened_handoff_regrade.json").read_text(encoding="utf-8"))
    assert regrade["sealed_m9_3_episodes_regraded"] == 28
    assert regrade["sealed_m9_3_episodes_passed_tightened_contract"] == 28
    adv = regrade["adversarial_negative_controls"]
    assert adv["src01_zeros7_hereditary_weights_rejected"] is True
    assert adv["rec01_R_nan1_hereditary_weights_rejected"] is True
    assert adv["suc01_T_perturbed_secant_handoff_rejected"] is True
    assert adv["src02_weights_vs_kkt_certificate_mismatch_rejected"] is True

    funnel = json.loads((M931_EVIDENCE / "m9_3_source_aware_l1_l2_adoption_funnel.json").read_text(encoding="utf-8"))
    e_sum = funnel["executable_arm_E_summary"]
    assert e_sum["new_source_L2_dev_call_count"] == 1021
    assert e_sum["new_source_L2_exact_ast_retained_and_reachable_episodes"] == 1
    assert e_sum["new_source_L1_dev_call_count"] == 0
    assert e_sum["prior_v7_L1_dev_call_count"] == 3


def test_f06_f10_uuid_attempt_whitelist_and_public_asset_index() -> None:
    src01_ft = m93_task_to_frontier_task(M93_TASKS_BY_ID["src01__variable_step_prony_spectrum_and_hereditary_creep_source"])
    with tempfile.TemporaryDirectory(prefix="m931_f06_f10_") as td:
        td_p = Path(td)
        stage_root = td_p / "stage"
        ep_subdir = f"run_f06__mock-model__Hegel_M6_9_Antithesis_LiveGPT__{src01_ft.task_id}".replace(":", "_")
        (stage_root / ep_subdir).mkdir(parents=True, exist_ok=True)
        # Plant a leftover/unmatched `attempts_999_leftover.json` in the same stage_subdir!
        (stage_root / ep_subdir / "attempts_999_leftover.json").write_text(
            json.dumps({"call_uuid": "foreign_uuid", "request_sha256": "foreign_sha", "attempt_receipts": [{"prompt_tokens": 9999}]}),
            encoding="utf-8",
        )

        def _mock_cli(cmd: list[str], **_kw: Any) -> subprocess.CompletedProcess[str]:
            resp_p = Path(cmd[cmd.index("--output-last-message") + 1])
            resp_p.write_text(
                json.dumps(
                    {
                        "action_type": "submit_completed_task",
                        "bash_command": None,
                        "final_answer": "done",
                        "active_hypothesis_and_worst_slice_metric": "ok",
                        "nogood_update": {
                            "has_new_nogood": False,
                            "nogood_rung": None,
                            "assumption_refuted": None,
                            "empirical_witness": None,
                            "causal_blanket_diagnosis": None,
                            "sublated_pivot": None,
                            "risk_kind": None,
                            "resolves_nogood_ids": [],
                        },
                        "considered_alternatives": [],
                        "rationale": "test",
                    }
                ),
                encoding="utf-8",
            )
            usage_ln = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 120, "cached_input_tokens": 0, "output_tokens": 30}})
            return subprocess.CompletedProcess(cmd, 0, usage_ln + "\n", "")

        codex_info = {
            "available": True,
            "auth_mode": "chatgpt",
            "provider_backend": "mock_codex",
            "executable": "mock",
            "is_windows_exe_under_wsl": False,
            "stage_root_wsl": stage_root,
            "stage_root_cli_str": str(stage_root),
        }
        ep_out = td_p / "ep_out"
        with patch.object(m6f.subprocess, "run", side_effect=_mock_cli):
            # Note: only `FrontierCodexClient` calls `m6f.subprocess.run` when container ops use `subprocess.run`?
            # Wait — `m6f.subprocess` is shared with `start_detached_container`, so patch `FrontierCodexClient.call_structured`
            # OR run `FrontierCodexClient.call_structured` directly on `_mock_cli` and pass its real return to `run_frontier_episode`!
            pass

        real_cli = m6f.FrontierCodexClient(codex_info=codex_info, model="mock-model")
        with patch.object(m6f.subprocess, "run", side_effect=_mock_cli):
            canned_res = real_cli.call_structured(
                system_prompt="s",
                conversation_turns=[{"role": "user", "content": "u"}],
                output_schema={"type": "object"},
                stage_subdir=ep_subdir,
            )
        assert canned_res["call_uuid"] and canned_res["attempts_ledger_filename"].startswith("attempts_001_")

        with patch.object(m6f.FrontierCodexClient, "call_structured", return_value=canned_res):
            summary = m6f.run_frontier_episode(
                src01_ft,
                "Hegel_M6_9_Antithesis_LiveGPT",
                model="mock-model",
                codex_info=codex_info,
                forest=PhilosophicalForest(),
                forest_archive={},
                hippo_idx=None,
                episode_out_dir=ep_out,
                run_id="run_f06",
                max_turns=1,
                defer_official_verifier=True,
            )
        # F10: `/holdout_task/public_asset_index.json` materialized on boot!
        assert summary["public_setup_receipt"]["public_asset_index_path"] == "/holdout_task/public_asset_index.json"
        assert "/holdout_task/public_asset_index.json" in summary["public_setup_receipt"]["files_sha256"]
        # F06: ONLY the whitelisted UUID attempt file is copied to `01b_provider_attempts`; leftover file is quarantined!
        copied = sorted(p.name for p in (ep_out / "01b_provider_attempts").glob("attempts_*.json"))
        assert copied == [canned_res["attempts_ledger_filename"]]
        assert summary["provider_attempt_accounting"]["quarantined_unmatched_attempt_files"] == ["attempts_999_leftover.json"]


def test_f08_cold_start_qualification_context_and_ast_guard_separation() -> None:
    _, v7_arch, _, _ = m7a.load_autogenous_forest_bundle(
        REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
    )
    sample_v7_l1 = deepcopy(
        next(
            m
            for _, r in m7a.iter_live_archive_nodes(v7_arch)
            for m in r.get("autogenous_wormhole_modules", [])
            if str(m.get("filename", "")).startswith("autogenous_prim_")
        )
    )
    m64.clear_verified_l1_env_digests_for_testing()
    ok_cold, rsn_cold, _ = m64.verify_l1_module_qualification_record(sample_v7_l1, require_4tuple_protocol=True)
    assert ok_cold is False and rsn_cold == "runtime_env_not_initialized"

    ctx = m64.ensure_runtime_qualification_context(sandbox_cid=None)
    assert ctx["initialized"] is True
    ok_warm, rsn_warm, _ = m64.verify_l1_module_qualification_record(sample_v7_l1, require_4tuple_protocol=True)
    assert ok_warm is True, rsn_warm

    # AST guard separation + anti-unification + F09 dev-seed non-overlap
    src_g = "def f(x: float) -> float:\n    if x <= -999.0:\n        raise ValueError('bad')\n    return 3.0 * x + 2.0\n"
    diag_g: dict[str, Any] = {}
    progs = m68.lift_python_scientific_expressions_to_programs(src_g, funnel_diagnostics_out=diag_g)
    assert len(progs) == 1 and diag_g["lifted_to_program_tree"] == 1
    assert "f" in diag_g["separated_input_guards_by_fn"]

    dev_ctrl = evaluate_post_freeze_dev_seed_plugin_controls()
    assert dev_ctrl["direct_locked_float_plugin_fails_all_successors"] is True
    assert set(dev_ctrl["dev_seeds_used"]).isdisjoint({9301, 9311, 9321, 9331, 9341, 9351, 9391, 9392, 99501, 99502, 99601, 99602})
