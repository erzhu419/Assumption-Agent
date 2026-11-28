"""Offline M10 CLI coverage: no provider or development/confirmation answer is used."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_m10_0_reliable_modular_amortization_experiment as runner


def test_condition_survives_process_boundary_and_ablation_abstains_on_missing_prefix(tmp_path, monkeypatch):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    monkeypatch.setattr(runner, "_git_and_source_tree_state", lambda: ("abc", False, "python-tree"))
    monkeypatch.setattr(runner, "ensure_clean_scipy_sandbox_image", lambda: None)
    monkeypatch.setattr(runner, "_docker_image_digest", lambda image: "sha256:test")
    monkeypatch.setattr(runner.m64, "ensure_runtime_qualification_context", lambda **kw: None)
    monkeypatch.setattr(runner.m6f, "discover_chatgpt_subscription_codex_cli", lambda: {"available": True, "auth_mode": "chatgpt"})
    monkeypatch.setattr(runner.m7a, "load_autogenous_forest_bundle", lambda fp: (SimpleNamespace(nodes={"a": 1}), {}, 0, 0))
    worlds = []
    def build(spec, **opts):
        assert spec.split == "dev"
        worlds.append((spec.world_id, opts))
        n = 7 if opts.get("grid_size") == "2x3" else 10
        return None, (None,) * n, [{"causal_event": str(i)} for i in range(n)], [None] * (n + 1)
    monkeypatch.setattr(runner, "build_m100_world_tasks", build)

    def source(spec, out_dir, **kwargs):
        result = {"admitted": True, "strict_pass": False}
        write(out_dir / "stage_S_shared_sources" / spec.world_id / "00_stage_S_summary.json", result)
        return dict(summary=result, source_monolithic_code="", source_archive_entry={}, source_pub_item={}, source_pub_sha256="sha")

    def stream(spec, arm, stream_tasks, stream_out_dir, **kw):
        n = len(stream_tasks)
        z = dict(world_id=spec.world_id, arm=arm, episode_count=n, strict_passes=0, zero_llm_episodes_total=0,
                 zero_turn_passed_closures=0, total_turns_used=0, separability_withdrawal_receipts=[],
                 episodes_meta=[{"strict_pass": False} for _ in stream_tasks])
        for field in ("pre_llm_probe", "post_llm_probe", "localization_probe", "decomposition_probe", "coverage_probe", "official_verifier"):
            z[f"total_{field}_wall_ms"] = 0
        write(stream_out_dir / "00_stream_summary.json", z)
        return z
    monkeypatch.setattr(runner, "_run_stage_s_shared_source", source)
    monkeypatch.setattr(runner, "_run_one_m100_world_arm_stream", stream)
    args = ["m10", "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", args + ["--grid-size", "2x3", "--order-mode", "coupled_first"])
    original_call = runner.m6f.FrontierCodexClient.call_structured
    runner.main()
    assert original_call is runner.m6f.FrontierCodexClient.call_structured
    start_bytes = (tmp_path/"execution_manifest.json").read_bytes()
    assert len(worlds) == 4 and all(choice == {"grid_size": "2x3", "order_mode": "coupled_first"} for _, choice in worlds)
    ablation = json.loads((tmp_path/"m10_0_step08_routing_ablation.json").read_bytes())
    assert len(ablation["rows"]) == 24
    assert all(row["prefix_checkpoint"] == "graph_ckpt_after_step_05.json" and row["private_strict_pass"] is None for row in ablation["rows"])
    monkeypatch.setattr(runner.m6f, "discover_chatgpt_subscription_codex_cli", lambda: pytest.fail("finalizer accessed provider"))
    monkeypatch.setattr(sys, "argv", args + ["--finalize-existing"])
    runner.main()
    assert (tmp_path/"execution_manifest.json").read_bytes() == start_bytes
    assert worlds[:4] == worlds[4:]
    summary = json.loads(next(tmp_path.glob("m10*summary.json")).read_bytes())
    assert summary["independent_source_cluster_count"] == 4 and summary["total_logical_calls"] == 0
    assert summary["development_condition"]["effective_grid"] == "2x3"
    curves = next(iter(summary["cumulative_amortization_curves_and_K_star"].values()))
    assert len(curves["by_step_K"]) == 7 and curves["by_step_K"][-1]["total_episodes_at_K"] == 28
    monkeypatch.setattr(sys, "argv", args + ["--finalize-existing", "--grid-size", "3x3"])
    with pytest.raises(SystemExit):
        runner.main()


@pytest.mark.parametrize("public_ok,private_ok", [(True, False), (False, True)])
def test_stage_s_private_score_is_only_a_report(tmp_path, monkeypatch, public_ok, private_ok):
    spec = runner.DEV_M100_WORLD_SPECS[0]
    task = SimpleNamespace(short_name="s0", task_id="test-source", declared_artifacts=("solver.py",))
    monkeypatch.setattr(runner, "_inject_equal_info_archive_into_m100_task", lambda *a: (task, ""))
    monkeypatch.setattr(runner, "m100_task_to_frontier_task", lambda *a, **kw: task)
    monkeypatch.setattr(runner, "HippoRAGWormholeIndex", lambda a: None)
    monkeypatch.setattr(runner.m6f, "run_frontier_episode", lambda *a, **kw: {"turns_used": 0})
    monkeypatch.setattr(runner, "_extract_submitted_files", lambda p: {"batch_solver.py": "solver body"})
    monkeypatch.setattr(runner, "_execute_bound_gate", lambda **kw: {"accepted": public_ok})
    monkeypatch.setattr(runner, "joint_scientific_receipt_matches", lambda *a: public_ok)
    monkeypatch.setattr(runner, "materialize_m99_public_item_and_sha", lambda t: ({}, "sandbox-sha"))
    monkeypatch.setattr(runner, "grade_m94_artifact_map", lambda *a, **kw: {"strict_pass": private_ok})
    result = runner._run_stage_s_shared_source(
        spec=spec, s_task=task, s0_prefix_logs=[], base_v7_forest={}, base_v7_archive={},
        codex_info={}, model="offline", reasoning_effort="none", out_dir=tmp_path, run_id="local", max_turns=1)
    assert result["summary"]["admitted"] is public_ok
    assert result["summary"]["strict_pass"] is private_ok
    assert result["summary"]["public_input_sha256"] == "sandbox-sha"
    stage_dir = tmp_path / "stage_S_shared_sources" / spec.world_id
    assert json.loads((stage_dir / "stage_s_public_gate.json").read_bytes())["accepted"] is public_ok
    assert json.loads((stage_dir / "03_official_verifier_report.json").read_bytes())["strict_pass"] is private_ok
