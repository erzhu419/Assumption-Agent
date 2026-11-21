"""Pre-flight verification tests for the 20-task Frontier Real Benchmark Harness (`TB-Science` + `BioMysteryBench`)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.m6_frontier_real_benchmarks import (
    build_m6_frontier_general_forest,
    compute_filesystem_diff,
    consolidate_wave1_into_m6_forest,
    exec_container_bash,
    extract_public_task_signatures,
    load_frontier_scout_20_tasks,
    run_biomystery_verifier,
    run_tb_separate_verifier,
    snapshot_container_files,
    start_detached_container,
    stop_container_safely,
)


def _docker_img_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def test_load_all_20_tasks_and_verify_docker_images() -> None:
    waves = load_frontier_scout_20_tasks()
    assert len(waves["wave_1"]) == 10
    assert len(waves["wave_2"]) == 10

    all_tasks = waves["wave_1"] + waves["wave_2"]
    tb_cnt = sum(1 for t in all_tasks if t.benchmark == "Terminal-Bench-Science-0.1")
    bm_cnt = sum(1 for t in all_tasks if t.benchmark == "BioMysteryBench-Preview")
    assert tb_cnt == 15 and bm_cnt == 5

    for t in all_tasks:
        assert len(t.instruction_text.strip()) > 40
        assert _docker_img_exists(t.agent_image_tag), f"Missing agent image {t.agent_image_tag}"
        if t.benchmark == "Terminal-Bench-Science-0.1":
            assert t.verifier_image_tag and _docker_img_exists(t.verifier_image_tag), f"Missing verifier {t.verifier_image_tag}"
            assert len(t.declared_artifacts) >= 1
        else:
            assert t.biomystery_data_dir is not None and t.biomystery_data_dir.is_dir()
            assert any(t.biomystery_data_dir.iterdir())
            assert t.biomystery_answer_rubric and len(t.biomystery_answer_rubric) > 0


def test_m6_general_forest_retrieval_and_wave1_consolidation() -> None:
    waves = load_frontier_scout_20_tasks()
    forest, archive, meta_dag = build_m6_frontier_general_forest()
    assert len(forest.nodes) == 12
    idx = HippoRAGWormholeIndex(forest)

    gw_task = next(t for t in waves["wave_2"] if t.short_name == "guided-wave-localization")
    sigs = extract_public_task_signatures(gw_task)
    rec = idx.retrieve(sigs, task_view={"goal_kind": "scientific_terminal_discovery", "tolerance": 1.0}, top_k=3)
    rec_ids = [c["node_id"] for c in rec["candidates"]]
    assert any("shm" in nid or "wave" in nid for nid in rec_ids)

    cons = consolidate_wave1_into_m6_forest(
        wave1_m6_summaries=[
            {
                "task_id": "tb__baseline-free-localization",
                "short_name": "baseline-free-localization",
                "wormhole_family": "fam_guided_wave_shm_localization",
                "strict_pass": True,
                "failure_taxonomy": "PASSED_ALL_CHECKS",
                "nogoods_recorded": [
                    {
                        "assumption_refuted": "Constant isotropic phase velocity subtraction",
                        "sublated_pivot": "Distance-binned Hilbert envelope arrival delay triangulation",
                    }
                ],
            }
        ],
        forest=forest,
        archive=archive,
        meta_dag=meta_dag,
    )
    assert cons["bayes_ball_mechanism_channel_active"] is True
    assert cons["bayes_ball_collider_channel_blocked"] is True
    assert len(archive["node_F04_guided_wave_physics_transfer"]["wave1_consolidated_nogoods"]) == 1
    assert len(archive["node_F05_digital_twin_and_baseline_free_wave"]["wave1_consolidated_nogoods"]) == 1


def test_airgap_container_fs_diff_and_noop_separate_verifier() -> None:
    waves = load_frontier_scout_20_tasks()
    ode_task = next(t for t in waves["wave_1"] if t.short_name == "ode-law-discovery")
    bm_task = next(t for t in waves["wave_1"] if t.short_name == "hb020")

    cid = None
    try:
        cid = start_detached_container(ode_task.agent_image_tag, memory_limit="2g", cpus="2")
        # 1. Verify strict air-gap: `/tests` and `/solution` must NOT exist in agent container!
        airgap = exec_container_bash(cid, "test ! -e /tests && test ! -e /solution && echo AIRGAP_OK", timeout_sec=15)
        assert airgap["exit_code"] == 0 and "AIRGAP_OK" in airgap["stdout"]

        # 2. Verify filesystem diff tracking
        s0 = snapshot_container_files(cid)
        exec_container_bash(cid, "mkdir -p /results && echo '{\"hello\": 1}' > /results/dummy_check.json", timeout_sec=15)
        s1 = snapshot_container_files(cid)
        diff = compute_filesystem_diff(s0, s1)
        assert any(x["path"] == "/results/dummy_check.json" for x in diff["created"])

        # 3. Verify official Harbor Separate-Verifier returns reward == 0.0 cleanly on incomplete submission!
        v_rep = run_tb_separate_verifier(ode_task, cid)
        assert v_rep["strict_pass"] is False
        assert v_rep["reward"] == 0.0
        assert v_rep["artifact_audit"]["all_present"] is False
    finally:
        stop_container_safely(cid)

    # 4. Verify BioMystery empty answer fails cleanly
    bm_rep = run_biomystery_verifier(bm_task, "", codex_info=None)
    assert bm_rep["strict_pass"] is False
    assert bm_rep["reward"] == 0.0
