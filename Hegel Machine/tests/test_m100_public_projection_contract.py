"""Public behavior and receipt retention for M10's lazy projection helper."""
import hashlib
import importlib.util
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world as world


def test_dev_alpha_schema_checks_rate_invariance_without_prescribing_estimator(tmp_path):
    stage, *_ = world.build_m100_world_tasks(world.DEV_M100_WORLD_SPECS[0])
    setup = tmp_path / "public_setup.py"
    setup.write_text(stage.public_setup_py.replace("/holdout_task", str(tmp_path)))
    subprocess.run([sys.executable, "-I", str(setup)], check=True)
    schema = tmp_path / "validate_public_schema.py"
    candidate = tmp_path / "candidate.py"
    candidate.write_text("""
import numpy as np
def solve_batch(ds):
    n = len(ds["test"]["lab_id"])
    return {"estimated_taus": [1, 2], "hereditary_weights": np.ones(7),
            "predicted_strain": np.zeros((n,len(ds["eval_times"]))),
            "initial_rate_pred": np.full(n, np.mean(ds["train"]["ref_coupons"]))}
""")
    proc = subprocess.run([sys.executable, "-I", str(schema), str(candidate)], text=True, capture_output=True)
    assert proc.stdout, proc.stderr
    result = json.loads(proc.stdout)
    assert proc.returncode == 1 and result["errors"] == ["pooled_reference_coupon_invariance_failed"]
    check = result["pooled_reference_check"]
    assert check["swapped_ref_rows_per_split"] > 0
    assert check["ref_swap_strain_max_diff"] == 0 and check["ref_swap_rate_max_diff"] > check["tol"]
    assert "inverse_cell_count" not in schema.read_text()
    assert "vstack_train_test" not in proc.stdout


@pytest.mark.parametrize("failure", ["scientific", "timeout", "wrong_candidate"])
def test_projection_preserves_source_and_target_evidence(monkeypatch, failure):
    source = SimpleNamespace(task_id="source_dev", public_setup_py="make source")
    target = SimpleNamespace(task_id="target_dev", public_setup_py="make target")
    calls = []

    def propose(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "parsed": {"downstream_code": "def solve_downstream(item): return {}"},
                "call_uuid": f"test-{len(calls)}"}

    def gated(*, task, candidate_code, expected_dependency_sha256, require_pooled_coupon_check, **kwargs):
        assert require_pooled_coupon_check
        passed = task is source
        if not passed and failure == "timeout":
            raise subprocess.TimeoutExpired("dev probe", 1)
        candidate_sha = hashlib.sha256(candidate_code.encode()).hexdigest()
        bundle_sha = hashlib.sha256(b"/holdout_task/batch_solver.py\0" + bytes.fromhex(candidate_sha)).hexdigest()
        assets = {"/holdout_task/public_batch.pkl": task.task_id, **expected_dependency_sha256}
        diag = {
            "execution_status": "completed", "signed_by_host_probe_executor": True, "receipt_kind": "scientific_probe_receipt",
            "probe_protocol_version": "test", "probe_code_digest": "test", "candidate_bundle_sha256": bundle_sha,
            "public_setup_files_sha256": assets, "public_input_digest": task.task_id,
            "canonical_numeric_sha256": "numeric-" + task.task_id, "scientific_invariants_verified": passed,
            "measured_scientific_metrics": {"error": int(not passed)}, "kernel_version": "test",
        }
        return {"accepted": passed, "schema_valid": True, "setup_ok": True, "schema_exit_code": 0,
                "candidate_bundle_sha256": bundle_sha, "public_setup_files_sha256": assets,
                "setup_source_sha256": hashlib.sha256(task.public_setup_py.encode()).hexdigest(),
                "candidate_raw_sha256": "wrong" if not passed and failure == "wrong_candidate" else candidate_sha,
                "readonly_dependencies_sha256": expected_dependency_sha256, "scientific_probe_receipt": diag}

    monkeypatch.setattr(world, "execute_single_pre_llm_public_gate_probe_in_container", gated)
    args = dict(prior_task=source, current_task=target, u0_code="def calibrate_upstream(ds, pay): return ds",
                current_up_code="def up(ds, pay): return ds", current_up_entrypoint="up",
                current_pay={}, prior_pay={}, prior_monolithic_code="", prior_ro_deps={}, prior_ro_shas={},
                current_ro_deps={}, current_ro_shas={}, family="alpha", stage_subdir="dev", max_attempts=2)
    result = world.propose_and_verify_lazy_retroactive_downstream_projection(client=SimpleNamespace(call_structured=propose), **args)
    assert not result["accepted"] and result["attempts_used"] == (2 if failure == "scientific" else 1)
    assert len(result["probes"]) == 2 * result["attempts_used"]
    prior, current = result["probes"][:2]
    assert prior["accepted"] and prior["executed_public_input_sha256"] == "source_dev"
    assert prior["candidate_sha256"] == prior["expected_candidate_sha256"]
    assert prior["public_input_sha256"] == prior["executed_public_input_sha256"]
    assert prior["canonical_numeric_sha256"] == prior["executed_canonical_numeric_sha256"]
    assert prior["dependency_bundle_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert prior["component_code_sha256s"] == [hashlib.sha256(s.encode()).hexdigest()
                                             for s in (args["u0_code"], "def solve_downstream(item): return {}")]
    assert current["source_call_uuid"] == "test-1" and current["task_id"] == "target_dev"
    assert current["probe_wall_ms"] >= 0 and current["expected_candidate_sha256"] != prior["expected_candidate_sha256"]
    assert result["total_probe_wall_ms"] == round(sum(r["wall_ms"] for r in result["probes"]), 2)
    if failure == "scientific":
        assert current["failure_tier"] == "SCIENTIFIC_FAILURE" and result["status"] == "rejected"
        assert current["executed_canonical_numeric_sha256"] == "numeric-target_dev"
        assert current["scientific_probe_receipt"]["measured_scientific_metrics"]["error"] == 1
    else:
        assert result["status"] == "unresolved" and current["failure_tier"] == "INFRA_FAILURE"
        assert len(calls) == 1


def test_real_two_target_projection_receipts_create_two_distinct_witnesses():
    from hegel_machine.cognition import m9_8_blind_mismatch_and_selective_dependency_world as m98
    from hegel_machine.cognition import m9_9_autonomous_decomposition_and_interaction_withdrawal_world as m99
    from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import ImmutableBranchLineageController
    runner_path = Path(__file__).resolve().parents[1] / "scripts/run_m10_0_reliable_modular_amortization_experiment.py"
    module_spec = importlib.util.spec_from_file_location("m100_projection_runner", runner_path)
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)

    spec = world.DEV_M100_WORLD_SPECS[0]
    _, tasks, manifest, logs = world.build_m100_world_tasks(spec)
    assert [manifest[i]["causal_event"] for i in (0, 3)] == ["S01_decomp_d1", "S04_comb_u1_d1"]
    dn = m99.build_m99_reference_dev_downstream_solver_code("alpha", "solve_downstream", "D1")
    u0 = m98.build_m98_default_u0_transducer_code("alpha")
    u1 = m98.build_m98_reference_dev_upstream_u1_code("alpha")
    pay = {"gains": list(spec.v2_gains), "offsets": list(spec.v2_offsets)}
    source = m98.compose_m98_two_stage_solver_code(
        upstream_code=u0, upstream_entrypoint="calibrate_upstream", downstream_code=dn,
        downstream_entrypoint="solve_downstream", param_payload=pay)
    prefix = "/tmp/hegel_continual_archive/archive_apparatus_log.json"
    def log_context(index):
        value = json.dumps(logs[:index + 2], indent=2)
        return {prefix: value}, {prefix: hashlib.sha256(value.encode()).hexdigest()}
    source_logs, source_shas = log_context(0)
    target_logs, target_shas = log_context(3)
    proposal = SimpleNamespace(call_structured=lambda **kw: {"ok": True, "call_uuid": "dev-fixture",
                                                             "parsed": {"downstream_code": dn}})
    result = world.propose_and_verify_lazy_retroactive_downstream_projection(
        client=proposal, stage_subdir="dev", family="alpha", u0_code=u0, current_up_code=u1,
        current_up_entrypoint="calibrate_upstream", current_pay=pay, prior_monolithic_code=source,
        prior_task=tasks[0], prior_pay=pay, prior_ro_deps=source_logs, prior_ro_shas=source_shas,
        current_task=tasks[3], current_ro_deps=target_logs, current_ro_shas=target_shas, max_attempts=1)
    assert result["accepted"] and len(result["probes"]) == 2, result
    forest, _, _ = runner.m7a.build_blank_axiomatic_philosophical_forest()
    graph, _, registry = m98.initialize_m98_factored_mechanism_graph(spec=spec.to_m98_compat_spec(), forest=forest)
    binding = runner._register_learned_binding(
        registry=registry, world_id=spec.world_id, binding_id="projected_dev_d1", chain_id="alpha_dn", branch_tag="d1",
        forest_node_id=next(iter(forest.nodes)), code_txt=dn, entrypoint="solve_downstream", graph=graph,
        parent_ids=["constitutive_root_dn"], output_schema_keys=m98.expected_m98_public_output_keys("alpha"),
        applicability_guard={}, selection_priority=1, env_image_digest="dev_fixture")
    controller = ImmutableBranchLineageController(world_id="dev_projection_dn", family="alpha_dn", graph=graph, registry=registry)
    for idx, receipt in zip((0, 3), result["probes"]):
        item, raw_sha = m99.materialize_m99_public_item_and_sha(tasks[idx])
        witness = controller.record_tested_witness_from_probe_receipt(
            binding=binding, probe_receipt=receipt, public_input_sha256=raw_sha, evidence_origin="joint_task_probe",
            canonical_numeric_sha256=runner.compute_canonical_numeric_item_sha256(item),
            features=m98.compute_m98_downstream_observables(logs[:idx+2]))
        assert witness.is_verified_positive_witness() and witness.probe_id == receipt["probe_id"]
    assert len({w.public_input_sha256 for w in controller.witness_ledger}) == 2
