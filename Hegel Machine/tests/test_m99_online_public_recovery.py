"""Prospective public routing, tested on archived inputs and synthetic cost/order controls."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from hegel_machine.cognition import m9_4_1_shared_verification_kernel as kernel

from hegel_machine.cognition import m9_8_blind_mismatch_and_selective_dependency_world as m98
from hegel_machine.cognition import m9_9_autonomous_decomposition_and_interaction_withdrawal_world as m99

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m99_public_recovery", ROOT / "scripts/run_m9_9_autonomous_decomposition_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def measured(passed):
    return {"accepted": passed, "schema_valid": True, "schema_exit_code": 0,
            "scientific_probe_receipt": {"execution_status": "completed", "scientific_invariants_verified": passed,
                                         "errors": [] if passed else ["public_constraint_failed"]}}


def initial_trial(locus):
    return dict(accepted_pre_llm=False, probe_result=measured(False), locus=locus, pre_llm_probes_used=1,
                probe_wall_ms=2.0, localization_probe_wall_ms=3.0, localization_probes_used=1,
                candidate_code=None, selected_upstream_id="u0", selected_downstream_code="def down(x): return x",
                selected_downstream_ep="down", conflict_locus_diagnosis={"execution_status": "completed",
                "component_probe_executed": True, "locus": locus})


def recover(trial, **kwargs):
    fields = dict(family="alpha", task=SimpleNamespace(task_id="target"), public_item={}, numeric_sha="current",
                  dependencies={}, dependency_shas={}, payload={}, primary_code="original", upstream_candidates=[],
                  selected_up=None, compose_options={}, prior_entries=[], archive_context={}, schema=["x"])
    runner._bounded_public_recovery(trial=trial, **{**fields, **kwargs})


@pytest.mark.parametrize("outcomes,accept,mode", [
    ([True], True, "archive_raw"),
    ([False, True, True], True, "archive_projection_target"),
    ([False, False], False, None),
    ([False, True, False], False, None),
])
def test_archive_public_witnesses_and_charges(monkeypatch, outcomes, accept, mode):
    trial = initial_trial("downstream_constitutive")
    archive = runner.make_continual_archive_entry(
        task_id="source", short_name="past", canonical_algorithm_code="def solve_batch(x): return x",
        canonical_entrypoint="solve_batch", output_schema_keys=["x"], public_qualification_status="public_gate_failed")
    archive["module_projection_status"] = "unprojected"
    calls = []

    def gate(**kw):
        calls.append(kw["task"].task_id)
        return measured(outcomes[len(calls)-1])

    monkeypatch.setattr(runner, "execute_single_pre_llm_public_gate_probe_in_container", gate)
    recover(trial, prior_entries=[archive], selected_up=("def up(x): return x", "up", None),
            archive_context={"source": (SimpleNamespace(task_id="source"), "prior", {}, {}, {})})
    assert calls == ["target", "source", "target"][:len(outcomes)]
    assert trial["pre_llm_probes_used"] == 1 + len(calls)
    assert len(trial["recovery_attempts"]) == len(calls)
    assert trial["accepted_pre_llm"] is accept and trial.get("recovery_mode") == mode
    assert (archive["module_projection_status"] == "verified_on_source_and_target") == (mode == "archive_projection_target")
    assert archive["public_qualification_status"] == "public_gate_failed"


def test_missing_initial_science_does_not_route(monkeypatch):
    trial = initial_trial("upstream_transducer")
    trial["probe_result"]["scientific_probe_receipt"] = None
    recover(trial, upstream_candidates=[("u1", "def up(x): return x", "up", None)])
    assert "recovery_attempts" not in trial and trial["pre_llm_probes_used"] == 1


@pytest.mark.parametrize("cluster,up_idx,dn_idx,passes", [("c02", 1, 2, False), ("c04", 2, 3, True)])
def test_online_alpha_gate_and_alternate_on_exposed_e7(cluster, up_idx, dn_idx, passes):
    wid = "alpha_m99_" + cluster
    world = next(s for s in m99.CONFIRMATION_M99_WORLD_SPECS if s.world_id == wid)
    _, tasks, _, logs = m99.build_m99_world_tasks(world)
    task = tasks[6]
    item, _ = m99.materialize_m99_public_item_and_sha(task)
    graph = runner.PersistentMechanismGraph.load_checkpoint(
        runner.DEFAULT_OUT_DIR / "continual_streams" / wid / "Hegel_AutonomousInteractionGraph_Online/graph_ckpt_after_step_06.json")
    registry = runner.M94BindingRegistry(graph)
    compat = world.to_m98_compat_spec()
    m99.update_m99_calibration_snapshot_and_graph(spec=compat, episode_id="public_recovery_dev_replay",
                                                  public_item=item, graph=graph, registry=registry)
    pay = m98.assemble_m98_graph_diagonal_payload(compat, graph)
    up = registry.bindings[f"bind_{wid}_up_b0{up_idx}"]
    dn = registry.bindings[f"bind_{wid}_dn_b0{dn_idx}"]
    ro = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(logs[:8], indent=2)}
    shas = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in ro.items()}
    if passes:
        first = registry.bindings[f"bind_{wid}_up_b01"]
        original = m98.compose_m98_two_stage_solver_code(
            upstream_code=first.canonical_algorithm_code, upstream_entrypoint=first.canonical_entrypoint,
            downstream_code=dn.canonical_algorithm_code, downstream_entrypoint=dn.canonical_entrypoint, param_payload=pay)
        primary = m99.execute_single_pre_llm_public_gate_probe_in_container(
            task=task, candidate_code=original, readonly_dependencies=ro, expected_dependency_sha256=shas)
        direct = m98.diagnose_m98_factored_conflict_locus(
            "alpha", item, chosen_up_code=first.canonical_algorithm_code, chosen_up_entrypoint=first.canonical_entrypoint,
            diag_payload=pay, force_direct_component_probe=True)
        assert not primary["accepted"] and direct["execution_status"] == "completed"
        assert direct["chosen_up_holdout_rmse"] > 0.5 and direct["locus"] == "upstream_transducer"
        trial = initial_trial(direct["locus"])
        trial.update(probe_result=primary, conflict_locus_diagnosis=direct, selected_upstream_id=first.binding_id,
                     selected_downstream_code=dn.canonical_algorithm_code, selected_downstream_ep=dn.canonical_entrypoint)
        recover(trial, task=task, public_item=item, numeric_sha=runner.compute_canonical_numeric_item_sha256(item),
                dependencies=ro, dependency_shas=shas, payload=pay,
                upstream_candidates=[(up.binding_id, up.canonical_algorithm_code, up.canonical_entrypoint, up)],
                compose_options=dict(downstream_binding=dn, graph=graph, registry=registry))
        assert trial["accepted_pre_llm"] and trial["recovery_mode"] == "alternative_upstream_joint"
        assert trial["pre_llm_probes_used"] == trial["localization_probes_used"] == 2
        assert trial["recovery_attempts"][0]["component_probe"]["chosen_up_holdout_rmse"] < 0.0001
        result = trial["probe_result"]
    else:
        candidate = m98.compose_m98_two_stage_solver_code(
            upstream_code=up.canonical_algorithm_code, upstream_entrypoint=up.canonical_entrypoint,
            downstream_code=dn.canonical_algorithm_code, downstream_entrypoint=dn.canonical_entrypoint, param_payload=pay)
        result = m99.execute_single_pre_llm_public_gate_probe_in_container(
            task=task, candidate_code=candidate, readonly_dependencies=ro, expected_dependency_sha256=shas)
    receipt = result["scientific_probe_receipt"]
    swap = receipt["measured_scientific_metrics"]["pooled_ref_coupon_check"]
    assert receipt["canonical_numeric_sha256"] == runner.compute_canonical_numeric_item_sha256(item)
    assert receipt["execution_status"] == "completed" and swap["swapped_ref_rows_per_split"] > 0
    assert result["accepted"] is passes and swap["valid"] is passes
    assert (swap["ref_swap_strain_max_diff"] <= swap["tol"]) is passes


def test_public_reference_weighted_imputation_on_independent_unbalanced_panel():
    # The six rows have two different lab calibrations. Three observations share
    # the first cell, so pooling observations uniformly gives a different fill.
    labs = np.array([0, 0, 0, 0, 1, 1])
    strata = np.array([0, 0, 0, 1, 0, 1])
    refs = np.array([-1, 1, -1, 1, 8, 12], dtype=float)[:, None]
    values = np.array([1, 3, 5, 9, 12, np.nan], dtype=float)[:, None]
    expected = np.array([1, 3, 5, 9, 1, 13/3])
    assert expected[-1] != np.mean(expected[:-1])

    def panel(indices):
        return dict(lab_id=labs[indices], stratum=strata[indices], ref_coupons=refs[indices],
                    coupon_features_raw=values[indices])

    for train, test in (([0, 1, 4], [2, 3, 5]), ([4, 1, 0], [5, 3, 2]), ([2, 3, 5], [0, 1, 4])):
        x_train, x_test, mask_tr, mask_te = kernel.standardize_reference_coupon_panel(panel(train), panel(test))
        np.testing.assert_allclose(x_train[:, 0], expected[train])
        np.testing.assert_allclose(x_test[:, 0], expected[test])
        np.testing.assert_array_equal(mask_tr, np.isfinite(values[train, 0]))
        np.testing.assert_array_equal(mask_te, np.isfinite(values[test, 0]))


def test_reuse_projection_charges_only_new_target(monkeypatch):
    archive = runner.make_continual_archive_entry(
        task_id="source", short_name="s", canonical_entrypoint="solve_batch", canonical_algorithm_code="def solve_batch(x): return x",
        output_schema_keys=["x"])
    archive["module_projection_status"] = "unprojected"
    params = dict(prior_entries=[archive], selected_up=("def up(x): return x", "up", None),
                  archive_context={"source": (SimpleNamespace(task_id="source"), "prior_numeric", {}, {}, {})})
    outcomes = iter([False, True, True, True])
    monkeypatch.setattr(runner, "execute_single_pre_llm_public_gate_probe_in_container", lambda **kw: measured(next(outcomes)))
    initial = initial_trial("downstream_constitutive")
    recover(initial, **params)
    next_trial = initial_trial("downstream_constitutive")
    recover(next_trial, task=SimpleNamespace(task_id="next"), numeric_sha="next_numeric", **params)
    assert next_trial["accepted_pre_llm"] and next_trial["pre_llm_probes_used"] == 2
    assert [a["kind"] for a in next_trial["recovery_attempts"]] == ["archive_projection_target"]
    assert next_trial["projection_source_receipt"]["qualifying_target_task_id"] == "target"


def test_coverage_missing_scientific_result_is_unknown(monkeypatch):
    task = SimpleNamespace(public_setup_py="setup")
    monkeypatch.setattr(runner, "materialize_m99_public_item_and_sha", lambda t: ({"x": [1]}, "x"))
    monkeypatch.setattr(runner, "execute_single_pre_llm_public_gate_probe_in_container",
                        lambda **kw: {"accepted": False, "schema_valid": True, "scientific_probe_receipt": None})
    receipt = runner._evaluate_empirical_coverage_m99(
        new_id="b1", new_candidate_builder=lambda t, pay: ("code", "a"*64), old_id="b0", old_solver_sha="b"*64,
        old_parent_vers={}, old_witness_specs=[("w0", task, {}, {}, {})])
    assert receipt.status == "unknown" and receipt.evaluated_witness_count == 1


def test_coupled_first_requires_prior_positive_and_survives_restart(tmp_path):
    """Flip the arrival order for a frozen Beta component. Re-evaluate it after a real positive witness."""
    wid = "beta_m99_c01"
    world = next(s for s in m99.CONFIRMATION_M99_WORLD_SPECS if s.world_id == wid)
    _, tasks, _, logs = m99.build_m99_world_tasks(world)
    graph = runner.PersistentMechanismGraph.load_checkpoint(
        runner.DEFAULT_OUT_DIR / "continual_streams" / wid / "Hegel_AutonomousInteractionGraph_Online/graph_ckpt_after_step_04.json")
    registry = runner.M94BindingRegistry(graph)
    # A separate experiment on the exposed components starts with empty witness histories.
    experiment = wid + "_order_control"

    def controller(role):
        return runner.ImmutableBranchLineageController(world_id=f"{experiment}_{role}", family=f"beta_{role}",
                                                       graph=graph, registry=registry)

    dn_ctl, up_ctl = controller("dn"), controller("up")
    assert not dn_ctl.witness_ledger and not up_ctl.witness_ledger
    compat = world.to_m98_compat_spec()
    issued = []
    # E7 and E6 have colliding downstream observations. The upstream for each
    # fixed candidate is declared here and verified directly on the real item.
    for step, (idx, up_idx, expected) in enumerate([(6, 1, False), (5, 0, True), (6, 1, False)]):
        task = tasks[idx]
        item, sha = m99.materialize_m99_public_item_and_sha(task)
        prefix = logs[:idx+2]
        ro = {"/tmp/hegel_continual_archive/archive_apparatus_log.json": json.dumps(prefix, indent=2)}
        dep_shas = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in ro.items()}
        m99.update_m99_calibration_snapshot_and_graph(spec=compat, episode_id=f"order_{step}",
                                                      public_item=item, graph=graph, registry=registry)
        pay = m98.assemble_m98_graph_diagonal_payload(compat, graph)
        up = registry.bindings[f"bind_{wid}_up_b0{up_idx}"]
        dn = registry.bindings[f"bind_{wid}_dn_b02"]
        obs = m98.compute_m98_downstream_observables(prefix)
        candidate = m98.compose_m98_two_stage_solver_code(
            upstream_code=up.canonical_algorithm_code, upstream_entrypoint=up.canonical_entrypoint,
            downstream_code=dn.canonical_algorithm_code, downstream_entrypoint=dn.canonical_entrypoint, param_payload=pay)
        direct = m98.diagnose_m98_factored_conflict_locus(
            "beta", item, chosen_up_code=up.canonical_algorithm_code, chosen_up_entrypoint=up.canonical_entrypoint,
            diag_payload=pay, force_direct_component_probe=True)
        assert direct["execution_status"] == "completed" and direct["locus"] == "downstream_constitutive"
        gate = m99.execute_single_pre_llm_public_gate_probe_in_container(
            task=task, candidate_code=candidate, readonly_dependencies=ro, expected_dependency_sha256=dep_shas)
        assert gate["accepted"] is expected
        identity = dict(public_input_sha256=sha, setup_source_sha256=gate["setup_source_sha256"],
                        canonical_numeric_sha256=gate["scientific_probe_receipt"]["canonical_numeric_sha256"],
                        dependency_bundle_sha256=hashlib.sha256(json.dumps(dep_shas, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        up_ctl.record_tested_witness(binding=up, passed=True, failure_kind="none", features=m98.compute_m98_upstream_observables("beta", item),
                                     evidence_origin="direct_component_probe", probe_id=direct["probe_id"],
                                     metric_value=direct["chosen_up_holdout_rmse"], **identity)
        dn_ctl.record_tested_witness(binding=dn, passed=gate["accepted"], failure_kind=runner.classify_public_probe_failure(gate)["failure_kind"],
                                     features=obs, probe_id=f"public_order_{step}", **identity)
        receipt = runner._issue_measured_withdrawal(
            world_id=experiment, family="beta", downstream_selected=True, upstream=up, downstream=dn, downstream_features=obs,
            downstream_ledger=dn_ctl.witness_ledger, direct=direct, gate=gate, candidate_sha=hashlib.sha256(candidate.encode()).hexdigest())
        assert (receipt is not None) == (step == 2)
        if receipt:
            assert receipt.prior_witness_id == dn_ctl.witness_ledger[1].witness_id
            issued.append(receipt)
        m99.persist_separability_withdrawals_to_graph(graph, experiment, issued)
        path = tmp_path / f"order_{step}.json"
        graph.save_checkpoint(path)
        graph = runner.PersistentMechanismGraph.load_checkpoint(path)
        registry = runner.M94BindingRegistry(graph)
        up_ctl, dn_ctl = controller("up"), controller("dn")
        restored = m99.hydrate_separability_withdrawals_from_graph(graph, registry, experiment)
        assert [r.receipt_id for r in restored] == [r.receipt_id for r in issued]


def test_alternative_upstream_timeout_is_charged_and_stops_at_one(monkeypatch):
    calls = []

    def timed_out(*args, **kwargs):
        calls.append(kwargs["chosen_up_code"])
        return {"component_probe_executed": True, "execution_status": "timeout", "locus": "unknown_infra",
                "chosen_up_holdout_rmse": None, "probe_wall_ms": 7.5}

    monkeypatch.setattr(runner, "diagnose_m98_factored_conflict_locus", timed_out)
    trial = initial_trial("upstream_transducer")
    recover(trial, upstream_candidates=[("sibling1", "first", "up", None), ("sibling2", "second", "up", None)])
    assert calls == ["first"] and trial["pre_llm_probes_used"] == 1
    assert trial["localization_probes_used"] == 2 and trial["localization_probe_wall_ms"] == 10.5
    assert not trial["accepted_pre_llm"] and trial["candidate_code"] is None
