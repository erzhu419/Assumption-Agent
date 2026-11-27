"""Automated Audit-Closure & Real Runner State-Transition Suite for `M9.5.1` (`F01–F14`)."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pytest

from hegel_machine.cognition import (
    m6_8_b_prime_capability_bridge as m68,
    m7_autogenous_forest_pretraining as m7a,
    m9_3_online_lifecycle_benchmark as m93,
    m9_4_1_shared_verification_kernel as m941k,
    m9_4_continual_mechanism_binding_world as m94,
    m9_5_equal_tool_continual_attribution_world as m95,
)
from hegel_machine.cognition.cost_policy import aggregate_call_usages, score_single_call_usage
from hegel_machine.cognition.wormhole_forest import ClaimContract, ObservationMap, ScaleMap, TaskView


def _materialize_pub(task: object) -> dict:
    with tempfile.TemporaryDirectory(prefix="t_m951_") as td:
        ns: dict = {}
        exec(compile(task.public_setup_py.replace("/holdout_task", td), "<pub>", "exec"), ns)  # type: ignore[attr-defined]
        return pickle.loads((Path(td) / f"public_{task.short_name}.pkl").read_bytes())[0]  # type: ignore[attr-defined]


def test_m951_f01_f14_multi_hop_runner_state_transition_no_double_wrap() -> None:
    """Real multi-hop continual transition (`S -> G1 -> N` and `C -> R -> C'`) verifying:
    1. `make_continual_archive_entry` preserves `canonical_algorithm_code` (`input_coordinate_contract='raw_measured'`)
       separately from `delivered_artifact_code`, so `Flat_ParameterizedCache` & `Flat_Multiversion_Validated` NEVER double-wrap!
    2. `render_parameterized_input_adapter` fail-closed rejects wrapping an already-wrapped deliverable!
    3. `F14`: `public_gate_failed` entries are ignored by online auto-replay (`0` post-hoc private oracle!).
    """
    dev_alpha = m95.DEV_WORLD_SPECS[0]
    s_task, stream_tasks, manifest = m95.build_world_tasks(dev_alpha)
    c0_code = m95.build_reference_dev_solver_code("alpha", s_task.declared_callable_entrypoints[0], "C0")
    c1_code = m95.build_reference_dev_solver_code("alpha", "solve_dev_c1", "C1")
    uni_keys = m95.expected_public_output_keys(s_task)

    # Reject double-wrapping explicitly
    wrapped_1 = m95.render_parameterized_input_adapter(
        family="alpha",
        source_code=c0_code,
        source_entrypoint=s_task.declared_callable_entrypoints[0],
        target_entrypoint="solve_step1",
        param_payload={"stress_gains": [1.2, 0.8, 1.2, 0.8], "stress_biases": [0.1, -0.1, 0.1, -0.1]},
    )
    with pytest.raises(ValueError, match="F01"):
        m95.render_parameterized_input_adapter(
            family="alpha",
            source_code=wrapped_1,
            source_entrypoint="solve_step1",
            target_entrypoint="solve_step2",
            param_payload={"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]},
        )
    with pytest.raises(ValueError, match="F01"):
        m95.make_continual_archive_entry(
            short_name="bad",
            task_id="bad_t",
            canonical_entrypoint="solve_step1",
            canonical_algorithm_code=wrapped_1,
            output_schema_keys=uni_keys,
        )
    with pytest.raises(ValueError, match="F14"):
        m95.make_continual_archive_entry(
            short_name="bad_q",
            task_id="bad_qt",
            canonical_entrypoint=s_task.declared_callable_entrypoints[0],
            canonical_algorithm_code=c0_code,
            output_schema_keys=uni_keys,
            public_qualification_status="private_verifier_pass",
        )

    # Real multi-hop `G1 -> N` transition on `dev_alpha_00`!
    by_ev = {m["causal_event"]: t for t, m in zip(stream_tasks, manifest)}
    g1_task = by_ev["G1"]
    n_task = by_ev["N"]
    g2_task = by_ev["G2"]

    prior_entries = [
        m95.make_continual_archive_entry(
            short_name=s_task.short_name,
            task_id=s_task.task_id,
            canonical_entrypoint=s_task.declared_callable_entrypoints[0],
            canonical_algorithm_code=c0_code,
            output_schema_keys=uni_keys,
            public_qualification_status="public_gate_verified",
        ),
        # `F14`: Inject a newer broken solver stamped `public_gate_failed` — online replay MUST skip it!
        m95.make_continual_archive_entry(
            short_name="broken_unverified",
            task_id="broken_unverified_t",
            canonical_entrypoint="solve_broken",
            canonical_algorithm_code="def solve_broken(dataset): raise RuntimeError('should_not_be_selected')\n",
            output_schema_keys=uni_keys,
            public_qualification_status="public_gate_failed",
        ),
    ]
    flat_snap = {"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0]}

    # Hop 1: `G1` (`v2` shift)
    pub_g1 = _materialize_pub(g1_task)
    flat_snap = m95.update_flat_validated_wing_parameter_snapshot_from_public_item(
        family="alpha", public_item=pub_g1, prior_snapshot=flat_snap
    )
    tr_g1 = m95.run_equal_tool_pre_llm_trial(
        arm="Flat_Multiversion_Validated",
        family="alpha",
        task=g1_task,
        public_item=pub_g1,
        prior_archive_entries=prior_entries,
        flat_param_snapshot=flat_snap,
    )
    assert tr_g1["accepted_pre_llm"] is True
    assert tr_g1["selected_source_short_name"] == s_task.short_name
    prior_entries.append(
        m95.make_continual_archive_entry(
            short_name=g1_task.short_name,
            task_id=g1_task.task_id,
            canonical_entrypoint=tr_g1["selected_canonical_entrypoint"],
            canonical_algorithm_code=tr_g1["selected_canonical_algorithm_code"],
            output_schema_keys=uni_keys,
            public_qualification_status="public_gate_verified",
            delivered_artifact_code=tr_g1["candidate_code"],
            delivered_entrypoint=g1_task.declared_callable_entrypoints[0],
        )
    )

    # Hop 2: `N` (`v1` shift AFTER `G1`!) — previously failed in M9.5 due to double-wrap; now succeeds 1-hop & passes private verifier!
    pub_n = _materialize_pub(n_task)
    flat_snap = m95.update_flat_validated_wing_parameter_snapshot_from_public_item(
        family="alpha", public_item=pub_n, prior_snapshot=flat_snap
    )
    tr_n = m95.run_equal_tool_pre_llm_trial(
        arm="Flat_Multiversion_Validated",
        family="alpha",
        task=n_task,
        public_item=pub_n,
        prior_archive_entries=prior_entries,
        flat_param_snapshot=flat_snap,
    )
    assert tr_n["accepted_pre_llm"] is True
    ver_n = m94.grade_m94_artifact_map(n_task, {n_task.declared_artifacts[0]: tr_n["candidate_code"]})
    assert ver_n["strict_pass"] is True

    # Hop 3 (`F03`): `G2` (`v3` Wing A + EMI on Wing B after `G1` `v2`) — validated flat OLS rejects EMI on Wing B!
    snap_v2 = m95.update_flat_validated_wing_parameter_snapshot_from_public_item(
        family="alpha", public_item=pub_g1, prior_snapshot={"stress_gains": [1.0]*4, "stress_biases": [0.0]*4}
    )
    pub_g2 = _materialize_pub(g2_task)
    snap_v3 = m95.update_flat_validated_wing_parameter_snapshot_from_public_item(
        family="alpha", public_item=pub_g2, prior_snapshot=snap_v2
    )
    assert snap_v3["wing_validation_decisions"]["wing_a"]["accepted"] is True
    assert snap_v3["wing_validation_decisions"]["wing_b"]["accepted"] is False
    assert snap_v3["stress_gains"][2:] == snap_v2["stress_gains"][2:]
    assert c1_code != c0_code


def test_m951_f02_readonly_dependency_manifest_and_unmanifested_masking() -> None:
    """`F02`: Manifested `/tmp/hegel_continual_archive/archive_calibration_history.json` is delivered `0444` to BOTH
    public probe and private verifier, whereas unmanifested `/tmp/hegel_continual_archive/unmanifested.json` is masked!"""
    dev_beta = m95.DEV_WORLD_SPECS[1]
    s_task, stream_tasks, manifest = m95.build_world_tasks(dev_beta)
    by_ev = {m["causal_event"]: t for t, m in zip(stream_tasks, manifest)}
    g1_task = by_ev["G1"]
    r_task = by_ev["R"]

    pub_s = _materialize_pub(s_task)
    pub_g1 = _materialize_pub(g1_task)
    hist_list = [
        {
            "episode_short_name": x_sn,
            "bank_id": np.asarray(p_it["study"]["sensor_calibration_sweeps"]["bank_id"], dtype=int).tolist(),
            "ref_signal": np.asarray(p_it["study"]["sensor_calibration_sweeps"]["ref_signal"], dtype=float).tolist(),
            "raw_readout": np.asarray(p_it["study"]["sensor_calibration_sweeps"]["raw_readout"], dtype=float).tolist(),
        }
        for x_sn, p_it in ((s_task.short_name, pub_s), (g1_task.short_name, pub_g1))
    ]
    ro_deps = {"/tmp/hegel_continual_archive/archive_calibration_history.json": json.dumps(hist_list)}
    ro_shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in ro_deps.items()}

    # Build a Beta R solver that EXPLICITLY reads `/tmp/hegel_continual_archive/archive_calibration_history.json` for sparse banks 1..3!
    base_c0 = m95.build_reference_dev_solver_code("beta", "_inner_beta_c0", "C0")
    hist_reader_solver = (
        base_c0
        + f"\nimport copy, json, os\ndef {r_task.declared_callable_entrypoints[0]}(study: dict) -> dict:\n"
        "    assert not os.path.exists('/tmp/hegel_continual_archive/unmanifested_secret.json')\n"
        "    hist = json.load(open('/tmp/hegel_continual_archive/archive_calibration_history.json'))\n"
        "    last_full = hist[-1]\n"
        "    st2 = copy.deepcopy(study)\n"
        "    st2['sensor_calibration_sweeps'] = {\n"
        "        'bank_id': np.asarray(last_full['bank_id'], dtype=int),\n"
        "        'ref_signal': np.asarray(last_full['ref_signal'], dtype=float),\n"
        "        'raw_readout': np.asarray(last_full['raw_readout'], dtype=float),\n"
        "    }\n"
        "    return _inner_beta_c0(st2)\n"
    )
    # Inject an unmanifested file into `public_setup_py` to prove `execute_host_trusted_scientific_probe` masks it!
    r_task_with_unman = type(r_task)(
        **{
            **r_task.__dict__,
            "public_setup_py": r_task.public_setup_py
            + "\nos.makedirs('/tmp/hegel_continual_archive', exist_ok=True)\n"
            + "open('/tmp/hegel_continual_archive/unmanifested_secret.json', 'w').write('{\"leak\": 1}')\n",
        }
    )
    pr = m95.execute_single_pre_llm_public_gate_probe_in_container(
        task=r_task_with_unman,
        candidate_code=hist_reader_solver,
        readonly_dependencies=ro_deps,
        expected_dependency_sha256=ro_shas,
    )
    assert pr["accepted"] is True
    priv = m94.grade_m94_artifact_map(
        r_task,
        {r_task.declared_artifacts[0]: hist_reader_solver},
        readonly_dependencies=ro_deps,
        expected_dependency_sha256=ro_shas,
    )
    assert priv["strict_pass"] is True


def test_m951_f04_f06_f07_f08_f09_f10_unit_invariants() -> None:
    # F04: CostPolicy exact arithmetic
    u = score_single_call_usage(prompt_tokens=1000, cached_tokens=400, completion_tokens=50)
    assert u["uncached_prompt_tokens"] == 600
    assert u["proxy_v1_u_025c_1o"] == 600 + 100 + 50
    assert u["proxy_v2_u_010c_4o"] == 600 + 40 + 200
    agg = aggregate_call_usages([{"prompt_tokens": 1000, "cached_tokens": 400, "completion_tokens": 50}])
    assert agg["proxy_v2_u_010c_4o"] == 840

    # F06: Pure JSON numerical RPC rejects arbitrary Python objects & object ndarrays
    with pytest.raises(TypeError):
        m941k.encode_pure_json_numerical_payload({"bad": object()})
    with pytest.raises(ValueError):
        m941k.decode_pure_json_numerical_payload({"__hegel_ndarray__": [1], "dtype": "object", "shape": [1]})

    # F07: QualificationReceipt FK & forest_node_id enforcement
    forest, _, _ = m7a.build_blank_axiomatic_philosophical_forest()
    graph, _, reg = m95.initialize_m95_wing_scoped_mechanism_graph(family="alpha", forest=forest)
    valid_nid = next(iter(forest.nodes.keys()))
    with pytest.raises(ValueError):
        reg.issue_qualification_receipt(
            qualification_protocol="hegel_public_scientific_gate_v4",
            verified_code_sha256="a" * 64,
            forest_node_id="missing_node_id",
        )
    with pytest.raises(ValueError):
        reg.register_binding(
            m94.MechanismExecutableBinding(
                binding_id="b_unissued", chain_id="alpha", branch_tag="C0_base", mechanism_id="m_u", mechanism_version=1,
                forest_node_id=valid_nid, component_id="c0", preserved_component_ids=[], violated_component_id=None,
                module_stem="s", module_digest="a" * 64, function_name="fn", qualification_id="hegel_public_scientific_gate_v4",
                qualification_receipt_id="qrec_missing", parameter_mechanism_id="alpha_facility_composite_calibration",
                parameter_version=1, parameterized_closure_code="def fn(d): return {}\n", parameterized_entrypoint="fn",
            ),
            strict_receipt_check=True,
        )

    # F08: M9.3 locked-float plugin smoke check
    ctrl = m93.evaluate_post_freeze_dev_seed_plugin_controls(
        locked_src01_out={"estimated_taus": [0.55, 3.8], "hereditary_weights": [0.7, 0.55, 0.09, 0.16, -0.13, 0.11, -0.08]},
        locked_src02_out={"weights": [0.2] * 12},
    )
    assert ctrl["direct_locked_float_plugin_fails_all_successors"] is True
    assert ctrl["src01_param_origin"] == "locked_stage_S_candidate_output"

    # F09: Guard-wrapped skel_slots path starts with [1, ...]
    s1 = "def f1(x):\n    if x < -10.0:\n        raise ValueError('bad')\n    return 3.0 * x + 1.0\n"
    s2 = "def f2(x):\n    if x > 10.0:\n        raise ValueError('bad')\n    return 3.0 * x + 2.0\n"
    fw, aw, _ = m7a.build_blank_axiomatic_philosophical_forest()
    for nid, tid, sig, sc in (("ng1", "tree_A01_conservation", "s:1", s1), ("ng2", "tree_A02_deconfounding", "s:2", s2)):
        fw.register_node(
            node_id=nid, tree_id=tid, layer_role="mid",
            task_view=TaskView("tv", ["x"], "y", "present_snapshot", []),
            observation_map=ObservationMap("y", temporal_readout_mode="present_snapshot"),
            scale_map=ScaleMap("identity", preserved_queries=["q"], closure_scope="exact"),
            claim_contract=ClaimContract(f"cc_{nid}", "p", [], {}, {}, 1.0, "SUPPORTED_IN_SCOPE", "local_solver_guidance"),
            probe_signatures=[sig],
        )
        aw[nid] = {"tree_id": tid, "layer_role": "mid", "signatures": [sig], "autogenous_wormhole_modules": [{"filename": f"{nid}.py", "python_code": sc}]}
    m7a.discover_cross_tree_operator_wormholes(fw, aw)
    sub = next(iter(fw.shared_base_subtrees.values()))
    skel = m68.Program.from_dict(sub["executable_skeleton_dict"])
    for sl in sub["open_completion_slots"]:
        assert sl["path"][0] == 1
        cur = skel
        for idx in sl["path"]:
            cur = cur.children[idx]
        assert cur.op == "arg"

    # F10: Missing test history -> UNKNOWN
    ag = m7a.ObservablePreconditionGuard(
        guard_id="g10",
        observable_extractor="overstress_excess_max_history_and_hold",
        op="<=",
        value=0.0,
        applies_to_component_id="linear_viscoelastic_memory",
    )
    res10 = m7a.evaluate_observable_precondition_guard(
        ag,
        {
            "yield_stress": 1.5,
            "past_stress": [[0.2, 0.3]],
            "stress_hold_0": [0.2, 0.3],
            "train": {"past_stress": [[0.2, 0.3]], "stress_hold_0": [0.2]},
            "test": {"stress_hold_0": [0.3]},
        },
    )
    assert res10["state"] == "UNKNOWN"
