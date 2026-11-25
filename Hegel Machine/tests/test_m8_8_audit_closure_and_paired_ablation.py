"""Hegel M8.8 Comprehensive Audit-Closure (`M8.6 Completion` + `M8.7 Audit`) & Unseen Post-Freeze Verification Suite.

Covers all deterministic findings across both GPT-6 audits:
  1. `M88-CACHE-VIEW`: `PersistentMechanismGraph` negative cache binds train/val view digest + incumbent state + scope guard
     (reproducing GPT-6's `81`-row validation-view swap `-26.86 -> +2.83` counterexample!).
  2. `M88-WAVE1-MULTIFILE-DIGEST`: `consolidate_wave1_via_aufhebung_compiler` fails closed on missing `expected_rep_sha`
     and multi-file mismatched `expected_rep_sha`, while accepting verified multi-file bundle SHA-256!
  3. `M88-NOGOOD-SCHEMA-RESOLVE`: Frontier response JSON schemas allow `risk_kind` & `resolves_nogood_ids` and clear
     resolved target-identifiability nogoods.
  4. `M88-PKG-TXN` & `M88-SCAN-FAILCLOSED` & `M88-REPAIR-SYNTH-PKG`:
     - Rejects `globals()` reflection inside skill modules and private `m._helper` access on whole-module aliases.
     - Fails closed (`scan_ok == False`, `["<SCAN_ERROR:...>"]`) when container scanner fails.
     - Rolls back multi-file container packaging atomically (`0` partial overwrite when file 2 fails!).
     - Returns `skill_packaging_receipt` on successful `execute_in_container_local_repair(..., staged_skill_sources_by_stem=...)`.
  5. `M88-TEST-ALIAS`, `M88-COMPILER-PAYLOADS`, `M88-PRIMITIVE-CONTRACTS` & `M88-PRIMITIVE-RETRIEVAL`:
     - Bare `from autogenous_prim_* import ...` in self-test resolves to same module singleton + enforces `require_all_public_fns_called=True`.
     - Archives full failure payload (`attempts` code + test + stderr) on failed `L1` verification, deduplicates existing verified `L1` modules,
       and sets honest `source_qualification_status` (`online_or_mixed_l1_primitive_sublation` when `require_pretraining_only=False`).
     - Enforces non-empty `mathematical_preconditions` and PSD-minorant guard on `box_ridge_cube_lower_bound` (rejecting `X=[[1]], diagonal=[2]`).
     - Capability-indexed `L1` retrieval surfaces matching `L1` primitives even when parent node was truncated, and `allow_l1_primitives=False` cleanly disables `L1`.
  6. `Hegel-Unseen-PostFreeze-4` (`uf01..uf04`): Analytical reference solvers pass all 4 private verifiers cleanly in air-gapped containers.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import pytest

from hegel_machine.cognition import m7_autogenous_forest_pretraining as m7a
from hegel_machine.cognition.m6_4_executable_wormhole_library import (
    auto_package_wormhole_imports_in_container,
    collect_recalled_executable_wormhole_sources,
    find_uninlined_wormhole_imports_in_container,
    inspect_final_deliverable_skill_retention,
    package_self_contained_wormhole_dependencies,
    scan_wormhole_imports_in_container,
)
from hegel_machine.cognition.m6_frontier_real_benchmarks import (
    HEGEL_M6_FRONTIER_DECISION_SCHEMA,
    HEGEL_M6_2_FRONTIER_DECISION_SCHEMA,
    FrontierBenchmarkTask,
    exec_container_bash,
    start_detached_container,
    stop_container_safely,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    HoldoutOODTask,
    ensure_clean_scipy_sandbox_image,
)
from hegel_machine.cognition.m8_8_unseen_postfreeze_benchmark import (
    build_unseen_postfreeze_4_tasks,
    run_unseen_postfreeze_private_verifier_by_id,
)
from hegel_machine.cognition.m8_container_mechanism_adapter import (
    execute_in_container_local_repair,
)
from hegel_machine.cognition.m8_offline_container_planner import (
    _has_unresolved_target_identifiability_nogood,
)
from hegel_machine.cognition.m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TypedMechanism,
    evaluate_committed_task_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m8_8_negative_cache_binds_validation_view_and_incumbent_state() -> None:
    """Reproduce GPT-6's `81`-row validation-view swap control: negative cache on view 1 MUST NOT block positive recall on view 2!"""
    g = PersistentMechanismGraph()
    g.register_mechanism(
        TypedMechanism("root", None, "shared_upstream_premise", ("a",), shared_weights={"a": 0.0})
    )
    mech = TypedMechanism("mech_quad_lift_m88", "root", "interaction_lift", ("a",), python_expression="x['a'] ** 2")
    g.register_mechanism(mech)

    a = np.linspace(-2.0, 2.0, 81)
    ds_tr = {"x": {"a": a}, "y": 2.5 * (a ** 2)}
    # View 1: adversarial/corrupted validation view of length `81` where `x['a'] ** 2` loses badly
    ds_va1 = {"x": {"a": a}, "y": np.full(81, 2.5 * float(np.mean(a ** 2)))}
    # View 2: true validation view of IDENTICAL length `81` where `x['a'] ** 2` wins strongly!
    ds_va2 = {"x": {"a": a}, "y": 2.5 * (a ** 2)}

    p = M8VariationalPlanner(g)
    tm = p.register_task(
        task_id="t_view_swap",
        shared_upstream_id="root",
        feature_names=("a",),
        train_data=ds_tr,
        val_data=ds_va1,
    )
    base1 = evaluate_committed_task_model(tm, g, ds_va1)
    cands_v1 = p._discover_reusable_mechanism_candidates(
        tm, ds_tr, ds_va1, float(base1["val_mean_log_predictive"]), float(base1["variational_free_energy"])
    )
    assert len(cands_v1) == 1 and cands_v1[0].val_log_predictive_gain < -5.0
    assert len(mech.inapplicable_task_ids) >= 1

    # Swap ONLY validation view `ds_va2` (still `81` rows!) -> MUST NOT be blocked by stale view-1 negative cache!
    g.task_datasets["t_view_swap"]["val"] = ds_va2
    base2 = evaluate_committed_task_model(tm, g, ds_va2)
    cands_v2 = p._discover_reusable_mechanism_candidates(
        tm, ds_tr, ds_va2, float(base2["val_mean_log_predictive"]), float(base2["variational_free_energy"])
    )
    assert len(cands_v2) == 1 and cands_v2[0].val_log_predictive_gain > 1.0


def test_m8_8_wave1_multifile_and_missing_hash_source_qualification_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `consolidate_wave1_via_aufhebung_compiler` rejects missing-hash 1-file AND mismatched-hash 2-file repaired dirs."""
    monkeypatch.setattr(m7a, "ensure_clean_scipy_sandbox_image", lambda: None)
    monkeypatch.setattr(m7a.m6f, "start_detached_container", lambda *a, **kw: "mock_cid")
    monkeypatch.setattr(m7a.m6f, "stop_container_safely", lambda *a, **kw: None)
    monkeypatch.setattr(m7a.m6f, "FrontierCodexClient", lambda **kw: object())

    compiled_calls: list[dict[str, Any]] = []

    def _fake_compile(**kwargs: Any) -> dict[str, Any]:
        compiled_calls.append(kwargs)
        return {
            "compiled_ok": False,
            "compilation_mode": "mock_skip",
            "filename": "wave1_mock.py",
            "function_names": [],
            "llm_tokens": {"prompt": 0, "cached": 0, "completion": 0, "wall_ms": 0.0},
        }

    monkeypatch.setattr(m7a, "compile_episode_dialectical_sublation", _fake_compile)
    monkeypatch.setattr(m7a, "consolidate_autogenous_episode_into_forest", lambda **kw: {"consolidated": False})

    forest, archive, _ = m7a.build_blank_axiomatic_philosophical_forest()
    dummy_task = FrontierBenchmarkTask(
        task_id="tb__dummy_multifile",
        short_name="dummy_multifile",
        benchmark="Terminal-Bench-Science",
        wave="wave_1",
        scientific_domain="Physics",
        wormhole_family="fam_test",
        instruction_text="Solve PDE.",
        agent_image_tag="mock",
        verifier_image_tag="mock",
        declared_artifacts=("/app/a.py", "/app/b.py"),
    )

    with tempfile.TemporaryDirectory(prefix="m88_w1_bind_") as td:
        td_p = Path(td)
        # Case 1: 1 file, missing `expected_rep_sha` -> MUST skip!
        d1 = td_p / "ep1"
        (d1 / "04a_repaired_antithesis_snapshot").mkdir(parents=True)
        (d1 / "04a_repaired_antithesis_snapshot" / "a.py").write_text("x = 1\n", encoding="utf-8")
        s1 = {"ended_by_finish": True, "local_container_repair_summary": {"repaired_verified": True, "repaired_candidate_sha256": ""}}

        # Case 2: 2 files, wrong `expected_rep_sha` -> MUST skip!
        d2 = td_p / "ep2"
        (d2 / "04a_repaired_antithesis_snapshot").mkdir(parents=True)
        (d2 / "04a_repaired_antithesis_snapshot" / "a.py").write_text("x = 1\n", encoding="utf-8")
        (d2 / "04a_repaired_antithesis_snapshot" / "b.py").write_text("y = 2\n", encoding="utf-8")
        s2 = {"ended_by_finish": True, "local_container_repair_summary": {"repaired_verified": True, "repaired_candidate_sha256": "f" * 64}}

        # Case 3: 2 files, matching bundle digest -> MUST enter compiler!
        d3 = td_p / "ep3"
        (d3 / "04a_repaired_antithesis_snapshot").mkdir(parents=True)
        (d3 / "04a_repaired_antithesis_snapshot" / "a.py").write_text("x = 1\n", encoding="utf-8")
        (d3 / "04a_repaired_antithesis_snapshot" / "b.py").write_text("y = 2\n", encoding="utf-8")
        hb = hashlib.sha256()
        hb.update(b"/app/a.py\0" + hashlib.sha256(b"x = 1\n").digest())
        hb.update(b"/app/b.py\0" + hashlib.sha256(b"y = 2\n").digest())
        s3 = {"ended_by_finish": True, "local_container_repair_summary": {"repaired_verified": True, "repaired_candidate_sha256": hb.hexdigest()}}

        out = m7a.consolidate_wave1_via_aufhebung_compiler(
            wave1_antithesis_episodes=[(dummy_task, s1, d1), (dummy_task, s2, d2), (dummy_task, s3, d3)],
            forest=forest,
            archive=archive,
            codex_info={"available": True, "auth_mode": "chatgpt", "executable": "codex", "provider_backend": "chatgpt"},
            model="gpt-6.2-sol",
            run_id="test_w1",
        )
        assert out["events"][0]["status"] == "skipped_missing_or_mismatched_repaired_snapshot"
        assert out["events"][1]["status"] == "skipped_missing_or_mismatched_repaired_snapshot"
        assert out["events"][2]["resolved_source_kind"] == "repaired_active_snapshot"
        assert len(compiled_calls) == 1


def _assert_openai_strict_json_schema(node: Any, ctx: str = "$") -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            props = node.get("properties") or {}
            req = node.get("required")
            assert isinstance(req, list), f"Missing `required` list at {ctx}"
            assert set(req) == set(props.keys()), f"OpenAI strict schema mismatch at {ctx}: required={req} vs props={list(props.keys())}"
            assert node.get("additionalProperties") is False, f"`additionalProperties` must be False at {ctx}"
            for k, v in props.items():
                _assert_openai_strict_json_schema(v, f"{ctx}.{k}")
        if "items" in node:
            _assert_openai_strict_json_schema(node["items"], f"{ctx}[]")


def test_m8_8_nogood_schema_allows_risk_kind_and_resolves_prior_nogood_ids() -> None:
    """Verify both Frontier decision JSON schemas expose `risk_kind` & `resolves_nogood_ids` and pass OpenAI strict-schema rules."""
    for sc in (HEGEL_M6_FRONTIER_DECISION_SCHEMA, HEGEL_M6_2_FRONTIER_DECISION_SCHEMA):
        _assert_openai_strict_json_schema(sc)
        ng_props = sc["properties"]["nogood_update"]["properties"]
        assert "risk_kind" in ng_props
        assert "resolves_nogood_ids" in ng_props

    ng_list = [
        {
            "nogood_id": "ng_t02",
            "nogood_rung": "RUNG_3_FALSIFIED_PHYSICAL_OR_CAUSAL_MECHANISM",
            "risk_kind": "target_identifiability",
            "resolved": False,
            "assumption_refuted": "Unidentifiable baseline drift.",
        }
    ]
    assert _has_unresolved_target_identifiability_nogood(ng_list) is True
    ng_list[0]["resolved"] = True
    ng_list[0]["superseded_by"] = "ng_t05"
    assert _has_unresolved_target_identifiability_nogood(ng_list) is False


def test_m8_8_packager_reflection_private_attr_atomic_multifile_rollback_and_failclosed_scanner() -> None:
    """Verify reflection/private-attr rejection, fail-closed scanner, atomic multi-file rollback, and repair packaging receipt."""
    # 1. Reflection inside skill module (`globals()`) MUST fail closed
    bad_refl = package_self_contained_wormhole_dependencies(
        "from hegel_wormhole_skills.m_refl import f\n",
        {"m_refl": "K = 2.0\ndef f(x):\n    return x * globals()['K']\n"},
    )
    assert bad_refl["ok"] is False and "reflection_in_skill_module_unsupported" in bad_refl["error"]

    # 2. Private `_helper` attribute access via whole-module alias MUST fail closed
    bad_priv = package_self_contained_wormhole_dependencies(
        "import hegel_wormhole_skills.m_priv as mp\ndef solve(x):\n    return mp._helper(x)\n",
        {"m_priv": "def _helper(x):\n    return x + 1\ndef pub(x):\n    return _helper(x)\n"},
    )
    assert bad_priv["ok"] is False and "unexported_or_private_module_attribute_unsupported" in bad_priv["error"]

    # 3. Scanner on invalid container ID MUST fail closed (`scan_ok == False`, non-empty error marker)
    sc_bad = scan_wormhole_imports_in_container("nonexistent_container_m88_xyz", ("/holdout_task/a.py",))
    assert sc_bad["scan_ok"] is False
    assert find_uninlined_wormhole_imports_in_container("nonexistent_container_m88_xyz", ("/holdout_task/a.py",))[0].startswith("<SCAN_ERROR:")

    # 4. Atomic multi-file packaging rollback + LocalContainerRepair `skill_packaging_receipt` inside real clean container
    ensure_clean_scipy_sandbox_image()
    cid = start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1024m", cpus="2", network_mode="none")
    try:
        exec_container_bash(cid, "mkdir -p /holdout_task", timeout_sec=10)
        f1_orig = "from hegel_wormhole_skills.good_skill import scale_arr\ndef run1(x):\n    return scale_arr(x)\n"
        f2_bad = "from hegel_wormhole_skills.good_skill import *\ndef run2(x):\n    return x\n"
        with tempfile.TemporaryDirectory(prefix="m88_tx_") as td:
            p1 = Path(td) / "f1.py"; p1.write_text(f1_orig, encoding="utf-8")
            p2 = Path(td) / "f2.py"; p2.write_text(f2_bad, encoding="utf-8")
            subprocess.run(["docker", "cp", str(p1), f"{cid}:/holdout_task/f1.py"], check=True, timeout=20)
            subprocess.run(["docker", "cp", str(p2), f"{cid}:/holdout_task/f2.py"], check=True, timeout=20)

        tx_out = auto_package_wormhole_imports_in_container(
            cid,
            ("/holdout_task/f1.py", "/holdout_task/f2.py"),
            {"good_skill": "import numpy as np\ndef scale_arr(x):\n    return np.asarray(x, float) * 2.0\n"},
        )
        assert tx_out["ok"] is False and tx_out["rolled_back"] is True
        # File 1 MUST remain byte-for-byte identical to `f1_orig`!
        f1_after = exec_container_bash(cid, "cat /holdout_task/f1.py", timeout_sec=10)["stdout"]
        assert f1_after == f1_orig

        # Also verify `execute_in_container_local_repair` bundles skill import AND returns `skill_packaging_receipt` on success!
        toy_task = HoldoutOODTask(
            task_id="ho06__plate_evaporation_deconfounded_ic50",
            short_name="toy_repair",
            wave="wave_1",
            scientific_domain="Assay",
            wormhole_family="fam_test",
            declared_artifacts=("/holdout_task/assay_ic50.py",),
            instruction_text="Estimate compound Hill parameters.",
            public_setup_py="",
            private_verifier_py="",
        )
        exec_container_bash(
            cid,
            "python3 -c \"import pickle; pickle.dump({'plates': [{'plate_id':'P1','wells':[{'row':0,'col':0,'well_type':'POS_KILL','fluorescence':10.0},{'row':0,'col':1,'well_type':'DMSO','fluorescence':110.0},{'row':1,'col':1,'well_type':'COMPOUND','compound_id':'C1','log10_conc_uM':-2.0,'fluorescence':105.0},{'row':1,'col':2,'well_type':'COMPOUND','compound_id':'C1','log10_conc_uM':0.0,'fluorescence':60.0},{'row':1,'col':3,'well_type':'COMPOUND','compound_id':'C1','log10_conc_uM':2.0,'fluorescence':15.0}]}], 'gt': {'C1': {'log10_ic50_uM': 0.0, 'hill_slope': 1.0}}}, open('/holdout_task/public_screen.pkl','wb'))\"",
            timeout_sec=15,
        )
        exec_container_bash(
            cid,
            "cat <<'EOF' > /holdout_task/assay_ic50.py\ndef estimate_compound_hill_parameters(plates):\n    return {}\nEOF",
            timeout_sec=10,
        )

        class _MockRepairClient:
            def call_structured(self, **_: Any) -> dict[str, Any]:
                return {
                    "ok": True,
                    "prompt_tokens": 50,
                    "cached_tokens": 0,
                    "completion_tokens": 40,
                    "wall_ms": 5.0,
                    "parsed": {
                        "defect_diagnosis": "Empty dict returned.",
                        "invariance_guard_name": "plate_norm",
                        "invariance_principle": "Normalize compound wells by DMSO and POS_KILL medians.",
                        "repaired_python_code": (
                            "import numpy as np\n"
                            "from hegel_wormhole_skills.autogenous_prim_demo import clip_pos\n"
                            "def estimate_compound_hill_parameters(plates):\n"
                            "    out = {}\n"
                            "    for p in plates:\n"
                            "        for w in p['wells']:\n"
                            "            if w['well_type'] == 'COMPOUND':\n"
                            "                out[w['compound_id']] = {'log10_ic50_uM': 0.0, 'hill_slope': float(clip_pos(1.0))}\n"
                            "    return out\n"
                        ),
                    },
                }

        rep_res = execute_in_container_local_repair(
            probe_cid=cid,
            task=toy_task,
            public_probe_before={"symmetry_probe_family": "compound_renaming_and_well_permutation_equivariance"},
            codex_client=_MockRepairClient(),
            staged_skill_sources_by_stem={"autogenous_prim_demo": "def clip_pos(v: float) -> float:\n    return max(float(v), 1e-3)\n"},
        )
        assert rep_res["repaired_verified"] is True
        assert rep_res["skill_packaging_receipt"] is not None
        assert rep_res["skill_packaging_receipt"]["ok"] is True
        assert "autogenous_prim_demo" in rep_res["skill_packaging_receipt"]["packaged_modules"]
    finally:
        stop_container_safely(cid)


def test_m8_8_l1_compiler_bare_alias_full_attempt_archival_psd_minorant_guard_and_capability_retrieval() -> None:
    """Verify bare-module test alias, full-export execution check, PSD minorant guard on `v3`, failure archival, and capability retrieval."""
    v3_fp = REPO_ROOT / "docs/evidence/m8_7_compositional_primitives_and_ood_adoption_20261125/pretrained_autogenous_forest_v3_primitives.json"
    forest, archive, _, _ = m7a.load_autogenous_forest_bundle(v3_fp)
    enr = m7a.enrich_and_harden_l1_primitive_contracts(archive, reverify_in_clean_container=True)
    assert len(enr["preconditions_populated_modules"]) == 4

    # Verify PSD minorant guard on `box_ridge_cube_lower_bound` rejects GPT-6's invalid `diagonal=[2.0]` when `X=[[1.0]]`!
    box_mod_src = next(
        m["python_code"]
        for _, r in m7a.iter_live_archive_nodes(archive)
        for m in (r.get("autogenous_wormhole_modules") or [])
        if m.get("filename") == "autogenous_prim_box_ridge_bounds.py"
    )
    pkg_box = package_self_contained_wormhole_dependencies(
        "from hegel_wormhole_skills.autogenous_prim_box_ridge_bounds import gram_diagonal_minorant, box_ridge_cube_lower_bound\n",
        {"autogenous_prim_box_ridge_bounds": box_mod_src},
    )
    assert pkg_box["ok"] is True
    ns_exec: dict[str, Any] = {}
    exec(pkg_box["packaged_python_code"], ns_exec)
    X1 = np.array([[1.0]])
    y1 = np.array([0.0])
    c1 = np.array([1.0])
    with pytest.raises(ValueError, match="valid PSD minorant"):
        ns_exec["box_ridge_cube_lower_bound"](X1, y1, c1, 1.0, 0.0, ridge=0.0, diagonal=np.array([2.0]))
    valid_d = ns_exec["gram_diagonal_minorant"](X1)
    lb_ok = ns_exec["box_ridge_cube_lower_bound"](X1, y1, c1, 1.0, 0.0, ridge=0.0, diagonal=valid_d)
    assert lb_ok <= 1e-9

    # Verify capability-indexed `L1` retrieval finds `autogenous_prim_box_ridge_bounds` even when ONLY an unrelated `recalled_note` is passed!
    unrelated_only = [deepcopy(archive["node_auto_04_creep_corrected_retraction_c"])]
    mods_l2_only, man_l2_only = collect_recalled_executable_wormhole_sources(
        unrelated_only,
        include_foundational_skills=False,
        allow_l1_primitives=False,
        full_forest_archive=archive,
        task_query_text="Certified box L0 branch-and-bound lower bound minorant",
    )
    assert all(not k.startswith("autogenous_prim_") for k in mods_l2_only)
    mods_l1_on, man_l1_on = collect_recalled_executable_wormhole_sources(
        unrelated_only,
        include_foundational_skills=False,
        allow_l1_primitives=True,
        full_forest_archive=archive,
        task_query_text="Certified box L0 branch-and-bound lower bound minorant",
    )
    assert "autogenous_prim_box_ridge_bounds.py" in mods_l1_on
    assert man_l1_on[0]["mathematical_preconditions"]

    # Verify bare-alias import in `verify_autogenous_module_in_clean_container` + failure payload archival + `inspect_final_deliverable_skill_retention`
    ensure_clean_scipy_sandbox_image()
    cid = start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="1024m", cpus="2", network_mode="none")
    try:
        ok_a, _, pub_f, pkg_c = m7a.validate_and_decorate_autogenous_python_code(
            "import numpy as np\ndef f_one(x: np.ndarray) -> np.ndarray:\n    return np.asarray(x, float) + 1.0\n"
            "def f_two(x: np.ndarray) -> np.ndarray:\n    return np.asarray(x, float) * 2.0\n",
            "autogenous_prim_bare_alias_demo",
            extra_forbidden_tokens=("sklearn",),
        )
        assert ok_a is True
        # Calling only 1 of 2 exported functions MUST fail when `require_all_public_fns_called=True`!
        ut_partial = m7a.verify_autogenous_module_in_clean_container(
            sandbox_cid=cid,
            mod_stem="autogenous_prim_bare_alias_demo",
            packaged_module_code=pkg_c,
            synthetic_unit_test_code="from autogenous_prim_bare_alias_demo import f_one\nassert float(f_one([2.0])[0]) == 3.0\n",
            public_fn_names=pub_f,
            require_all_public_fns_called=True,
        )
        assert ut_partial["passed"] is False and ut_partial["uncalled_public_functions"] == ["f_two"]

        # Calling BOTH via bare `from autogenous_prim_bare_alias_demo import ...` MUST pass!
        ut_full = m7a.verify_autogenous_module_in_clean_container(
            sandbox_cid=cid,
            mod_stem="autogenous_prim_bare_alias_demo",
            packaged_module_code=pkg_c,
            synthetic_unit_test_code=(
                "from autogenous_prim_bare_alias_demo import f_one, f_two\n"
                "assert float(f_one([2.0])[0]) == 3.0\n"
                "assert float(f_two([3.0])[0]) == 6.0\n"
            ),
            public_fn_names=pub_f,
            require_all_public_fns_called=True,
        )
        assert ut_full["passed"] is True and ut_full["uncalled_public_functions"] == []
    finally:
        stop_container_safely(cid)

    # Verify `inspect_final_deliverable_skill_retention` on sealed M8.7 `ho06` artifact
    ho06_snap = REPO_ROOT / "docs/evidence/m8_7_compositional_primitives_and_ood_adoption_20261125/runs/ho06__plate_evaporation_deconfounded_ic50/04_submitted_artifacts_snapshot"
    ref_map = {
        Path(str(m["filename"])).stem: str(m["python_code"])
        for _, r in m7a.iter_live_archive_nodes(archive)
        for m in (r.get("autogenous_wormhole_modules") or [])
    }
    ret_ho06 = inspect_final_deliverable_skill_retention(ho06_snap, ref_map)
    assert ret_ho06["final_retained_l1_modules"] == [
        "autogenous_prim_ridge_profiled_matching",
        "autogenous_prim_robust_affine_projection",
    ]
    assert ret_ho06["all_retained_ast_match_source"] is True
    assert len(ret_ho06["entrypoint_reachable_l1_functions"]) == 2


def test_m8_8_unseen_postfreeze_4_tasks_reference_solvers_pass_private_verifiers() -> None:
    """Verify all 4 post-freeze unseen OOD tasks (`uf01..uf04`) are well-posed and pass private verifiers via compact reference solvers."""
    tasks = build_unseen_postfreeze_4_tasks()
    assert len(tasks) == 4
    ref_solvers = {
        "uf01__electrical_resistivity_electrode_contact_tomography": (
            "ert_solver.py",
            r'''
import numpy as np
from scipy.linalg import helmert
from scipy.optimize import least_squares

def invert_ert_anomaly(survey: dict) -> dict:
    L, D, a0 = float(survey["L_m"]), float(survey["D_m"]), float(survey.get("a0_m", 6.0))
    zt = np.asarray(survey["z_tx_m"], float)
    zr = np.asarray(survey["z_rx_m"], float)
    V = np.asarray(survey["V_obs"], float).ravel()
    nt, nr = len(zt), len(zr)
    zti = np.repeat(zt, nr); zrj = np.tile(zr, nt)
    dz = zrj - zti; R = np.hypot(L, dz)
    gt = helmert(nt).T; gr = helmert(nr).T
    Bt = np.repeat(gt, nr, axis=0); Br = np.tile(gr, (nt, 1))
    A = np.column_stack([1.0 / R, (dz ** 2) / (R ** 3), Bt, Br])
    def kern(x, z):
        return 1.0 / (np.sqrt(x**2 + (z - zti)**2 + a0**2) * np.sqrt((L - x)**2 + (zrj - z)**2 + a0**2))
    w = np.ones_like(V)
    for _ in range(8):
        coef, *_ = np.linalg.lstsq(A * w[:, None], V * w, rcond=None)
        r = V - A @ coef
        w = 1.0 / (1.0 + (r / 0.002) ** 2)
    Aw = A * w[:, None]; Vw = V * w
    P_coef, *_ = np.linalg.lstsq(Aw, Vw, rcond=None)
    res0 = Vw - Aw @ P_coef
    gx, gz = np.meshgrid(np.linspace(2.0, L - 2.0, 25), np.linspace(4.0, D - 4.0, 35))
    best_s, best_p = -1.0, (L / 2, D / 2, 5.0)
    for x0, z0 in zip(gx.ravel(), gz.ravel()):
        kw = kern(x0, z0) * w
        c_proj, *_ = np.linalg.lstsq(Aw, kw, rcond=None)
        kp = kw - Aw @ c_proj
        en = float(kp @ kp)
        if en > 1e-12:
            amp = max(0.0, float(res0 @ kp) / en)
            sc = (amp ** 2) * en
            if sc > best_s:
                best_s = sc; best_p = (x0, z0, amp)
    c_init, *_ = np.linalg.lstsq(Aw, (V - best_p[2] * kern(best_p[0], best_p[1])) * w, rcond=None)
    p0 = np.r_[best_p[0], best_p[1], max(best_p[2], 0.5), c_init]
    lo = np.r_[0.0, 0.0, 0.0, np.full(len(c_init), -np.inf)]
    hi = np.r_[L, D, np.inf, np.full(len(c_init), np.inf)]
    fit = least_squares(lambda p: p[2] * kern(p[0], p[1]) + A @ p[3:] - V, p0, bounds=(lo, hi), loss="cauchy", f_scale=0.001)
    p = fit.x
    inliers = np.abs(p[2] * kern(p[0], p[1]) + A @ p[3:] - V) < 0.004
    fit2 = least_squares(lambda q: (q[2] * kern(q[0], q[1]) + A @ q[3:] - V)[inliers], p, bounds=(lo, hi))
    q = fit2.x
    rx_off = gr @ q[5 + (nt - 1):]
    return {"anomaly_xz_m": [float(q[0]), float(q[1])], "rx_contact_offset_v": rx_off.tolist()}
''',
        ),
        "uf02__batch_drifted_substrate_inhibition_kinetics": (
            "enzyme_plate_solver.py",
            r'''
import numpy as np
from scipy.optimize import least_squares

def _poly_design(r, c):
    u, v = (r - 3.5) / 3.5, (c - 5.5) / 5.5
    return np.column_stack([np.ones_like(u), u, v, u**2, v**2, u * v])

def estimate_variant_haldane_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    by_var = {}
    for pl in plates:
        wells = pl["wells"]
        blanks = np.array([w["raw_rate"] for w in wells if w["well_type"] == "BLANK"], float)
        bg = float(np.median(blanks))
        refs = [w for w in wells if w["well_type"] == "REF_WT"]
        rr = np.array([w["row"] for w in refs], float); rc = np.array([w["col"] for w in refs], float)
        ry = np.array([w["raw_rate"] - bg for w in refs], float)
        Xr = _poly_design(rr, rc)
        fit_g = least_squares(lambda b: Xr @ b - ry, np.r_[np.median(ry), np.zeros(5)], loss="cauchy", f_scale=2.5)
        beta = fit_g.x
        for w in wells:
            if w["well_type"] == "VARIANT":
                g_hat = float((_poly_design(np.array([float(w["row"])]), np.array([float(w["col"])])) @ beta)[0])
                by_var.setdefault(w["variant_id"], []).append((float(w["substrate_mM"]), (float(w["raw_rate"]) - bg) / g_hat))
    out = {}
    for vid, pts in by_var.items():
        arr = np.asarray(pts, float)
        S, y = arr[:, 0], arr[:, 1]
        fit = least_squares(
            lambda p: np.exp(p[0]) * S / (np.exp(p[1]) + S + (S**2) / np.exp(p[2])) - y,
            [np.log(max(np.percentile(y, 85), 0.5)), 0.0, np.log(10.0)],
            loss="cauchy", f_scale=0.04,
        )
        vm, km, ki = np.exp(fit.x)
        out[vid] = {"V_max": float(vm), "K_m": float(km), "K_i": float(ki)}
    return out
''',
        ),
        "uf03__orthotropic_robin_heat_sink_inversion": (
            "thermal_sink_solver.py",
            r'''
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.optimize import least_squares

def _build_fvm(kappa, gamma):
    N = kappa.shape[0]; h = 1.0 / N
    rows, cols, vals = [], [], []
    diag = np.zeros(N * N, float)
    for i in range(N):
        for j in range(N):
            u = i * N + j
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    v = ni * N + nj
                    g = 2.0 * kappa[i, j] * kappa[ni, nj] / (kappa[i, j] + kappa[ni, nj])
                    diag[u] += g; rows.append(u); cols.append(v); vals.append(-g)
                else:
                    diag[u] += gamma * h
    idx = np.arange(N * N)
    return sparse.csr_matrix((np.r_[vals, diag], (np.r_[rows, idx], np.r_[cols, idx])), shape=(N * N, N * N))

def localize_thermal_source(measurement: dict) -> dict:
    N = int(measurement["N"]); h = 1.0 / N
    kappa = np.asarray(measurement["kappa"], float)
    gamma = float(measurement["gamma_robin"]); sigma = float(measurement["sigma_m"])
    s_ij = np.asarray(measurement["sensor_ij"], int)
    T = np.asarray(measurement["T_obs"], float)
    s_flat = s_ij[:, 0] * N + s_ij[:, 1]
    solve_A = factorized(_build_fvm(kappa, gamma).tocsc())
    xc = (np.arange(N) + 0.5) * h
    gx, gy = np.meshgrid(xc, xc, indexing="ij")
    def fwd_basis(xs, ys):
        q = np.exp(-0.5 * ((gx - xs)**2 + (gy - ys)**2) / (sigma**2)) * (h**2)
        return solve_A(q.ravel())[s_flat]
    best_cost, best_p = np.inf, (0.5, 0.5, 10.0, 0.0)
    for xs in np.linspace(0.15, 0.85, 11):
        for ys in np.linspace(0.15, 0.85, 11):
            b = fwd_basis(xs, ys)
            X = np.column_stack([b, np.ones_like(b)])
            fit = least_squares(lambda c: X @ c - T, [15.0, 0.0], bounds=([0.0, -np.inf], [np.inf, np.inf]), loss="cauchy", f_scale=0.005)
            if fit.cost < best_cost:
                best_cost = fit.cost; best_p = (xs, ys, fit.x[0], fit.x[1])
    fit2 = least_squares(
        lambda p: p[2] * fwd_basis(p[0], p[1]) + p[3] - T,
        list(best_p),
        bounds=([0.02, 0.02, 0.0, -np.inf], [0.98, 0.98, np.inf, np.inf]),
        loss="cauchy", f_scale=0.004,
    )
    p = fit2.x
    inl = np.abs(p[2] * fwd_basis(p[0], p[1]) + p[3] - T) < 0.01
    fit3 = least_squares(lambda q: (q[2] * fwd_basis(q[0], q[1]) + q[3] - T)[inl], p, bounds=([0.02, 0.02, 0.0, -np.inf], [0.98, 0.98, np.inf, np.inf]))
    q = fit3.x
    return {"source_xy": [float(q[0]), float(q[1])], "amplitude": float(q[2]), "junction_offset": float(q[3])}
''',
        ),
        "uf04__certified_box_l0_sensor_selection": (
            "certified_l0_solver.py",
            r'''
import heapq, numpy as np
from scipy.optimize import minimize

def solve_certified_box_l0(instance: dict) -> dict:
    X = np.asarray(instance["X"], float); y = np.asarray(instance["y"], float)
    ridge = float(instance["ridge"]); box = float(instance["box"]); lam0 = float(instance["l0_penalty"])
    n, p = X.shape
    G = X.T @ X + ridge * np.eye(p); rhs = X.T @ y; y2 = 0.5 * float(y @ y)

    def eval_subset(active_idx):
        if not active_idx:
            return y2, np.zeros(p)
        Gs = G[np.ix_(active_idx, active_idx)]; rs = rhs[active_idx]
        w_u = np.linalg.solve(Gs, rs)
        if np.all(np.abs(w_u) <= box):
            val = y2 - 0.5 * float(rs @ w_u) + lam0 * len(active_idx)
            w_sub = w_u
        else:
            res = minimize(lambda z: 0.5 * float(z @ Gs @ z - 2.0 * (rs @ z)), np.clip(w_u, -box, box),
                           jac=lambda z: Gs @ z - rs, bounds=[(-box, box)] * len(active_idx), method="L-BFGS-B")
            val = y2 + float(res.fun) + lam0 * len(active_idx); w_sub = res.x
        w_full = np.zeros(p); w_full[active_idx] = w_sub
        return val, w_full

    def relax_lb(fixed_zero, fixed_one):
        free = [j for j in range(p) if j not in fixed_zero]
        if not free:
            return y2, np.zeros(p)
        Gf = G[np.ix_(free, free)]; rf = rhs[free]
        # Perspective / Huber-like convex envelope of `0.5 * ridge_part + lam0 * I(w != 0)` on `[-box, box]`:
        # Exact Perspective on `[-box, box]` with extracted diagonal `d_j <= G_{jj}`:
        e_min = max(0.0, float(np.linalg.eigvalsh(Gf)[0]) - 1e-9)
        d = np.full(len(free), e_min)
        G_rem = Gf - np.diag(d)
        def fun_and_grad(z):
            gz = G_rem @ z
            f_val = y2 + float(0.5 * (z @ gz) - (rf @ z))
            grad = gz - rf
            for idx_f, j_orig in enumerate(free):
                zj = z[idx_f]; dj = d[idx_f]
                if j_orig in fixed_one:
                    f_val += 0.5 * dj * zj * zj + lam0
                    grad[idx_f] += dj * zj
                else:
                    # Convex envelope of `0.5 * d * z^2 + lam0 * I(z!=0)` on `|z| <= box`
                    thr = min(box, np.sqrt(2.0 * lam0 / max(dj, 1e-12))) if dj > 1e-12 else box
                    slope = (0.5 * dj * (thr ** 2) + lam0) / max(thr, 1e-12)
                    az = abs(zj)
                    if az <= thr:
                        # Smooth subgradient at 0 handled via L-BFGS-B split variable `(z+, z-)` below
                        pass
            return f_val, grad
        # Exact split `z = u - v`, `u, v in [0, box]` for smooth L-BFGS-B convex relaxation!
        m = len(free)
        thr = np.array([0.0 if free[k] in fixed_one else (min(box, np.sqrt(2.0 * lam0 / d[k])) if d[k] > 1e-12 else box) for k in range(m)])
        slp = np.array([0.0 if free[k] in fixed_one else ((0.5 * d[k] * (thr[k] ** 2) + lam0) / max(thr[k], 1e-12)) for k in range(m)])
        const_one = lam0 * sum(1 for k in range(m) if free[k] in fixed_one)
        def obj_uv(uv):
            u, v = uv[:m], uv[m:]; z = u - v; s = u + v
            gz = G_rem @ z
            val = y2 + const_one + 0.5 * float(z @ gz) - float(rf @ z)
            g_z = gz - rf; g_s = np.zeros(m)
            for k in range(m):
                if free[k] in fixed_one:
                    val += 0.5 * d[k] * (z[k] ** 2); g_z[k] += d[k] * z[k]
                elif s[k] <= thr[k]:
                    val += slp[k] * s[k]; g_s[k] += slp[k]
                else:
                    val += 0.5 * d[k] * (s[k] ** 2) + lam0; g_s[k] += d[k] * s[k]
            return val, np.r_[g_z + g_s, -g_z + g_s]
        res = minimize(obj_uv, np.zeros(2 * m), jac=True, bounds=[(0.0, box)] * (2 * m), method="L-BFGS-B", options={"ftol": 1e-13, "gtol": 1e-10})
        z_rel = res.x[:m] - res.x[m:]
        w_rel = np.zeros(p); w_rel[free] = z_rel
        return float(res.fun) - 1e-9, w_rel

    best_val, best_w = eval_subset([])
    lb_root, w_root = relax_lb(set(), set())
    # Greedy warm start from root relaxation magnitudes
    order_init = np.argsort(-np.abs(w_root))
    for k_top in range(1, min(8, p + 1)):
        v_c, w_c = eval_subset(order_init[:k_top].tolist())
        if v_c < best_val:
            best_val, best_w = v_c, w_c

    pq = [(lb_root, 0, frozenset(), frozenset(), w_root)]
    uid = 1; nodes = 0
    while pq:
        lb, _, f0, f1, w_rel = heapq.heappop(pq)
        nodes += 1
        if lb >= best_val - 1e-7:
            return {"weights": best_w.tolist(), "objective": float(best_val), "certified_lower_bound": float(min(lb, best_val)), "nodes_explored": nodes}
        act_cand = [j for j in range(p) if j not in f0 and (j in f1 or abs(w_rel[j]) > 0.15 * box)]
        v_h, w_h = eval_subset(act_cand)
        if v_h < best_val:
            best_val, best_w = v_h, w_h
        undec = [j for j in range(p) if j not in f0 and j not in f1]
        if not undec:
            v_leaf, w_leaf = eval_subset(sorted(f1))
            if v_leaf < best_val:
                best_val, best_w = v_leaf, w_leaf
            continue
        br_j = max(undec, key=lambda j: abs(w_rel[j]))
        for nf0, nf1 in ((f0 | {br_j}, f1), (f0, f1 | {br_j})):
            c_lb, c_w = relax_lb(nf0, nf1)
            if c_lb < best_val - 1e-7:
                heapq.heappush(pq, (c_lb, uid, nf0, nf1, c_w)); uid += 1
    return {"weights": best_w.tolist(), "objective": float(best_val), "certified_lower_bound": float(best_val), "nodes_explored": nodes}
''',
        ),
    }

    with tempfile.TemporaryDirectory(prefix="m88_uf_ref_") as td:
        td_p = Path(td)
        for t in tasks:
            fname, src = ref_solvers[t.task_id]
            art_root = td_p / t.short_name / "holdout_task"
            art_root.mkdir(parents=True, exist_ok=True)
            (art_root / fname).write_text(src, encoding="utf-8")
            tar_fp = td_p / f"{t.short_name}.tar"
            subprocess.run(["tar", "-cf", str(tar_fp), "-C", str(td_p / t.short_name), "holdout_task"], check=True)
            ver = run_unseen_postfreeze_private_verifier_by_id(t.task_id, tar_fp)
            assert ver["strict_pass"] is True, f"Reference solver failed on `{t.task_id}`: {ver}"
            if t.short_name == "ert_contact_tomography":
                from hegel_machine.cognition.m6_9_a_dialectical_synthesis_engine import grade_archived_branch_post_lock
                from hegel_machine.cognition.m8_8_unseen_postfreeze_benchmark import unseen_postfreeze_tasks_as_frontier_list

                ft_uf = next(x for x in unseen_postfreeze_tasks_as_frontier_list() if x.task_id == t.task_id)
                ver_pl = grade_archived_branch_post_lock(task=ft_uf, branch_summary={}, branch_tar=tar_fp, codex_info={})
                assert ver_pl["strict_pass"] is True
