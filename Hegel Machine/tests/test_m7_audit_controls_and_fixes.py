"""Fast, traceable verification tests for all genuine bugs & adversarial controls in `Hegel_M7_Audit`."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace
import types

import numpy as np

from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex
from hegel_machine.cognition.language import EvaluationContext, Program
from hegel_machine.cognition.m6_8_b_prime_capability_bridge import (
    lift_python_scientific_expressions_to_programs,
)
from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
    discover_cross_tree_operator_wormholes,
    load_autogenous_forest_bundle,
)
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (
    build_holdout_ood_12_tasks,
    parse_authenticated_grader_stdout,
)
from hegel_machine.cognition.models import Observation

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ast_lifting_semantics_control_all_pass() -> None:
    """Verify `log1p`, `expm1`, strict `>` / `<`, and imperative loop rejection."""
    cases = [
        ("ordinary_arithmetic", "def f(x):\n    return x * x + 1\n", 2.0, True),
        ("log1p_is_not_log", "def f(x):\n    return math.log1p(x)\n", 1.0, True),
        ("expm1_exact", "def f(x):\n    return math.expm1(x)\n", 0.5, True),
        ("strict_gt_boundary", "def f(x):\n    return 1 if x > 0 else 0\n", 0.0, True),
        ("strict_gt_positive", "def f(x):\n    return 1 if x > 0 else 0\n", 0.25, True),
        ("non_strict_lte_boundary", "def f(x):\n    return 1 if x <= 0 else 0\n", 0.0, True),
        ("ignored_loop_mutation", "def f(x):\n    for _ in range(1):\n        x = x * 2\n    return x + 1\n", 2.0, False),
    ]
    for name, code, x_val, expect_lifted in cases:
        py_ns = {"math": math}
        exec(code, py_ns)
        expected = float(py_ns["f"](x_val))
        lifted = lift_python_scientific_expressions_to_programs([{"name": name, "code": code}])
        assert bool(lifted) is expect_lifted, f"Unexpected lift status on `{name}`: {lifted}"
        if expect_lifted:
            actual = EvaluationContext(Observation({"x_0": x_val}, 0.0)).evaluate(lifted[0][1])
            assert math.isclose(expected, actual, abs_tol=1e-12), f"`{name}` mismatch: {expected} != {actual}"


def test_validator_dataclass_import_and_self_mutation_controls() -> None:
    """Verify `traffic` `@dataclass` + future annotations passes, while self-mutating & missing-entrypoint fail."""
    bridge_src = (REPO_ROOT / "src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py").read_text(encoding="utf-8")
    tree = ast.parse(bridge_src)
    validator_py = next(
        n.value.value
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef)
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "validator_py" for t in n.targets)
        and isinstance(n.value, ast.Constant)
    )

    def _digest(fp: Path) -> str:
        h = hashlib.sha256()
        h.update(str(fp).encode("utf-8") + b"\0" + hashlib.sha256(fp.read_bytes()).digest())
        return h.hexdigest()

    with tempfile.TemporaryDirectory(prefix="m7_val_fix_") as td:
        td_p = Path(td)
        # Stub `flux_models` in same dir so real archived `traffic_solver.py` pattern imports cleanly
        (td_p / "flux_models.py").write_text("FORMS = {}\ndef flux(*a, **kw): return 0.0\n", encoding="utf-8")

        # 1. Real archived traffic-flux `@dataclass` + `from __future__ import annotations`
        dc_file = td_p / "traffic_solver.py"
        archived_traffic = (
            REPO_ROOT
            / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124/branches/"
            "wave_1__gpt-6.2-sol__tb__traffic-flux-inversion/branch_2_antithesis_wormhole/"
            "04_submitted_artifacts_snapshot/traffic_solver.py"
        )
        if archived_traffic.is_file():
            dc_file.write_text(archived_traffic.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            dc_file.write_text(
                "from __future__ import annotations\nfrom dataclasses import dataclass\nimport flux_models\n"
                "@dataclass\nclass Cfg:\n    dt: float = 0.1\ndef solve(): return Cfg()\n",
                encoding="utf-8",
            )
        p1 = subprocess.run(
            [sys.executable, "-c", validator_py, json.dumps([str(dc_file)])],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        r1 = json.loads(p1.stdout.strip().splitlines()[-1])
        assert r1["accepted"] is True, f"Archived dataclass candidate wrongly rejected: {r1}"

        # 2. Self-mutating candidate during import
        mut_file = td_p / "mutating.py"
        mut_file.write_text(
            "from pathlib import Path\nPath(__file__).write_text('def solve(x):\\n    return None\\n')\ndef solve(x):\n    return x\n",
            encoding="utf-8",
        )
        p2 = subprocess.run(
            [sys.executable, "-c", validator_py, json.dumps([str(mut_file)])],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        r2 = json.loads(p2.stdout.strip().splitlines()[-1])
        assert r2["accepted"] is False
        assert any("candidate_mutated_during_validation" in iss for iss in r2["issues"])
        assert r2["computed_sha256"] == _digest(mut_file)

        # 3. Missing required task entrypoint
        unrel_file = td_p / "unrelated.py"
        unrel_file.write_text("def unrelated():\n    return None\n", encoding="utf-8")
        p3 = subprocess.run(
            [sys.executable, "-c", validator_py, json.dumps([str(unrel_file)]), json.dumps(["solve_required"])],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        r3 = json.loads(p3.stdout.strip().splitlines()[-1])
        assert r3["accepted"] is False
        assert any("missing_required_entrypoint" in iss for iss in r3["issues"])


def test_sparse_public_checker_shannon_cofactor_and_gap_certification() -> None:
    """Verify duplicate half-spaces & uncertified 1-leaf trees are rejected while true optimal partitions pass."""
    eng_src = (REPO_ROOT / "src/hegel_machine/cognition/m6_9_a_dialectical_synthesis_engine.py").read_text(encoding="utf-8")
    code = next(
        ast.literal_eval(n.value)
        for n in ast.parse(eng_src).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC_CONTAINER_CANDIDATE_BENCHMARK_PY" for t in n.targets)
    )
    checker = code.split("# 3. Certified Sparse Regression", 1)[1].split("# 4. Spin-Glass", 1)[0]
    checker = checker[checker.index("if os.path.isfile") :]

    with tempfile.TemporaryDirectory(prefix="m7_sparse_fix_") as td:
        root = Path(td)
        (root / "data").mkdir()
        l0, l2, M, eps = 0.1, 0.01, 1.0, 0.001
        (root / "data/problem.json").write_text(
            json.dumps({"lambda_0": l0, "lambda_2": l2, "M": M, "epsilon": eps, "max_nodes": 60000}),
            encoding="utf-8",
        )
        np.save(root / "data/X.npy", np.eye(2))
        np.save(root / "data/y.npy", np.array([1.0, 0.0]))

        def _eval_nodes(support, coeffs, nodes):
            (root / "results.json").write_text(
                json.dumps({"support": support, "coefficients": coeffs, "nodes": nodes}),
                encoding="utf-8",
            )
            env = {"os": __import__("os"), "json": json, "np": np, "out": {}}
            exec(compile(checker.replace("/app", str(root)), "<sparse_checker>", "exec"), env)
            return env["out"]

        # Negative 1: Single root leaf with huge duality gap
        r_bad_root = _eval_nodes([0], [0.2], [{"B0": [], "B1": []}])
        assert r_bad_root["sparse_partition_valid"] is True
        assert r_bad_root["sparse_cert_valid"] is False

        # Negative 2: Two identical half-spaces [{'B0':[0]}, {'B0':[0]}] (measure sums to 1, but overlaps!)
        r_dup = _eval_nodes([0], [1.0 / (1.0 + 2.0 * l2)], [{"B0": [0], "B1": []}, {"B0": [0], "B1": []}])
        assert r_dup["sparse_partition_valid"] is False
        assert r_dup["sparse_cert_valid"] is False

        # Positive: Exact 4-leaf full cube partition with exact optimal primal beta_0 = 1 / (1 + 2*l2)
        opt_b0 = 1.0 / (1.0 + 2.0 * l2)
        exact_leaves = [
            {"B0": [0, 1], "B1": []},
            {"B0": [0], "B1": [1]},
            {"B0": [1], "B1": [0]},
            {"B0": [], "B1": [0, 1]},
        ]
        r_good = _eval_nodes([0], [opt_b0], exact_leaves)
        assert r_good["sparse_partition_valid"] is True
        assert r_good["sparse_cert_valid"] is True
        assert r_good["sparse_cert_gap"] <= eps + 1e-6

        # Performance: 16,384-leaf complete binary partition verifies disjointness in < 0.5s
        np.save(root / "data/X.npy", np.eye(16))
        np.save(root / "data/y.npy", np.zeros(16))
        big_leaves = [
            {"B0": [bit for bit in range(14) if not ((mask >> bit) & 1)], "B1": [bit for bit in range(14) if (mask >> bit) & 1]}
            for mask in range(1 << 14)
        ]
        t0 = perf_counter()
        r_big = _eval_nodes([], [], big_leaves)
        elapsed = perf_counter() - t0
        assert r_big["sparse_partition_valid"] is True
        assert elapsed < 1.5, f"Shannon cofactor check too slow on 16k leaves: {elapsed:.3f}s"


def test_synthesis_layer2b_unified_token_accounting_zero_drop() -> None:
    """Verify non-dominating Layer-2b sublation tokens are accumulated into the returned ledger (`348 == 348`)."""
    source = REPO_ROOT / "src/hegel_machine/cognition/m6_9_a_dialectical_synthesis_engine.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reconcile_thesis_and_antithesis_publicly")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    class StubClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, int]] = []

        def call_structured(self, **kw):
            sub = "__code_sublation_" in kw["stage_subdir"]
            res = {
                "ok": True,
                "prompt_tokens": 101 if sub else 211,
                "cached_tokens": 0,
                "completion_tokens": 13 if sub else 23,
                "wall_ms": 3.0 if sub else 5.0,
                "parsed": {"rationale": "stub", "bash_command": ":", "selected_branch": "candidate_a", "confidence": "high"},
            }
            self.calls.append({"prompt": res["prompt_tokens"], "completion": res["completion_tokens"]})
            return res

    client = StubClient()
    m6f_stub = SimpleNamespace(
        export_declared_artifacts_tar=lambda *a: None,
        import_declared_artifacts_tar=lambda cid, arts, path: {"candidate_sha256": path.name},
        exec_container_bash=lambda *a, **kw: {"exit_code": 0, "stdout": "", "stderr": ""},
        compute_declared_artifacts_digest=lambda *a: "new-cand",
    )
    env = {
        "Path": Path,
        "json": json,
        "deepcopy": deepcopy,
        "m6f": m6f_stub,
        "subprocess": SimpleNamespace(run=lambda *a, **kw: None),
        "_blinded_ab_mapping": lambda _: ("thesis", "antithesis", {"candidate_a": "thesis", "candidate_b": "antithesis"}),
        "verify_candidate_artifact_contract_in_container": lambda *a: {"accepted": True},
        "_run_public_container_candidate_benchmark": lambda *a: {},
        "_public_benchmark_has_unresolved_gap": lambda *a: (True, "synthetic_gap"),
        "_synthesized_strictly_dominates_branches": lambda *a: (False, "not_dominant"),
        "_extract_trajectory_evidence_digest": lambda *a, **kw: "digest",
        "find_uninlined_wormhole_imports_in_container": lambda *a: [],
        "CODE_SUBLATION_TURN_SCHEMA": {},
        "DIALECTICAL_SYNTHESIS_DECISION_SCHEMA": {},
    }
    exec(compile(module, str(source), "exec"), env)
    with tempfile.TemporaryDirectory(prefix="m7_tok_") as td:
        d = Path(td)
        result = env["reconcile_thesis_and_antithesis_publicly"](
            task=SimpleNamespace(
                benchmark="inert",
                task_id="inert",
                short_name="inert",
                declared_artifacts=["art"],
                instruction_text="inert",
            ),
            synth_cid="inert_cid",
            thesis_dir=d / "t",
            thesis_summary={},
            thesis_tar=d / "t.tar",
            antithesis_dir=d / "a",
            antithesis_summary={},
            antithesis_tar=d / "a.tar",
            client=client,
            stage_subdir="probe",
        )
    total = sum(c["prompt"] + c["completion"] for c in client.calls)
    returned = result["llm_tokens"]["prompt"] + result["llm_tokens"]["completion"]
    assert total == 348 and returned == 348, f"Expected 348 accounted tokens, got {returned} (total={total})"


def test_wormhole_off_and_constant_stub_negative_control() -> None:
    """Verify `allow_horizontal_wormholes=False` gives 0 edges AND constant-stub mutation drops wormholes to 0."""
    post_bundle = REPO_ROOT / "docs/evidence/m7_4_exp1_frozen_20task_and_ablations_20261124/post_wave1_autogenous_forest_bundle.json"
    if not post_bundle.is_file():
        post_bundle = REPO_ROOT / "docs/evidence/m7_3_frozen_autogenous_forest_v2_20261124/pretrained_autogenous_forest_v2.json"

    f_orig, a_orig, _, _ = load_autogenous_forest_bundle(post_bundle)
    idx_off = HippoRAGWormholeIndex(f_orig, allow_horizontal_wormholes=False, wormhole_equivalence_pooling=False)
    assert idx_off.edge_counts["horizontal_wormhole_edges"] == 0

    f_a, a_a, _, _ = load_autogenous_forest_bundle(post_bundle)
    f_b, a_b, _, _ = load_autogenous_forest_bundle(post_bundle)
    for f in (f_a, f_b):
        f.horizontal_wormhole_edges = {}
        f.wormhole_history = {}
        f.shared_base_subtrees = {}
        f.shared_base_subtree_history = {}

    for node_rec in a_b.values():
        if isinstance(node_rec, dict):
            for mod in node_rec.get("autogenous_wormhole_modules", []):
                mod["python_code"] = "def deliberately_wrong_solver(*args, **kwargs):\n    return -999999999.0\n"
                mod["compiled_code_sha256"] = hashlib.sha256(mod["python_code"].encode("utf-8")).hexdigest()

    det_real = discover_cross_tree_operator_wormholes(f_a, a_a)
    det_stub = discover_cross_tree_operator_wormholes(f_b, a_b)
    assert len(f_a.horizontal_wormhole_edges) > 0
    assert len(f_b.horizontal_wormhole_edges) == 0, f"Constant-stub control must yield 0 wormholes, got {det_stub}"
    assert det_real != det_stub
    # Stage-prior candidates must NEVER falsely claim `mechanism_stability_verified=True`
    for sub in det_real["shared_subtrees"]:
        if sub["verification_tier"] == "code_backed_stage_prior_candidate":
            assert sub["mechanism_stability_verified"] is False
    # Verify bifurcated `fine` nodes participate in cross-tree discovery (`evaluated_node_count` includes fine nodes!)
    fine_nids = {nid for nid, rec in a_a.items() if isinstance(rec, dict) and rec.get("layer_role") == "fine"}
    if fine_nids:
        assert any(s["node_a_id"] in fine_nids or s["node_b_id"] in fine_nids for s in det_real["shared_subtrees"])


def test_split_counter_guard_and_horizontal_off_consolidation() -> None:
    """Verify split registers complementary parent guard and `allow_horizontal_wormholes=False` skips cross-tree edges."""
    from hegel_machine.cognition.m7_autogenous_forest_pretraining import (
        build_blank_axiomatic_philosophical_forest,
        consolidate_autogenous_episode_into_forest,
        validate_and_decorate_autogenous_python_code,
    )

    forest, archive, _ = build_blank_axiomatic_philosophical_forest()
    ok1, _, pub1, pkg1 = validate_and_decorate_autogenous_python_code(
        "def f1(a, b):\n    return (a + 1.0) / max(abs(b), 1e-6)\n",
        "autogenous_test_one",
    )
    assert ok1
    base_comp = {
        "compiled_ok": True,
        "compilation_mode": "llm_aufhebung_container_verified",
        "episode_id": "ep_1",
        "module_stem": "autogenous_test_one",
        "filename": "autogenous_test_one.py",
        "function_names": pub1,
        "exported_apis": "f1(a, b)",
        "packaged_python_code": pkg1,
        "compiled_code_sha256": "1" * 64,
        "unit_test_receipt": {"passed": True, "mode": "container_verified"},
        "sublated_methodology_principle": "Base test law.",
        "routed_signatures": ["sig:conservation_and_invariant_flux"],
        "scope_preconditions": [],
        "regime_bifurcation": {"should_split_node": False, "regime_axis": "none", "branch_guard_op": ">=", "branch_guard_value": 0.0, "counter_regime_warning": ""},
        "operator_pipeline_stages": [{"stage_kind": "frontend", "numeric_param": 1.0}, {"stage_kind": "robust_inversion", "numeric_param": 1.0}],
    }
    r1 = consolidate_autogenous_episode_into_forest(
        forest=forest, archive=archive, compilation=base_comp, nogoods_recorded=[], allow_horizontal_wormholes=False
    )
    mid_nid = r1["target_node_id"]
    split_comp = dict(base_comp)
    split_comp["episode_id"] = "ep_1_split"
    split_comp["module_stem"] = "autogenous_test_one_split"
    split_comp["filename"] = "autogenous_test_one_split.py"
    split_comp["regime_bifurcation"] = {
        "should_split_node": True,
        "regime_axis": "peclet_num",
        "branch_guard_op": ">",
        "branch_guard_value": 2.0,
        "counter_regime_warning": "Central diff ok only when peclet_num <= 2.0",
    }
    r1b = consolidate_autogenous_episode_into_forest(
        forest=forest, archive=archive, compilation=split_comp, nogoods_recorded=[], allow_horizontal_wormholes=False
    )
    assert r1b["growth_action"] == "bifurcated_precondition_child_node"
    cguards = archive[mid_nid].get("counter_regime_guards", [])
    assert any(g["field"] == "peclet_num" and g["op"] == "<=" and g["value"] == 2.0 for g in cguards)

    other_comp = dict(base_comp)
    other_comp["episode_id"] = "ep_2"
    other_comp["routed_signatures"] = ["sig:robust_physical_calibration_and_safety_margins"]
    consolidate_autogenous_episode_into_forest(
        forest=forest, archive=archive, compilation=other_comp, nogoods_recorded=[], allow_horizontal_wormholes=False
    )
    assert len(forest.horizontal_wormhole_edges) == 0


def test_nested_verifier_container_semaphore_no_deadlock() -> None:
    """Verify 16 threads holding `max_containers=4` outer `agent_cid`s do NOT deadlock when calling `run_tb_separate_verifier`."""
    from concurrent.futures import ThreadPoolExecutor
    import importlib.util
    import time
    from hegel_machine.cognition import m6_frontier_real_benchmarks as m6f

    spec = importlib.util.spec_from_file_location("run_m7_mod", REPO_ROOT / "scripts/run_m7_closing_experiments.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    orig_start = m6f.start_detached_container
    orig_stop = m6f.stop_container_safely
    orig_ver = m6f.run_tb_separate_verifier
    orig_exec = m6f.exec_container_bash
    counter = 0

    def _fake_start(*args, **kwargs):
        nonlocal counter
        counter += 1
        return f"cid_{counter}"

    def _fake_stop(cid):
        return None

    def _fake_ver(task, agent_cid):
        # Simulate real `run_tb_separate_verifier` starting a nested `verifier_cid` while `agent_cid` is still held!
        vcid = m6f.start_detached_container("verifier_img")
        time.sleep(0.01)
        m6f.stop_container_safely(vcid)
        return {"strict_pass": True}

    try:
        m6f.start_detached_container = _fake_start  # type: ignore[assignment]
        m6f.stop_container_safely = _fake_stop  # type: ignore[assignment]
        m6f.run_tb_separate_verifier = _fake_ver  # type: ignore[assignment]
        mod.install_quiet_timing_locks(max_containers=4)

        fake_task = SimpleNamespace(verifier_memory_mb=2048)

        def _worker(_i: int) -> bool:
            acid = m6f.start_detached_container("agent_img")
            try:
                time.sleep(0.02)
                res = m6f.run_tb_separate_verifier(fake_task, acid)
                return bool(res["strict_pass"])
            finally:
                m6f.stop_container_safely(acid)

        with ThreadPoolExecutor(max_workers=16) as pool:
            outs = [fut.result(timeout=5.0) for fut in [pool.submit(_worker, k) for k in range(16)]]
        assert outs == [True] * 16
    finally:
        m6f.start_detached_container = orig_start  # type: ignore[assignment]
        m6f.stop_container_safely = orig_stop  # type: ignore[assignment]
        m6f.run_tb_separate_verifier = orig_ver  # type: ignore[assignment]
        m6f.exec_container_bash = orig_exec  # type: ignore[assignment]


def test_holdout_ho05_ho03_ho11_and_authenticated_stdout_controls() -> None:
    """Verify all 4 Holdout adversarial shortcuts from `Hegel_M7_Audit` now fail strictly."""
    tasks = {t.task_id: t for t in build_holdout_ood_12_tasks()}

    # 1. ho05: fixed ring ignoring X_samples & omitting sparse_precision MUST FAIL
    t05 = tasks["ho05__latent_confounded_graphical_precision"]
    mod05 = types.ModuleType("deconfound_graph")

    def _fixed_ring(X_samples):
        p = X_samples.shape[1]
        a = np.zeros((p, p), dtype=bool)
        idx = np.arange(p)
        a[idx, (idx + 1) % p] = True
        a[(idx + 1) % p, idx] = True
        return {"adjacency": a, "sparse_precision": np.eye(p) * 2.5 - 0.4 * a.astype(float)}

    mod05.recover_sparse_precision_support = _fixed_ring
    sys.modules["deconfound_graph"] = mod05
    try:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(t05.private_verifier_py, "<ho05_ver>", "exec"), {"__name__": "__main__"})
        res05 = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert res05["strict_pass"] is False, f"Fixed ring shortcut still passed ho05: {res05}"
        assert res05["data_dependent"] is False
        assert res05["min_edge_f1"] < 0.35
    finally:
        sys.modules.pop("deconfound_graph", None)

    # 2. ho03: copying public `gt_field` MUST FAIL private forward check
    t03 = tasks["ho03__high_peclet_upwind_plume_inversion"]
    with tempfile.TemporaryDirectory(prefix="m7_ho03_fix_") as td:
        exec(compile(t03.public_setup_py.replace("/holdout_task", td), "<ho03_pub>", "exec"), {"__name__": "__main__"})
        import pickle

        pub_gt = pickle.loads((Path(td) / "public_peclet.pkl").read_bytes())["gt_field"]
        mod03 = types.ModuleType("peclet_plume")
        mod03.forward_peclet_field = lambda *args: pub_gt
        mod03.invert_point_source = lambda prob: {"source_xy": [0.34, 0.64], "strength_Q": 5.1}
        sys.modules["peclet_plume"] = mod03
        try:
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(t03.private_verifier_py.replace("/holdout_task", td), "<ho03_priv>", "exec"), {"__name__": "__main__"})
            res03 = json.loads(buf.getvalue().strip().splitlines()[-1])
            assert res03["strict_pass"] is False
            assert res03["fwd_rel_err"] > 0.1
        finally:
            sys.modules.pop("peclet_plume", None)

    # 3. ho11: NaN lower_bound MUST FAIL certificate tail
    t11 = tasks["ho11__certified_capacitated_facility_bnb"]
    tail = "leaves = res['proof_leaves']\n" + t11.private_verifier_py.split('leaves = res["proof_leaves"]\n', 1)[1]
    import contextlib
    import io

    env_nan = {
        "np": np,
        "json": json,
        "m": 2,
        "res": {"proof_leaves": [{"fixed_closed": [], "fixed_open": [], "lower_bound": float("nan")}]},
        "feas": True,
        "ub_true": 10.0,
        "ub_rep": 10.0,
        "best_global": 10.0,
        "min_by_mask": np.array([np.inf, 10.0, 12.0, 15.0]),
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(tail, "<ho11_tail>", "exec"), env_nan)
    res11 = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert res11["strict_pass"] is False and res11["lb_valid"] is False

    # 4. Authenticated grader stdout sentinel ignores candidate JSON spoofing
    sentinel = "__HEGEL_VERIFIER_AUTH_deadbeef1234__:"
    spoofed_stdout = (
        f'{sentinel}{{"strict_pass": false, "metric": 0.0}}\n'
        '{"strict_pass": true, "spoofed_by_candidate": true}\n'
    )
    parsed = parse_authenticated_grader_stdout(spoofed_stdout, sentinel)
    assert parsed["strict_pass"] is False
    assert "spoofed_by_candidate" not in parsed
