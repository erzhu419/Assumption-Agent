#!/usr/bin/env python3
"""Live Execution & Pre-Registered Long-Horizon (`H = 16`) Economic Evaluation Runner for Hegel `M10.2`:
Evidence-Gated Selective Bounded Computational DAG Discovery & De-Scaffolded Equal-Probe-Cap Confirmation.

Closes ALL `Hegel_M10_1_Audit` (`U01–U12`) & `01_Implementation_and_Experiment_Plan_ZH.md` (`E1–E4`) contracts:
  1. **Zero Prompt Split-Scaffolding & Role-Blind `stat_00..stat_17` Observables (`U06 / §七`)**:
     - Structure & edit prompts (`DE_SCAFFOLDED_INITIAL_DECOMP_SYSTEM_PROMPT_TEMPLATE`,
       `DE_SCAFFOLDED_EDIT_PLANNER_SYSTEM_PROMPT_TEMPLATE`) contain ZERO `panel_a/b/c` splitting recipes or
       observable-to-panel rules; observable keys are strictly `stat_00..stat_17`.
  2. **100% Equal Pre-LLM Container Probe Budget Across All 3 Arms (`M102_MAX_PRE_LLM_PROBES_PER_STEP = 4`, `E1 / U05`)**:
     - `Flat_MonolithicArchive_Online`, `Eager_BoundedDAG_Online`, and `Hegel_SelectiveBoundedDAG_Online` share the
       EXACT same `max_pre_llm_probes = 4` container CV budget, `select_feasible_candidates_ranked_by_public_evidence`,
       and borderline public CV (`0.026 < mae <= 0.038`) 2-candidate comparison rule.
  3. **Single-Factor Causal Attribution (`Eager_BoundedDAG_Online` vs `Hegel_SelectiveBoundedDAG_Online`, `E2`)**:
     - Both DAG arms share identical `1 <= K <= 4, depth <= 4, fan-out <= 2` bounds, de-scaffolded prompts, runtime
       `_TrackedPayloadDict` causal edge consumption & novel-numeric anti-wrapper-shell checks, and strict v2 4-way
       semantic `CutWithdrawalMergeReceipt` verification; they differ ONLY in `evaluate_m102_selective_investment_gate`!
  4. **Zero-LLM Guard Satisfiability Pre-Check & Verified Feasible-Domain Routing (`U02`)**:
     - `build_feasible_domain_negative_witnesses` + `check_axis_aligned_all_of_guard_satisfiability` prevent impossible
       mutual-exclusion guard calls on overlapping domains across all 3 arms.
  5. **Pre-Registered `H = 16` Primary Endpoint (`H = 4, 8, 12` Nested Descriptive Checkpoints), Dual Zero-Call Metrics
     (`solver_free_successes` + `all_role_call_free_successes`), 3 Separate Verdict Booleans, Automatic Structural Gates,
     and Step-14 Cold-Hydration + Plain-Dict Routing Parity Check (`U01, U07, U08, U10, U11`)**.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import monotonic
from typing import Any
import uuid

import numpy as np
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m9_9_autonomous_decomposition_experiment import (  # noqa: E402
    _bootstrap_mean_ci,
    _exact_two_sided_sign_flip_p,
)
from hegel_machine.cognition import (  # noqa: E402
    m6_4_executable_wormhole_library as m64,
    m6_frontier_real_benchmarks as m6f,
    m7_autogenous_forest_pretraining as m7a,
)
from hegel_machine.cognition.cost_policy import (  # noqa: E402
    COST_POLICY_VERSION,
    aggregate_call_usages,
    assert_hierarchical_ledger_conservation,
    score_single_call_usage,
)
from hegel_machine.cognition.hipporag_wormhole_index import HippoRAGWormholeIndex  # noqa: E402
from hegel_machine.cognition.m7_holdout_ood_blind_benchmark import (  # noqa: E402
    HoldoutOODTask,
    ensure_clean_scipy_sandbox_image,
)
from hegel_machine.cognition.m8_persistent_variational_planner import (  # noqa: E402
    PersistentMechanismGraph,
)
from hegel_machine.cognition.m9_4_1_shared_verification_kernel import (  # noqa: E402
    SHARED_KERNEL_VERSION,
)
from hegel_machine.cognition.m9_6_autonomous_structure_continual_world import (  # noqa: E402
    M10_0_1_PROBE_RECEIPT_INDEX_FOREST_KEY,
    ProbeReceiptStore,
    _canonical_json_sha256,
)
from hegel_machine.cognition.m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world import (  # noqa: E402
    propose_and_validate_m99_guard_partition,
)
from hegel_machine.cognition.m10_1_unknown_cut_dag_discovery_world import (  # noqa: E402
    CutWithdrawalMergeReceipt,
    ExecutableComputationDAG,
    ExecutableModuleNode,
    M101_MAX_ACTIVE_NODES,
    M101_MAX_DEPTH,
    M101_MAX_FANOUT,
    M10_1_1_VERIFIER_SCHEMA_VERSION,
    build_feasible_domain_negative_witnesses,
    check_axis_aligned_all_of_guard_satisfiability,
    compile_m101_dag_to_instrumented_solver_code,
    execute_m101_public_gate_in_container,
    grade_m101_candidate_in_container,
    hydrate_m101_full_prefix_state,
    m101_task_to_frontier_task,
    materialize_m101_public_study_and_sha,
    persist_m101_full_prefix_state,
    select_feasible_candidates_ranked_by_public_evidence,
    verify_m101_cut_withdrawal_external_fk,
    verify_m101_feasible_domain_guards,
)
from hegel_machine.cognition.m10_2_selective_bounded_dag_world import (  # noqa: E402
    CONFIRMATION_M102_WORLD_SPECS,
    DEV_M102_WORLD_SPECS,
    DE_SCAFFOLDED_EDIT_PLANNER_SYSTEM_PROMPT_TEMPLATE,
    DE_SCAFFOLDED_INITIAL_DECOMP_SYSTEM_PROMPT_TEMPLATE,
    M102WorldSpec,
    M102_ARMS,
    M102_BORDERLINE_CV_COMPARE_THRESHOLD,
    M102_DAG_LEDGER_FOREST_KEY,
    M102_HORIZON_STEPS,
    M102_MAX_PRE_LLM_PROBES_PER_STEP,
    M102_NESTED_CHECKPOINTS,
    M102_OBSERVABLE_NAMES,
    build_m102_world_tasks,
    compute_m102_blind_observables,
    evaluate_m102_selective_investment_gate,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
PREREG_PROTOCOL_PATH = REPO_ROOT / "docs/evidence/m10_2_preregistered_analysis_protocol.json"
DEFAULT_DEV_OUT_DIR = REPO_ROOT / "docs/evidence/m10_2_dev_selective_bounded_dag_calibration_20261128"
DEFAULT_CONF_OUT_DIR = REPO_ROOT / "docs/evidence/m10_2_selective_bounded_dag_confirmation_20261128"
PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE = -0.05

M102_SCOPE1_ROLES: tuple[str, ...] = (
    "stream_solve",
    "dag_structure_proposal",
    "dag_compile_and_node_synthesis",
    "dag_failed_search",
    "dag_merge_withdrawal_repair",
    "guard_induction",
)

_CALL_CONTEXT = threading.local()
_RECEIPT_HOOK_LOCK = threading.Lock()
_ALL_RECEIPTS_REF: list[dict[str, Any]] | None = None


# ============================================================================
# 1. Provenance, Statistical Inference & Probe Receipt Helpers
# ============================================================================


def _git_and_source_tree_state() -> tuple[str, bool, str]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    st = subprocess.run(["git", "status", "--porcelain", "."], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    h = hashlib.sha256()
    for sub in ("src", "scripts", "tests"):
        for fp in sorted((REPO_ROOT / sub).rglob("*.py")):
            if fp.is_file():
                rel = fp.relative_to(REPO_ROOT).as_posix()
                h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256(fp.read_bytes()).digest())
    return sha, bool(st), h.hexdigest()


def _docker_image_digest(image_name: str) -> str:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image_name], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip().startswith("sha256:"):
        raise RuntimeError(f"Could not determine Docker image digest for `{image_name}`")
    return r.stdout.strip()


def _paired_t_95ci(diffs: list[float]) -> list[float] | None:
    if len(diffs) < 2:
        return None
    arr = np.asarray(diffs, dtype=float)
    sd = float(np.std(arr, ddof=1))
    mean = float(np.mean(arr))
    if sd == 0.0:
        return [round(mean, 6), round(mean, 6)]
    t_crit = float(scipy_stats.t.ppf(0.975, df=len(arr) - 1))
    hw = t_crit * sd / math.sqrt(len(arr))
    return [round(mean - hw, 6), round(mean + hw, 6)]


def paired_cluster_inference_stats(diffs: list[float]) -> dict[str, Any]:
    if not diffs:
        return {
            "n_clusters": 0,
            "mean_diff": None,
            "two_sided_sign_flip_p": None,
            "bootstrap_95ci": None,
            "paired_t_95ci": None,
        }
    return {
        "n_clusters": len(diffs),
        "mean_diff": round(float(np.mean(diffs)), 6),
        "cluster_diffs": [round(float(x), 6) for x in diffs],
        "two_sided_sign_flip_p": _exact_two_sided_sign_flip_p(diffs),
        "bootstrap_95ci": _bootstrap_mean_ci(diffs),
        "paired_t_95ci": _paired_t_95ci(diffs),
    }


def _retag_failed_search_receipts(start_idx: int, world_id: str, arm: str, episode_short_name: str) -> int:
    if _ALL_RECEIPTS_REF is None:
        return 0
    cnt = 0
    with _RECEIPT_HOOK_LOCK:
        for row in _ALL_RECEIPTS_REF[start_idx:]:
            if (
                row.get("world_id") == world_id
                and row.get("arm") == arm
                and row.get("episode_short_name") == episode_short_name
                and row.get("call_role") in {"dag_structure_proposal", "dag_compile_and_node_synthesis"}
            ):
                row["original_call_role"] = row["call_role"]
                row["call_role"] = "dag_failed_search"
                cnt += 1
    return cnt


def _current_receipts_len() -> int:
    if _ALL_RECEIPTS_REF is None:
        return 0
    with _RECEIPT_HOOK_LOCK:
        return len(_ALL_RECEIPTS_REF)


def _persist_m102_probe(
    probe_store: ProbeReceiptStore,
    *,
    probe_kind: str,
    receipt: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    pid = f"m102p_{uuid.uuid4().hex[:16]}"
    rec_with_id = {**receipt, "probe_id": pid}
    env = probe_store.persist_probe_receipt(
        probe_id=pid,
        receipt=rec_with_id,
        probe_kind=probe_kind,
        metadata=metadata,
    )
    return pid, str(env["receipt_content_sha256"])


def _extract_submitted_files(ep_dir: Path) -> dict[str, str]:
    sdir = ep_dir / "04_submitted_artifacts_snapshot"
    out: dict[str, str] = {}
    if not sdir.is_dir():
        return out
    for fp in sorted(sdir.rglob("*.py")):
        bname = fp.name.split("__", 1)[-1] if "__" in fp.name else fp.name
        out[bname] = fp.read_text(encoding="utf-8")
    return out


def _obs_l1_distance(
    a: dict[str, float | None],
    b: dict[str, float | None],
    keys: tuple[str, ...] = M102_OBSERVABLE_NAMES,
) -> float:
    vals = [abs(float(a[k]) - float(b[k])) for k in keys if a.get(k) is not None and b.get(k) is not None]
    return float(sum(vals) / max(1, len(vals)))


def _gate_cv_score(gate: dict[str, Any]) -> tuple[float, float, float]:
    cvm = gate.get("cv_metrics") or {}
    ym = float(cvm.get("public_cv_y_mae", 0.038))
    tm = float(cvm.get("public_cv_traj_mae", 0.038))
    return ym + tm, ym, tm


def _rank_compositional_node_version_combinations_m102(
    *,
    template_dag: ExecutableComputationDAG,
    node_library: dict[str, ExecutableModuleNode],
    node_pos_obs: dict[str, list[dict[str, float | None]]],
    obs_now: dict[str, float | None],
    already_tried_fingerprints: set[str],
    max_candidates: int = 3,
) -> list[ExecutableComputationDAG]:
    """Data-driven `0`-LLM compositional subgraph replay over already-verified node versions using `stat_00..stat_17`."""
    versions_by_node: dict[str, list[str]] = {}
    for nid in template_dag.nodes:
        vids = sorted(vid for vid, m in node_library.items() if m.node_id == nid)
        versions_by_node[nid] = vids if vids else [template_dag.node_versions[nid]]

    if all(len(v) <= 1 for v in versions_by_node.values()):
        return []

    scored_by_node: dict[str, list[tuple[float, str]]] = {}
    for nid, vids in versions_by_node.items():
        if len(vids) == 1:
            scored_by_node[nid] = [(0.0, vids[0])]
            continue
        vals_by_v: dict[str, dict[str, list[float]]] = {}
        for vid in vids:
            wl = node_pos_obs.get(vid) or []
            vals_by_v[vid] = {
                k: [float(w[k]) for w in wl if w.get(k) is not None]
                for k in M102_OBSERVABLE_NAMES
            }
        fisher_feats: list[tuple[float, float, str]] = []
        for k in M102_OBSERVABLE_NAMES:
            v_means = [float(np.mean(vals_by_v[v][k])) for v in vids if vals_by_v[v][k]]
            if len(v_means) < 2:
                continue
            sp = max(v_means) - min(v_means)
            if sp < 0.07:
                continue
            w_vars = [float(np.var(vals_by_v[v][k])) for v in vids if len(vals_by_v[v][k]) >= 2]
            within_v = float(np.mean(w_vars)) if w_vars else 0.0
            f_ratio = (sp ** 2) / (within_v + 0.01)
            fisher_feats.append((f_ratio, sp, k))
        fisher_feats.sort(reverse=True)
        if fisher_feats:
            f_max = fisher_feats[0][0]
            top_feats = [x for x in fisher_feats if x[0] >= 0.40 * f_max][:3]
            if not top_feats:
                top_feats = fisher_feats[:2]
        else:
            top_feats = [(1.0, 1.0, k) for k in M102_OBSERVABLE_NAMES]

        v_scores: list[tuple[float, str]] = []
        for vid in vids:
            wl = node_pos_obs.get(vid) or []
            if not wl:
                v_scores.append((999.0, vid))
                continue
            best_d = 999.0
            for w in wl:
                d_acc = 0.0
                w_acc = 0.0
                for f_r, sp, k in top_feats:
                    if w.get(k) is not None and obs_now.get(k) is not None:
                        d_acc += f_r * (abs(float(w[k]) - float(obs_now[k])) / max(sp, 0.05))
                        w_acc += f_r
                if w_acc > 0:
                    best_d = min(best_d, d_acc / w_acc)
            v_scores.append((best_d, vid))
        v_scores.sort(key=lambda x: (x[0], x[1]))
        scored_by_node[nid] = v_scores

    combos: list[tuple[float, dict[str, str]]] = [(0.0, {})]
    for nid in template_dag.nodes:
        nxt_combos: list[tuple[float, dict[str, str]]] = []
        for sc0, mapping in combos:
            for sc_v, vid in scored_by_node[nid][:3]:
                m2 = dict(mapping)
                m2[nid] = vid
                nxt_combos.append((sc0 + sc_v, m2))
        combos = nxt_combos
    combos.sort(key=lambda x: (x[0], sorted(x[1].items())))

    out_dags: list[ExecutableComputationDAG] = []
    for idx, (_sc, nv_map) in enumerate(combos):
        cand_dag = ExecutableComputationDAG(
            dag_id=f"{template_dag.dag_id}_r{idx:02d}",
            nodes=list(template_dag.nodes),
            edges=list(template_dag.edges),
            sink_node_id=template_dag.sink_node_id,
            node_versions=nv_map,
            applicability_guard={},
            priority=template_dag.priority + 1,
            origin_action="reuse_subgraph",
            origin_step=0,
        )
        fp = cand_dag.topology_fingerprint()
        if fp in already_tried_fingerprints:
            continue
        already_tried_fingerprints.add(fp)
        out_dags.append(cand_dag)
        if len(out_dags) >= max_candidates:
            break
    return out_dags


# ============================================================================
# 2. Equal-Info Archive Injection & De-Scaffolded Node-Update Sandbox Builder
# ============================================================================


def _inject_m102_equal_info_archive(
    task: HoldoutOODTask,
    prior_entries: list[dict[str, Any]],
    pre_llm_diag: dict[str, Any] | None = None,
) -> tuple[HoldoutOODTask, str]:
    """Identical `/tmp/hegel_continual_archive/` staging across all 3 arms (`0` helper ban, `0` topology hints!)."""
    extra = [
        "\nimport json as _j, os as _o",
        "_o.makedirs('/tmp/hegel_continual_archive', exist_ok=True)",
    ]
    manifest = []
    for ent in prior_entries:
        tag = str(ent["short_name"])
        code_txt = str(ent["code"])
        extra.append(f"open('/tmp/hegel_continual_archive/{tag}_solver.py', 'w').write({code_txt!r})")
        manifest.append(
            {
                "short_name": tag,
                "task_id": ent["task_id"],
                "solver_path": f"/tmp/hegel_continual_archive/{tag}_solver.py",
                "stat_summary": ent.get("obs"),
            }
        )
    extra.append(f"open('/tmp/hegel_continual_archive/archive_manifest.json', 'w').write({json.dumps(manifest, indent=2)!r})")
    if pre_llm_diag is not None:
        extra.append(
            f"open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json', 'w').write({json.dumps(pre_llm_diag, indent=2)!r})"
        )
    aug = HoldoutOODTask(
        **{
            **task.__dict__,
            "public_setup_py": task.public_setup_py + "\n" + "\n".join(extra) + "\n",
        }
    )
    hdr = (
        "CONTINUAL CAMPAIGN ARCHIVE:\n"
        "- Previously verified campaign solvers: `/tmp/hegel_continual_archive/archive_manifest.json`\n"
        "- Pre-LLM public 52/20 CV gate diagnostic: `/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json`\n"
        "- You may freely inspect prior solvers, reuse/refactor helper functions or sub-computations, and validate via "
        "internal 52/20 held-out CV on `train` (`<= 0.038`) before finishing.\n"
    )
    return aug, hdr


def _build_m102_targeted_node_synthesis_task(
    *,
    base_task: HoldoutOODTask,
    target_dag: ExecutableComputationDAG,
    frozen_nodes: dict[str, ExecutableModuleNode],
    editable_node_ids: list[str],
    edit_action: str,
    edit_rationale: str,
    prior_entries: list[dict[str, Any]],
    pre_llm_diag: dict[str, Any],
) -> HoldoutOODTask:
    """De-scaffolded Docker sandbox task where ONLY `editable_node_ids` are written into `/holdout_task/{nid}.py`.
    Includes the EXACT `_TrackedPayloadDict` incoming-edge consumption & novel-numeric anti-wrapper-shell checks in
    `/holdout_task/test_dag_candidate.py` and ZERO `panel_a/b/c` domain coaching (`U06`)!
    """
    aug_task, _ = _inject_m102_equal_info_archive(base_task, prior_entries, pre_llm_diag)
    topo_info = target_dag.validate_topology()
    incoming: dict[str, list[str]] = {n: [] for n in target_dag.nodes}
    for u, v in target_dag.edges:
        incoming[v].append(u)
    for n in incoming:
        incoming[n].sort()

    frozen_map = {nid: m.code for nid, m in frozen_nodes.items()}
    dag_spec = {
        "dag_id": target_dag.dag_id,
        "nodes": list(target_dag.nodes),
        "edges": [list(e) for e in target_dag.edges],
        "topo_order": topo_info["topo_order"],
        "incoming": incoming,
        "sink_node_id": target_dag.sink_node_id,
        "editable_node_ids": list(editable_node_ids),
    }

    tester_py = r'''from __future__ import annotations
from collections.abc import Mapping
import copy, hashlib, json, math, os, pickle, sys
import numpy as np

spec_obj = json.loads(open("/holdout_task/_dag_spec.json", "r", encoding="utf-8").read())
frozen_dir = "/holdout_task/_frozen_nodes"

def _canon_digest(obj) -> str:
    h = hashlib.sha256()
    def _feed(v):
        if v is None: h.update(b"N;")
        elif isinstance(v, bool): h.update(b"B1;" if v else b"B0;")
        elif isinstance(v, (int, np.integer)): h.update(f"I:{int(v)};".encode())
        elif isinstance(v, (float, np.floating)):
            fv = float(v)
            if math.isnan(fv): h.update(b"F:NaN;")
            elif math.isinf(fv): h.update(b"F:Inf;")
            else: h.update(b"F:" + np.asarray([0.0 if fv == 0.0 else fv], dtype="<f8").tobytes() + b";")
        elif isinstance(v, str): h.update(f"S:{len(v)}:{v};".encode())
        elif isinstance(v, np.ndarray):
            h.update(f"A:{v.dtype.kind}:{v.shape}:".encode())
            a = np.asarray(v, dtype=np.float64 if v.dtype.kind == "f" else np.int64)
            h.update(np.where(a == 0, 0, a).tobytes("C") + b";")
        elif isinstance(v, (dict, Mapping)):
            h.update(f"D:{len(v)}:".encode())
            for k in sorted(v.keys(), key=str): _feed(str(k)); _feed(v[k])
            h.update(b";")
        elif isinstance(v, (list, tuple)):
            h.update(f"L:{len(v)}:".encode())
            for it in v: _feed(it)
            h.update(b";")
        else: h.update(repr(v).encode())
    _feed(obj)
    return h.hexdigest()

def _collect_subtree_digests(obj) -> set[str]:
    out = {_canon_digest(obj)}
    if isinstance(obj, (dict, Mapping)):
        for v in obj.values(): out.update(_collect_subtree_digests(v))
    elif isinstance(obj, (list, tuple)):
        for it in obj: out.update(_collect_subtree_digests(it))
    return out

def _collect_numeric_leaf_digests(obj) -> set[str]:
    out = set()
    if isinstance(obj, np.ndarray) and obj.size > 0 and obj.dtype.kind in ("f", "i", "u"):
        out.add(_canon_digest(obj))
    elif isinstance(obj, (float, np.floating)):
        out.add(_canon_digest(float(obj)))
    elif isinstance(obj, (dict, Mapping)):
        for v in obj.values(): out.update(_collect_numeric_leaf_digests(v))
    elif isinstance(obj, (list, tuple)):
        for it in obj: out.update(_collect_numeric_leaf_digests(it))
    return out

class _TrackedPayloadDict(Mapping):
    def __init__(self, data: dict, src_key: str, tracker: set):
        self._data = data
        self._src_key = src_key
        self._tracker = tracker
    def __getitem__(self, k):
        self._tracker.add(self._src_key)
        return self._data[k]
    def get(self, k, default=None):
        self._tracker.add(self._src_key)
        return self._data.get(k, default)
    def __iter__(self):
        self._tracker.add(self._src_key)
        return iter(self._data)
    def __len__(self): return len(self._data)
    def keys(self):
        self._tracker.add(self._src_key)
        return self._data.keys()
    def items(self):
        self._tracker.add(self._src_key)
        return self._data.items()
    def values(self):
        self._tracker.add(self._src_key)
        return self._data.values()
    def __contains__(self, k):
        self._tracker.add(self._src_key)
        return k in self._data

fns = {}
for nid in spec_obj["topo_order"]:
    fp = f"/holdout_task/{nid}.py" if nid in spec_obj["editable_node_ids"] else os.path.join(frozen_dir, f"{nid}.py")
    if not os.path.isfile(fp):
        print(json.dumps({"accepted": False, "error": f"missing_node_file:{fp}"}))
        sys.exit(1)
    ns = {"__name__": f"_node_{nid}"}
    exec(compile(open(fp, "r", encoding="utf-8").read(), fp, "exec"), ns)
    fn = ns.get("run_node") or ns.get("solve_batch")
    if not callable(fn):
        print(json.dumps({"accepted": False, "error": f"missing_run_node:{nid}"}))
        sys.exit(1)
    fns[nid] = fn

def run_dag(study: dict):
    raw_in = copy.deepcopy(study)
    raw_dig = _canon_digest(raw_in)
    outputs = {"__input__": raw_in}
    ec = {n: 0 for n in spec_obj["topo_order"]}
    is_multi = len(spec_obj["topo_order"]) > 1
    for nid in spec_obj["topo_order"]:
        srcs = spec_obj["incoming"][nid]
        consumed = set()
        if not is_multi and srcs == ["__input__"]:
            node_in = _TrackedPayloadDict(copy.deepcopy(outputs["__input__"]), "__input__", consumed)
        else:
            node_in = {s: (_TrackedPayloadDict(copy.deepcopy(outputs[s]), s, consumed) if isinstance(outputs[s], dict) else copy.deepcopy(outputs[s])) for s in srcs}
        in_d = _canon_digest(outputs["__input__"] if (not is_multi and srcs == ["__input__"]) else {s: outputs[s] for s in srcs})
        in_num = set()
        for s in srcs: in_num.update(_collect_numeric_leaf_digests(outputs[s]))
        consumed.clear()
        out_v = fns[nid](node_in)
        ec[nid] += 1
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f"Node {nid} returned non-dict or empty dict")
        if is_multi:
            unused = sorted(set(srcs) - consumed)
            if unused:
                raise RuntimeError(f"DAG node {nid} left declared incoming edges unconsumed: {unused}")
        if nid != spec_obj["sink_node_id"]:
            od = _canon_digest(out_v)
            subs = _collect_subtree_digests(out_v)
            out_num = _collect_numeric_leaf_digests(out_v)
            if od == in_d or raw_dig in subs or any(od == _canon_digest(outputs[s]) for s in srcs) or not out_num or not (out_num - in_num):
                raise RuntimeError(f"DAG node {nid} failed anti-wrapper-shell / novel numeric transformation check")
        outputs[nid] = {k: (dict(v) if isinstance(v, _TrackedPayloadDict) else v) for k, v in out_v.items()}
    return outputs[spec_obj["sink_node_id"]], ec

st = pickle.load(open("/holdout_task/public_batch.pkl", "rb"))[0]["study"]
out_full, ec = run_dag(st)
n_te, k_t = len(st["test"]["x"]), len(st["eval_times"])
errs = []
for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
    if k not in out_full:
        errs.append(f"missing:{k}"); continue
    a = np.asarray(out_full[k], dtype=float)
    if a.shape != shp or not np.all(np.isfinite(a)):
        errs.append(f"bad:{k}:{a.shape}")
if not errs:
    yh, sp, dv = np.asarray(out_full["y_hat"], float), np.asarray(out_full["state_pred"], float), np.asarray(out_full["diagnostics_vec"], float)
    exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
        errs.append("diagnostics_vec_handoff_mismatch")

cv_metrics = {}
if not errs:
    st_cv = copy.deepcopy(st)
    sp_idx = max(40, len(st["train"]["x"]) - 20)
    vx, vy, vtr = st["train"]["x"][sp_idx:].copy(), st["train"]["y"][sp_idx:].copy(), st["train"]["traj"][sp_idx:].copy()
    st_cv["train"] = {
        "x": st["train"]["x"][:sp_idx].copy(),
        "y": st["train"]["y"][:sp_idx].copy(),
        "traj": st["train"]["traj"][:sp_idx].copy(),
        "summary": np.array([float(np.mean(st["train"]["y"][:sp_idx])), float(np.std(st["train"]["y"][:sp_idx])), float(np.mean(st["train"]["traj"][:sp_idx, -1])), float(np.std(st["train"]["traj"][:sp_idx, -1]))], dtype=float),
    }
    st_cv["test"] = {"x": vx}
    out_cv, _ = run_dag(st_cv)
    ym = float(np.mean(np.abs(np.asarray(out_cv["y_hat"], float) - vy)))
    tm = float(np.mean(np.abs(np.asarray(out_cv["state_pred"], float) - vtr)))
    cv_metrics = {"public_cv_y_mae": round(ym, 6), "public_cv_traj_mae": round(tm, 6)}
    if ym > 0.038: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.038: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

ok = len(errs) == 0 and all(v == 1 for v in ec.values())
print(json.dumps({"accepted": ok, "errors": errs, "cv_metrics": cv_metrics, "exec_counts": ec}, allow_nan=False))
sys.exit(0 if ok else 1)
'''

    setup_extra = [
        "\nimport base64 as _b64, json as _j2, os as _o2",
        "_o2.makedirs('/holdout_task/_frozen_nodes', exist_ok=True)",
        f"open('/holdout_task/_dag_spec.json', 'w').write({json.dumps(dag_spec, indent=2)!r})",
        "_o2.chmod('/holdout_task/_dag_spec.json', 0o444)",
        f"open('/holdout_task/test_dag_candidate.py', 'w').write({tester_py!r})",
    ]
    for nid, code_txt in sorted(frozen_map.items()):
        b64 = base64.b64encode(code_txt.encode("utf-8")).decode("ascii")
        setup_extra.append(f"open('/holdout_task/_frozen_nodes/{nid}.py', 'wb').write(_b64.b64decode({b64!r}))")
        setup_extra.append(f"_o2.chmod('/holdout_task/_frozen_nodes/{nid}.py', 0o444)")

    edit_files = tuple(f"/holdout_task/{nid}.py" for nid in editable_node_ids)
    in_desc = "\n".join(
        f"  - `/holdout_task/{nid}.py`: receives `inputs` dict with keys `{incoming[nid]}`"
        + (
            " (SINK node: must read every declared incoming key and return `{'y_hat', 'state_pred', 'diagnostics_vec'}`)"
            if nid == target_dag.sink_node_id
            else " (intermediate node: must read every declared incoming key and return a non-empty dict of newly computed `np.ndarray` features matching downstream consumers)"
        )
        for nid in editable_node_ids
    )
    frozen_desc = ", ".join(f"`/holdout_task/_frozen_nodes/{nid}.py`" for nid in sorted(frozen_map.keys())) or "(none)"

    inst = (
        f"TARGETED COMPUTATIONAL-DAG NODE SYNTHESIS (`action={edit_action}`):\n"
        f"- DAG topology (`nodes={target_dag.nodes}`, `edges={target_dag.edges}`, `sink={target_dag.sink_node_id}`).\n"
        f"- Verified frozen nodes (read-only; inspect them to match exact dict key contracts!): {frozen_desc}\n"
        f"- Planner rationale: {edit_rationale}\n"
        f"- Write ONLY the editable node file(s):\n{in_desc}\n"
        "  Each file must expose `def run_node(inputs: dict) -> dict:`.\n"
        "- Inspect `/holdout_task/public_batch.pkl` and `/tmp/hegel_continual_archive/` numerically, infer the exact "
        "functional transformation for the edited node(s) from the data, and run `python3 /holdout_task/test_dag_candidate.py` "
        "until `accepted: true` (`public_cv_y_mae <= 0.038`, `public_cv_traj_mae <= 0.038`, `exec_count == 1`, all incoming "
        "edges consumed, and novel numeric output verified)!\n"
    )
    return HoldoutOODTask(
        task_id=f"{base_task.task_id}__{edit_action}",
        short_name=base_task.short_name,
        wave=base_task.wave,
        scientific_domain=base_task.scientific_domain,
        wormhole_family=base_task.wormhole_family,
        declared_artifacts=edit_files,
        declared_callable_entrypoints=("run_node",),
        public_pickle_filename=base_task.public_pickle_filename,
        host_only_public_gt_json="",
        instruction_text=inst,
        public_setup_py=aug_task.public_setup_py + "\n" + "\n".join(setup_extra) + "\n",
        private_verifier_py=base_task.private_verifier_py,
    )


# ============================================================================
# 3. De-Scaffolded Unknown-Cut DAG Proposal & Edit Planner (`U06 / §七`)
# ============================================================================


def _propose_descaffolded_initial_dag_or_keep_monolithic(
    *,
    client: Any,
    stage_subdir: str,
    source_code: str,
    source_obs: dict[str, float | None],
    target_obs: dict[str, float | None],
    max_nodes: int,
    max_fanout: int,
) -> dict[str, Any]:
    node_item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["node_id", "code"],
        "properties": {
            "node_id": {"type": "string"},
            "code": {"type": "string"},
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "rationale", "nodes", "edges", "sink_node_id", "node_modules"],
        "properties": {
            "action": {"type": "string", "enum": ["keep_monolithic", "extract_module", "split_node"]},
            "rationale": {"type": "string"},
            "nodes": {"type": "array", "items": {"type": "string"}},
            "edges": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
            },
            "sink_node_id": {"type": "string"},
            "node_modules": {"type": "array", "items": node_item_schema},
        },
    }
    obs_diff = {
        k: {
            "source": source_obs.get(k),
            "target": target_obs.get(k),
            "abs_delta": (
                round(abs(float(target_obs[k]) - float(source_obs[k])), 5)
                if source_obs.get(k) is not None and target_obs.get(k) is not None
                else None
            ),
        }
        for k in M102_OBSERVABLE_NAMES
    }
    sys_prompt = DE_SCAFFOLDED_INITIAL_DECOMP_SYSTEM_PROMPT_TEMPLATE.format(
        max_nodes=max_nodes,
        max_depth=M101_MAX_DEPTH,
        max_fanout=max_fanout,
    )
    payload = {
        "topology_bounds": {"max_nodes": max_nodes, "max_depth": M101_MAX_DEPTH, "max_fanout": max_fanout},
        "source_monolithic_code": source_code,
        "blind_statistic_comparison_source_vs_target": obs_diff,
    }
    return client.call_structured(
        system_prompt=sys_prompt,
        conversation_turns=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        output_schema=schema,
        stage_subdir=stage_subdir,
    )


def _propose_descaffolded_dag_edit_action(
    *,
    client: Any,
    stage_subdir: str,
    active_dags_summary: list[dict[str, Any]],
    node_library_summary: list[dict[str, Any]],
    obs_now: dict[str, float | None],
    ref_obs_by_dag: dict[str, dict[str, float | None]],
    pre_llm_probe_errors: list[str],
    node_health_summary: dict[str, Any] | None,
    max_nodes: int,
    max_fanout: int,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "rationale",
            "base_dag_id",
            "new_nodes",
            "new_edges",
            "new_sink_node_id",
            "reused_node_versions",
            "editable_node_ids",
        ],
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "keep_monolithic",
                    "extract_module",
                    "split_node",
                    "merge_nodes",
                    "reuse_subgraph",
                    "rewire_edge",
                ],
            },
            "rationale": {"type": "string"},
            "base_dag_id": {"type": "string"},
            "new_nodes": {"type": "array", "items": {"type": "string"}},
            "new_edges": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
            },
            "new_sink_node_id": {"type": "string"},
            "reused_node_versions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["node_id", "version_id"],
                    "properties": {
                        "node_id": {"type": "string"},
                        "version_id": {"type": "string"},
                    },
                },
            },
            "editable_node_ids": {"type": "array", "items": {"type": "string"}},
        },
    }
    sys_prompt = DE_SCAFFOLDED_EDIT_PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        max_nodes=max_nodes,
        max_depth=M101_MAX_DEPTH,
        max_fanout=max_fanout,
    )
    payload = {
        "bounds": {"max_nodes": max_nodes, "max_depth": M101_MAX_DEPTH, "max_fanout": max_fanout},
        "current_blind_statistics": obs_now,
        "reference_blind_statistics_by_dag": ref_obs_by_dag,
        "existing_verified_dags": active_dags_summary,
        "verified_node_versions": node_library_summary,
        "pre_llm_probe_errors": pre_llm_probe_errors,
        "node_health_summary": node_health_summary,
    }
    return client.call_structured(
        system_prompt=sys_prompt,
        conversation_turns=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        output_schema=schema,
        stage_subdir=stage_subdir,
    )


def _probe_existing_dag_nodes_health_in_container(
    *,
    task: HoldoutOODTask,
    dag: ExecutableComputationDAG,
    node_library: dict[str, ExecutableModuleNode],
) -> dict[str, Any]:
    code_txt, comp_meta = compile_m101_dag_to_instrumented_solver_code(dag, node_library)
    gate = execute_m101_public_gate_in_container(task=task, candidate_code=code_txt, expect_dag_trace=True)
    tr = gate.get("dag_runtime_trace") or {}
    nodes_clean = bool(
        gate.get("schema_valid")
        and isinstance(tr, dict)
        and tr.get("all_exec_count_eq_1")
        and tr.get("all_declared_edges_consumed", True)
    )
    return {
        "healthy": nodes_clean,
        "dag_id": dag.dag_id,
        "active_node_count": comp_meta["active_node_count"],
        "compiled_sha256": comp_meta["compiled_sha256"],
        "schema_valid": bool(gate.get("schema_valid")),
        "accepted": bool(gate.get("accepted")),
        "cv_metrics": gate.get("cv_metrics") or {},
        "dag_runtime_trace": tr,
        "errors": gate.get("errors") or [],
        "wall_ms": float(gate.get("wall_ms") or 0.0),
    }


# ============================================================================
# 4. Shared `Stage S` (`batch_00`) Runner & 3-Arm Equal-Probe-Cap `H = 16` Engine
# ============================================================================


def _run_stage_s_shared_source(
    *,
    spec: M102WorldSpec,
    s_task: HoldoutOODTask,
    base_v7_forest: Any,
    base_v7_archive: Any,
    codex_info: dict[str, Any],
    model: str,
    reasoning_effort: str,
    out_dir: Path,
    run_id: str,
    max_turns: int,
) -> dict[str, Any]:
    """Solve `Stage S` (`batch_00`) ONCE per independent cluster monolithically (`<= 2` public-only attempts, `0` private peek!).
    Note (`U01`): even if `public_gate` were `False` after 2 attempts, proceeds cleanly with the best candidate rather than
    aborting/restarting!
    """
    wid = spec.world_id
    s_dir = out_dir / "stage_S_shared_sources" / wid
    s_dir.mkdir(parents=True, exist_ok=True)

    _CALL_CONTEXT.role = "stage_S_solve"
    _CALL_CONTEXT.world_id = wid
    _CALL_CONTEXT.cluster_id = spec.cluster_id
    _CALL_CONTEXT.arm = "Stage_S_Shared_Source"
    _CALL_CONTEXT.episode_short_name = s_task.short_name

    s_code = ""
    pub_gate: dict[str, Any] = {"accepted": False}
    tot_s_turns = 0
    attempts_used = 0
    for att_s in (1, 2):
        attempts_used = att_s
        att_dir = s_dir if att_s == 1 else (s_dir / f"retry_att_{att_s}")
        att_dir.mkdir(parents=True, exist_ok=True)
        ctx_retry = (
            ""
            if att_s == 1
            else (
                f"PRIOR PUBLIC 52/20 CV GATE REJECTION (`errors={pub_gate.get('errors')}`, `cv_metrics={pub_gate.get('cv_metrics')}`): "
                "Inspect `/holdout_task/public_batch.pkl` numerically, avoid generic high-dimensional smoothers that underfit 72 rows, "
                "and verify internal 52/20 cross-validation (`<= 0.038`) via `python3 /holdout_task/validate_public_schema.py`!"
            )
        )
        ft_s = m101_task_to_frontier_task(s_task, continual_context_block=ctx_retry)
        forest = deepcopy(base_v7_forest)
        archive = deepcopy(base_v7_archive)
        ep_sum = m6f.run_frontier_episode(
            ft_s,
            "Hegel_M6_9_Antithesis_LiveGPT",
            model=model,
            codex_info=codex_info,
            forest=forest,
            forest_archive=archive,
            hippo_idx=HippoRAGWormholeIndex(forest),
            episode_out_dir=att_dir,
            run_id=f"{run_id}__stageS__{wid}__att{att_s}",
            max_turns=max_turns,
            cmd_timeout_sec=180,
            reasoning_effort=reasoning_effort,
            defer_official_verifier=True,
            enable_auto_skill_packaging=False,
            allow_l1_primitives=False,
            disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated",
        )
        tot_s_turns += int(ep_sum.get("turns_used", 0) or 0)
        sub = _extract_submitted_files(att_dir)
        cand_code = sub.get("batch_solver.py", "")
        cand_gate = (
            execute_m101_public_gate_in_container(task=s_task, candidate_code=cand_code, expect_dag_trace=False)
            if cand_code
            else {"accepted": False}
        )
        if cand_code and (not s_code or cand_gate.get("accepted")):
            s_code = cand_code
            pub_gate = cand_gate
        if pub_gate.get("accepted"):
            break

    # Official private grading executes ONCE after public loop finishes (`0` private feedback to policy!)
    priv_ver = grade_m101_candidate_in_container(task=s_task, candidate_code=s_code) if s_code else {"strict_pass": False}
    s_study, s_pub_sha, s_num_sha = materialize_m101_public_study_and_sha(s_task)
    s_obs = compute_m102_blind_observables(s_study)

    (s_dir / "stage_s_public_gate.json").write_text(json.dumps(pub_gate, indent=2) + "\n", encoding="utf-8")
    (s_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")

    rec = {
        "world_id": wid,
        "cluster_id": spec.cluster_id,
        "stratum": spec.stratum,
        "topology_class_hidden_analytics_only": spec.topology_class,
        "short_name": s_task.short_name,
        "public_attempts_used": attempts_used,
        "turns_used": tot_s_turns,
        "public_gate_accepted": bool(pub_gate.get("accepted")),
        "admitted": bool(s_code.strip()),
        "strict_pass": bool(priv_ver.get("strict_pass")),
        "source_code_sha256": hashlib.sha256(s_code.encode("utf-8")).hexdigest(),
        "public_input_sha256": str(pub_gate.get("public_input_sha256") or s_pub_sha),
        "canonical_numeric_sha256": s_num_sha,
    }
    (s_dir / "00_stage_S_summary.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(
        f"[M10.2 Stage S | {wid} ({spec.stratum})] pass={rec['strict_pass']} | pub_gate={rec['public_gate_accepted']} "
        f"| att={attempts_used} | turns={rec['turns_used']}",
        flush=True,
    )
    return {
        "summary": rec,
        "source_monolithic_code": s_code,
        "source_pub_study": s_study,
        "source_pub_sha256": str(pub_gate.get("public_input_sha256") or s_pub_sha),
        "source_obs": s_obs,
        "source_public_gate": pub_gate,
    }


def _persist_m102_prefix_state(
    graph: PersistentMechanismGraph,
    *,
    world_id: str,
    arm: str,
    node_library: dict[str, ExecutableModuleNode],
    dag_library: list[ExecutableComputationDAG],
    withdrawals: list[CutWithdrawalMergeReceipt],
    dag_pos_obs: dict[str, list[dict[str, float | None]]],
    dag_neg_obs: dict[str, list[dict[str, float | None]]],
    node_pos_obs: dict[str, list[dict[str, float | None]]],
    dag_last_pos_probe: dict[str, tuple[str, str]],
    candidate_input_verdicts: dict[str, dict[str, Any]],
    attempted_decomp_from_dag_ids: set[str],
    covered_dag_pairs: set[tuple[str, str]],
    version_counter: int,
    dag_counter: int,
    event_seq: int,
) -> dict[str, Any]:
    blob = persist_m101_full_prefix_state(
        graph,
        world_id=world_id,
        arm=arm,
        node_library=node_library,
        dag_library=dag_library,
        withdrawals=withdrawals,
        dag_pos_obs=dag_pos_obs,
        dag_neg_obs=dag_neg_obs,
        node_pos_obs=node_pos_obs,
        dag_last_pos_probe=dag_last_pos_probe,
        candidate_input_verdicts=candidate_input_verdicts,
        attempted_decomp_from_dag_ids=attempted_decomp_from_dag_ids,
        covered_dag_pairs=covered_dag_pairs,
        version_counter=version_counter,
        dag_counter=dag_counter,
        event_seq=event_seq,
    )
    graph.forest.questions[M102_DAG_LEDGER_FOREST_KEY] = blob
    graph.sync_authoritative_state_to_forest()
    return blob


def _run_one_m102_world_arm_stream(
    *,
    spec: M102WorldSpec,
    arm: str,
    s_task: HoldoutOODTask,
    stream_tasks: tuple[HoldoutOODTask, ...],
    schedule_manifest: list[dict[str, Any]],
    base_v7_forest: Any,
    base_v7_archive: Any,
    source_monolithic_code: str,
    source_pub_sha256: str,
    source_obs: dict[str, float | None],
    source_public_gate: dict[str, Any],
    model: str,
    codex_info: dict[str, Any],
    stream_out_dir: Path,
    run_id: str,
    max_turns: int,
    reasoning_effort: str,
    mock_backend: Any = None,
) -> dict[str, Any]:
    """Execute one `H = 16` continual stream for `(spec, arm)` under strict equal-probe-cap (`<= 4`) & de-scaffolded prompts."""
    wid = spec.world_id
    total_horizon = len(stream_tasks)
    stream_t0 = monotonic()
    stream_out_dir.mkdir(parents=True, exist_ok=True)

    forest = deepcopy(base_v7_forest)
    graph = PersistentMechanismGraph(forest=forest)
    probe_store = ProbeReceiptStore(stream_out_dir / "probe_receipts", graph=graph)

    s0_pid, s0_psha = _persist_m102_probe(
        probe_store,
        probe_kind="stage_s_initial_gate",
        receipt=source_public_gate,
        metadata={"world_id": wid, "arm": arm, "step": 0, "task_id": s_task.task_id},
    )

    prior_archive_entries: list[dict[str, Any]] = [
        {
            "short_name": s_task.short_name,
            "task_id": s_task.task_id,
            "code": source_monolithic_code,
            "obs": source_obs,
        }
    ]
    verified_tasks_by_dag: dict[str, HoldoutOODTask] = {"dag_00": s_task}

    mono_rows: list[dict[str, Any]] = [
        {
            "id": "mono_r00",
            "guard": {},
            "code": source_monolithic_code,
            "code_sha256": hashlib.sha256(source_monolithic_code.encode("utf-8")).hexdigest(),
            "priority": 10,
            "origin_step": 0,
            "probe_id": s0_pid,
            "probe_sha256": s0_psha,
        }
    ]
    mono_pos_obs: dict[str, list[dict[str, float | None]]] = {"mono_r00": [source_obs]}
    mono_neg_obs: dict[str, list[dict[str, float | None]]] = {}

    node_library: dict[str, ExecutableModuleNode] = {
        "nv_00_00": ExecutableModuleNode(
            node_id="node_00",
            version_id="nv_00_00",
            code=source_monolithic_code,
            priority=10,
            origin_action="keep_monolithic",
            origin_step=0,
            probe_receipt_id=s0_pid,
            probe_receipt_sha256=s0_psha,
        )
    }
    dag_library: list[ExecutableComputationDAG] = [
        ExecutableComputationDAG(
            dag_id="dag_00",
            nodes=["node_00"],
            edges=[("__input__", "node_00")],
            sink_node_id="node_00",
            node_versions={"node_00": "nv_00_00"},
            applicability_guard={},
            priority=10,
            origin_action="keep_monolithic",
            origin_step=0,
        )
    ]
    dag_pos_obs: dict[str, list[dict[str, float | None]]] = {"dag_00": [source_obs]}
    dag_neg_obs: dict[str, list[dict[str, float | None]]] = {}
    dag_last_pos_probe: dict[str, tuple[str, str]] = {"dag_00": (s0_pid, s0_psha)}
    node_pos_obs: dict[str, list[dict[str, float | None]]] = {"nv_00_00": [source_obs]}
    candidate_input_verdicts: dict[str, dict[str, Any]] = {
        f"dag_00::{source_pub_sha256}": {"accepted": bool(source_public_gate.get("accepted")), "probe_id": s0_pid}
    }
    covered_dag_pairs: set[tuple[str, str]] = set()
    withdrawals: list[CutWithdrawalMergeReceipt] = []
    attempted_decomp_from_dag_ids: set[str] = set()
    version_counter = 1
    dag_counter = 1
    event_seq = 0
    consecutive_failed_dag_searches = 0
    solver_free_dag_reuse_hits = 0

    def _next_vid(nid: str) -> str:
        nonlocal version_counter
        v = f"nv_{nid.replace('node_', '')}_{version_counter:02d}"
        version_counter += 1
        return v

    def _next_dag_id() -> str:
        nonlocal dag_counter
        d = f"dag_{dag_counter:02d}"
        dag_counter += 1
        return d

    def _make_codex_client(role: str, ep_sn: str) -> Any:
        if mock_backend is not None:
            return mock_backend.make_client(role=role, world_id=wid, arm=arm, ep_sn=ep_sn)
        cli = m6f.FrontierCodexClient(
            codex_info=codex_info,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=300,
        )
        cli._hegel_call_meta = {
            "role": role,
            "world_id": wid,
            "cluster_id": spec.cluster_id,
            "arm": arm,
            "episode_short_name": ep_sn,
        }
        return cli

    def _run_sandbox_solve(
        *,
        task_to_solve: HoldoutOODTask,
        ep_dir: Path,
        sub_tag: str,
        role: str,
        ep_sn: str,
        continual_ctx: str = "",
        max_turns_override: int | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        if mock_backend is not None:
            return mock_backend.solve_in_sandbox(
                task=task_to_solve,
                role=role,
                world_id=wid,
                arm=arm,
                ep_sn=ep_sn,
                sub_tag=sub_tag,
            )
        _CALL_CONTEXT.role = role
        _CALL_CONTEXT.world_id = wid
        _CALL_CONTEXT.cluster_id = spec.cluster_id
        _CALL_CONTEXT.arm = arm
        _CALL_CONTEXT.episode_short_name = ep_sn
        ft = m101_task_to_frontier_task(task_to_solve, continual_context_block=continual_ctx)
        s_out = ep_dir / sub_tag
        s_out.mkdir(parents=True, exist_ok=True)
        ep_s = m6f.run_frontier_episode(
            ft,
            "Hegel_M6_9_Antithesis_LiveGPT",
            model=model,
            codex_info=codex_info,
            forest=deepcopy(base_v7_forest),
            forest_archive=deepcopy(base_v7_archive),
            hippo_idx=HippoRAGWormholeIndex(deepcopy(base_v7_forest)),
            episode_out_dir=s_out,
            run_id=f"{run_id}__{wid}__{arm}__{ep_sn}__{sub_tag}",
            max_turns=int(max_turns_override or max_turns),
            cmd_timeout_sec=180,
            reasoning_effort=reasoning_effort,
            defer_official_verifier=True,
            enable_auto_skill_packaging=False,
            allow_l1_primitives=False,
            disclosure_mode="on_demand_executable",
            staging_policy="net_value_gated",
        )
        return _extract_submitted_files(s_out), ep_s

    def _induce_guards_for_table(
        *,
        ep_sn: str,
        new_id: str,
        existing_ids: list[str],
        pos_map: dict[str, list[dict[str, float | None]]],
        neg_map: dict[str, list[dict[str, float | None]]],
        covered_pairs: set[tuple[str, str]] | None = None,
    ) -> dict[str, Any] | None:
        if not existing_ids:
            return None
        all_bids = list(existing_ids) + [new_id]
        feas = build_feasible_domain_negative_witnesses(
            branch_ids=all_bids,
            positive_witnesses_by_branch=pos_map,
            executed_negative_witnesses_by_branch=neg_map,
            covered_branch_pairs=covered_pairs,
        )
        sat = check_axis_aligned_all_of_guard_satisfiability(
            branch_ids=all_bids,
            feature_names=M102_OBSERVABLE_NAMES,
            positive_witnesses_by_branch=pos_map,
            effective_negatives_by_branch=feas["effective_negatives_by_branch"],
            min_clause_margin=0.035,
        )
        if not sat["has_distinguishing_negatives"]:
            return {
                "accepted": False,
                "skipped_zero_llm": True,
                "reason": "overlapping_valid_domains_no_distinguishing_negatives",
                "verified_overlap_pairs": [list(p) for p in feas["verified_overlap_pairs"]],
            }
        if not sat["satisfiable"]:
            return {
                "accepted": False,
                "skipped_zero_llm": True,
                "reason": "unidentifiable_from_current_observables",
                "inseparable_branches": sat["inseparable_branches"],
            }
        if mock_backend is not None and hasattr(mock_backend, "induce_guards"):
            return mock_backend.induce_guards(
                new_id=new_id,
                existing_ids=existing_ids,
                pos_map=pos_map,
                neg_map=feas["effective_negatives_by_branch"],
                feature_names=M102_OBSERVABLE_NAMES,
            )
        eff_cov: set[tuple[str, str]] = set(covered_pairs or set())
        for b1, b2 in feas["verified_overlap_pairs"]:
            eff_cov.add((b1, b2))
            eff_cov.add((b2, b1))
            covered_dag_pairs.add((b1, b2))
            covered_dag_pairs.add((b2, b1))
        cli = _make_codex_client("guard_induction", ep_sn)
        llm_g = propose_and_validate_m99_guard_partition(
            client=cli,
            stage_subdir=f"{run_id}__{wid}__{arm}__guard_{ep_sn}_{new_id}",
            scope_label=f"{arm}_m102",
            feature_names=M102_OBSERVABLE_NAMES,
            existing_branch_ids=existing_ids,
            new_branch_id_placeholder=new_id,
            positive_witness_features_by_branch=pos_map,
            negative_witness_features_by_branch=feas["effective_negatives_by_branch"],
            covered_branch_pairs=eff_cov,
            min_clause_margin=0.035,
            max_attempts=1,
        )
        if llm_g and llm_g.get("accepted"):
            return llm_g
        # Symmetrical verified max-margin fallback across ALL 3 arms when `sat["satisfiable"] == True` (`0` extra LLM retries!)
        cg = sat.get("constructive_guards") or {}
        ver = verify_m101_feasible_domain_guards(
            branch_guards=cg,
            positive_witnesses_by_branch=pos_map,
            effective_negatives_by_branch=feas["effective_negatives_by_branch"],
            feature_spreads=sat["feature_spreads"],
            min_clause_margin=0.035,
        )
        if ver["valid"] and new_id in cg:
            return {
                "accepted": True,
                "used_constructive_verified_fallback": True,
                "new_branch_guard": deepcopy(cg[new_id]),
                "existing_branch_guards": {bid: deepcopy(cg[bid]) for bid in existing_ids if bid in cg},
            }
        return llm_g

    _persist_m102_prefix_state(
        graph,
        world_id=wid,
        arm=arm,
        node_library=node_library,
        dag_library=dag_library,
        withdrawals=withdrawals,
        dag_pos_obs=dag_pos_obs,
        dag_neg_obs=dag_neg_obs,
        node_pos_obs=node_pos_obs,
        dag_last_pos_probe=dag_last_pos_probe,
        candidate_input_verdicts=candidate_input_verdicts,
        attempted_decomp_from_dag_ids=attempted_decomp_from_dag_ids,
        covered_dag_pairs=covered_dag_pairs,
        version_counter=version_counter,
        dag_counter=dag_counter,
        event_seq=event_seq,
    )
    graph.save_checkpoint(stream_out_dir / "graph_ckpt_after_step_00.json")

    episode_results: list[dict[str, Any]] = []
    action_log: list[dict[str, Any]] = []
    failed_search_events: list[dict[str, Any]] = []
    investment_gate_log: list[dict[str, Any]] = []

    for step_i, (task, m_info) in enumerate(zip(stream_tasks, schedule_manifest), start=1):
        event_seq += 1
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_sn = task.short_name
        pub_study, host_pub_sha, _ = materialize_m101_public_study_and_sha(task)
        obs_now = compute_m102_blind_observables(pub_study)

        pre_probes_wall_ms = 0.0
        post_probes_wall_ms = 0.0
        pre_probes_used = 0
        turns_used_step = 0
        final_code = ""
        final_gate: dict[str, Any] = {"accepted": False}
        final_probe_id = ""
        final_probe_sha = ""
        chosen_action = "zero_llm_replay"
        execution_mode = "zero_llm_replay"
        active_dag_trace: dict[str, Any] | None = None
        guard_ind_receipt: dict[str, Any] | None = None
        withdrawal_this_step: dict[str, Any] | None = None
        step_investment_decisions: list[dict[str, Any]] = []

        # =====================================================================
        # ARM 1: `Flat_MonolithicArchive_Online` (`MAX_PRE_LLM_PROBES_PER_STEP = 4`)
        # =====================================================================
        if arm == "Flat_MonolithicArchive_Online":
            ranked_rows = select_feasible_candidates_ranked_by_public_evidence(
                mono_rows, obs_now, mono_pos_obs, mono_neg_obs
            )
            best_acc: tuple[float, dict[str, Any], dict[str, Any], str, str] | None = None
            accepted_probe_count = 0
            first_probe_gate: dict[str, Any] = {"accepted": False}

            for r_idx, cand_row in enumerate(ranked_rows):
                if pre_probes_used >= M102_MAX_PRE_LLM_PROBES_PER_STEP:
                    break
                gr = execute_m101_public_gate_in_container(task=task, candidate_code=cand_row["code"], expect_dag_trace=False)
                pr_id, pr_sha = _persist_m102_probe(
                    probe_store,
                    probe_kind="pre_llm_monolithic_probe",
                    receipt=gr,
                    metadata={"world_id": wid, "arm": arm, "step": step_i, "row_id": cand_row["id"], "rank_idx": r_idx},
                )
                pre_probes_wall_ms += float(gr.get("wall_ms") or 0.0)
                pre_probes_used += 1
                if r_idx == 0:
                    first_probe_gate = gr
                c_in_sha = str(gr.get("public_input_sha256") or host_pub_sha)
                candidate_input_verdicts[f"{cand_row['id']}::{c_in_sha}"] = {
                    "accepted": bool(gr.get("accepted")),
                    "probe_id": pr_id,
                    "cv_metrics": gr.get("cv_metrics"),
                }
                if gr.get("accepted"):
                    accepted_probe_count += 1
                    sc_tot, ym, tm = _gate_cv_score(gr)
                    if best_acc is None or sc_tot < best_acc[0]:
                        best_acc = (sc_tot, cand_row, gr, pr_id, pr_sha)
                    if max(ym, tm) <= M102_BORDERLINE_CV_COMPARE_THRESHOLD or accepted_probe_count >= 2:
                        break
                else:
                    mono_neg_obs.setdefault(cand_row["id"], []).append(obs_now)
                    if best_acc is not None:
                        # Probed 1 alternative for borderline comparison and it failed -> keep `best_acc`!
                        break

            assert pre_probes_used <= M102_MAX_PRE_LLM_PROBES_PER_STEP
            if best_acc is not None:
                _, winning_row, winning_gate, winning_pid, winning_psha = best_acc
                final_code = winning_row["code"]
                final_gate = winning_gate
                final_probe_id, final_probe_sha = winning_pid, winning_psha
                mono_pos_obs.setdefault(winning_row["id"], []).append(obs_now)
                chosen_action = "zero_llm_monolithic_hit"
                execution_mode = "zero_llm_monolithic_hit"
                if winning_row["id"] != ranked_rows[0]["id"] and len(mono_rows) >= 2:
                    ex_ids = [r["id"] for r in mono_rows if r["id"] != winning_row["id"]]
                    guard_ind_receipt = _induce_guards_for_table(
                        ep_sn=ep_sn,
                        new_id=winning_row["id"],
                        existing_ids=ex_ids,
                        pos_map=mono_pos_obs,
                        neg_map=mono_neg_obs,
                    )
                    if guard_ind_receipt and guard_ind_receipt.get("accepted"):
                        winning_row["guard"] = deepcopy(guard_ind_receipt["new_branch_guard"])
                        for r in mono_rows:
                            if r["id"] in (guard_ind_receipt.get("existing_branch_guards") or {}):
                                r["guard"] = deepcopy(guard_ind_receipt["existing_branch_guards"][r["id"]])
            else:
                aug_t, ctx_hdr = _inject_m102_equal_info_archive(task, prior_archive_entries, first_probe_gate)
                sub_files, ep_s = _run_sandbox_solve(
                    task_to_solve=aug_t,
                    ep_dir=ep_dir,
                    sub_tag="mono_solve",
                    role="stream_solve",
                    ep_sn=ep_sn,
                    continual_ctx=ctx_hdr,
                )
                turns_used_step += int(ep_s.get("turns_used", 0) or 0)
                cand = sub_files.get("batch_solver.py", "")
                g_post = (
                    execute_m101_public_gate_in_container(task=task, candidate_code=cand, expect_dag_trace=False)
                    if cand
                    else {"accepted": False}
                )
                pp_id, pp_sha = _persist_m102_probe(
                    probe_store,
                    probe_kind="post_llm_monolithic_gate",
                    receipt=g_post,
                    metadata={"world_id": wid, "arm": arm, "step": step_i},
                )
                post_probes_wall_ms += float(g_post.get("wall_ms") or 0.0)
                final_code = cand
                final_gate = g_post
                final_probe_id, final_probe_sha = pp_id, pp_sha
                chosen_action = "keep_monolithic"
                execution_mode = "llm_monolithic_solve"
                if g_post.get("accepted"):
                    new_rid = f"mono_r{step_i:02d}"
                    ex_ids = [r["id"] for r in mono_rows]
                    mono_pos_obs[new_rid] = [obs_now]
                    guard_ind_receipt = _induce_guards_for_table(
                        ep_sn=ep_sn,
                        new_id=new_rid,
                        existing_ids=ex_ids,
                        pos_map=mono_pos_obs,
                        neg_map=mono_neg_obs,
                    )
                    new_g = deepcopy(guard_ind_receipt["new_branch_guard"]) if (guard_ind_receipt and guard_ind_receipt.get("accepted")) else {}
                    if guard_ind_receipt and guard_ind_receipt.get("accepted"):
                        for r in mono_rows:
                            if r["id"] in (guard_ind_receipt.get("existing_branch_guards") or {}):
                                r["guard"] = deepcopy(guard_ind_receipt["existing_branch_guards"][r["id"]])
                    mono_rows.append(
                        {
                            "id": new_rid,
                            "guard": new_g,
                            "code": cand,
                            "code_sha256": hashlib.sha256(cand.encode("utf-8")).hexdigest(),
                            "priority": 10 + step_i,
                            "origin_step": step_i,
                            "probe_id": pp_id,
                            "probe_sha256": pp_sha,
                        }
                    )
                    prior_archive_entries.append({"short_name": ep_sn, "task_id": task.task_id, "code": cand, "obs": obs_now})

        # =====================================================================
        # ARMS 2 & 3: `Eager_BoundedDAG_Online` vs `Hegel_SelectiveBoundedDAG_Online`
        #             (100% Equal `MAX_PRE_LLM_PROBES_PER_STEP = 4`; Single-Factor Gate!)
        # =====================================================================
        else:
            is_selective = (arm == "Hegel_SelectiveBoundedDAG_Online")
            dag_items = [
                {
                    "id": d.dag_id,
                    "guard": d.applicability_guard,
                    "priority": d.priority,
                    "dag": d,
                }
                for d in dag_library
            ]
            ranked_dag_its = select_feasible_candidates_ranked_by_public_evidence(
                dag_items, obs_now, dag_pos_obs, dag_neg_obs
            )
            chosen_dag: ExecutableComputationDAG = ranked_dag_its[0]["dag"]
            tried_fps: set[str] = {chosen_dag.topology_fingerprint()}

            c1_code, _ = compile_m101_dag_to_instrumented_solver_code(chosen_dag, node_library)
            g1 = execute_m101_public_gate_in_container(task=task, candidate_code=c1_code, expect_dag_trace=True)
            p1_id, p1_sha = _persist_m102_probe(
                probe_store,
                probe_kind="pre_llm_dag_probe",
                receipt=g1,
                metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": chosen_dag.dag_id, "rank_idx": 0},
            )
            pre_probes_wall_ms += float(g1.get("wall_ms") or 0.0)
            pre_probes_used += 1
            c1_in_sha = str(g1.get("public_input_sha256") or host_pub_sha)
            candidate_input_verdicts[f"{chosen_dag.dag_id}::{c1_in_sha}"] = {
                "accepted": bool(g1.get("accepted")),
                "probe_id": p1_id,
                "cv_metrics": g1.get("cv_metrics"),
            }

            best_dag_acc: tuple[float, ExecutableComputationDAG, str, dict[str, Any], str, str] | None = None
            pre_fail_errors: list[str] = list(g1.get("errors") or [])

            if g1.get("accepted"):
                sc1, ym1, tm1 = _gate_cv_score(g1)
                best_dag_acc = (sc1, chosen_dag, c1_code, g1, p1_id, p1_sha)
                # Borderline CV comparison (`0.026 < mae <= 0.038`) on overlapping DAGs (`U05 / E1`)!
                if (
                    max(ym1, tm1) > M102_BORDERLINE_CV_COMPARE_THRESHOLD
                    and len(ranked_dag_its) >= 2
                    and pre_probes_used < M102_MAX_PRE_LLM_PROBES_PER_STEP
                ):
                    alt_d0: ExecutableComputationDAG = ranked_dag_its[1]["dag"]
                    fp_alt = alt_d0.topology_fingerprint()
                    if fp_alt not in tried_fps:
                        tried_fps.add(fp_alt)
                        ca_code, _ = compile_m101_dag_to_instrumented_solver_code(alt_d0, node_library)
                        ga = execute_m101_public_gate_in_container(task=task, candidate_code=ca_code, expect_dag_trace=True)
                        pa_id, pa_sha = _persist_m102_probe(
                            probe_store,
                            probe_kind="pre_llm_borderline_alt_dag_probe",
                            receipt=ga,
                            metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": alt_d0.dag_id},
                        )
                        pre_probes_wall_ms += float(ga.get("wall_ms") or 0.0)
                        pre_probes_used += 1
                        if ga.get("accepted"):
                            sca, _, _ = _gate_cv_score(ga)
                            if sca < best_dag_acc[0]:
                                best_dag_acc = (sca, alt_d0, ca_code, ga, pa_id, pa_sha)
                        else:
                            dag_neg_obs.setdefault(alt_d0.dag_id, []).append(obs_now)
            else:
                dag_neg_obs.setdefault(chosen_dag.dag_id, []).append(obs_now)
                remaining_budget = M102_MAX_PRE_LLM_PROBES_PER_STEP - pre_probes_used  # exactly `3` remaining probes!

                comp_queue: list[tuple[str, ExecutableComputationDAG]] = []
                multi_dags = [d for d in dag_library if len(d.nodes) >= 2]
                if multi_dags:
                    multi_dags_sorted = sorted(multi_dags, key=lambda d: (-len(d.nodes), -d.priority, d.dag_id))
                    for tmpl in multi_dags_sorted[:2]:
                        cands_comp = _rank_compositional_node_version_combinations_m102(
                            template_dag=tmpl,
                            node_library=node_library,
                            node_pos_obs=node_pos_obs,
                            obs_now=obs_now,
                            already_tried_fingerprints=tried_fps,
                            max_candidates=2,
                        )
                        for cd in cands_comp:
                            comp_queue.append(("compositional_replay", cd))

                alt_queue: list[tuple[str, ExecutableComputationDAG]] = []
                for it in ranked_dag_its[1:]:
                    ad: ExecutableComputationDAG = it["dag"]
                    fp = ad.topology_fingerprint()
                    if fp not in tried_fps:
                        tried_fps.add(fp)
                        alt_queue.append(("alt_verified_dag", ad))

                # Strictly <= `remaining_budget` (`3` extra probes -> `<= 4` total probes INCLUDING `chosen_dag`!)
                n_comp = min(2, len(comp_queue)) if alt_queue else min(remaining_budget, len(comp_queue))
                pre_queue = comp_queue[:n_comp] + alt_queue[: max(0, remaining_budget - n_comp)]

                accepted_fb_count = 0
                for q_kind, q_dag in pre_queue[:remaining_budget]:
                    if pre_probes_used >= M102_MAX_PRE_LLM_PROBES_PER_STEP:
                        break
                    q_code, _ = compile_m101_dag_to_instrumented_solver_code(q_dag, node_library)
                    gq = execute_m101_public_gate_in_container(task=task, candidate_code=q_code, expect_dag_trace=True)
                    pq_id, pq_sha = _persist_m102_probe(
                        probe_store,
                        probe_kind=f"pre_llm_{q_kind}_probe",
                        receipt=gq,
                        metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": q_dag.dag_id, "q_kind": q_kind},
                    )
                    pre_probes_wall_ms += float(gq.get("wall_ms") or 0.0)
                    pre_probes_used += 1
                    if q_kind == "alt_verified_dag":
                        qin_sha = str(gq.get("public_input_sha256") or host_pub_sha)
                        candidate_input_verdicts[f"{q_dag.dag_id}::{qin_sha}"] = {
                            "accepted": bool(gq.get("accepted")),
                            "probe_id": pq_id,
                            "cv_metrics": gq.get("cv_metrics"),
                        }
                    if gq.get("accepted"):
                        accepted_fb_count += 1
                        scq, ymq, tmq = _gate_cv_score(gq)
                        if best_dag_acc is None or scq < best_dag_acc[0]:
                            best_dag_acc = (scq, q_dag, q_code, gq, pq_id, pq_sha)
                        if max(ymq, tmq) <= M102_BORDERLINE_CV_COMPARE_THRESHOLD or accepted_fb_count >= 2:
                            break
                    else:
                        if q_kind == "alt_verified_dag":
                            dag_neg_obs.setdefault(q_dag.dag_id, []).append(obs_now)
                        if best_dag_acc is not None:
                            break

            assert pre_probes_used <= M102_MAX_PRE_LLM_PROBES_PER_STEP

            if best_dag_acc is not None:
                _, winning_dag, winning_code, winning_gate, winning_pid, winning_psha = best_dag_acc
                final_code = winning_code
                final_gate = winning_gate
                final_probe_id, final_probe_sha = winning_pid, winning_psha
                active_dag_trace = winning_gate.get("dag_runtime_trace")
                is_existing = any(d.dag_id == winning_dag.dag_id for d in dag_library)
                if not is_existing:
                    winning_dag.dag_id = _next_dag_id()
                    winning_dag.priority = 10 + step_i
                    winning_dag.origin_step = step_i
                    winning_dag.origin_action = "reuse_subgraph"
                    dag_library.append(winning_dag)
                    verified_tasks_by_dag[winning_dag.dag_id] = task
                    chosen_action = "reuse_subgraph"
                    execution_mode = "zero_llm_compositional_reuse"
                    solver_free_dag_reuse_hits += 1
                else:
                    chosen_action = "zero_llm_dag_hit"
                    execution_mode = "zero_llm_dag_hit"
                    if len(winning_dag.nodes) >= 2:
                        solver_free_dag_reuse_hits += 1

                dag_pos_obs.setdefault(winning_dag.dag_id, []).append(obs_now)
                dag_last_pos_probe[winning_dag.dag_id] = (winning_pid, winning_psha)
                for _nid, _vid in winning_dag.node_versions.items():
                    node_pos_obs.setdefault(_vid, []).append(obs_now)

                if winning_dag.dag_id != chosen_dag.dag_id and len(dag_library) >= 2:
                    ex_ids = [d.dag_id for d in dag_library if d.dag_id != winning_dag.dag_id]
                    guard_ind_receipt = _induce_guards_for_table(
                        ep_sn=ep_sn,
                        new_id=winning_dag.dag_id,
                        existing_ids=ex_ids,
                        pos_map=dag_pos_obs,
                        neg_map=dag_neg_obs,
                        covered_pairs=covered_dag_pairs,
                    )
                    if guard_ind_receipt and guard_ind_receipt.get("accepted"):
                        winning_dag.applicability_guard = deepcopy(guard_ind_receipt["new_branch_guard"])
                        for d in dag_library:
                            if d.dag_id in (guard_ind_receipt.get("existing_branch_guards") or {}):
                                d.applicability_guard = deepcopy(guard_ind_receipt["existing_branch_guards"][d.dag_id])
            else:
                # --- Step B: De-Scaffolded DAG Structure Discovery / Edit Proposal ---
                mono_dags = [d for d in dag_library if len(d.nodes) == 1]
                multi_dags = [d for d in dag_library if len(d.nodes) >= 2]
                nearest_mono = (
                    min(
                        mono_dags,
                        key=lambda d: min(_obs_l1_distance(obs_now, w) for w in (dag_pos_obs.get(d.dag_id) or [source_obs])),
                    )
                    if mono_dags
                    else None
                )

                should_check_decomp = False
                if nearest_mono is not None and nearest_mono.dag_id not in attempted_decomp_from_dag_ids:
                    if not multi_dags:
                        should_check_decomp = True
                    else:
                        d_m = min(_obs_l1_distance(obs_now, w) for w in (dag_pos_obs.get(nearest_mono.dag_id) or [source_obs]))
                        d_mul = min(_obs_l1_distance(obs_now, w) for d in multi_dags for w in (dag_pos_obs.get(d.dag_id) or [source_obs]))
                        if d_m + 0.05 < d_mul:
                            should_check_decomp = True

                keep_mono_decided = False
                if should_check_decomp and nearest_mono is not None:
                    src_vid = nearest_mono.node_versions[nearest_mono.sink_node_id]
                    src_code_m = node_library[src_vid].code
                    src_obs_m = (dag_pos_obs.get(nearest_mono.dag_id) or [source_obs])[0]
                    src_task_m = verified_tasks_by_dag.get(nearest_mono.dag_id, s_task)

                    allow_init_decomp = True
                    if is_selective:
                        g_init = evaluate_m102_selective_investment_gate(
                            phase="initial_decomposition",
                            source_code=src_code_m,
                            ref_obs=src_obs_m,
                            target_obs=obs_now,
                            step_index=step_i,
                            total_horizon=total_horizon,
                            consecutive_failed_dag_searches=consecutive_failed_dag_searches,
                            total_failed_dag_searches=len(failed_search_events),
                            solver_free_dag_reuse_hits=solver_free_dag_reuse_hits,
                        )
                        g_init_row = {"step": step_i, "phase": "initial_decomposition", "from_dag": nearest_mono.dag_id, **g_init}
                        step_investment_decisions.append(g_init_row)
                        investment_gate_log.append(g_init_row)
                        allow_init_decomp = bool(g_init["invest_in_structure"])
                        if not allow_init_decomp:
                            if g_init.get("permanent_source_verdict"):
                                attempted_decomp_from_dag_ids.add(nearest_mono.dag_id)
                            if not multi_dags:
                                keep_mono_decided = True
                            action_log.append(
                                {
                                    "step": step_i,
                                    "action": "keep_monolithic_zero_llm_gate",
                                    "reason": g_init["reason"],
                                    "from_dag": nearest_mono.dag_id,
                                }
                            )

                    if allow_init_decomp:
                        attempted_decomp_from_dag_ids.add(nearest_mono.dag_id)
                        idx_before_prop = _current_receipts_len()
                        cli_prop = _make_codex_client("dag_structure_proposal", ep_sn)
                        prop_res = _propose_descaffolded_initial_dag_or_keep_monolithic(
                            client=cli_prop,
                            stage_subdir=f"{run_id}__{wid}__{arm}__init_decomp_{ep_sn}",
                            source_code=src_code_m,
                            source_obs=src_obs_m,
                            target_obs=obs_now,
                            max_nodes=M101_MAX_ACTIVE_NODES,
                            max_fanout=M101_MAX_FANOUT,
                        )
                        parsed_p = prop_res.get("parsed") if isinstance(prop_res, dict) and prop_res.get("ok") else None
                        if not isinstance(parsed_p, dict):
                            _retag_failed_search_receipts(idx_before_prop, wid, arm, ep_sn)
                            failed_search_events.append({"step": step_i, "phase": "init_decomp_call_failed"})
                            consecutive_failed_dag_searches += 1
                        elif parsed_p.get("action") == "keep_monolithic":
                            keep_mono_decided = True
                            action_log.append({"step": step_i, "action": "keep_monolithic", "from_dag": nearest_mono.dag_id})
                        else:
                            try:
                                p_nodes = [str(x) for x in parsed_p["nodes"]]
                                p_edges = [(str(e[0]), str(e[1])) for e in parsed_p["edges"]]
                                p_sink = str(parsed_p["sink_node_id"])
                                mod_by_nid = {str(m["node_id"]): str(m["code"]) for m in parsed_p["node_modules"]}
                                if not (2 <= len(p_nodes) <= M101_MAX_ACTIVE_NODES):
                                    raise ValueError(f"Proposed node count {len(p_nodes)} outside [2, {M101_MAX_ACTIVE_NODES}]")
                                if set(mod_by_nid.keys()) != set(p_nodes):
                                    raise ValueError("node_modules keys != nodes")
                                tmp_nlib = dict(node_library)
                                new_nv_map: dict[str, str] = {}
                                created_nodes: dict[str, ExecutableModuleNode] = {}
                                for nid in p_nodes:
                                    vid = _next_vid(nid)
                                    mn = ExecutableModuleNode(
                                        node_id=nid,
                                        version_id=vid,
                                        code=mod_by_nid[nid],
                                        priority=10 + step_i,
                                        origin_action=str(parsed_p["action"]),
                                        origin_step=step_i,
                                    )
                                    tmp_nlib[vid] = mn
                                    created_nodes[vid] = mn
                                    new_nv_map[nid] = vid
                                cand_d0 = ExecutableComputationDAG(
                                    dag_id=_next_dag_id(),
                                    nodes=p_nodes,
                                    edges=p_edges,
                                    sink_node_id=p_sink,
                                    node_versions=new_nv_map,
                                    applicability_guard=deepcopy(nearest_mono.applicability_guard),
                                    priority=nearest_mono.priority + 1,
                                    origin_action=str(parsed_p["action"]),
                                    origin_step=step_i,
                                )
                                t_info = cand_d0.validate_topology()
                                if t_info["max_fanout"] > M101_MAX_FANOUT:
                                    raise ValueError(f"Proposed fan-out {t_info['max_fanout']} > {M101_MAX_FANOUT}")
                                comp_src_code, _ = compile_m101_dag_to_instrumented_solver_code(cand_d0, tmp_nlib)
                                g_src = execute_m101_public_gate_in_container(
                                    task=src_task_m, candidate_code=comp_src_code, expect_dag_trace=True
                                )
                                psrc_id, psrc_sha = _persist_m102_probe(
                                    probe_store,
                                    probe_kind="decomp_source_verification_probe",
                                    receipt=g_src,
                                    metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": cand_d0.dag_id, "action": parsed_p["action"]},
                                )
                                post_probes_wall_ms += float(g_src.get("wall_ms") or 0.0)
                                if g_src.get("accepted"):
                                    for vid, mn in created_nodes.items():
                                        mn.probe_receipt_id = psrc_id
                                        mn.probe_receipt_sha256 = psrc_sha
                                        node_library[vid] = mn
                                        node_pos_obs.setdefault(vid, []).append(src_obs_m)
                                    dag_library.append(cand_d0)
                                    verified_tasks_by_dag[cand_d0.dag_id] = src_task_m
                                    dag_pos_obs[cand_d0.dag_id] = [src_obs_m]
                                    dag_last_pos_probe[cand_d0.dag_id] = (psrc_id, psrc_sha)
                                    covered_dag_pairs.add((nearest_mono.dag_id, cand_d0.dag_id))
                                    covered_dag_pairs.add((cand_d0.dag_id, nearest_mono.dag_id))
                                    multi_dags = [d for d in dag_library if len(d.nodes) >= 2]
                                    action_log.append(
                                        {
                                            "step": step_i,
                                            "action": str(parsed_p["action"]),
                                            "adopted_dag_id": cand_d0.dag_id,
                                            "active_node_count": t_info["active_node_count"],
                                            "depth": t_info["depth"],
                                            "max_fanout": t_info["max_fanout"],
                                            "source_probe_id": psrc_id,
                                        }
                                    )
                                else:
                                    _retag_failed_search_receipts(idx_before_prop, wid, arm, ep_sn)
                                    failed_search_events.append(
                                        {"step": step_i, "phase": "init_decomp_docker_rejected", "errors": g_src.get("errors")}
                                    )
                                    consecutive_failed_dag_searches += 1
                            except Exception as exc:
                                _retag_failed_search_receipts(idx_before_prop, wid, arm, ep_sn)
                                failed_search_events.append({"step": step_i, "phase": "init_decomp_invalid", "error": str(exc)})
                                consecutive_failed_dag_searches += 1

                # 2. Execute graph-maintenance edit action if approved!
                solved_via_dag_edit = False
                suspended_by_maintenance_gate = False
                if not keep_mono_decided and multi_dags:
                    ref_multi = sorted(multi_dags, key=lambda d: (-len(d.nodes), -d.priority, d.dag_id))[0]
                    allow_edit = True
                    if is_selective:
                        ref_obs_m = (dag_pos_obs.get(ref_multi.dag_id) or [source_obs])[0]
                        g_maint = evaluate_m102_selective_investment_gate(
                            phase="graph_maintenance_edit",
                            source_code="",
                            ref_obs=ref_obs_m,
                            target_obs=obs_now,
                            step_index=step_i,
                            total_horizon=total_horizon,
                            consecutive_failed_dag_searches=consecutive_failed_dag_searches,
                            total_failed_dag_searches=len(failed_search_events),
                            solver_free_dag_reuse_hits=solver_free_dag_reuse_hits,
                        )
                        g_m_row = {"step": step_i, "phase": "graph_maintenance_edit", "ref_dag": ref_multi.dag_id, **g_maint}
                        step_investment_decisions.append(g_m_row)
                        investment_gate_log.append(g_m_row)
                        allow_edit = bool(g_maint["invest_in_structure"])
                        if not allow_edit:
                            suspended_by_maintenance_gate = True
                            action_log.append(
                                {
                                    "step": step_i,
                                    "action": "suspend_surgery_keep_monolithic_step",
                                    "reason": g_maint["reason"],
                                }
                            )

                    if allow_edit:
                        health_rec = _probe_existing_dag_nodes_health_in_container(
                            task=task, dag=ref_multi, node_library=node_library
                        )
                        health_pid, health_psha = _persist_m102_probe(
                            probe_store,
                            probe_kind="dag_node_health_probe",
                            receipt=health_rec,
                            metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": ref_multi.dag_id},
                        )
                        post_probes_wall_ms += float(health_rec.get("wall_ms") or 0.0)

                        dags_sum = [
                            {
                                "dag_id": d.dag_id,
                                "nodes": d.nodes,
                                "edges": [list(e) for e in d.edges],
                                "sink_node_id": d.sink_node_id,
                                "node_versions": d.node_versions,
                                "origin_action": d.origin_action,
                            }
                            for d in dag_library
                        ]
                        nlib_sum = [
                            {
                                "node_id": m.node_id,
                                "version_id": m.version_id,
                                "origin_step": m.origin_step,
                                "code": m.code,
                            }
                            for m in sorted(node_library.values(), key=lambda x: (x.node_id, x.version_id))
                        ]
                        ref_obs_map = {
                            d.dag_id: (dag_pos_obs.get(d.dag_id) or [source_obs])[0] for d in dag_library
                        }

                        idx_before_edit = _current_receipts_len()
                        cli_edit = _make_codex_client("dag_structure_proposal", ep_sn)
                        edit_res = _propose_descaffolded_dag_edit_action(
                            client=cli_edit,
                            stage_subdir=f"{run_id}__{wid}__{arm}__edit_plan_{ep_sn}",
                            active_dags_summary=dags_sum,
                            node_library_summary=nlib_sum,
                            obs_now=obs_now,
                            ref_obs_by_dag=ref_obs_map,
                            pre_llm_probe_errors=pre_fail_errors,
                            node_health_summary=health_rec,
                            max_nodes=M101_MAX_ACTIVE_NODES,
                            max_fanout=M101_MAX_FANOUT,
                        )
                        ep_plan = edit_res.get("parsed") if isinstance(edit_res, dict) and edit_res.get("ok") else None
                        if isinstance(ep_plan, dict) and ep_plan.get("action") != "keep_monolithic":
                            act = str(ep_plan["action"])
                            try:
                                n_nodes = [str(x) for x in ep_plan["new_nodes"]]
                                n_edges = [(str(e[0]), str(e[1])) for e in ep_plan["new_edges"]]
                                n_sink = str(ep_plan["new_sink_node_id"])
                                reused_map = {str(x["node_id"]): str(x["version_id"]) for x in ep_plan["reused_node_versions"]}
                                edit_nids = [str(x) for x in ep_plan["editable_node_ids"]]
                                if not (1 <= len(n_nodes) <= M101_MAX_ACTIVE_NODES):
                                    raise ValueError(f"Edit node count {len(n_nodes)} outside [1, {M101_MAX_ACTIVE_NODES}]")
                                if not edit_nids or set(reused_map.keys()) & set(edit_nids):
                                    raise ValueError("editable_node_ids must be non-empty and disjoint from reused_node_versions")
                                if set(reused_map.keys()) | set(edit_nids) != set(n_nodes):
                                    raise ValueError("reused + editable != new_nodes")
                                for nid, vid in reused_map.items():
                                    if vid not in node_library:
                                        raise ValueError(f"Unknown reused version `{vid}`")

                                frozen_n = {nid: node_library[vid] for nid, vid in reused_map.items()}
                                placeholder_v = {**reused_map, **{nid: f"__edit_{nid}__" for nid in edit_nids}}
                                cand_dag = ExecutableComputationDAG(
                                    dag_id=_next_dag_id(),
                                    nodes=n_nodes,
                                    edges=n_edges,
                                    sink_node_id=n_sink,
                                    node_versions=placeholder_v,
                                    priority=10 + step_i,
                                    origin_action=act,
                                    origin_step=step_i,
                                    is_cross_scoped=(act in {"merge_nodes", "rewire_edge"}),
                                )
                                t_inf = cand_dag.validate_topology()
                                if t_inf["max_fanout"] > M101_MAX_FANOUT:
                                    raise ValueError(f"Edit fan-out {t_inf['max_fanout']} > {M101_MAX_FANOUT}")

                                synth_role = (
                                    "dag_merge_withdrawal_repair"
                                    if act in {"merge_nodes", "rewire_edge"}
                                    else "dag_compile_and_node_synthesis"
                                )
                                t_node_task = _build_m102_targeted_node_synthesis_task(
                                    base_task=task,
                                    target_dag=cand_dag,
                                    frozen_nodes=frozen_n,
                                    editable_node_ids=edit_nids,
                                    edit_action=act,
                                    edit_rationale=str(ep_plan.get("rationale") or ""),
                                    prior_entries=prior_archive_entries,
                                    pre_llm_diag=g1,
                                )
                                sub_nodes, ep_s = _run_sandbox_solve(
                                    task_to_solve=t_node_task,
                                    ep_dir=ep_dir,
                                    sub_tag=f"dag_edit_{act}",
                                    role=synth_role,
                                    ep_sn=ep_sn,
                                    max_turns_override=max(8, max_turns - 2),
                                )
                                turns_used_step += int(ep_s.get("turns_used", 0) or 0)
                                tmp_lib = dict(node_library)
                                final_nv = dict(reused_map)
                                created_n: dict[str, ExecutableModuleNode] = {}
                                for nid in edit_nids:
                                    fn = f"{nid}.py"
                                    if fn not in sub_nodes or not sub_nodes[fn].strip():
                                        raise ValueError(f"Agent did not submit `{fn}`")
                                    vid = _next_vid(nid)
                                    mn = ExecutableModuleNode(
                                        node_id=nid,
                                        version_id=vid,
                                        code=sub_nodes[fn],
                                        priority=10 + step_i,
                                        origin_action=act,
                                        origin_step=step_i,
                                    )
                                    tmp_lib[vid] = mn
                                    created_n[vid] = mn
                                    final_nv[nid] = vid
                                cand_dag.node_versions = final_nv
                                comp_code, _ = compile_m101_dag_to_instrumented_solver_code(cand_dag, tmp_lib)
                                g_edit = execute_m101_public_gate_in_container(
                                    task=task, candidate_code=comp_code, expect_dag_trace=True
                                )
                                pe_id, pe_sha = _persist_m102_probe(
                                    probe_store,
                                    probe_kind=f"post_llm_dag_{act}_gate",
                                    receipt=g_edit,
                                    metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": cand_dag.dag_id, "action": act},
                                )
                                post_probes_wall_ms += float(g_edit.get("wall_ms") or 0.0)
                                if g_edit.get("accepted"):
                                    consecutive_failed_dag_searches = 0
                                    for vid, mn in created_n.items():
                                        mn.probe_receipt_id = pe_id
                                        mn.probe_receipt_sha256 = pe_sha
                                        node_library[vid] = mn
                                    for _nid, _vid in cand_dag.node_versions.items():
                                        node_pos_obs.setdefault(_vid, []).append(obs_now)

                                    # Strict v2 4-way semantic `CutWithdrawalMergeReceipt` decoupled from adoption (`U03 + U04`)!
                                    if act in {"merge_nodes", "rewire_edge"}:
                                        dag_by_id = {d.dag_id: d for d in dag_library}
                                        old_d_id = str(ep_plan.get("base_dag_id") or chosen_dag.dag_id)
                                        if old_d_id not in dag_by_id or old_d_id not in dag_last_pos_probe:
                                            old_d_id = ref_multi.dag_id
                                        old_d_obj = dag_by_id.get(old_d_id)
                                        if old_d_obj is not None and int(old_d_obj.origin_step) < int(step_i):
                                            pr_pid, pr_psha = dag_last_pos_probe[old_d_id]
                                            old_shas = {nid: node_library[vid].code_sha256 for nid, vid in old_d_obj.node_versions.items()}
                                            new_shas = {nid: node_library[vid].code_sha256 for nid, vid in cand_dag.node_versions.items()}
                                            if chosen_dag.dag_id == old_d_id and (g1.get("dag_runtime_trace") or {}).get("node_code_shas") == old_shas:
                                                fail_pid, fail_psha = p1_id, p1_sha
                                            else:
                                                old_c_code, _ = compile_m101_dag_to_instrumented_solver_code(old_d_obj, node_library)
                                                g_old_fail = execute_m101_public_gate_in_container(
                                                    task=task, candidate_code=old_c_code, expect_dag_trace=True
                                                )
                                                fail_pid, fail_psha = _persist_m102_probe(
                                                    probe_store,
                                                    probe_kind="withdrawal_old_dag_current_failure_probe",
                                                    receipt=g_old_fail,
                                                    metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": old_d_id},
                                                )
                                                post_probes_wall_ms += float(g_old_fail.get("wall_ms") or 0.0)
                                            try:
                                                tgt_in_sha = str(g_edit.get("public_input_sha256") or host_pub_sha)
                                                wd_rec = CutWithdrawalMergeReceipt(
                                                    receipt_id="",
                                                    world_id=wid,
                                                    arm=arm,
                                                    old_dag_id=old_d_id,
                                                    new_dag_id=cand_dag.dag_id,
                                                    action=act,
                                                    prior_positive_probe_id=pr_pid,
                                                    prior_positive_receipt_sha256=pr_psha,
                                                    preserved_node_probe_id=health_pid,
                                                    preserved_node_receipt_sha256=health_psha,
                                                    failing_joint_probe_id=fail_pid,
                                                    failing_joint_receipt_sha256=fail_psha,
                                                    new_passing_probe_id=pe_id,
                                                    new_passing_receipt_sha256=pe_sha,
                                                    preserved_old_dag_id=old_d_id,
                                                    expanded_guard_features=list(M102_OBSERVABLE_NAMES),
                                                    old_dag_node_code_shas=old_shas,
                                                    new_dag_node_code_shas=new_shas,
                                                    target_public_input_sha256=tgt_in_sha,
                                                    preserved_evidence_level="runtime_interface_healthy",
                                                )
                                                wd_rec.receipt_id = wd_rec.compute_expected_receipt_id()
                                                known_shas = {
                                                    **{
                                                        d.dag_id: {n: node_library[v].code_sha256 for n, v in d.node_versions.items()}
                                                        for d in dag_library
                                                    },
                                                    cand_dag.dag_id: new_shas,
                                                }
                                                verify_m101_cut_withdrawal_external_fk(
                                                    wd_rec,
                                                    probe_store,
                                                    allowed_guard_features=M102_OBSERVABLE_NAMES,
                                                    known_dag_node_shas=known_shas,
                                                )
                                                withdrawals.append(wd_rec)
                                                withdrawal_this_step = wd_rec.to_dict()
                                            except ValueError as wd_exc:
                                                withdrawal_this_step = {
                                                    "withdrawal_status": "unproved_candidate_adopted",
                                                    "reason": str(wd_exc),
                                                    "adopted_dag_id": cand_dag.dag_id,
                                                }
                                        else:
                                            withdrawal_this_step = {
                                                "withdrawal_status": "first_decomposition_and_adaptation",
                                                "adopted_dag_id": cand_dag.dag_id,
                                            }

                                    ex_ids = [d.dag_id for d in dag_library]
                                    dag_pos_obs[cand_dag.dag_id] = [obs_now]
                                    dag_last_pos_probe[cand_dag.dag_id] = (pe_id, pe_sha)
                                    verified_tasks_by_dag[cand_dag.dag_id] = task
                                    dag_library.append(cand_dag)
                                    guard_ind_receipt = _induce_guards_for_table(
                                        ep_sn=ep_sn,
                                        new_id=cand_dag.dag_id,
                                        existing_ids=ex_ids,
                                        pos_map=dag_pos_obs,
                                        neg_map=dag_neg_obs,
                                        covered_pairs=covered_dag_pairs,
                                    )
                                    if guard_ind_receipt and guard_ind_receipt.get("accepted"):
                                        cand_dag.applicability_guard = deepcopy(guard_ind_receipt["new_branch_guard"])
                                        for d in dag_library:
                                            if d.dag_id in (guard_ind_receipt.get("existing_branch_guards") or {}):
                                                d.applicability_guard = deepcopy(guard_ind_receipt["existing_branch_guards"][d.dag_id])
                                    final_code = comp_code
                                    final_gate = g_edit
                                    final_probe_id, final_probe_sha = pe_id, pe_sha
                                    active_dag_trace = g_edit.get("dag_runtime_trace")
                                    chosen_action = act
                                    execution_mode = f"llm_dag_{act}"
                                    solved_via_dag_edit = True
                                    action_log.append(
                                        {
                                            "step": step_i,
                                            "action": act,
                                            "dag_id": cand_dag.dag_id,
                                            "active_node_count": t_inf["active_node_count"],
                                            "depth": t_inf["depth"],
                                            "max_fanout": t_inf["max_fanout"],
                                            "editable_nodes": edit_nids,
                                            "reused_nodes": sorted(reused_map.keys()),
                                        }
                                    )
                                else:
                                    _retag_failed_search_receipts(idx_before_edit, wid, arm, ep_sn)
                                    failed_search_events.append(
                                        {"step": step_i, "phase": f"dag_edit_{act}_docker_rejected", "errors": g_edit.get("errors")}
                                    )
                                    consecutive_failed_dag_searches += 1
                            except Exception as exc:
                                _retag_failed_search_receipts(idx_before_edit, wid, arm, ep_sn)
                                failed_search_events.append({"step": step_i, "phase": f"dag_edit_{act}_exception", "error": str(exc)})
                                consecutive_failed_dag_searches += 1
                        elif isinstance(ep_plan, dict) and ep_plan.get("action") == "keep_monolithic":
                            keep_mono_decided = True
                        else:
                            _retag_failed_search_receipts(idx_before_edit, wid, arm, ep_sn)
                            failed_search_events.append({"step": step_i, "phase": "edit_plan_call_failed"})
                            consecutive_failed_dag_searches += 1

                # 3. Clean `K = 1` monolithic solve (either intentionally chosen by gate / `keep_monolithic` OR fallback!)
                if not solved_via_dag_edit:
                    aug_t, ctx_hdr = _inject_m102_equal_info_archive(task, prior_archive_entries, g1)
                    s_role = (
                        "stream_solve"
                        if (keep_mono_decided or suspended_by_maintenance_gate or not multi_dags)
                        else "dag_merge_withdrawal_repair"
                    )
                    sub_files, ep_s = _run_sandbox_solve(
                        task_to_solve=aug_t,
                        ep_dir=ep_dir,
                        sub_tag="mono_fallback_solve",
                        role=s_role,
                        ep_sn=ep_sn,
                        continual_ctx=ctx_hdr,
                    )
                    turns_used_step += int(ep_s.get("turns_used", 0) or 0)
                    if suspended_by_maintenance_gate:
                        # Reset single-step cooldown after clean monolithic step
                        consecutive_failed_dag_searches = 0
                    cand = sub_files.get("batch_solver.py", "")
                    if cand:
                        new_vid = _next_vid("node_00")
                        new_mn = ExecutableModuleNode(
                            node_id="node_00",
                            version_id=new_vid,
                            code=cand,
                            priority=10 + step_i,
                            origin_action=("keep_monolithic" if not multi_dags else "merge_nodes"),
                            origin_step=step_i,
                        )
                        new_d = ExecutableComputationDAG(
                            dag_id=_next_dag_id(),
                            nodes=["node_00"],
                            edges=[("__input__", "node_00")],
                            sink_node_id="node_00",
                            node_versions={"node_00": new_vid},
                            priority=10 + step_i,
                            origin_action=("keep_monolithic" if not multi_dags else "merge_nodes"),
                            origin_step=step_i,
                            is_cross_scoped=bool(multi_dags),
                        )
                        tmp_lib = {**node_library, new_vid: new_mn}
                        comp_m, _ = compile_m101_dag_to_instrumented_solver_code(new_d, tmp_lib)
                        g_post = execute_m101_public_gate_in_container(task=task, candidate_code=comp_m, expect_dag_trace=True)
                        pp_id, pp_sha = _persist_m102_probe(
                            probe_store,
                            probe_kind="post_llm_monolithic_dag_gate",
                            receipt=g_post,
                            metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": new_d.dag_id},
                        )
                        post_probes_wall_ms += float(g_post.get("wall_ms") or 0.0)
                        final_code = comp_m
                        final_gate = g_post
                        final_probe_id, final_probe_sha = pp_id, pp_sha
                        active_dag_trace = g_post.get("dag_runtime_trace")
                        chosen_action = "keep_monolithic" if (keep_mono_decided or suspended_by_maintenance_gate or not multi_dags) else "merge_nodes"
                        execution_mode = "llm_keep_monolithic" if (keep_mono_decided or suspended_by_maintenance_gate or not multi_dags) else "llm_merge_fallback"
                        if g_post.get("accepted"):
                            new_mn.probe_receipt_id = pp_id
                            new_mn.probe_receipt_sha256 = pp_sha
                            node_library[new_vid] = new_mn
                            node_pos_obs[new_vid] = [obs_now]
                            ex_ids = [d.dag_id for d in dag_library]
                            dag_pos_obs[new_d.dag_id] = [obs_now]
                            dag_last_pos_probe[new_d.dag_id] = (pp_id, pp_sha)
                            verified_tasks_by_dag[new_d.dag_id] = task
                            dag_library.append(new_d)
                            guard_ind_receipt = _induce_guards_for_table(
                                ep_sn=ep_sn,
                                new_id=new_d.dag_id,
                                existing_ids=ex_ids,
                                pos_map=dag_pos_obs,
                                neg_map=dag_neg_obs,
                                covered_pairs=covered_dag_pairs,
                            )
                            if guard_ind_receipt and guard_ind_receipt.get("accepted"):
                                new_d.applicability_guard = deepcopy(guard_ind_receipt["new_branch_guard"])
                                for d in dag_library:
                                    if d.dag_id in (guard_ind_receipt.get("existing_branch_guards") or {}):
                                        d.applicability_guard = deepcopy(guard_ind_receipt["existing_branch_guards"][d.dag_id])
                            prior_archive_entries.append({"short_name": ep_sn, "task_id": task.task_id, "code": cand, "obs": obs_now})

        # Official private Docker grading (never visible to policy!)
        priv_ver = grade_m101_candidate_in_container(task=task, candidate_code=final_code) if final_code else {"strict_pass": False, "reward": 0.0}
        eff_pub_sha = str(final_gate.get("public_input_sha256") or host_pub_sha)

        ep_rec = {
            "world_id": wid,
            "cluster_id": spec.cluster_id,
            "stratum": spec.stratum,
            "topology_class_hidden_analytics_only": spec.topology_class,
            "arm": arm,
            "stream_step": step_i,
            "task_id": task.task_id,
            "short_name": ep_sn,
            "causal_event": m_info["causal_event"],
            "topology_kind_hidden": m_info["topology_kind"],
            "stage_cfg_hidden": m_info["stage_cfg"],
            "execution_mode": execution_mode,
            "chosen_action": chosen_action,
            "turns_used": int(turns_used_step),
            "solver_free_closure": bool(turns_used_step == 0 and final_gate.get("accepted")),
            "public_gate_accepted": bool(final_gate.get("accepted")),
            "public_gate_probe_id": final_probe_id,
            "public_gate_probe_sha256": final_probe_sha,
            "strict_pass": bool(priv_ver.get("strict_pass")),
            "reward": 1.0 if priv_ver.get("strict_pass") else 0.0,
            "pre_llm_probes_used": pre_probes_used,
            "pre_llm_probe_wall_ms": round(pre_probes_wall_ms, 2),
            "post_llm_probe_wall_ms": round(post_probes_wall_ms, 2),
            "official_verifier_wall_ms": float(priv_ver.get("verifier_wall_ms") or float(priv_ver.get("student_elapsed_sec", 0.0) or 0.0) * 1000.0),
            "dag_runtime_trace": active_dag_trace,
            "cut_withdrawal_receipt": withdrawal_this_step,
            "investment_gate_decisions": step_investment_decisions,
            "guard_induction_accepted": (bool(guard_ind_receipt.get("accepted")) if guard_ind_receipt else None),
            "guard_skipped_zero_llm": (bool(guard_ind_receipt.get("skipped_zero_llm")) if guard_ind_receipt else False),
            "public_input_sha256": eff_pub_sha,
            "verifier_report": priv_ver,
        }
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(ep_rec, indent=2) + "\n", encoding="utf-8")
        (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")
        if final_code:
            (ep_dir / "final_candidate_batch_solver.py").write_text(final_code, encoding="utf-8")

        _persist_m102_prefix_state(
            graph,
            world_id=wid,
            arm=arm,
            node_library=node_library,
            dag_library=dag_library,
            withdrawals=withdrawals,
            dag_pos_obs=dag_pos_obs,
            dag_neg_obs=dag_neg_obs,
            node_pos_obs=node_pos_obs,
            dag_last_pos_probe=dag_last_pos_probe,
            candidate_input_verdicts=candidate_input_verdicts,
            attempted_decomp_from_dag_ids=attempted_decomp_from_dag_ids,
            covered_dag_pairs=covered_dag_pairs,
            version_counter=version_counter,
            dag_counter=dag_counter,
            event_seq=event_seq,
        )
        graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_rec)
        print(
            f"  [M10.2 {wid} | {arm} | step {step_i:02d}/{total_horizon} ({m_info['causal_event']})] "
            f"{execution_mode} | act={chosen_action} | pass={ep_rec['strict_pass']} | probes={pre_probes_used} "
            f"| turns={turns_used_step} | K={(active_dag_trace or {}).get('active_node_count', 1)}",
            flush=True,
        )

    max_k_obs = max((int((r.get("dag_runtime_trace") or {}).get("active_node_count", 1)) for r in episode_results), default=1)
    max_depth_obs = max((int((r.get("dag_runtime_trace") or {}).get("depth", 1)) for r in episode_results), default=1)
    max_fanout_obs = max((int((r.get("dag_runtime_trace") or {}).get("max_fanout", 0)) for r in episode_results), default=0)
    shared_exec_1_verified = any(
        bool((r.get("dag_runtime_trace") or {}).get("shared_fanout_nodes"))
        and bool((r.get("dag_runtime_trace") or {}).get("all_exec_count_eq_1"))
        and bool((r.get("dag_runtime_trace") or {}).get("all_declared_edges_consumed", True))
        and r.get("strict_pass")
        for r in episode_results
    )
    all_traced_exec_1 = all(
        bool((r.get("dag_runtime_trace") or {}).get("all_exec_count_eq_1"))
        and bool((r.get("dag_runtime_trace") or {}).get("all_declared_edges_consumed", True))
        for r in episode_results
        if isinstance(r.get("dag_runtime_trace"), dict)
    )
    compositional_reuse_passes = sum(
        1 for r in episode_results if r["execution_mode"] == "zero_llm_compositional_reuse" and r["strict_pass"]
    )

    st_sum = {
        "world_id": wid,
        "cluster_id": spec.cluster_id,
        "stratum": spec.stratum,
        "topology_class_hidden_analytics_only": spec.topology_class,
        "arm": arm,
        "wall_sec": round(monotonic() - stream_t0, 2),
        "strict_passes": sum(1 for r in episode_results if r["strict_pass"]),
        "episode_count": len(episode_results),
        "solver_free_episodes_total": sum(1 for r in episode_results if r["turns_used"] == 0),
        "solver_free_successes": sum(1 for r in episode_results if r["turns_used"] == 0 and r["strict_pass"]),
        "compositional_reuse_passes": compositional_reuse_passes,
        "total_turns_used": sum(int(r["turns_used"]) for r in episode_results),
        "max_pre_llm_probes_observed": max((int(r["pre_llm_probes_used"]) for r in episode_results), default=0),
        "total_pre_llm_probes_used": sum(int(r["pre_llm_probes_used"]) for r in episode_results),
        "max_active_nodes_observed": max_k_obs,
        "max_depth_observed": max_depth_obs,
        "max_fanout_observed": max_fanout_obs,
        "shared_fanout_exec_count_1_verified": shared_exec_1_verified,
        "all_traced_episodes_exec_count_1": all_traced_exec_1,
        "cut_withdrawal_count": len(withdrawals),
        "cut_withdrawal_receipts": [w.to_dict() for w in withdrawals],
        "failed_search_events_count": len(failed_search_events),
        "failed_search_events": failed_search_events,
        "investment_gate_log": investment_gate_log,
        "action_log": action_log,
        "probe_receipt_store_count": len(list((stream_out_dir / "probe_receipts").glob("*.json"))),
        "total_pre_llm_probe_wall_ms": round(sum(float(r["pre_llm_probe_wall_ms"]) for r in episode_results), 2),
        "total_post_llm_probe_wall_ms": round(sum(float(r["post_llm_probe_wall_ms"]) for r in episode_results), 2),
        "total_official_verifier_wall_ms": round(sum(float(r["official_verifier_wall_ms"]) for r in episode_results), 2),
        "episodes_meta": [
            {
                "stream_step": r["stream_step"],
                "causal_event": r["causal_event"],
                "short_name": r["short_name"],
                "execution_mode": r["execution_mode"],
                "chosen_action": r["chosen_action"],
                "strict_pass": r["strict_pass"],
                "turns_used": r["turns_used"],
                "pre_llm_probes_used": r["pre_llm_probes_used"],
                "active_node_count": (r.get("dag_runtime_trace") or {}).get("active_node_count", 1),
                "depth": (r.get("dag_runtime_trace") or {}).get("depth", 1),
                "max_fanout": (r.get("dag_runtime_trace") or {}).get("max_fanout", 0),
                "all_exec_count_eq_1": (r.get("dag_runtime_trace") or {}).get("all_exec_count_eq_1"),
                "all_declared_edges_consumed": (r.get("dag_runtime_trace") or {}).get("all_declared_edges_consumed"),
                "public_gate_probe_id": r["public_gate_probe_id"],
            }
            for r in episode_results
        ],
    }
    (stream_out_dir / "00_stream_summary.json").write_text(json.dumps(st_sum, indent=2) + "\n", encoding="utf-8")
    return st_sum


# ============================================================================
# 5. Post-Run `0`-LLM Step-14 Cold-Hydration + Plain-Dict Parity Check (`U11`)
# ============================================================================


def run_m102_step14_cold_hydration_and_plain_dict_parity_ablation(
    *,
    world_specs: tuple[M102WorldSpec, ...],
    built_worlds: dict[str, Any],
    out_dir: Path,
    checkpoint_step: int = 14,
) -> dict[str, Any]:
    """Post-run `0`-LLM Cold-Hydration FK Verification + Plain-Dict Routing Parity Check on frozen Step-14 checkpoints
    across the final 2 episodes (`S15` & `S16`)!
    """
    rows: list[dict[str, Any]] = []
    hydration_rows: list[dict[str, Any]] = []
    matched = 0
    evaluated = 0
    hydrated_ok = 0

    for sp in world_specs:
        wid = sp.world_id
        _, tasks, manifest = built_worlds[wid]
        eff_ckpt = min(checkpoint_step, max(0, len(tasks) - 2))
        s_dir = out_dir / "continual_streams" / wid / "Hegel_SelectiveBoundedDAG_Online"
        ck_path = s_dir / f"graph_ckpt_after_step_{eff_ckpt:02d}.json"
        ck_final = s_dir / f"graph_ckpt_after_step_{len(tasks):02d}.json"
        if not ck_path.is_file():
            continue

        # 1) Verify full bijective end-of-stream cold hydration (`disk_pids == index_pids`, `0` orphan files!)
        if ck_final.is_file():
            g_fin = PersistentMechanismGraph.load_checkpoint(ck_final)
            pstore_fin = ProbeReceiptStore(s_dir / "probe_receipts", graph=g_fin)
            hydrate_m101_full_prefix_state(g_fin, pstore_fin, allowed_guard_features=M102_OBSERVABLE_NAMES)

        # 2) Verify mid-stream Step-14 prefix cold hydration (`entries_14 <= disk_pids` via isolated prefix store!)
        g_ck = PersistentMechanismGraph.load_checkpoint(ck_path)
        idx_14 = (g_ck.forest.questions.get(M10_0_1_PROBE_RECEIPT_INDEX_FOREST_KEY) or {}).get("entries") or {}
        import tempfile as _tf
        with _tf.TemporaryDirectory(prefix="m102_step14_pstore_") as tmp_ps:
            tmp_ps_p = Path(tmp_ps)
            src_ps_p = s_dir / "probe_receipts"
            for pid_14 in idx_14:
                src_fp = src_ps_p / f"{pid_14}.json"
                if not src_fp.is_file():
                    raise FileNotFoundError(f"Step-{eff_ckpt} prefix receipt `{pid_14}` missing on disk!")
                (tmp_ps_p / f"{pid_14}.json").write_bytes(src_fp.read_bytes())
            pstore_ck = ProbeReceiptStore(tmp_ps_p, graph=g_ck)
            hyd = hydrate_m101_full_prefix_state(g_ck, pstore_ck, allowed_guard_features=M102_OBSERVABLE_NAMES)
        hydrated_ok += 1
        hydration_rows.append(
            {
                "world_id": wid,
                "checkpoint_step": eff_ckpt,
                "cold_hydration_verified": True,
                "state_sha256": hyd["state_sha256"],
                "dag_count": len(hyd["dag_library"]),
                "node_version_count": len(hyd["node_library"]),
                "withdrawal_count": len(hyd["withdrawals"]),
            }
        )

        raw_ck = json.loads(ck_path.read_text(encoding="utf-8"))
        st_d = raw_ck["questions"].get(M102_DAG_LEDGER_FOREST_KEY) or raw_ck["questions"].get("__m10_1_unknown_cut_dag_state__") or {}

        nlib_g = hyd["node_library"]
        dags_g = hyd["dag_library"]
        pos_g = hyd["dag_pos_obs"]
        neg_g = hyd["dag_neg_obs"]

        nlib_d = {k: ExecutableModuleNode.from_dict(v) for k, v in (st_d.get("node_library") or {}).items()}
        dags_d = [ExecutableComputationDAG.from_dict(x) for x in (st_d.get("dag_library") or [])]
        pos_d = st_d.get("dag_pos_obs") or {}
        neg_d = st_d.get("dag_neg_obs") or {}

        eval_indices = [i for i in range(eff_ckpt, len(tasks))][:2]
        for idx in eval_indices:
            tk = tasks[idx]
            pub_st, pub_sha, _ = materialize_m101_public_study_and_sha(tk)
            obs = compute_m102_blind_observables(pub_st)
            rk_g = select_feasible_candidates_ranked_by_public_evidence(
                [{"id": d.dag_id, "guard": d.applicability_guard, "priority": d.priority, "dag": d} for d in dags_g],
                obs,
                pos_g,
                neg_g,
            )
            rk_d = select_feasible_candidates_ranked_by_public_evidence(
                [{"id": d.dag_id, "guard": d.applicability_guard, "priority": d.priority, "dag": d} for d in dags_d],
                obs,
                pos_d,
                neg_d,
            )
            sel_g = rk_g[0]
            sel_d = rk_d[0]
            cg, _ = compile_m101_dag_to_instrumented_solver_code(sel_g["dag"], nlib_g)
            cd, _ = compile_m101_dag_to_instrumented_solver_code(sel_d["dag"], nlib_d)
            sg_sha = hashlib.sha256(cg.encode("utf-8")).hexdigest()
            sd_sha = hashlib.sha256(cd.encode("utf-8")).hexdigest()
            evaluated += 1
            eq = bool(sel_g["id"] == sel_d["id"] and sg_sha == sd_sha)
            if eq:
                matched += 1
            rows.append(
                {
                    "world_id": wid,
                    "stratum": sp.stratum,
                    "event": manifest[idx]["causal_event"],
                    "public_input_sha256": pub_sha,
                    "graph_dag_id": sel_g["id"],
                    "plain_dag_id": sel_d["id"],
                    "graph_candidate_sha256": sg_sha,
                    "plain_candidate_sha256": sd_sha,
                    "exact_parity": eq,
                }
            )
    return {
        "schema": "hegel_m10_2_step14_cold_hydration_and_plain_dict_parity_v1",
        "paid_model_calls": 0,
        "hydrated_clusters_verified": hydrated_ok,
        "expected_clusters": len(world_specs),
        "all_clusters_cold_hydrated_ok": (hydrated_ok == len(world_specs)),
        "matched_episodes": matched,
        "evaluated_episodes": evaluated,
        "parity_rate_pct": round(100.0 * matched / max(1, evaluated), 2),
        "hydration_rows": hydration_rows,
        "rows": rows,
    }


# ============================================================================
# 6. Nested Horizon Checkpoints (`H = 4, 8, 12, 16`), `K*`, & Structural Gates
# ============================================================================


def compute_m102_nested_horizons_and_k_star(
    *,
    world_specs: tuple[M102WorldSpec, ...],
    all_receipts: list[dict[str, Any]],
    stream_summaries: list[dict[str, Any]],
    n_steps: int = M102_HORIZON_STEPS,
) -> dict[str, Any]:
    wids = [sp.world_id for sp in world_specs]
    st_by_key = {(s["world_id"], s["arm"]): s for s in stream_summaries}

    step_cost: dict[tuple[str, str, int], int] = {}
    for r in all_receipts:
        arm = r["arm"]
        if arm not in M102_ARMS:
            continue
        ep_sn = str(r["episode_short_name"])
        if "_batch_" not in ep_sn:
            continue
        k_num = int(ep_sn.rsplit("_batch_", 1)[-1])
        if 1 <= k_num <= n_steps:
            key = (r["world_id"], arm, k_num)
            step_cost[key] = step_cost.get(key, 0) + int(r["proxy_v2_u_010c_4o"])

    comparisons = (
        ("Hegel_Selective_minus_Flat_Monolithic", "Hegel_SelectiveBoundedDAG_Online", "Flat_MonolithicArchive_Online"),
        ("Hegel_Selective_minus_Eager_BoundedDAG", "Hegel_SelectiveBoundedDAG_Online", "Eager_BoundedDAG_Online"),
        ("Eager_BoundedDAG_minus_Flat_Monolithic", "Eager_BoundedDAG_Online", "Flat_MonolithicArchive_Online"),
    )
    out: dict[str, Any] = {}
    checkpoints = [h for h in M102_NESTED_CHECKPOINTS if h <= n_steps]
    if n_steps not in checkpoints:
        checkpoints.append(n_steps)

    for comp_key, arm_a, arm_b in comparisons:
        step_rows = []
        for k in range(1, n_steps + 1):
            a_c = [sum(step_cost.get((w, arm_a, s), 0) for s in range(1, k + 1)) for w in wids]
            b_c = [sum(step_cost.get((w, arm_b, s), 0) for s in range(1, k + 1)) for w in wids]
            diffs = [float(a - b) for a, b in zip(a_c, b_c)]
            a_p = sum(1 for w in wids for ep in st_by_key[(w, arm_a)]["episodes_meta"][:k] if ep["strict_pass"])
            b_p = sum(1 for w in wids for ep in st_by_key[(w, arm_b)]["episodes_meta"][:k] if ep["strict_pass"])
            tot_eps = len(wids) * k
            pr_diff = (a_p - b_p) / float(tot_eps)
            step_rows.append(
                {
                    "step_K": k,
                    "arm_a_cum_scope1_v2_total": int(sum(a_c)),
                    "arm_b_cum_scope1_v2_total": int(sum(b_c)),
                    "mean_cluster_delta_v2": round(float(np.mean(diffs)), 2),
                    "total_delta_v2": int(sum(a_c) - sum(b_c)),
                    "pct_cost_change": round(100.0 * (sum(a_c) - sum(b_c)) / max(1, sum(b_c)), 4),
                    "cluster_wins_arm_a_cheaper": sum(1 for d in diffs if d < 0),
                    "arm_a_cum_strict_passes": a_p,
                    "arm_b_cum_strict_passes": b_p,
                    "total_episodes_at_K": tot_eps,
                    "pass_rate_diff": round(pr_diff, 6),
                    "quality_noninferior_point_at_K": bool(pr_diff >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE),
                }
            )

        # Nested checkpoint statistical tables (`H = 4, 8, 12, 16`)
        nested_H: dict[str, Any] = {}
        for h_k in checkpoints:
            a_c_h = [sum(step_cost.get((w, arm_a, s), 0) for s in range(1, h_k + 1)) for w in wids]
            b_c_h = [sum(step_cost.get((w, arm_b, s), 0) for s in range(1, h_k + 1)) for w in wids]
            c_diffs_h = [float(a - b) for a, b in zip(a_c_h, b_c_h)]
            q_diffs_h = [
                (
                    sum(1 for ep in st_by_key[(w, arm_a)]["episodes_meta"][:h_k] if ep["strict_pass"])
                    - sum(1 for ep in st_by_key[(w, arm_b)]["episodes_meta"][:h_k] if ep["strict_pass"])
                )
                / float(h_k)
                for w in wids
            ]
            c_inf = paired_cluster_inference_stats(c_diffs_h)
            q_inf = paired_cluster_inference_stats(q_diffs_h)
            c_boot = c_inf["bootstrap_95ci"]
            q_boot = q_inf["bootstrap_95ci"]
            q_t = q_inf["paired_t_95ci"]
            cost_sup = bool(
                c_inf["mean_diff"] is not None
                and c_inf["mean_diff"] < 0
                and c_inf["two_sided_sign_flip_p"] is not None
                and c_inf["two_sided_sign_flip_p"] < 0.05
                and c_boot is not None
                and c_boot[1] < 0
            )
            qual_ni_boot = bool(q_boot is not None and q_boot[0] >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE)
            qual_ni_t = bool(q_t is not None and q_t[0] >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE)
            nested_H[f"H_{h_k:02d}"] = {
                "horizon_H": h_k,
                "is_primary_endpoint": (h_k == n_steps),
                "arm_a_total_v2": int(sum(a_c_h)),
                "arm_b_total_v2": int(sum(b_c_h)),
                "total_cost_delta_v2": int(sum(a_c_h) - sum(b_c_h)),
                "pct_cost_change": round(100.0 * (sum(a_c_h) - sum(b_c_h)) / max(1, sum(b_c_h)), 4),
                "cluster_wins_arm_a_cheaper": sum(1 for d in c_diffs_h if d < 0),
                "arm_a_passes": sum(1 for w in wids for ep in st_by_key[(w, arm_a)]["episodes_meta"][:h_k] if ep["strict_pass"]),
                "arm_b_passes": sum(1 for w in wids for ep in st_by_key[(w, arm_b)]["episodes_meta"][:h_k] if ep["strict_pass"]),
                "total_episodes": len(wids) * h_k,
                "cost_inference": c_inf,
                "quality_inference": q_inf,
                "verdicts": {
                    "cost_superiority_supported": cost_sup,
                    "quality_noninferiority_bootstrap_supported": qual_ni_boot,
                    "quality_noninferiority_paired_t_supported": qual_ni_t,
                },
            }

        k_star: int | str = "not_reached"
        prim_ni = nested_H[f"H_{n_steps:02d}"]["verdicts"]["quality_noninferiority_bootstrap_supported"]
        if prim_ni:
            for idx_k in range(len(step_rows)):
                if all(r["total_delta_v2"] < 0 for r in step_rows[idx_k:]):
                    k_star = step_rows[idx_k]["step_K"]
                    break
        missing_u = sum(
            int(r.get("usage_unknown_attempt_count", 0))
            for r in all_receipts
            if r["arm"] in {arm_a, arm_b}
        )
        if missing_u:
            k_star = "unknown_partial_usage"

        nested_H[f"H_{n_steps:02d}"]["verdicts"]["finite_observed_break_even_reached"] = isinstance(k_star, int)

        per_cluster_k_star: dict[str, Any] = {}
        for sp in world_specs:
            w = sp.world_id
            deltas_w = [
                sum(step_cost.get((w, arm_a, s), 0) - step_cost.get((w, arm_b, s), 0) for s in range(1, k + 1))
                for k in range(1, n_steps + 1)
            ]
            a_pw = st_by_key[(w, arm_a)]["strict_passes"]
            b_pw = st_by_key[(w, arm_b)]["strict_passes"]
            kw: int | str = "not_reached"
            if (a_pw - b_pw) / float(n_steps) >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE:
                for idx_k in range(n_steps):
                    if all(d < 0 for d in deltas_w[idx_k:]):
                        kw = idx_k + 1
                        break
            per_cluster_k_star[w] = {
                "stratum": sp.stratum,
                "topology_class_hidden_analytics_only": sp.topology_class,
                "K_star": kw,
                "final_delta_v2": int(deltas_w[-1]),
                "arm_a_passes": a_pw,
                "arm_b_passes": b_pw,
            }

        out[comp_key] = {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "empirical_quality_constrained_K_star": k_star,
            "finite_observed_break_even_reached": isinstance(k_star, int),
            "preregistered_quality_noninferiority_margin_per_episode": PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE,
            "nested_checkpoints": nested_H,
            "per_cluster_K_star": per_cluster_k_star,
            "by_step_K": step_rows,
        }
    return out


def evaluate_m102_structural_verification_gates(
    *,
    world_specs: tuple[M102WorldSpec, ...],
    stream_summaries: list[dict[str, Any]],
    arm_ledgers: dict[str, Any],
    parity_ablation: dict[str, Any],
) -> list[dict[str, Any]]:
    st_by_key = {(s["world_id"], s["arm"]): s for s in stream_summaries}
    hg_streams = [s for s in stream_summaries if s["arm"] == "Hegel_SelectiveBoundedDAG_Online"]

    # Gate 1: Monolithic economic non-fragmentation
    mono_specs = [sp for sp in world_specs if sp.topology_class == "monolithic_coupled"]
    g1_ok = True
    g1_obs_parts = []
    for sp in mono_specs:
        sh = st_by_key[(sp.world_id, "Hegel_SelectiveBoundedDAG_Online")]
        sm = st_by_key[(sp.world_id, "Flat_MonolithicArchive_Online")]
        max_k = int(sh["max_active_nodes_observed"])
        p_ok = (sh["strict_passes"] - sm["strict_passes"]) / float(sh["episode_count"]) >= -0.05
        if max_k != 1 or not p_ok:
            g1_ok = False
        g1_obs_parts.append(f"{sp.world_id}: max_K={max_k}, passes={sh['strict_passes']}/{sh['episode_count']} vs Mono {sm['strict_passes']}/{sm['episode_count']}")

    # Gate 2: De-scaffolded multi-stage (`K>=3`) & shared fan-out (`exec_count==1`) discovery + solver-free compositional reuse
    k3_clusters = [s["world_id"] for s in hg_streams if int(s["max_active_nodes_observed"]) >= 3]
    fanout_clusters = [s["world_id"] for s in hg_streams if bool(s["shared_fanout_exec_count_1_verified"])]
    comp_reuse_tot = sum(int(s["compositional_reuse_passes"]) for s in hg_streams)
    min_k3_req = 2 if len(world_specs) <= 4 else 4
    g2_ok = len(k3_clusters) >= min_k3_req and len(fanout_clusters) >= 1 and comp_reuse_tot >= 1

    # Gate 3: Strict v2 semantic withdrawal & runtime invariants
    all_exec_1 = all(bool(s["all_traced_episodes_exec_count_1"]) for s in hg_streams)
    wd_tot = sum(int(s["cut_withdrawal_count"]) for s in hg_streams)
    g3_ok = bool(all_exec_1)

    # Gate 4: Step-14 cold hydration & plain-dict parity
    g4_ok = bool(parity_ablation["all_clusters_cold_hydrated_ok"] and parity_ablation["parity_rate_pct"] == 100.0)

    return [
        {
            "gate_id": "G1_monolithic_economic_non_fragmentation",
            "preregistered_criterion": "On `monolithic_coupled`, `Hegel_SelectiveBoundedDAG_Online` retains `max_K == 1` via 0-LLM investment gate with non-inferior quality vs Monolithic.",
            "observed": "; ".join(g1_obs_parts) if g1_obs_parts else "N/A",
            "status": "PASS" if g1_ok else "FAIL",
        },
        {
            "gate_id": "G2_descaffolded_multi_stage_and_shared_fanout_discovery",
            "preregistered_criterion": f"Under de-scaffolded prompts & `stat_00..stat_17`, discover `K >= 3` on `>= {min_k3_req}` clusters, shared fan-out `exec_count == 1` on `>= 1` cluster, and `>= 1` solver-free compositional reuse pass.",
            "observed": f"K>=3 clusters={k3_clusters} ({len(k3_clusters)}), shared_fanout_exec_1={fanout_clusters} ({len(fanout_clusters)}), compositional_reuse_passes={comp_reuse_tot}",
            "status": "PASS" if g2_ok else "FAIL",
        },
        {
            "gate_id": "G3_strict_semantic_withdrawal_and_runtime_edge_consumption",
            "preregistered_criterion": "100% traced episodes satisfy `all_exec_count_eq_1 == True` & `all_declared_edges_consumed == True`; 100% issued `CutWithdrawalMergeReceipt` pass v2 4-way semantic FK.",
            "observed": f"all_traced_exec_1_and_edges_consumed={all_exec_1}, verified_v2_withdrawals={wd_tot}",
            "status": "PASS" if g3_ok else "FAIL",
        },
        {
            "gate_id": "G4_step14_cold_hydration_and_plain_dict_parity",
            "preregistered_criterion": "100% cold-hydration FK verification and 100.0% plain-dict SHA-256 routing parity on frozen Step-14 checkpoints across S15 & S16.",
            "observed": f"hydrated={parity_ablation['hydrated_clusters_verified']}/{parity_ablation['expected_clusters']}, parity={parity_ablation['matched_episodes']}/{parity_ablation['evaluated_episodes']} ({parity_ablation['parity_rate_pct']}%)",
            "status": "PASS" if g4_ok else "FAIL",
        },
    ]


# ============================================================================
# 7. Main Campaign Orchestrator
# ============================================================================


def _main() -> None:
    global _ALL_RECEIPTS_REF
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--v7-bundle", type=Path, default=DEFAULT_V7_BUNDLE)
    ap.add_argument("--split", type=str, default="dev", choices=["dev", "confirmation"])
    ap.add_argument("--model", type=str, default="gpt-6.2-sol")
    ap.add_argument("--reasoning-effort", type=str, default="medium")
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--max-codex-concurrency", "--max-concurrency", dest="max_codex_concurrency", type=int, default=32)
    ap.add_argument("--max-stream-workers", type=int, default=24)
    ap.add_argument("--finalize-existing", action="store_true")
    args = ap.parse_args()

    selected_specs = DEV_M102_WORLD_SPECS if args.split == "dev" else CONFIRMATION_M102_WORLD_SPECS
    default_dir = DEFAULT_DEV_OUT_DIR if args.split == "dev" else DEFAULT_CONF_OUT_DIR
    out_dir: Path = (args.out_dir or default_dir).resolve()
    exec_manifest_path = out_dir / "execution_manifest.json"
    exec_done_path = out_dir / "execution_completion.json"

    os.environ["HEGEL_MAX_CODEX_CONCURRENCY"] = str(args.max_codex_concurrency)

    if not args.finalize_existing and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory `{out_dir}`!")

    git_sha_start, is_dirty_start, src_tree_sha_start = _git_and_source_tree_state()
    if is_dirty_start and not args.finalize_existing:
        raise RuntimeError(f"Refusing to start M10.2 `{args.split}` paid execution on a dirty working tree (`U01`)!")

    prereg_sha = hashlib.sha256(PREREG_PROTOCOL_PATH.read_bytes()).hexdigest()

    codex_info: dict[str, Any] = {}
    if not args.finalize_existing:
        ensure_clean_scipy_sandbox_image()
        m64.ensure_runtime_qualification_context(sandbox_cid=None)
        env_img_digest = _docker_image_digest(m7a.CLEAN_SCIPY_SANDBOX_IMAGE)
        codex_info = m6f.discover_chatgpt_subscription_codex_cli()
        if not (codex_info.get("available") and codex_info.get("auth_mode") == "chatgpt"):
            raise RuntimeError("Official ChatGPT Codex CLI unavailable for paid execution")
    else:
        start_obj = json.loads(exec_manifest_path.read_text(encoding="utf-8"))
        env_img_digest = start_obj["sandbox_image_digest"]

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_mirror_dir = out_dir / "provider_raw_attempts_mirror"
    raw_mirror_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"m102_{args.split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    if not args.finalize_existing:
        exec_manifest = {
            "schema": "hegel_m10_2_execution_provenance_manifest_v1",
            "execution_run_id": run_id,
            "split": args.split,
            "preregistered_protocol_sha256": prereg_sha,
            "code_commit_at_start": git_sha_start,
            "working_tree_dirty_at_start": bool(is_dirty_start),
            "source_tree_sha256_at_start": src_tree_sha_start,
            "sandbox_image_digest": env_img_digest,
            "v7_bundle_sha256": hashlib.sha256(args.v7_bundle.read_bytes()).hexdigest(),
            "shared_kernel_version": SHARED_KERNEL_VERSION,
            "verifier_schema_version": M10_1_1_VERIFIER_SCHEMA_VERSION,
            "cost_policy_version": COST_POLICY_VERSION,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_turns": int(args.max_turns),
            "configured_max_codex_concurrency": int(args.max_codex_concurrency),
            "max_stream_workers": int(args.max_stream_workers),
            "equal_resource_contracts": {
                "max_pre_llm_probes_per_step_all_arms": M102_MAX_PRE_LLM_PROBES_PER_STEP,
                "borderline_cv_compare_threshold": M102_BORDERLINE_CV_COMPARE_THRESHOLD,
                "primary_horizon_steps": M102_HORIZON_STEPS,
                "nested_checkpoints": list(M102_NESTED_CHECKPOINTS),
                "prompt_split_scaffolding_removed": True,
                "observable_role_names_removed": True,
            },
            "provenance_flags": {
                "source_private_previously_exposed": False,
                "protocol_amended_after_source_results": False,
                "downstream_streams_previously_exposed": False,
            },
            "clock_domain": "monotonic_elapsed_plus_utc_iso8601",
            "utc_started_at": datetime.now(timezone.utc).isoformat(),
        }
        with exec_manifest_path.open("x", encoding="utf-8") as fh:
            json.dump(exec_manifest, fh, indent=2)
            fh.write("\n")

    cli_sem = threading.BoundedSemaphore(max(1, int(args.max_codex_concurrency)))
    all_receipts: list[dict[str, Any]] = []
    _ALL_RECEIPTS_REF = all_receipts
    active_counter = {"cur": 0, "peak": 0}
    orig_call_structured = m6f.FrontierCodexClient.call_structured

    def _wrapped_call_structured(self_cli: Any, **c_kwargs: Any) -> dict[str, Any]:
        self_cli.immediate_archive_dir = raw_mirror_dir
        meta = getattr(self_cli, "_hegel_call_meta", None) or {}
        role = str(meta.get("role") or getattr(_CALL_CONTEXT, "role", "unknown"))
        wid = str(meta.get("world_id") or getattr(_CALL_CONTEXT, "world_id", "unknown"))
        cid = str(meta.get("cluster_id") or getattr(_CALL_CONTEXT, "cluster_id", "unknown"))
        arm = str(meta.get("arm") or getattr(_CALL_CONTEXT, "arm", "Stage_S_Shared_Source"))
        ep_sn = str(meta.get("episode_short_name") or getattr(_CALL_CONTEXT, "episode_short_name", "unknown"))
        mono_t0 = monotonic()
        utc_t0 = datetime.now(timezone.utc).isoformat()
        with cli_sem:
            with _RECEIPT_HOOK_LOCK:
                active_counter["cur"] += 1
                if active_counter["cur"] > active_counter["peak"]:
                    active_counter["peak"] = active_counter["cur"]
            try:
                res = orig_call_structured(self_cli, **c_kwargs)
            finally:
                with _RECEIPT_HOOK_LOCK:
                    active_counter["cur"] -= 1
        mono_t1 = monotonic()
        utc_t1 = datetime.now(timezone.utc).isoformat()
        sc = score_single_call_usage(
            prompt_tokens=int(res.get("prompt_tokens", 0) or 0),
            cached_tokens=int(res.get("cached_tokens", 0) or 0),
            completion_tokens=int(res.get("completion_tokens", 0) or 0),
            wall_ms=float(res.get("wall_ms", 0) or 0.0),
        )
        cu_uuid = str(res.get("call_uuid") or "")
        att_fn = str(res.get("attempts_ledger_filename") or "")
        rec_row = {
            "call_uuid": cu_uuid,
            "world_id": wid,
            "cluster_id": cid,
            "arm": arm,
            "episode_short_name": ep_sn,
            "call_role": role,
            "stage_subdir": str(c_kwargs.get("stage_subdir") or ""),
            "attempts_ledger_filename": att_fn,
            "mirrored_raw_filename": f"{cu_uuid}__{att_fn}",
            "utc_started_at": utc_t0,
            "utc_ended_at": utc_t1,
            "monotonic_start_sec": round(mono_t0, 4),
            "monotonic_end_sec": round(mono_t1, 4),
            "monotonic_elapsed_sec": round(mono_t1 - mono_t0, 4),
            "usage_unknown_attempt_count": int(res.get("usage_unknown_attempt_count", 0) or 0),
            **sc,
        }
        with _RECEIPT_HOOK_LOCK:
            all_receipts.append(rec_row)
        return res

    if not args.finalize_existing:
        m6f.FrontierCodexClient.call_structured = _wrapped_call_structured  # type: ignore[method-assign]

    campaign_t0 = monotonic()
    built_worlds = {sp.world_id: build_m102_world_tasks(sp) for sp in selected_specs}

    if not args.finalize_existing:
        v7_forest, v7_archive, _, _ = m7a.load_autogenous_forest_bundle(args.v7_bundle.resolve())
        stage_s_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(selected_specs)) as pool:
            futs = {
                pool.submit(
                    _run_stage_s_shared_source,
                    spec=sp,
                    s_task=built_worlds[sp.world_id][0],
                    base_v7_forest=v7_forest,
                    base_v7_archive=v7_archive,
                    codex_info=codex_info,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    out_dir=out_dir,
                    run_id=run_id,
                    max_turns=args.max_turns,
                ): sp.world_id
                for sp in selected_specs
            }
            for fut in as_completed(futs):
                stage_s_results[futs[fut]] = fut.result()

        stream_jobs = [(sp, arm_nm) for sp in selected_specs for arm_nm in M102_ARMS]
        stream_summaries: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(int(args.max_stream_workers), len(stream_jobs))) as pool:
            s_futs = []
            for sp, arm_nm in stream_jobs:
                wid = sp.world_id
                s_tk, st_tks, s_man = built_worlds[wid]
                st_dir = out_dir / "continual_streams" / wid / arm_nm
                s_res = stage_s_results[wid]
                s_futs.append(
                    pool.submit(
                        _run_one_m102_world_arm_stream,
                        spec=sp,
                        arm=arm_nm,
                        s_task=s_tk,
                        stream_tasks=st_tks,
                        schedule_manifest=s_man,
                        base_v7_forest=v7_forest,
                        base_v7_archive=v7_archive,
                        source_monolithic_code=s_res["source_monolithic_code"],
                        source_pub_sha256=s_res["source_pub_sha256"],
                        source_obs=s_res["source_obs"],
                        source_public_gate=s_res["source_public_gate"],
                        model=args.model,
                        codex_info=codex_info,
                        stream_out_dir=st_dir,
                        run_id=run_id,
                        max_turns=args.max_turns,
                        reasoning_effort=args.reasoning_effort,
                    )
                )
            for fut in as_completed(s_futs):
                stream_summaries.append(fut.result())

        stream_summaries.sort(key=lambda s: (s["world_id"], M102_ARMS.index(s["arm"])))
        all_receipts.sort(key=lambda x: (x["monotonic_start_sec"], x["call_uuid"]))
        (out_dir / "all_provider_call_receipts.json").write_text(json.dumps(all_receipts, indent=2) + "\n", encoding="utf-8")

        git_sha_end, is_dirty_end, src_tree_sha_end = _git_and_source_tree_state()
        exec_done = {
            "execution_run_id": run_id,
            "utc_ended_at": datetime.now(timezone.utc).isoformat(),
            "monotonic_elapsed_sec": round(monotonic() - campaign_t0, 2),
            "execution_elapsed_wall_sec": round(monotonic() - campaign_t0, 2),
            "clock_domain": "monotonic_elapsed_plus_utc_iso8601",
            "peak_active_codex_concurrency": int(active_counter["peak"]),
            "code_commit_at_end": git_sha_end,
            "working_tree_dirty_at_end": bool(is_dirty_end),
            "source_tree_sha256_at_end": src_tree_sha_end,
            "source_tree_unchanged_during_paid_execution": bool(src_tree_sha_end == src_tree_sha_start),
        }
        exec_done_path.write_text(json.dumps(exec_done, indent=2) + "\n", encoding="utf-8")
    else:
        all_receipts = json.loads((out_dir / "all_provider_call_receipts.json").read_text(encoding="utf-8"))
        stage_s_results = {
            sp.world_id: {"summary": json.loads((out_dir / "stage_S_shared_sources" / sp.world_id / "00_stage_S_summary.json").read_text(encoding="utf-8"))}
            for sp in selected_specs
        }
        stream_summaries = [
            json.loads((out_dir / "continual_streams" / sp.world_id / arm_nm / "00_stream_summary.json").read_text(encoding="utf-8"))
            for sp in selected_specs
            for arm_nm in M102_ARMS
        ]

    abl_path = out_dir / "m10_2_step14_cold_hydration_and_plain_dict_parity.json"
    abl_manifest_path = out_dir / "ablation_manifest.json"
    if abl_path.exists():
        parity_ablation = json.loads(abl_path.read_text(encoding="utf-8"))
    else:
        t_ab0 = monotonic()
        parity_ablation = run_m102_step14_cold_hydration_and_plain_dict_parity_ablation(
            world_specs=selected_specs, built_worlds=built_worlds, out_dir=out_dir, checkpoint_step=14
        )
        abl_path.write_text(json.dumps(parity_ablation, indent=2) + "\n", encoding="utf-8")
        abl_manifest_path.write_text(
            json.dumps(
                {
                    "schema": "hegel_m10_2_ablation_provenance_manifest_v1",
                    "code_commit": git_sha_start,
                    "source_tree_sha256": src_tree_sha_start,
                    "monotonic_elapsed_sec": round(monotonic() - t_ab0, 2),
                    "utc_completed_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    exec_start_obj = json.loads(exec_manifest_path.read_text(encoding="utf-8"))
    exec_done_obj = json.loads(exec_done_path.read_text(encoding="utf-8"))

    # Exact per-episode all-role call counts for Dual Zero-Call Metrics (`U08`)!
    ep_call_counts: dict[tuple[str, str, str], int] = {}
    for r in all_receipts:
        k_ep = (str(r["world_id"]), str(r["arm"]), str(r["episode_short_name"]))
        ep_call_counts[k_ep] = ep_call_counts.get(k_ep, 0) + 1

    for s in stream_summaries:
        all_role_zero_passes = sum(
            1
            for ep in s["episodes_meta"]
            if ep["strict_pass"] and ep_call_counts.get((s["world_id"], s["arm"], ep["short_name"]), 0) == 0
        )
        all_role_zero_total = sum(
            1
            for ep in s["episodes_meta"]
            if ep_call_counts.get((s["world_id"], s["arm"], ep["short_name"]), 0) == 0
        )
        s["all_role_call_free_successes"] = all_role_zero_passes
        s["all_role_call_free_episodes_total"] = all_role_zero_total

    stage_s_calls = [r for r in all_receipts if r["arm"] == "Stage_S_Shared_Source"]
    stage_s_ledger = {
        **aggregate_call_usages(stage_s_calls),
        "usage_unknown_attempt_count": sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in stage_s_calls),
    }

    arm_ledgers: dict[str, Any] = {}
    arm_outcomes: dict[str, Any] = {}
    for arm_nm in M102_ARMS:
        a_calls = [r for r in all_receipts if r["arm"] == arm_nm]
        tot_led = aggregate_call_usages(a_calls)
        role_leds = {role: aggregate_call_usages([r for r in a_calls if r["call_role"] == role]) for role in M102_SCOPE1_ROLES}
        by_cluster = {}
        for sp in selected_specs:
            wc = [r for r in a_calls if r["world_id"] == sp.world_id]
            by_cluster[sp.world_id] = {
                **aggregate_call_usages(wc),
                "usage_unknown_attempt_count": sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in wc),
            }
        assert_hierarchical_ledger_conservation(
            reference_arm_ledger=tot_led,
            named_partition_ledgers={"clusters": list(by_cluster.values()), "roles": list(role_leds.values())},
            context_label=arm_nm,
        )
        arm_ledgers[arm_nm] = {
            "scope1_in_stream_total": {
                **tot_led,
                "usage_unknown_attempt_count": sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in a_calls),
            },
            "by_role": role_leds,
            "by_cluster": by_cluster,
        }
        sub_s = [s for s in stream_summaries if s["arm"] == arm_nm]
        arm_outcomes[arm_nm] = {
            "strict_passes": sum(int(s["strict_passes"]) for s in sub_s),
            "episode_count": sum(int(s["episode_count"]) for s in sub_s),
            "solver_free_episodes_total": sum(int(s["solver_free_episodes_total"]) for s in sub_s),
            "solver_free_successes": sum(int(s["solver_free_successes"]) for s in sub_s),
            "all_role_call_free_episodes_total": sum(int(s["all_role_call_free_episodes_total"]) for s in sub_s),
            "all_role_call_free_successes": sum(int(s["all_role_call_free_successes"]) for s in sub_s),
            "compositional_reuse_passes": sum(int(s["compositional_reuse_passes"]) for s in sub_s),
            "total_turns_used": sum(int(s["total_turns_used"]) for s in sub_s),
            "max_pre_llm_probes_observed": max((int(s["max_pre_llm_probes_observed"]) for s in sub_s), default=0),
            "total_pre_llm_probes_used": sum(int(s["total_pre_llm_probes_used"]) for s in sub_s),
            "cut_withdrawals": sum(int(s["cut_withdrawal_count"]) for s in sub_s),
            "failed_search_events": sum(int(s["failed_search_events_count"]) for s in sub_s),
            "probe_receipt_store_total_files": sum(int(s["probe_receipt_store_count"]) for s in sub_s),
            "max_active_nodes_observed": max((int(s["max_active_nodes_observed"]) for s in sub_s), default=1),
            "max_depth_observed": max((int(s["max_depth_observed"]) for s in sub_s), default=1),
            "max_fanout_observed": max((int(s["max_fanout_observed"]) for s in sub_s), default=0),
            "shared_fanout_exec_count_1_verified_clusters": sum(1 for s in sub_s if s["shared_fanout_exec_count_1_verified"]),
            "all_traced_episodes_exec_count_1": all(bool(s["all_traced_episodes_exec_count_1"]) for s in sub_s),
            "pre_llm_probe_wall_ms": round(sum(float(s["total_pre_llm_probe_wall_ms"]) for s in sub_s), 2),
            "post_llm_probe_wall_ms": round(sum(float(s["total_post_llm_probe_wall_ms"]) for s in sub_s), 2),
            "official_verifier_wall_ms": round(sum(float(s["total_official_verifier_wall_ms"]) for s in sub_s), 2),
        }

    grand_tot = aggregate_call_usages(all_receipts)
    assert_hierarchical_ledger_conservation(
        reference_arm_ledger=grand_tot,
        named_partition_ledgers={"arms_and_source": [stage_s_ledger] + [arm_ledgers[a]["scope1_in_stream_total"] for a in M102_ARMS]},
        context_label="M102_grand_total",
    )

    # Stratified inference at Primary Endpoint `H = 16`
    wids = [sp.world_id for sp in selected_specs]
    strata_groups = {
        "pooled_all": wids,
        "stratum_a_seen_topology": [sp.world_id for sp in selected_specs if sp.stratum == "stratum_a_seen_topology"],
        "stratum_b1_unseen_static_topology": [sp.world_id for sp in selected_specs if sp.stratum == "stratum_b1_unseen_static_topology"],
        "stratum_b2_shift_order_and_regime_switch": [sp.world_id for sp in selected_specs if sp.stratum == "stratum_b2_shift_order_and_regime_switch"],
        "stratum_b_combined": [sp.world_id for sp in selected_specs if sp.stratum != "stratum_a_seen_topology"],
        "monolithic_control": [sp.world_id for sp in selected_specs if sp.topology_class == "monolithic_coupled"],
        "multi_node_non_monolithic": [sp.world_id for sp in selected_specs if sp.topology_class != "monolithic_coupled"],
    }
    st_by_key = {(s["world_id"], s["arm"]): s for s in stream_summaries}

    stratified_h16: dict[str, Any] = {}
    for g_name, sub_w in strata_groups.items():
        if not sub_w:
            continue
        stratified_h16[g_name] = {}
        for comp_lbl, arm_a, arm_b in (
            ("Hegel_Selective_minus_Flat_Monolithic", "Hegel_SelectiveBoundedDAG_Online", "Flat_MonolithicArchive_Online"),
            ("Hegel_Selective_minus_Eager_BoundedDAG", "Hegel_SelectiveBoundedDAG_Online", "Eager_BoundedDAG_Online"),
            ("Eager_BoundedDAG_minus_Flat_Monolithic", "Eager_BoundedDAG_Online", "Flat_MonolithicArchive_Online"),
        ):
            ca = [int(arm_ledgers[arm_a]["by_cluster"][w]["proxy_v2_u_010c_4o"]) for w in sub_w]
            cb = [int(arm_ledgers[arm_b]["by_cluster"][w]["proxy_v2_u_010c_4o"]) for w in sub_w]
            c_diffs = [float(a - b) for a, b in zip(ca, cb)]
            q_diffs = [
                (float(st_by_key[(w, arm_a)]["strict_passes"]) - float(st_by_key[(w, arm_b)]["strict_passes"]))
                / float(st_by_key[(w, arm_b)]["episode_count"])
                for w in sub_w
            ]
            c_inf = paired_cluster_inference_stats(c_diffs)
            q_inf = paired_cluster_inference_stats(q_diffs)
            c_boot = c_inf["bootstrap_95ci"]
            q_boot = q_inf["bootstrap_95ci"]
            q_t = q_inf["paired_t_95ci"]
            stratified_h16[g_name][comp_lbl] = {
                "n_clusters": len(sub_w),
                "arm_a_passes": sum(int(st_by_key[(w, arm_a)]["strict_passes"]) for w in sub_w),
                "arm_b_passes": sum(int(st_by_key[(w, arm_b)]["strict_passes"]) for w in sub_w),
                "total_episodes": sum(int(st_by_key[(w, arm_b)]["episode_count"]) for w in sub_w),
                "arm_a_total_v2": int(sum(ca)),
                "arm_b_total_v2": int(sum(cb)),
                "total_cost_delta_v2": int(sum(ca) - sum(cb)),
                "pct_cost_change": round(100.0 * (sum(ca) - sum(cb)) / max(1, sum(cb)), 4),
                "cost_inference": c_inf,
                "quality_inference": q_inf,
                "cost_superiority_supported": bool(
                    c_inf["mean_diff"] is not None
                    and c_inf["mean_diff"] < 0
                    and c_inf["two_sided_sign_flip_p"] is not None
                    and c_inf["two_sided_sign_flip_p"] < 0.05
                    and c_boot is not None
                    and c_boot[1] < 0
                ),
                "quality_noninferiority_bootstrap_supported": bool(
                    q_boot is not None and q_boot[0] >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE
                ),
                "quality_noninferiority_paired_t_supported": bool(
                    q_t is not None and q_t[0] >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE
                ),
            }

    nested_curves = compute_m102_nested_horizons_and_k_star(
        world_specs=selected_specs,
        all_receipts=all_receipts,
        stream_summaries=stream_summaries,
        n_steps=M102_HORIZON_STEPS,
    )
    structural_gates = evaluate_m102_structural_verification_gates(
        world_specs=selected_specs,
        stream_summaries=stream_summaries,
        arm_ledgers=arm_ledgers,
        parity_ablation=parity_ablation,
    )
    total_unknown = sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in all_receipts)

    summary = {
        "schema": "hegel_m10_2_selective_bounded_dag_summary_v1",
        "execution_run_id": exec_start_obj["execution_run_id"],
        "split": args.split,
        "execution_provenance": {**exec_start_obj, **exec_done_obj},
        "independent_source_cluster_count": len({sp.cluster_id for sp in selected_specs}),
        "primary_horizon_steps": M102_HORIZON_STEPS,
        "nested_checkpoints": list(M102_NESTED_CHECKPOINTS),
        "max_pre_llm_probes_per_step_all_arms": M102_MAX_PRE_LLM_PROBES_PER_STEP,
        "sandbox_image_digest": exec_start_obj["sandbox_image_digest"],
        "shared_kernel_version": exec_start_obj["shared_kernel_version"],
        "verifier_schema_version": M10_1_1_VERIFIER_SCHEMA_VERSION,
        "cost_policy_version": exec_start_obj["cost_policy_version"],
        "model": exec_start_obj["model"],
        "reasoning_effort": exec_start_obj["reasoning_effort"],
        "configured_max_codex_concurrency": exec_start_obj["configured_max_codex_concurrency"],
        "peak_active_codex_concurrency": exec_done_obj["peak_active_codex_concurrency"],
        "monotonic_elapsed_sec": exec_done_obj["monotonic_elapsed_sec"],
        "usage_completeness": ("complete" if total_unknown == 0 else "partial_unknown_usage"),
        "total_usage_unknown_attempt_count": total_unknown,
        "total_logical_calls": len(all_receipts),
        "grand_physical_ledger": grand_tot,
        "stage_S_shared_ledger": stage_s_ledger,
        "stage_S_summaries": {wid: stage_s_results[wid]["summary"] for wid in wids},
        "arm_outcome_totals": arm_outcomes,
        "arm_cost_ledgers": arm_ledgers,
        "stratified_primary_endpoint_h16": stratified_h16,
        "nested_horizon_curves_and_K_star": nested_curves,
        "structural_verification_gates": structural_gates,
        "step14_cold_hydration_and_plain_dict_parity": parity_ablation,
        "streams": stream_summaries,
    }
    (out_dir / "m10_2_selective_bounded_dag_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    sha_lines = []
    for fp in sorted(out_dir.rglob("*")):
        if fp.is_file() and fp.name != "SHA256SUMS.txt":
            rel = fp.relative_to(out_dir).as_posix()
            sha_lines.append(f"{hashlib.sha256(fp.read_bytes()).hexdigest()}  {rel}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "execution_run_id": summary["execution_run_id"],
                "split": args.split,
                "total_calls": len(all_receipts),
                "unknown_attempts": total_unknown,
                "peak_concurrency": summary["peak_active_codex_concurrency"],
                "arm_outcomes": arm_outcomes,
                "K_star_Hegel_vs_Monolithic": nested_curves["Hegel_Selective_minus_Flat_Monolithic"]["empirical_quality_constrained_K_star"],
                "K_star_Hegel_vs_EagerDAG": nested_curves["Hegel_Selective_minus_Eager_BoundedDAG"]["empirical_quality_constrained_K_star"],
                "structural_gates": {g["gate_id"]: g["status"] for g in structural_gates},
                "plain_dict_parity_pct": parity_ablation["parity_rate_pct"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    original_call = m6f.FrontierCodexClient.call_structured
    try:
        _main()
    finally:
        m6f.FrontierCodexClient.call_structured = original_call


if __name__ == "__main__":
    main()
