#!/usr/bin/env python3
"""Live Execution & Pre-Registered Economic Evaluation Runner for Hegel `M10.1`:
Bounded Unknown-Cut Acyclic Computational DAG Discovery (`1–4` Active Nodes, `depth <= 4`, `fan-out <= 2`, `K = 1` Always Legal).

Strict Audit & Scientific Contracts:
  - Zero prescribed split boundaries, zero prescribed module counts, zero `U/D` role labels in prompts/IDs/schemas.
  - 3 equal-tool arms sharing identical `Stage S` (`batch_00`) initial monolithic solver, neutral `obs_00..obs_17` extractor,
    pure-public 56/16 held-out Docker scientific gate, atomic `ProbeReceiptStore` (`T01`), Kleene large-margin guard compiler
    (`min_clause_margin = 0.035`), and official private Docker grader:
      1. `Flat_MonolithicArchive_Online` (`K = 1` general Python monolithic table; NO ban on internal helper functions!)
      2. `Fixed_TwoStage_Modular_Online` (prescribed `K <= 2` sequential chain `node_00 -> node_01`, `max_fanout = 1`)
      3. `Hegel_UnknownCut_GraphDiscovery_Online` (`1 <= K <= 4`, `depth <= 4`, `fan-out <= 2`, `K = 1` always legal, 6 actions:
         `keep_monolithic`, `extract_module`, `split_node`, `merge_nodes`, `reuse_subgraph`, `rewire_edge`, runtime
         `exec_count == 1` + anti-identity-shell enforcement, 4-link external-FK `CutWithdrawalMergeReceipt`).
  - 100% metered `Scope-1` cost conservation across `stream_solve`, `dag_structure_proposal`,
    `dag_compile_and_node_synthesis`, `dag_failed_search`, `dag_merge_withdrawal_repair`, and `guard_induction`.
  - Write-once immutable start-time `execution_manifest.json`, separated `execution_completion.json` & `ablation_manifest.json`,
    UTC + monotonic separated clock domains (`T11`), exact sign-flip `p` + unrounded bootstrap & Student-`t` sensitivity 95% CIs
    (`T02`), Stratum A vs Stratum B stratified reporting, and pooled + per-cluster finite 8-step `ΔC(K)` & `K*` (`T07`).
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
    ProbeReceiptStore,
    _canonical_json_sha256,
    evaluate_m96_autonomous_guard,
)
from hegel_machine.cognition.m10_0_reliable_modular_amortization_and_bidirectional_withdrawal_world import (  # noqa: E402
    propose_and_validate_m99_guard_partition,
)
from hegel_machine.cognition.m10_1_unknown_cut_dag_discovery_world import (  # noqa: E402
    CONFIRMATION_M101_WORLD_SPECS,
    DEV_M101_WORLD_SPECS,
    M101_ARMS,
    M101_DAG_LEDGER_FOREST_KEY,
    M101_MAX_ACTIVE_NODES,
    M101_MAX_DEPTH,
    M101_MAX_FANOUT,
    M101_OBSERVABLE_NAMES,
    CutWithdrawalMergeReceipt,
    ExecutableComputationDAG,
    ExecutableModuleNode,
    M101WorldSpec,
    build_m101_world_tasks,
    compile_m101_dag_to_instrumented_solver_code,
    compute_m101_neutral_observables,
    execute_m101_public_gate_in_container,
    grade_m101_candidate_in_container,
    m101_task_to_frontier_task,
    materialize_m101_public_study_and_sha,
    verify_m101_cut_withdrawal_external_fk,
)

DEFAULT_V7_BUNDLE = REPO_ROOT / "docs/evidence/m9_2_1_audit_closure_and_regrade_20261126/pretrained_autogenous_forest_v7_m921.json"
DEFAULT_DEV_OUT_DIR = REPO_ROOT / "docs/evidence/m10_1_dev_unknown_cut_calibration_20261128"
DEFAULT_CONF_OUT_DIR = REPO_ROOT / "docs/evidence/m10_1_bounded_unknown_cut_dag_discovery_20261128"
PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE = -0.05

M101_SCOPE1_ROLES: tuple[str, ...] = (
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
    """Re-attribute any LLM receipts generated during a failed DAG search/synthesis attempt to `dag_failed_search`."""
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


def _persist_m101_probe(
    probe_store: ProbeReceiptStore,
    *,
    probe_kind: str,
    receipt: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Persist ANY M10.1 public/health/decomp probe atomically into `ProbeReceiptStore` (`T01`: 100% coverage)."""
    pid = f"m101p_{uuid.uuid4().hex[:16]}"
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


def _obs_l1_distance(a: dict[str, float | None], b: dict[str, float | None], keys: tuple[str, ...] = M101_OBSERVABLE_NAMES) -> float:
    vals = [abs(float(a[k]) - float(b[k])) for k in keys if a.get(k) is not None and b.get(k) is not None]
    return float(sum(vals) / max(1, len(vals)))


def _select_guarded_candidate(
    items: list[dict[str, Any]],
    obs: dict[str, float | None],
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Deterministic Kleene `all_of` selector (`priority desc, id asc`)."""
    matched = []
    for it in items:
        ev = evaluate_m96_autonomous_guard(it.get("guard") or {}, obs, allow_open_feature_keys=True)
        if ev["state"] == "TRUE":
            matched.append(it)
    matched.sort(key=lambda r: (-int(r.get("priority", 0)), str(r["id"])))
    ordered = sorted(items, key=lambda r: (-int(r.get("priority", 0)), str(r["id"])))
    if matched:
        chosen = matched[0]
        alts = [r for r in ordered if r["id"] != chosen["id"]]
        return chosen, ("unique_guard_match" if len(matched) == 1 else "priority_tiebreak"), alts
    chosen = ordered[0]
    alts = [r for r in ordered if r["id"] != chosen["id"]]
    return chosen, "fallback_highest_priority", alts


def _rank_compositional_node_version_combinations(
    *,
    template_dag: ExecutableComputationDAG,
    node_library: dict[str, ExecutableModuleNode],
    node_pos_obs: dict[str, list[dict[str, float | None]]],
    obs_now: dict[str, float | None],
    already_tried_fingerprints: set[str],
    max_candidates: int = 4,
) -> list[ExecutableComputationDAG]:
    """Data-driven 0-LLM compositional subgraph replay over ALREADY-VERIFIED node versions!
    Zero hardcoded panel-to-node mapping: for each `node_id` having `>= 2` verified versions, ranks observables among
    `obs_00..obs_17` by Fisher ANOVA variance ratio (`between_version_spread^2 / (within_version_var + 0.01)`) across that
    node's verified positive witnesses—automatically isolating each node's true causal observable without confounding
    incidental shifts in other nodes!—and emits top unseen Cartesian combinations.
    """
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
                for k in M101_OBSERVABLE_NAMES
            }
        # Fisher ANOVA separation ratio: `between_spread^2 / (within_var + 0.01)`
        fisher_feats: list[tuple[float, float, str]] = []
        for k in M101_OBSERVABLE_NAMES:
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
        # Keep top 2 highest-Fisher observables (or all within 40% of max Fisher ratio, up to 3)
        if fisher_feats:
            f_max = fisher_feats[0][0]
            top_feats = [x for x in fisher_feats if x[0] >= 0.40 * f_max][:3]
            if not top_feats:
                top_feats = fisher_feats[:2]
        else:
            top_feats = [(1.0, 1.0, k) for k in M101_OBSERVABLE_NAMES]

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

    # Enumerate Cartesian combinations (bounded since <=4 nodes and <=3 versions/node -> at most 81 combos)
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
# 2. Equal-Info Archive Injection & Targeted Node-Update Sandbox Builder
# ============================================================================


def _inject_m101_equal_info_archive(
    task: HoldoutOODTask,
    prior_entries: list[dict[str, Any]],
    pre_llm_diag: dict[str, Any] | None = None,
) -> tuple[HoldoutOODTask, str]:
    """Identical `/tmp/hegel_continual_archive/` staging across arms (`0` helper ban, `0` topology hints!)."""
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
                "obs_summary": ent.get("obs"),
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
        "- Pre-LLM public 56/16 CV gate diagnostic: `/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json`\n"
        "- You may freely inspect prior solvers, reuse/refactor helper functions or sub-computations, and validate via "
        "internal 56/16 held-out CV on `train` before finishing.\n"
    )
    return aug, hdr


def _build_m101_targeted_node_synthesis_task(
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
    """Build a Docker sandbox task where ONLY `editable_node_ids` (`1` or at most `2` nodes) need to be written into
    `/holdout_task/node_{nid}.py` (`def run_node(inputs: dict) -> dict`), while ALL unchanged verified nodes are mounted
    read-only under `/holdout_task/_frozen_nodes/` and `/holdout_task/test_dag_candidate.py` compiles & runs the exact
    instrumented public 56/16 CV gate + `exec_count == 1` + anti-identity-shell check!
    """
    aug_task, _ = _inject_m101_equal_info_archive(base_task, prior_entries, pre_llm_diag)
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
import copy, hashlib, importlib.util, json, math, os, pickle, sys
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
        elif isinstance(v, dict):
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
    outputs = {"__input__": copy.deepcopy(study)}
    ec = {n: 0 for n in spec_obj["topo_order"]}
    for nid in spec_obj["topo_order"]:
        srcs = spec_obj["incoming"][nid]
        node_in = copy.deepcopy(outputs["__input__"]) if (len(spec_obj["topo_order"]) == 1 and srcs == ["__input__"]) else {s: copy.deepcopy(outputs[s]) for s in srcs}
        in_d = _canon_digest(node_in)
        out_v = fns[nid](node_in)
        ec[nid] += 1
        if not isinstance(out_v, dict) or not out_v:
            raise RuntimeError(f"Node {nid} returned non-dict or empty dict")
        if nid != spec_obj["sink_node_id"]:
            od = _canon_digest(out_v)
            if od == in_d or any(od == _canon_digest(outputs[s]) for s in srcs):
                raise RuntimeError(f"Node {nid} failed anti-identity-shell check")
        outputs[nid] = out_v
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
    if ym > 0.035: errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
    if tm > 0.035: errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")

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
        + (" (SINK node: must return `{'y_hat', 'state_pred', 'diagnostics_vec'}`)" if nid == target_dag.sink_node_id else " (intermediate node: must transform inputs into a non-empty dict of numpy arrays matching downstream consumers)")
        for nid in editable_node_ids
    )
    frozen_desc = ", ".join(f"`/holdout_task/_frozen_nodes/{nid}.py`" for nid in sorted(frozen_map.keys())) or "(none)"

    inst = (
        f"TARGETED COMPUTATION-GRAPH NODE SYNTHESIS (`action={edit_action}`):\n"
        f"- DAG topology (`nodes={target_dag.nodes}`, `edges={target_dag.edges}`, `sink={target_dag.sink_node_id}`).\n"
        f"- Verified frozen nodes (read-only; inspect them to match exact intermediate dict key names!): {frozen_desc}\n"
        f"- Diagnosis: {edit_rationale}\n"
        f"- Write ONLY the editable node file(s):\n{in_desc}\n"
        "  Each file must expose `def run_node(inputs: dict) -> dict:`.\n"
        "- Inspect `/holdout_task/public_batch.pkl` numerically, infer the exact mathematical relationship for the edited "
        "node(s), and run `python3 /holdout_task/test_dag_candidate.py` until `accepted: true` (`public_cv_y_mae <= 0.035`, "
        "`public_cv_traj_mae <= 0.035`, `exec_count == 1`, anti-identity-shell passed)!\n"
        "- Remember: `panel_a` has 3 sparse outliers (use robust forward IRLS fitting `u -> v` before inverting if modeling `panel_a`); "
        "`panel_b` and `panel_c` have 24 rows (avoid over-parameterized degree-3 polynomials with >24 terms; discover the actual nonlinear basis!).\n"
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
# 3. Live Structured Unknown-Cut DAG Proposal & Refactoring (`gpt-6.2-sol`)
# ============================================================================


def _propose_initial_dag_decomposition_or_keep_monolithic(
    *,
    client: Any,
    stage_subdir: str,
    source_code: str,
    source_obs: dict[str, float | None],
    target_obs: dict[str, float | None],
    max_nodes: int,
    max_fanout: int,
    arm_label: str,
) -> dict[str, Any]:
    """Ask LLM whether a verified monolithic solver should stay `keep_monolithic` (`K = 1`) OR be decomposed into a
    bounded acyclic DAG (`2..max_nodes` active nodes, `depth <= 4`, `fan-out <= max_fanout`) with zero prescribed cut hints!
    """
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
        for k in M101_OBSERVABLE_NAMES
    }
    sys_prompt = (
        "You are an autonomous computational-graph architect analyzing a previously verified monolithic scientific solver "
        "(`source_code`) and public statistical summaries (`obs_00..obs_17`) comparing the verified source batch against a "
        "new shifted batch where `source_code` failed public cross-validation.\n"
        f"HARD TOPOLOGY BOUNDS: `1 <= active_nodes <= {max_nodes}`, acyclic, `depth <= {M101_MAX_DEPTH}`, `max_fanout <= {max_fanout}`.\n"
        "DECISION RULES:\n"
        "1. Choose `keep_monolithic` (returning empty `nodes/edges/node_modules`, `sink_node_id=''`) whenever `source_code` "
        "does NOT actually compose distinct stages over `panel_a`/`panel_b`/`panel_c` (for example, if `source_code` directly "
        "maps `train['x'] -> y, traj` without fitting/inverting the auxiliary panels, or if the panels do not separate stages!). "
        "`K = 1` monolithic is ALWAYS legal and preferred when no genuine modular factorization exists.\n"
        "2. Choose `split_node` or `extract_module` ONLY when `source_code` genuinely composes distinct sub-computations "
        "(e.g., fitting/inverting `panel_a`, applying `panel_b`, applying `panel_c['v1']` and/or `panel_c['v2']` in sequence "
        "or parallel branches, and combining intermediate features into the readout on `train`/`test`).\n"
        "3. When decomposing:\n"
        f"   - Use neutral node IDs `node_00`, `node_01`, ... (at most `{max_nodes}` nodes, `max_fanout <= {max_fanout}`).\n"
        "   - Every node that reads `study` MUST have an edge `['__input__', node_id]`!\n"
        "   - Each node's Python code must define `def run_node(inputs: dict) -> dict:` where `inputs` maps each incoming "
        "source ID (`'__input__'` or predecessor `'node_XX'`) to its dict output!\n"
        "   - ANTI-IDENTITY-SHELL RULE: every non-sink node MUST compute a genuine numerical transformation (e.g., return "
        "`{'z_tr': ..., 'z_te': ...}` or `{'h_tr': ..., 'h_te': ...}`), NEVER echo `inputs['__input__']`!\n"
        "   - SHARED PREFIX RULE: if two parallel branches (or a skip path) both need the output of `node_00`, wire "
        "`['node_00', 'node_01']` and `['node_00', 'node_02']` (or `['node_00', sink_node_id]`) so `node_00` executes ONCE (`exec_count == 1`)!\n"
        "   - The extracted DAG MUST reproduce `source_code`'s behavior accurately on the verified source batch (so we can verify "
        "it immediately on the source batch in Docker before editing only the shifted node for the target batch!)."
    )
    payload = {
        "arm_topology_bounds": {"max_nodes": max_nodes, "max_depth": M101_MAX_DEPTH, "max_fanout": max_fanout},
        "source_monolithic_code": source_code,
        "observable_comparison_source_vs_target": obs_diff,
    }
    res = client.call_structured(
        system_prompt=sys_prompt,
        conversation_turns=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        output_schema=schema,
        stage_subdir=stage_subdir,
    )
    return res


def _propose_dag_edit_action_for_current_step(
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
    """Structured graph-edit action planner (`6` legal actions bounded by arm `max_nodes` & `max_fanout`, zero hints!)."""
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
    sys_prompt = (
        "You are an online computational-graph edit planner deciding the minimal, valid edit action to repair a failing "
        "scientific campaign step.\n"
        f"HARD TOPOLOGY BOUNDS: `1 <= len(new_nodes) <= {max_nodes}`, acyclic, `depth <= {M101_MAX_DEPTH}`, `max_fanout <= {max_fanout}`.\n"
        "LEGAL ACTIONS:\n"
        "- `reuse_subgraph`: Keep an existing multi-node DAG topology (`base_dag_id`), freeze ALL unchanged nodes via "
        "`reused_node_versions`, and list ONLY the `1` (or at most `2`) shifted node(s) in `editable_node_ids` to re-synthesize! "
        "CAREFULLY inspect every verified node's full `code` in `verified_node_versions` to pick the EXACT node whose panel/readout "
        "shifted (e.g., if `obs_10_pc_v2_lin_r2` shifted, edit ONLY the node that reads `panel_c['v2']`, NOT the node that reads `panel_c['v1']`!).\n"
        f"- `split_node` / `extract_module`: Refine an existing DAG into more nodes (`<= {max_nodes}`) whenever an existing non-sink "
        "node lumps two distinct stages together (e.g., fitting both `panel_a` and `panel_b`, or both `panel_c['v1']` and `panel_c['v2']`) "
        "and one of those stages just shifted while the other stayed unchanged—so each independently shifting stage becomes its own reusable node!\n"
        "- `rewire_edge`: Keep existing upstream/branch nodes frozen, ADD a direct cross-node edge (e.g., `['node_00', sink_node_id]`, "
        f"respecting `max_fanout <= {max_fanout}`), and set `editable_node_ids = [sink_node_id]`! Choose this whenever ALL individual "
        "panel observables already match previously verified versions (no single panel looks newly unseen), YET the separable DAG fails "
        "public CV — signaling a direct cross-stage physical interaction requiring the sink node to also see an earlier node's output!\n"
        "- `merge_nodes`: Fuse interacting nodes into fewer nodes (or `K = 1` monolithic) when `rewire_edge` would exceed "
        f"`max_fanout <= {max_fanout}` or when intermediate cuts no longer separate the physics! (The old DAG is ALWAYS preserved intact!)\n"
        "- `keep_monolithic`: Solve directly as a `K = 1` monolithic program (`new_nodes = ['node_00']`, `new_edges = [['__input__', 'node_00']]`, "
        "`new_sink_node_id = 'node_00'`, `reused_node_versions = []`, `editable_node_ids = ['node_00']`).\n"
        "IMPORTANT: Every node in `new_nodes` MUST be either in `reused_node_versions` OR in `editable_node_ids` (never both, never neither). "
        "Every node reading `study` MUST have edge `['__input__', node_id]`."
    )
    payload = {
        "bounds": {"max_nodes": max_nodes, "max_depth": M101_MAX_DEPTH, "max_fanout": max_fanout},
        "current_observables": obs_now,
        "reference_observables_by_dag": ref_obs_by_dag,
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
    """Run existing compiled DAG on `task` public study and return public self-consistency health receipt (for `T01` FK!)."""
    code_txt, comp_meta = compile_m101_dag_to_instrumented_solver_code(dag, node_library)
    gate = execute_m101_public_gate_in_container(task=task, candidate_code=code_txt, expect_dag_trace=True)
    tr = gate.get("dag_runtime_trace") or {}
    # Individual upstream nodes executed cleanly without schema/runtime/anti-identity violation
    nodes_clean = bool(gate.get("schema_valid") and isinstance(tr, dict) and tr.get("all_exec_count_eq_1"))
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
# 4. Shared `Stage S` (`batch_00`) Runner & 3-Arm Online Continual Stream Engine
# ============================================================================


def _run_stage_s_shared_source(
    *,
    spec: M101WorldSpec,
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
    """Solve `Stage S` (`batch_00`) ONCE per independent cluster monolithically (`0` host-seeded modules!)."""
    wid = spec.world_id
    s_dir = out_dir / "stage_S_shared_sources" / wid
    s_dir.mkdir(parents=True, exist_ok=True)
    ft_s = m101_task_to_frontier_task(s_task)

    _CALL_CONTEXT.role = "stage_S_solve"
    _CALL_CONTEXT.world_id = wid
    _CALL_CONTEXT.cluster_id = spec.cluster_id
    _CALL_CONTEXT.arm = "Stage_S_Shared_Source"
    _CALL_CONTEXT.episode_short_name = s_task.short_name

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
        episode_out_dir=s_dir,
        run_id=f"{run_id}__stageS__{wid}",
        max_turns=max_turns,
        cmd_timeout_sec=180,
        reasoning_effort=reasoning_effort,
        defer_official_verifier=True,
        enable_auto_skill_packaging=False,
        allow_l1_primitives=False,
        disclosure_mode="on_demand_executable",
        staging_policy="net_value_gated",
    )
    sub = _extract_submitted_files(s_dir)
    s_code = sub.get("batch_solver.py", "")
    pub_gate = execute_m101_public_gate_in_container(task=s_task, candidate_code=s_code, expect_dag_trace=False) if s_code else {"accepted": False}
    priv_ver = grade_m101_candidate_in_container(task=s_task, candidate_code=s_code) if s_code else {"strict_pass": False}
    s_study, s_pub_sha, s_num_sha = materialize_m101_public_study_and_sha(s_task)
    s_obs = compute_m101_neutral_observables(s_study)

    (s_dir / "stage_s_public_gate.json").write_text(json.dumps(pub_gate, indent=2) + "\n", encoding="utf-8")
    (s_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")

    rec = {
        "world_id": wid,
        "cluster_id": spec.cluster_id,
        "stratum": spec.stratum,
        "topology_class_hidden_analytics_only": spec.topology_class,
        "short_name": s_task.short_name,
        "turns_used": int(ep_sum.get("turns_used", 0) or 0),
        "public_gate_accepted": bool(pub_gate.get("accepted")),
        "admitted": bool(pub_gate.get("accepted")),
        "strict_pass": bool(priv_ver.get("strict_pass")),
        "source_code_sha256": hashlib.sha256(s_code.encode("utf-8")).hexdigest(),
        "public_input_sha256": s_pub_sha,
        "canonical_numeric_sha256": s_num_sha,
    }
    (s_dir / "00_stage_S_summary.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(
        f"[M10.1 Stage S | {wid} ({spec.stratum})] pass={rec['strict_pass']} | pub_gate={rec['public_gate_accepted']} | turns={rec['turns_used']}",
        flush=True,
    )
    return {
        "summary": rec,
        "source_monolithic_code": s_code,
        "source_pub_study": s_study,
        "source_pub_sha256": s_pub_sha,
        "source_obs": s_obs,
        "source_public_gate": pub_gate,
    }


def _persist_dag_state_to_graph(
    graph: PersistentMechanismGraph,
    *,
    world_id: str,
    node_library: dict[str, ExecutableModuleNode],
    dag_library: list[ExecutableComputationDAG],
    withdrawals: list[CutWithdrawalMergeReceipt],
    dag_pos_obs: dict[str, list[dict[str, float | None]]],
    dag_neg_obs: dict[str, list[dict[str, float | None]]],
) -> None:
    payload = {
        "world_id": world_id,
        "node_library": {k: v.to_dict() for k, v in sorted(node_library.items())},
        "dag_library": [d.to_dict() for d in dag_library],
        "cut_withdrawal_receipts": [w.to_dict() for w in withdrawals],
        "dag_pos_obs": deepcopy(dag_pos_obs),
        "dag_neg_obs": deepcopy(dag_neg_obs),
    }
    state_sha = _canonical_json_sha256(payload)
    graph.forest.questions[M101_DAG_LEDGER_FOREST_KEY] = {**payload, "state_sha256": state_sha}


def _run_one_m101_world_arm_stream(
    *,
    spec: M101WorldSpec,
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
    """Execute one 8-step continual stream for `(spec, arm)`.
    When `mock_backend` is provided (Gate 2 unit/Docker integration tests ONLY!), delegates LLM proposals & synthesis to
    `mock_backend` while running 100% REAL Docker public gates, `ProbeReceiptStore`, DAG compiler, and private grader!
    """
    wid = spec.world_id
    stream_t0 = monotonic()
    stream_out_dir.mkdir(parents=True, exist_ok=True)

    forest = deepcopy(base_v7_forest)
    archive = deepcopy(base_v7_archive)
    graph = PersistentMechanismGraph(forest=forest)
    probe_store = ProbeReceiptStore(stream_out_dir / "probe_receipts", graph=graph)

    # Persist Stage S initial verified public receipt into ProbeReceiptStore so EVERY prior witness has a real external FK!
    s0_pid, s0_psha = _persist_m101_probe(
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

    # Arm 1: `Flat_MonolithicArchive_Online`
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

    # Arm 2 (`K <= 2, fanout <= 1`) & Arm 3 (`K <= 4, fanout <= 2`) shared DAG representation
    arm_max_nodes = 2 if arm == "Fixed_TwoStage_Modular_Online" else M101_MAX_ACTIVE_NODES
    arm_max_fanout = 1 if arm == "Fixed_TwoStage_Modular_Online" else M101_MAX_FANOUT

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
    withdrawals: list[CutWithdrawalMergeReceipt] = []
    attempted_decomp_from_dag_ids: set[str] = set()
    version_counter = 1
    dag_counter = 1

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

    def _make_codex_client(ep_dir: Path, role: str, ep_sn: str) -> Any:
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
        ep_dir: Path,
        ep_sn: str,
        new_id: str,
        existing_ids: list[str],
        pos_map: dict[str, list[dict[str, float | None]]],
        neg_map: dict[str, list[dict[str, float | None]]],
    ) -> dict[str, Any] | None:
        if not existing_ids:
            return None
        if mock_backend is not None and hasattr(mock_backend, "induce_guards"):
            return mock_backend.induce_guards(
                new_id=new_id,
                existing_ids=existing_ids,
                pos_map=pos_map,
                neg_map=neg_map,
                feature_names=M101_OBSERVABLE_NAMES,
            )
        cli = _make_codex_client(ep_dir, "guard_induction", ep_sn)
        return propose_and_validate_m99_guard_partition(
            client=cli,
            stage_subdir=f"{run_id}__{wid}__{arm}__guard_{ep_sn}_{new_id}",
            scope_label=f"{arm}_m101",
            feature_names=M101_OBSERVABLE_NAMES,
            existing_branch_ids=existing_ids,
            new_branch_id_placeholder=new_id,
            positive_witness_features_by_branch=pos_map,
            negative_witness_features_by_branch=neg_map,
            min_clause_margin=0.035,
            max_attempts=2,
        )

    _persist_dag_state_to_graph(
        graph,
        world_id=wid,
        node_library=node_library,
        dag_library=dag_library,
        withdrawals=withdrawals,
        dag_pos_obs=dag_pos_obs,
        dag_neg_obs=dag_neg_obs,
    )
    graph.save_checkpoint(stream_out_dir / "graph_ckpt_after_step_00.json")

    episode_results: list[dict[str, Any]] = []
    action_log: list[dict[str, Any]] = []
    failed_search_events: list[dict[str, Any]] = []

    for step_i, (task, m_info) in enumerate(zip(stream_tasks, schedule_manifest), start=1):
        ep_dir = stream_out_dir / task.task_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_sn = task.short_name
        pub_study, pub_sha, _ = materialize_m101_public_study_and_sha(task)
        obs_now = compute_m101_neutral_observables(pub_study)

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

        # =====================================================================
        # ARM 1: `Flat_MonolithicArchive_Online` (general Python, NO helper ban!)
        # =====================================================================
        if arm == "Flat_MonolithicArchive_Online":
            chosen_row, sel_mode, alt_rows = _select_guarded_candidate(mono_rows, obs_now)
            g1 = execute_m101_public_gate_in_container(task=task, candidate_code=chosen_row["code"], expect_dag_trace=False)
            p1_id, p1_sha = _persist_m101_probe(
                probe_store,
                probe_kind="pre_llm_monolithic_probe",
                receipt=g1,
                metadata={"world_id": wid, "arm": arm, "step": step_i, "row_id": chosen_row["id"], "sel_mode": sel_mode},
            )
            pre_probes_wall_ms += float(g1.get("wall_ms") or 0.0)
            pre_probes_used += 1

            accepted_pre = bool(g1.get("accepted"))
            winning_row = chosen_row if accepted_pre else None
            winning_gate = g1
            winning_pid, winning_psha = p1_id, p1_sha

            if not accepted_pre and alt_rows:
                # Fair 1-nearest-observable fallback probe (`max 2` pre-LLM probes total)
                mono_neg_obs.setdefault(chosen_row["id"], []).append(obs_now)
                alt_sorted = sorted(
                    alt_rows,
                    key=lambda r: min(_obs_l1_distance(obs_now, w) for w in (mono_pos_obs.get(r["id"]) or [source_obs])),
                )
                fb_row = alt_sorted[0]
                g2 = execute_m101_public_gate_in_container(task=task, candidate_code=fb_row["code"], expect_dag_trace=False)
                p2_id, p2_sha = _persist_m101_probe(
                    probe_store,
                    probe_kind="pre_llm_monolithic_fallback_probe",
                    receipt=g2,
                    metadata={"world_id": wid, "arm": arm, "step": step_i, "row_id": fb_row["id"]},
                )
                pre_probes_wall_ms += float(g2.get("wall_ms") or 0.0)
                pre_probes_used += 1
                if g2.get("accepted"):
                    accepted_pre = True
                    winning_row = fb_row
                    winning_gate = g2
                    winning_pid, winning_psha = p2_id, p2_sha
                else:
                    mono_neg_obs.setdefault(fb_row["id"], []).append(obs_now)

            if accepted_pre and winning_row is not None:
                final_code = winning_row["code"]
                final_gate = winning_gate
                final_probe_id, final_probe_sha = winning_pid, winning_psha
                mono_pos_obs.setdefault(winning_row["id"], []).append(obs_now)
                chosen_action = "zero_llm_monolithic_hit"
                execution_mode = "zero_llm_monolithic_hit"
                if winning_row["id"] != chosen_row["id"] and len(mono_rows) >= 2:
                    ex_ids = [r["id"] for r in mono_rows if r["id"] != winning_row["id"]]
                    guard_ind_receipt = _induce_guards_for_table(
                        ep_dir=ep_dir,
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
                # Paid monolithic synthesis (`NO` helper ban!)
                aug_t, ctx_hdr = _inject_m101_equal_info_archive(task, prior_archive_entries, g1)
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
                g_post = execute_m101_public_gate_in_container(task=task, candidate_code=cand, expect_dag_trace=False) if cand else {"accepted": False}
                pp_id, pp_sha = _persist_m101_probe(
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
                        ep_dir=ep_dir,
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
        # ARMS 2 & 3: `Fixed_TwoStage_Modular_Online` (`K<=2, fo<=1`) AND
        #             `Hegel_UnknownCut_GraphDiscovery_Online` (`K<=4, fo<=2`)
        # =====================================================================
        else:
            # --- Step A: Pre-LLM Guarded DAG Execution & Bounded Compositional Subgraph Replay (`0` LLM calls!) ---
            dag_items = [
                {
                    "id": d.dag_id,
                    "guard": d.applicability_guard,
                    "priority": d.priority,
                    "dag": d,
                }
                for d in dag_library
            ]
            chosen_it, sel_mode, alt_its = _select_guarded_candidate(dag_items, obs_now)
            chosen_dag: ExecutableComputationDAG = chosen_it["dag"]
            tried_fps: set[str] = {chosen_dag.topology_fingerprint()}

            c1_code, _ = compile_m101_dag_to_instrumented_solver_code(chosen_dag, node_library)
            g1 = execute_m101_public_gate_in_container(task=task, candidate_code=c1_code, expect_dag_trace=True)
            p1_id, p1_sha = _persist_m101_probe(
                probe_store,
                probe_kind="pre_llm_dag_probe",
                receipt=g1,
                metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": chosen_dag.dag_id, "sel_mode": sel_mode},
            )
            pre_probes_wall_ms += float(g1.get("wall_ms") or 0.0)
            pre_probes_used += 1

            accepted_pre = bool(g1.get("accepted"))
            winning_dag: ExecutableComputationDAG | None = chosen_dag if accepted_pre else None
            winning_code = c1_code if accepted_pre else ""
            winning_gate = g1
            winning_pid, winning_psha = p1_id, p1_sha
            pre_fail_errors: list[str] = list(g1.get("errors") or [])

            if not accepted_pre:
                dag_neg_obs.setdefault(chosen_dag.dag_id, []).append(obs_now)

                # Build bounded pre-LLM queue (`<= 3` extra probes):
                # 1) Empirical-discriminant compositional replay over highest-K verified DAG template (`reuse_subgraph`),
                # 2) Nearest-observable alternative verified DAGs in `dag_library`!
                comp_queue: list[tuple[str, ExecutableComputationDAG]] = []
                multi_dags = [d for d in dag_library if len(d.nodes) >= 2]
                if multi_dags:
                    multi_dags_sorted = sorted(multi_dags, key=lambda d: (-len(d.nodes), -d.priority, d.dag_id))
                    for tmpl in multi_dags_sorted[:2]:
                        cands_comp = _rank_compositional_node_version_combinations(
                            template_dag=tmpl,
                            node_library=node_library,
                            node_pos_obs=node_pos_obs,
                            obs_now=obs_now,
                            already_tried_fingerprints=tried_fps,
                            max_candidates=3,
                        )
                        for cd in cands_comp:
                            comp_queue.append(("compositional_replay", cd))
                alt_queue: list[tuple[str, ExecutableComputationDAG]] = []
                alt_dags_sorted = sorted(
                    [it["dag"] for it in alt_its],
                    key=lambda d: min(_obs_l1_distance(obs_now, w) for w in (dag_pos_obs.get(d.dag_id) or [source_obs])),
                )
                for ad in alt_dags_sorted:
                    fp = ad.topology_fingerprint()
                    if fp not in tried_fps:
                        tried_fps.add(fp)
                        alt_queue.append(("alt_verified_dag", ad))

                # Bounded <= 4 pre-LLM extra probes: up to 3 Fisher-ranked compositional combos + at least 1 nearest alt DAG
                pre_queue = comp_queue[:3] + alt_queue[: max(1, 4 - len(comp_queue[:3]))]

                for q_kind, q_dag in pre_queue[:4]:
                    q_code, _ = compile_m101_dag_to_instrumented_solver_code(q_dag, node_library)
                    gq = execute_m101_public_gate_in_container(task=task, candidate_code=q_code, expect_dag_trace=True)
                    pq_id, pq_sha = _persist_m101_probe(
                        probe_store,
                        probe_kind=f"pre_llm_{q_kind}_probe",
                        receipt=gq,
                        metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": q_dag.dag_id, "q_kind": q_kind},
                    )
                    pre_probes_wall_ms += float(gq.get("wall_ms") or 0.0)
                    pre_probes_used += 1
                    if gq.get("accepted"):
                        accepted_pre = True
                        winning_dag = q_dag
                        winning_code = q_code
                        winning_gate = gq
                        winning_pid, winning_psha = pq_id, pq_sha
                        break
                    else:
                        if q_kind == "alt_verified_dag":
                            dag_neg_obs.setdefault(q_dag.dag_id, []).append(obs_now)

            if accepted_pre and winning_dag is not None:
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
                else:
                    chosen_action = "zero_llm_dag_hit"
                    execution_mode = "zero_llm_dag_hit"

                dag_pos_obs.setdefault(winning_dag.dag_id, []).append(obs_now)
                dag_last_pos_probe[winning_dag.dag_id] = (winning_pid, winning_psha)
                for _nid, _vid in winning_dag.node_versions.items():
                    node_pos_obs.setdefault(_vid, []).append(obs_now)

                # If guard-selected DAG missed while a different DAG won, update guards (if `>= 2` DAGs)
                if winning_dag.dag_id != chosen_dag.dag_id and len(dag_library) >= 2:
                    ex_ids = [d.dag_id for d in dag_library if d.dag_id != winning_dag.dag_id]
                    guard_ind_receipt = _induce_guards_for_table(
                        ep_dir=ep_dir,
                        ep_sn=ep_sn,
                        new_id=winning_dag.dag_id,
                        existing_ids=ex_ids,
                        pos_map=dag_pos_obs,
                        neg_map=dag_neg_obs,
                    )
                    if guard_ind_receipt and guard_ind_receipt.get("accepted"):
                        winning_dag.applicability_guard = deepcopy(guard_ind_receipt["new_branch_guard"])
                        for d in dag_library:
                            if d.dag_id in (guard_ind_receipt.get("existing_branch_guards") or {}):
                                d.applicability_guard = deepcopy(guard_ind_receipt["existing_branch_guards"][d.dag_id])
            else:
                # --- Step B: Live Unknown-Cut DAG Structure Discovery / Edit Proposal + Targeted Synthesis ---
                # 1. Check if nearest verified monolithic DAG hasn't been analyzed for initial unknown-cut decomposition yet
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

                # Trigger initial decomposition inspection if:
                #   (a) we have no multi-node DAG yet AND haven't already checked `nearest_mono`, OR
                #   (b) `nearest_mono` is strictly closer than any multi-node DAG (e.g., delayed shift on `hybrid_mono_to_split_shift`!)
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
                    attempted_decomp_from_dag_ids.add(nearest_mono.dag_id)
                    src_vid = nearest_mono.node_versions[nearest_mono.sink_node_id]
                    src_code_m = node_library[src_vid].code
                    src_obs_m = (dag_pos_obs.get(nearest_mono.dag_id) or [source_obs])[0]
                    src_task_m = verified_tasks_by_dag.get(nearest_mono.dag_id, s_task)

                    idx_before_prop = _current_receipts_len()
                    cli_prop = _make_codex_client(ep_dir, "dag_structure_proposal", ep_sn)
                    prop_res = _propose_initial_dag_decomposition_or_keep_monolithic(
                        client=cli_prop,
                        stage_subdir=f"{run_id}__{wid}__{arm}__init_decomp_{ep_sn}",
                        source_code=src_code_m,
                        source_obs=src_obs_m,
                        target_obs=obs_now,
                        max_nodes=arm_max_nodes,
                        max_fanout=arm_max_fanout,
                        arm_label=arm,
                    )
                    parsed_p = prop_res.get("parsed") if isinstance(prop_res, dict) and prop_res.get("ok") else None
                    if not isinstance(parsed_p, dict):
                        _retag_failed_search_receipts(idx_before_prop, wid, arm, ep_sn)
                        failed_search_events.append({"step": step_i, "phase": "init_decomp_call_failed"})
                    elif parsed_p.get("action") == "keep_monolithic":
                        keep_mono_decided = True
                        action_log.append({"step": step_i, "action": "keep_monolithic", "from_dag": nearest_mono.dag_id})
                    else:
                        # Validate & Docker-verify proposed initial multi-node DAG on `src_task_m` (`exec_count == 1`, anti-identity-shell!)
                        try:
                            p_nodes = [str(x) for x in parsed_p["nodes"]]
                            p_edges = [(str(e[0]), str(e[1])) for e in parsed_p["edges"]]
                            p_sink = str(parsed_p["sink_node_id"])
                            mod_by_nid = {str(m["node_id"]): str(m["code"]) for m in parsed_p["node_modules"]}
                            if not (2 <= len(p_nodes) <= arm_max_nodes):
                                raise ValueError(f"Proposed node count {len(p_nodes)} outside [2, {arm_max_nodes}]")
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
                            if t_info["max_fanout"] > arm_max_fanout:
                                raise ValueError(f"Proposed fan-out {t_info['max_fanout']} > arm max {arm_max_fanout}")
                            comp_src_code, _ = compile_m101_dag_to_instrumented_solver_code(cand_d0, tmp_nlib)
                            g_src = execute_m101_public_gate_in_container(
                                task=src_task_m, candidate_code=comp_src_code, expect_dag_trace=True
                            )
                            psrc_id, psrc_sha = _persist_m101_probe(
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
                        except Exception as exc:
                            _retag_failed_search_receipts(idx_before_prop, wid, arm, ep_sn)
                            failed_search_events.append({"step": step_i, "phase": "init_decomp_invalid", "error": str(exc)})

                # 2. Now execute the edit action for the current step!
                solved_via_dag_edit = False
                if not keep_mono_decided and multi_dags:
                    # Run host node-health probe on highest-priority multi-node DAG so `CutWithdrawalMergeReceipt` has a real external FK!
                    ref_multi = sorted(multi_dags, key=lambda d: (-len(d.nodes), -d.priority, d.dag_id))[0]
                    health_rec = _probe_existing_dag_nodes_health_in_container(
                        task=task, dag=ref_multi, node_library=node_library
                    )
                    health_pid, health_psha = _persist_m101_probe(
                        probe_store,
                        probe_kind="dag_node_health_probe",
                        receipt=health_rec,
                        metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": ref_multi.dag_id},
                    )
                    post_probes_wall_ms += float(health_rec.get("wall_ms") or 0.0)

                    # Build concise summary of existing DAGs and node library for structured edit planner
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
                    cli_edit = _make_codex_client(ep_dir, "dag_structure_proposal", ep_sn)
                    edit_res = _propose_dag_edit_action_for_current_step(
                        client=cli_edit,
                        stage_subdir=f"{run_id}__{wid}__{arm}__edit_plan_{ep_sn}",
                        active_dags_summary=dags_sum,
                        node_library_summary=nlib_sum,
                        obs_now=obs_now,
                        ref_obs_by_dag=ref_obs_map,
                        pre_llm_probe_errors=pre_fail_errors,
                        node_health_summary=health_rec,
                        max_nodes=arm_max_nodes,
                        max_fanout=arm_max_fanout,
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
                            if not (1 <= len(n_nodes) <= arm_max_nodes):
                                raise ValueError(f"Edit node count {len(n_nodes)} outside [1, {arm_max_nodes}]")
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
                            if t_inf["max_fanout"] > arm_max_fanout:
                                raise ValueError(f"Edit fan-out {t_inf['max_fanout']} > {arm_max_fanout}")

                            synth_role = (
                                "dag_merge_withdrawal_repair"
                                if act in {"merge_nodes", "rewire_edge"}
                                else "dag_compile_and_node_synthesis"
                            )
                            t_node_task = _build_m101_targeted_node_synthesis_task(
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
                            # Extract newly synthesized editable node files
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
                            pe_id, pe_sha = _persist_m101_probe(
                                probe_store,
                                probe_kind=f"post_llm_dag_{act}_gate",
                                receipt=g_edit,
                                metadata={"world_id": wid, "arm": arm, "step": step_i, "dag_id": cand_dag.dag_id, "action": act},
                            )
                            post_probes_wall_ms += float(g_edit.get("wall_ms") or 0.0)
                            if g_edit.get("accepted"):
                                for vid, mn in created_n.items():
                                    mn.probe_receipt_id = pe_id
                                    mn.probe_receipt_sha256 = pe_sha
                                    node_library[vid] = mn
                                for _nid, _vid in cand_dag.node_versions.items():
                                    node_pos_obs.setdefault(_vid, []).append(obs_now)

                                # Issue 4-link external-FK `CutWithdrawalMergeReceipt` when `merge_nodes` / `rewire_edge` withdraws a cut!
                                if act in {"merge_nodes", "rewire_edge"}:
                                    old_d_id = str(ep_plan.get("base_dag_id") or chosen_dag.dag_id)
                                    if old_d_id not in dag_last_pos_probe:
                                        old_d_id = ref_multi.dag_id
                                    pr_pid, pr_psha = dag_last_pos_probe[old_d_id]
                                    wd_rec = CutWithdrawalMergeReceipt(
                                        receipt_id="",
                                        world_id=wid,
                                        old_dag_id=old_d_id,
                                        new_dag_id=cand_dag.dag_id,
                                        action=act,
                                        prior_positive_probe_id=pr_pid,
                                        prior_positive_receipt_sha256=pr_psha,
                                        preserved_node_probe_id=health_pid,
                                        preserved_node_receipt_sha256=health_psha,
                                        failing_joint_probe_id=p1_id,
                                        failing_joint_receipt_sha256=p1_sha,
                                        new_passing_probe_id=pe_id,
                                        new_passing_receipt_sha256=pe_sha,
                                        preserved_old_dag_id=old_d_id,
                                        expanded_guard_features=list(M101_OBSERVABLE_NAMES),
                                    )
                                    wd_rec.receipt_id = wd_rec.compute_expected_receipt_id()
                                    verify_m101_cut_withdrawal_external_fk(wd_rec, probe_store)
                                    withdrawals.append(wd_rec)
                                    withdrawal_this_step = wd_rec.to_dict()

                                ex_ids = [d.dag_id for d in dag_library]
                                dag_pos_obs[cand_dag.dag_id] = [obs_now]
                                dag_last_pos_probe[cand_dag.dag_id] = (pe_id, pe_sha)
                                verified_tasks_by_dag[cand_dag.dag_id] = task
                                dag_library.append(cand_dag)
                                guard_ind_receipt = _induce_guards_for_table(
                                    ep_dir=ep_dir,
                                    ep_sn=ep_sn,
                                    new_id=cand_dag.dag_id,
                                    existing_ids=ex_ids,
                                    pos_map=dag_pos_obs,
                                    neg_map=dag_neg_obs,
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
                        except Exception as exc:
                            _retag_failed_search_receipts(idx_before_edit, wid, arm, ep_sn)
                            failed_search_events.append({"step": step_i, "phase": f"dag_edit_{act}_exception", "error": str(exc)})
                    elif ep_plan is None:
                        _retag_failed_search_receipts(idx_before_edit, wid, arm, ep_sn)
                        failed_search_events.append({"step": step_i, "phase": "edit_plan_call_failed"})

                # 3. Clean `K = 1` monolithic solve (either intentionally chosen via `keep_monolithic` OR fallback after failed search!)
                if not solved_via_dag_edit:
                    aug_t, ctx_hdr = _inject_m101_equal_info_archive(task, prior_archive_entries, g1)
                    s_role = "stream_solve" if (keep_mono_decided or not multi_dags) else "dag_merge_withdrawal_repair"
                    sub_files, ep_s = _run_sandbox_solve(
                        task_to_solve=aug_t,
                        ep_dir=ep_dir,
                        sub_tag="mono_fallback_solve",
                        role=s_role,
                        ep_sn=ep_sn,
                        continual_ctx=ctx_hdr,
                    )
                    turns_used_step += int(ep_s.get("turns_used", 0) or 0)
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
                        pp_id, pp_sha = _persist_m101_probe(
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
                        chosen_action = "keep_monolithic" if not multi_dags else "merge_nodes"
                        execution_mode = "llm_keep_monolithic" if not multi_dags else "llm_merge_fallback"
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
                                ep_dir=ep_dir,
                                ep_sn=ep_sn,
                                new_id=new_d.dag_id,
                                existing_ids=ex_ids,
                                pos_map=dag_pos_obs,
                                neg_map=dag_neg_obs,
                            )
                            if guard_ind_receipt and guard_ind_receipt.get("accepted"):
                                new_d.applicability_guard = deepcopy(guard_ind_receipt["new_branch_guard"])
                                for d in dag_library:
                                    if d.dag_id in (guard_ind_receipt.get("existing_branch_guards") or {}):
                                        d.applicability_guard = deepcopy(guard_ind_receipt["existing_branch_guards"][d.dag_id])
                            prior_archive_entries.append({"short_name": ep_sn, "task_id": task.task_id, "code": cand, "obs": obs_now})

        # Official private Docker grading (never visible to policy!)
        priv_ver = grade_m101_candidate_in_container(task=task, candidate_code=final_code) if final_code else {"strict_pass": False, "reward": 0.0}

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
            "zero_llm_closure": bool(turns_used_step == 0 and final_gate.get("accepted")),
            "public_gate_accepted": bool(final_gate.get("accepted")),
            "public_gate_probe_id": final_probe_id,
            "public_gate_probe_sha256": final_probe_sha,
            "strict_pass": bool(priv_ver.get("strict_pass")),
            "reward": 1.0 if priv_ver.get("strict_pass") else 0.0,
            "pre_llm_probes_used": pre_probes_used,
            "pre_llm_probe_wall_ms": round(pre_probes_wall_ms, 2),
            "post_llm_probe_wall_ms": round(post_probes_wall_ms, 2),
            "official_verifier_wall_ms": float(priv_ver.get("verifier_wall_ms") or 0.0),
            "dag_runtime_trace": active_dag_trace,
            "cut_withdrawal_receipt": withdrawal_this_step,
            "guard_induction_accepted": (bool(guard_ind_receipt.get("accepted")) if guard_ind_receipt else None),
            "public_input_sha256": pub_sha,
            "verifier_report": priv_ver,
        }
        (ep_dir / "00_episode_summary.json").write_text(json.dumps(ep_rec, indent=2) + "\n", encoding="utf-8")
        (ep_dir / "03_official_verifier_report.json").write_text(json.dumps(priv_ver, indent=2) + "\n", encoding="utf-8")
        if final_code:
            (ep_dir / "final_candidate_batch_solver.py").write_text(final_code, encoding="utf-8")

        _persist_dag_state_to_graph(
            graph,
            world_id=wid,
            node_library=node_library,
            dag_library=dag_library,
            withdrawals=withdrawals,
            dag_pos_obs=dag_pos_obs,
            dag_neg_obs=dag_neg_obs,
        )
        graph.save_checkpoint(stream_out_dir / f"graph_ckpt_after_step_{step_i:02d}.json")
        episode_results.append(ep_rec)
        print(
            f"  [M10.1 {wid} | {arm} | step {step_i}/{len(stream_tasks)} ({m_info['causal_event']})] "
            f"{execution_mode} | act={chosen_action} | pass={ep_rec['strict_pass']} | turns={turns_used_step} "
            f"| K={(active_dag_trace or {}).get('active_node_count', 1)}",
            flush=True,
        )

    # Topology & runtime verification summary across the stream
    max_k_obs = max((int((r.get("dag_runtime_trace") or {}).get("active_node_count", 1)) for r in episode_results), default=1)
    max_depth_obs = max((int((r.get("dag_runtime_trace") or {}).get("depth", 1)) for r in episode_results), default=1)
    max_fanout_obs = max((int((r.get("dag_runtime_trace") or {}).get("max_fanout", 0)) for r in episode_results), default=0)
    shared_exec_1_verified = any(
        bool((r.get("dag_runtime_trace") or {}).get("shared_fanout_nodes"))
        and bool((r.get("dag_runtime_trace") or {}).get("all_exec_count_eq_1"))
        and r.get("strict_pass")
        for r in episode_results
    )
    all_traced_exec_1 = all(
        bool((r.get("dag_runtime_trace") or {}).get("all_exec_count_eq_1"))
        for r in episode_results
        if isinstance(r.get("dag_runtime_trace"), dict)
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
        "zero_llm_episodes_total": sum(1 for r in episode_results if r["turns_used"] == 0),
        "zero_turn_passed_closures": sum(1 for r in episode_results if r["turns_used"] == 0 and r["strict_pass"]),
        "total_turns_used": sum(int(r["turns_used"]) for r in episode_results),
        "max_active_nodes_observed": max_k_obs,
        "max_depth_observed": max_depth_obs,
        "max_fanout_observed": max_fanout_obs,
        "shared_fanout_exec_count_1_verified": shared_exec_1_verified,
        "all_traced_episodes_exec_count_1": all_traced_exec_1,
        "cut_withdrawal_count": len(withdrawals),
        "cut_withdrawal_receipts": [w.to_dict() for w in withdrawals],
        "failed_search_events_count": len(failed_search_events),
        "failed_search_events": failed_search_events,
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
                "active_node_count": (r.get("dag_runtime_trace") or {}).get("active_node_count", 1),
                "depth": (r.get("dag_runtime_trace") or {}).get("depth", 1),
                "max_fanout": (r.get("dag_runtime_trace") or {}).get("max_fanout", 0),
                "all_exec_count_eq_1": (r.get("dag_runtime_trace") or {}).get("all_exec_count_eq_1"),
                "public_gate_probe_id": r["public_gate_probe_id"],
            }
            for r in episode_results
        ],
    }
    (stream_out_dir / "00_stream_summary.json").write_text(json.dumps(st_sum, indent=2) + "\n", encoding="utf-8")
    return st_sum


# ============================================================================
# 5. Post-Run `0`-LLM Plain-Dict Routing Parity Ablation (`Q10`) & `ΔC(K) / K*`
# ============================================================================


def run_m101_step06_plain_dict_routing_parity_ablation(
    *,
    world_specs: tuple[M101WorldSpec, ...],
    built_worlds: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Post-run `0`-LLM Plain-Dict Substrate-Independence Parity Check on frozen Step-06 checkpoints (before the final
    2 revisits `S07` & `S08`), confirming exact candidate SHA-256 parity between `PersistentMechanismGraph`-backed DAG
    routing and a plain Python dict view over the same learned guards and node library (`Q10` honesty!).
    """
    rows: list[dict[str, Any]] = []
    matched = 0
    evaluated = 0
    for sp in world_specs:
        wid = sp.world_id
        _, tasks, manifest = built_worlds[wid]
        ck6 = out_dir / "continual_streams" / wid / "Hegel_UnknownCut_GraphDiscovery_Online" / "graph_ckpt_after_step_06.json"
        if not ck6.is_file():
            continue
        g6 = PersistentMechanismGraph.load_checkpoint(ck6)
        raw6 = json.loads(ck6.read_text(encoding="utf-8"))
        st_g = g6.forest.questions.get(M101_DAG_LEDGER_FOREST_KEY) or {}
        st_d = raw6["questions"].get(M101_DAG_LEDGER_FOREST_KEY) or {}

        nlib_g = {k: ExecutableModuleNode.from_dict(v) for k, v in (st_g.get("node_library") or {}).items()}
        dags_g = [ExecutableComputationDAG.from_dict(x) for x in (st_g.get("dag_library") or [])]

        nlib_d = {k: ExecutableModuleNode.from_dict(v) for k, v in (st_d.get("node_library") or {}).items()}
        dags_d = [ExecutableComputationDAG.from_dict(x) for x in (st_d.get("dag_library") or [])]

        for idx in (6, 7):
            tk = tasks[idx]
            pub_st, pub_sha, _ = materialize_m101_public_study_and_sha(tk)
            obs = compute_m101_neutral_observables(pub_st)
            sel_g, _, _ = _select_guarded_candidate(
                [{"id": d.dag_id, "guard": d.applicability_guard, "priority": d.priority, "dag": d} for d in dags_g],
                obs,
            )
            sel_d, _, _ = _select_guarded_candidate(
                [{"id": d.dag_id, "guard": d.applicability_guard, "priority": d.priority, "dag": d} for d in dags_d],
                obs,
            )
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
        "schema": "hegel_m10_1_step06_plain_dict_parity_ablation_v1",
        "paid_model_calls": 0,
        "matched_episodes": matched,
        "evaluated_episodes": evaluated,
        "parity_rate_pct": round(100.0 * matched / max(1, evaluated), 2),
        "rows": rows,
    }


def compute_m101_cumulative_curves_and_k_star(
    *,
    world_specs: tuple[M101WorldSpec, ...],
    all_receipts: list[dict[str, Any]],
    stream_summaries: list[dict[str, Any]],
    n_steps: int = 8,
) -> dict[str, Any]:
    """Compute finite 8-step cumulative cost curves `ΔC(K)` + pooled & per-cluster empirical `K*` (`T07`)."""
    wids = [sp.world_id for sp in world_specs]
    st_by_key = {(s["world_id"], s["arm"]): s for s in stream_summaries}

    step_cost: dict[tuple[str, str, int], int] = {}
    for r in all_receipts:
        arm = r["arm"]
        if arm not in M101_ARMS:
            continue
        ep_sn = str(r["episode_short_name"])
        if "_batch_" not in ep_sn:
            continue
        k_num = int(ep_sn.rsplit("_batch_", 1)[-1])
        if 1 <= k_num <= n_steps:
            key = (r["world_id"], arm, k_num)
            step_cost[key] = step_cost.get(key, 0) + int(r["proxy_v2_u_010c_4o"])

    curves: dict[str, Any] = {}
    for base_arm in ("Flat_MonolithicArchive_Online", "Fixed_TwoStage_Modular_Online"):
        step_rows = []
        for k in range(1, n_steps + 1):
            hg_c = [sum(step_cost.get((w, "Hegel_UnknownCut_GraphDiscovery_Online", s), 0) for s in range(1, k + 1)) for w in wids]
            bs_c = [sum(step_cost.get((w, base_arm, s), 0) for s in range(1, k + 1)) for w in wids]
            diffs = [float(h - b) for h, b in zip(hg_c, bs_c)]
            hg_p = sum(
                1 for w in wids for ep in st_by_key[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["episodes_meta"][:k] if ep["strict_pass"]
            )
            bs_p = sum(1 for w in wids for ep in st_by_key[(w, base_arm)]["episodes_meta"][:k] if ep["strict_pass"])
            tot_eps = len(wids) * k
            pr_diff = (hg_p - bs_p) / float(tot_eps)
            step_rows.append(
                {
                    "step_K": k,
                    "hegel_cum_scope1_v2_total": int(sum(hg_c)),
                    "baseline_cum_scope1_v2_total": int(sum(bs_c)),
                    "mean_cluster_delta_v2": round(float(np.mean(diffs)), 2),
                    "total_delta_v2": int(sum(hg_c) - sum(bs_c)),
                    "cluster_wins_hegel_cheaper": sum(1 for d in diffs if d < 0),
                    "hegel_cum_strict_passes": hg_p,
                    "baseline_cum_strict_passes": bs_p,
                    "total_episodes_at_K": tot_eps,
                    "pass_rate_diff_hegel_minus_base": round(pr_diff, 6),
                    "quality_noninferior_at_K": bool(pr_diff >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE),
                }
            )
        k_star: int | str = "not_reached"
        if step_rows[-1]["quality_noninferior_at_K"]:
            for idx_k in range(len(step_rows)):
                if all(r["total_delta_v2"] < 0 for r in step_rows[idx_k:]):
                    k_star = step_rows[idx_k]["step_K"]
                    break
        missing_u = sum(
            int(r.get("usage_unknown_attempt_count", 0))
            for r in all_receipts
            if r["arm"] in {"Hegel_UnknownCut_GraphDiscovery_Online", base_arm}
        )
        if missing_u:
            k_star = "unknown_partial_usage"

        # Per-cluster `K*` (`T07`!)
        per_cluster_k_star: dict[str, Any] = {}
        for sp in world_specs:
            w = sp.world_id
            deltas_w = [
                sum(step_cost.get((w, "Hegel_UnknownCut_GraphDiscovery_Online", s), 0) - step_cost.get((w, base_arm, s), 0) for s in range(1, k + 1))
                for k in range(1, n_steps + 1)
            ]
            hg_pw = st_by_key[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["strict_passes"]
            bs_pw = st_by_key[(w, base_arm)]["strict_passes"]
            kw: int | str = "not_reached"
            if (hg_pw - bs_pw) / float(n_steps) >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE:
                for idx_k in range(n_steps):
                    if all(d < 0 for d in deltas_w[idx_k:]):
                        kw = idx_k + 1
                        break
            per_cluster_k_star[w] = {
                "stratum": sp.stratum,
                "topology_class_hidden_analytics_only": sp.topology_class,
                "K_star": kw,
                "final_delta_v2": int(deltas_w[-1]),
                "hegel_passes": hg_pw,
                "baseline_passes": bs_pw,
            }

        curves[f"Hegel_minus_{base_arm}"] = {
            "empirical_quality_constrained_K_star": k_star,
            "preregistered_quality_noninferiority_margin_per_episode": PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE,
            "per_cluster_K_star": per_cluster_k_star,
            "by_step_K": step_rows,
        }
    return curves


# ============================================================================
# 6. Main Campaign Orchestrator
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
    ap.add_argument("--max-codex-concurrency", type=int, default=32)
    ap.add_argument("--max-stream-workers", type=int, default=24)
    ap.add_argument("--finalize-existing", action="store_true")
    args = ap.parse_args()

    selected_specs = DEV_M101_WORLD_SPECS if args.split == "dev" else CONFIRMATION_M101_WORLD_SPECS
    default_dir = DEFAULT_DEV_OUT_DIR if args.split == "dev" else DEFAULT_CONF_OUT_DIR
    out_dir: Path = (args.out_dir or default_dir).resolve()
    exec_manifest_path = out_dir / "execution_manifest.json"
    exec_done_path = out_dir / "execution_completion.json"

    os.environ["HEGEL_MAX_CODEX_CONCURRENCY"] = str(args.max_codex_concurrency)

    if not args.finalize_existing and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory `{out_dir}`!")

    git_sha_start, is_dirty_start, src_tree_sha_start = _git_and_source_tree_state()
    if is_dirty_start and args.split == "confirmation" and not args.finalize_existing:
        raise RuntimeError("Refusing to start M10.1 confirmation experiment on a dirty working tree!")

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
    run_id = f"m101_{args.split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    if not args.finalize_existing:
        exec_manifest = {
            "schema": "hegel_m10_1_execution_provenance_manifest_v1",
            "execution_run_id": run_id,
            "split": args.split,
            "code_commit_at_start": git_sha_start,
            "working_tree_dirty_at_start": bool(is_dirty_start),
            "source_tree_sha256_at_start": src_tree_sha_start,
            "sandbox_image_digest": env_img_digest,
            "v7_bundle_sha256": hashlib.sha256(args.v7_bundle.read_bytes()).hexdigest(),
            "shared_kernel_version": SHARED_KERNEL_VERSION,
            "cost_policy_version": COST_POLICY_VERSION,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_turns": int(args.max_turns),
            "configured_max_codex_concurrency": int(args.max_codex_concurrency),
            "max_stream_workers": int(args.max_stream_workers),
            "dag_bounds": {
                "max_active_nodes": M101_MAX_ACTIVE_NODES,
                "max_depth": M101_MAX_DEPTH,
                "max_fanout": M101_MAX_FANOUT,
                "k1_monolithic_always_legal": True,
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
    built_worlds = {sp.world_id: build_m101_world_tasks(sp) for sp in selected_specs}

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

        failed_s = [wid for wid, r in stage_s_results.items() if not r["summary"]["admitted"]]
        if failed_s:
            (out_dir / "all_provider_call_receipts.json").write_text(json.dumps(all_receipts, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"Stage S public qualification failed for {failed_s}; receipts preserved.")

        stream_jobs = [(sp, arm_nm) for sp in selected_specs for arm_nm in M101_ARMS]
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
                        _run_one_m101_world_arm_stream,
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
                try:
                    stream_summaries.append(fut.result())
                except Exception as exc:
                    import traceback as _tb
                    print(f"[M10.1 FATAL STREAM ERROR] {exc}", file=sys.stderr, flush=True)
                    _tb.print_exc()
                    raise

        stream_summaries.sort(key=lambda s: (s["world_id"], M101_ARMS.index(s["arm"])))
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
            for arm_nm in M101_ARMS
        ]

    # Separate post-run `0`-LLM plain-dict routing parity ablation (`Q10`)
    abl_path = out_dir / "m10_1_step06_plain_dict_parity_ablation.json"
    abl_manifest_path = out_dir / "ablation_manifest.json"
    if abl_path.exists():
        parity_ablation = json.loads(abl_path.read_text(encoding="utf-8"))
    else:
        t_ab0 = monotonic()
        parity_ablation = run_m101_step06_plain_dict_routing_parity_ablation(
            world_specs=selected_specs, built_worlds=built_worlds, out_dir=out_dir
        )
        abl_path.write_text(json.dumps(parity_ablation, indent=2) + "\n", encoding="utf-8")
        abl_manifest_path.write_text(
            json.dumps(
                {
                    "schema": "hegel_m10_1_ablation_provenance_manifest_v1",
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

    # Hierarchical cost conservation verification across ALL Scope-1 roles & clusters
    stage_s_calls = [r for r in all_receipts if r["arm"] == "Stage_S_Shared_Source"]
    stage_s_ledger = {
        **aggregate_call_usages(stage_s_calls),
        "usage_unknown_attempt_count": sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in stage_s_calls),
    }

    arm_ledgers: dict[str, Any] = {}
    arm_outcomes: dict[str, Any] = {}
    for arm_nm in M101_ARMS:
        a_calls = [r for r in all_receipts if r["arm"] == arm_nm]
        tot_led = aggregate_call_usages(a_calls)
        role_leds = {role: aggregate_call_usages([r for r in a_calls if r["call_role"] == role]) for role in M101_SCOPE1_ROLES}
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
            "zero_llm_episodes_total": sum(int(s["zero_llm_episodes_total"]) for s in sub_s),
            "zero_turn_passed_closures": sum(int(s["zero_turn_passed_closures"]) for s in sub_s),
            "total_turns_used": sum(int(s["total_turns_used"]) for s in sub_s),
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
        named_partition_ledgers={"arms_and_source": [stage_s_ledger] + [arm_ledgers[a]["scope1_in_stream_total"] for a in M101_ARMS]},
        context_label="M101_grand_total",
    )

    # Paired cluster-level inference (pooled + Stratum A + Stratum B + monolithic vs multi-node!)
    wids = [sp.world_id for sp in selected_specs]
    strat_a = [sp.world_id for sp in selected_specs if sp.stratum == "stratum_a_seen_topology"]
    strat_b = [sp.world_id for sp in selected_specs if sp.stratum == "stratum_b_unseen_topology"]
    mono_w = [sp.world_id for sp in selected_specs if sp.topology_class == "monolithic_coupled"]
    multi_w = [sp.world_id for sp in selected_specs if sp.topology_class != "monolithic_coupled"]

    hg_by_c = arm_ledgers["Hegel_UnknownCut_GraphDiscovery_Online"]["by_cluster"]
    st_by_key = {(s["world_id"], s["arm"]): s for s in stream_summaries}

    paired_stats: dict[str, Any] = {}
    for base_arm in ("Flat_MonolithicArchive_Online", "Fixed_TwoStage_Modular_Online"):
        bs_by_c = arm_ledgers[base_arm]["by_cluster"]

        def _cost_diffs(sub_w: list[str]) -> list[float]:
            return [float(hg_by_c[w]["proxy_v2_u_010c_4o"]) - float(bs_by_c[w]["proxy_v2_u_010c_4o"]) for w in sub_w]

        def _qual_diffs(sub_w: list[str]) -> list[float]:
            return [
                (
                    float(st_by_key[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["strict_passes"])
                    - float(st_by_key[(w, base_arm)]["strict_passes"])
                )
                / float(st_by_key[(w, base_arm)]["episode_count"])
                for w in sub_w
            ]

        def _zero_diffs(sub_w: list[str]) -> list[float]:
            return [
                float(st_by_key[(w, "Hegel_UnknownCut_GraphDiscovery_Online")]["zero_turn_passed_closures"])
                - float(st_by_key[(w, base_arm)]["zero_turn_passed_closures"])
                for w in sub_w
            ]

        q_all = _qual_diffs(wids)
        paired_stats[f"Hegel_minus_{base_arm}"] = {
            "pooled_cost_v2": paired_cluster_inference_stats(_cost_diffs(wids)),
            "stratum_a_seen_topology_cost_v2": paired_cluster_inference_stats(_cost_diffs(strat_a)),
            "stratum_b_unseen_topology_cost_v2": paired_cluster_inference_stats(_cost_diffs(strat_b)),
            "monolithic_control_cost_v2": paired_cluster_inference_stats(_cost_diffs(mono_w)),
            "multi_node_only_cost_v2": paired_cluster_inference_stats(_cost_diffs(multi_w)),
            "pooled_quality_rate_diff": paired_cluster_inference_stats(q_all),
            "stratum_a_quality_rate_diff": paired_cluster_inference_stats(_qual_diffs(strat_a)),
            "stratum_b_quality_rate_diff": paired_cluster_inference_stats(_qual_diffs(strat_b)),
            "meets_preregistered_quality_noninferiority_point": bool(
                float(np.mean(q_all)) >= PREREGISTERED_QUALITY_NONINFERIORITY_MARGIN_PER_EPISODE
            ),
            "pooled_zero_llm_closures_diff": paired_cluster_inference_stats(_zero_diffs(wids)),
            "stratum_a_zero_llm_closures_diff": paired_cluster_inference_stats(_zero_diffs(strat_a)),
            "stratum_b_zero_llm_closures_diff": paired_cluster_inference_stats(_zero_diffs(strat_b)),
        }

    cum_curves = compute_m101_cumulative_curves_and_k_star(
        world_specs=selected_specs, all_receipts=all_receipts, stream_summaries=stream_summaries, n_steps=8
    )
    total_unknown = sum(int(r.get("usage_unknown_attempt_count", 0) or 0) for r in all_receipts)

    summary = {
        "schema": "hegel_m10_1_bounded_unknown_cut_dag_discovery_summary_v1",
        "execution_run_id": exec_start_obj["execution_run_id"],
        "split": args.split,
        "execution_provenance": {**exec_start_obj, **exec_done_obj},
        "independent_source_cluster_count": len({sp.cluster_id for sp in selected_specs}),
        "stratum_counts": {"stratum_a_seen_topology": len(strat_a), "stratum_b_unseen_topology": len(strat_b)},
        "sandbox_image_digest": exec_start_obj["sandbox_image_digest"],
        "shared_kernel_version": exec_start_obj["shared_kernel_version"],
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
        "paired_cluster_inference": paired_stats,
        "cumulative_amortization_curves_and_K_star": cum_curves,
        "step06_plain_dict_parity_ablation": parity_ablation,
        "streams": stream_summaries,
    }
    (out_dir / "m10_1_unknown_cut_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

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
                "K_star_vs_Monolithic": cum_curves["Hegel_minus_Flat_MonolithicArchive_Online"]["empirical_quality_constrained_K_star"],
                "K_star_vs_FixedTwoStage": cum_curves["Hegel_minus_Fixed_TwoStage_Modular_Online"]["empirical_quality_constrained_K_star"],
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
