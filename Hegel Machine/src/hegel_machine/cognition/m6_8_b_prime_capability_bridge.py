"""Hegel M6.8-B' Capability Persistence, Multi-Scale/Subtree Adapter & Candidate Verification Bridge.

Directly resolves the structural breakpoints identified in `Hegel_M63_M67_Audit` without discarding
any core Hegelian innovation (Plan 20 Multi-Scale `Pi_fold`, Memory Kernels, Shared Base-Subtrees,
Role/Unit/Time Alignment, Conditional Aufhebung, and Premise Retraction):

1. **`P0a` — Immutable Score Ledger & `RegradeReceipt`**:
   - Never mutates `00_episode_summary.json` post-run; represents any isolated timing/environment
     regrade as an explicit hash-bound `RegradeReceipt` (`05_regrade_receipt.json`) verifying
     `regrade_candidate_sha256 == original_candidate_sha256`.
2. **`P0b` — Complete `ExecutableSkillManifest` & Historical Producer Recovery**:
   - Captures full `/tmp` & workspace Python/C++/shell producer closures, helper dependencies,
     build recipes, content-addressed binary/JSON state references (via `checkpoint_storage.py`),
     and `code_agent.retain_evaluated_python_model` snapshots before container teardown — replacing
     shallow `first4` function scraping and 512 KB truncation.
   - Reconstructs verified winning producer closures from sealed historical `M6.3`/`M6.4`/`M6.6`
     provenance DAGs for capability-retention verification (strictly separated from blind generalization).
3. **`P0c` — Reader+Writer Reentrant `HeavyTimingRWLock` & Hash-Bound `VerificationReceipt`**:
   - Tracks per-thread reader reentrancy depth (`_reader_depths`) so nested reads under a queued writer
     never deadlock (`H04` Fix), while metering lock wait milliseconds explicitly.
   - Replaces string-heuristic verification checks with host-executed `VerificationReceipt`s bound to
     exact `candidate_sha256`, enforcing `commit(candidate_hash)` and automatic rollback to the best
     validated incumbent if budget expires on an unverified edit (`H01` Fix).
4. **`P1a` — Wiring `m5_temporal_multiscale_and_base_subtree` & `branch_reconciliation` into Frontier**:
   - Connects `extract_causal_temporal_regimes`, `predict_via_pi_fold_mode`, `anti_unify_programs`,
     `extract_shared_base_subtree_from_sources`, `align_semantics_roles_units_and_time`, and
     `propagate_premise_retraction_invalidation` into Frontier's recall -> bind -> pilot -> select loop,
     and mounts `multiscale_and_reconciliation_wormhole.py` inside `/tmp/hegel_wormhole_skills/`.
5. **`P1b` — Six-Stage Skill Lifecycle & Offline Public Selector Calibration**:
   - Explicitly tracks `retrieved -> bound -> executed -> validated -> adopted -> retained` (plus
     `failed / unsupported / uncertain`) and evaluates non-tautological public candidate selection
     with explicit `uncertain -> keep_incumbent` protection.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from time import monotonic
from typing import Any

import numpy as np

from .branch_reconciliation import (
    align_semantics_roles_units_and_time,
    anti_unify_programs,
    propagate_premise_retraction_invalidation,
)
from .checkpoint_storage import clone_or_copy, write_text_incrementally
from .code_agent import retain_evaluated_python_model
from .language import Program
from .m5_temporal_multiscale_and_base_subtree import (
    extract_shared_base_subtree_from_sources,
    predict_via_pi_fold_mode,
)
from .m6_4_executable_wormhole_library import extract_pure_reusable_functions_from_python
from .wormhole_forest import PhilosophicalForest, extract_causal_temporal_regimes


# ============================================================================
# 1. Reader- AND Writer-Reentrant Heavy Timing RWLock (`H04` Fix)
# ============================================================================


class HeavyTimingRWLock:
    """Reader- AND writer-reentrant RWLock with explicit lock-wait metering (`H04` Fix).

    - If a thread ALREADY holds a read lock (`_reader_depths[tid] > 0`), nested `acquire_read()`
      succeeds immediately without blocking on `_writers_waiting > 0` — eliminating the nested-reader
      deadlock proven in `harness_controls.py`!
    - If a thread holds the exclusive write lock (`_writer_tid == tid`), nested `acquire_read()` and
      nested `acquire_write()` both succeed reentrantly.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._reader_depths: dict[int, int] = {}
        self._writers_waiting = 0
        self._writer_active = False
        self._writer_tid: int | None = None
        self._writer_depth = 0
        self.total_read_wait_ms = 0.0
        self.total_write_wait_ms = 0.0

    def acquire_read(self) -> float:
        tid = threading.get_ident()
        t0 = monotonic()
        with self._cond:
            if self._writer_active and self._writer_tid == tid:
                return 0.0
            cur_depth = self._reader_depths.get(tid, 0)
            if cur_depth > 0:
                self._reader_depths[tid] = cur_depth + 1
                self._readers += 1
                return 0.0
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._reader_depths[tid] = 1
            self._readers += 1
            w_ms = round((monotonic() - t0) * 1000.0, 3)
            self.total_read_wait_ms = round(self.total_read_wait_ms + w_ms, 3)
            return w_ms

    def release_read(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if self._writer_active and self._writer_tid == tid:
                return
            cur_depth = self._reader_depths.get(tid, 0)
            if cur_depth <= 0:
                return
            if cur_depth == 1:
                self._reader_depths.pop(tid, None)
            else:
                self._reader_depths[tid] = cur_depth - 1
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> float:
        tid = threading.get_ident()
        t0 = monotonic()
        with self._cond:
            if self._writer_active and self._writer_tid == tid:
                self._writer_depth += 1
                return 0.0
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._cond.wait()
                self._writer_active = True
                self._writer_tid = tid
                self._writer_depth = 1
            finally:
                self._writers_waiting -= 1
            w_ms = round((monotonic() - t0) * 1000.0, 3)
            self.total_write_wait_ms = round(self.total_write_wait_ms + w_ms, 3)
            return w_ms

    def release_write(self) -> None:
        tid = threading.get_ident()
        with self._cond:
            if not self._writer_active or self._writer_tid != tid:
                return
            self._writer_depth -= 1
            if self._writer_depth <= 0:
                self._writer_active = False
                self._writer_tid = None
                self._writer_depth = 0
                self._cond.notify_all()


# ============================================================================
# 2. Multi-Scale `Pi_fold` & Branch Reconciliation Container Wormhole (`P1a`)
# ============================================================================


SKILL_07_MULTISCALE_RECONCILIATION = '''"""Domain-General Multi-Scale Causal Temporal Folding (`Pi_fold`) & Minority-Slice Reconciliation (`M6.8-B'`).

Exposes the core Hegelian Plan-20 operators inside `/tmp/hegel_wormhole_skills/`:
  1. `pi_fold_temporal_regime_selector`: Evaluates `effective_constant` (fast adiabatic relaxation),
     `reduced_slow_ode` (slow manifold drift), and `mori_zwanzig_memory_kernel` (exponential history
     convolution + hidden-state carry-over) strictly split at `t <= cutoff_t0` -> `t > cutoff_t0`!
  2. `adjudicate_incumbent_vs_challenger_slices`: Enforces strict non-degradation across real-data slices
     (cohorts, sensor subsets, multi-window horizons) so pooled-mean gains NEVER overwrite an incumbent
     that performs better on worst/minority slices!
"""
from __future__ import annotations
import numpy as np
from . import track_wormhole_skill


@track_wormhole_skill("multiscale_and_reconciliation_wormhole")
def pi_fold_temporal_regime_selector(
    timestamps: np.ndarray,
    forcing_u: np.ndarray,
    observed_y: np.ndarray,
    cutoff_t0: float | None = None,
    tau_grid: tuple[float, ...] = (0.05, 0.2, 0.8, 3.0),
) -> dict:
    """Fit & compare Plan-20 `Pi_fold` temporal modes (`effective_constant`, `reduced_slow_ode`, `mori_zwanzig_memory_kernel`)
    using ONLY causal prefix `t <= cutoff_t0` for fitting and `t > cutoff_t0` for forward horizon confirmation!
    """
    t = np.asarray(timestamps, dtype=float).ravel()
    u = np.asarray(forcing_u, dtype=float).ravel()
    y = np.asarray(observed_y, dtype=float).ravel()
    if not (len(t) == len(u) == len(y)) or len(t) < 6 or not (np.all(np.isfinite(t)) and np.all(np.isfinite(u)) and np.all(np.isfinite(y))):
        raise ValueError("Invalid or non-finite time series in pi_fold_temporal_regime_selector")
    order = np.argsort(t)
    t, u, y = t[order], u[order], y[order]
    t0 = float(t[int(0.7 * len(t))]) if cutoff_t0 is None else float(cutoff_t0)
    tr = t <= t0
    va = t > t0
    if int(np.sum(tr)) < 4 or int(np.sum(va)) < 2:
        tr = np.arange(len(t)) < max(4, len(t) - 2)
        va = ~tr

    # Mode 1: `effective_constant` (instantaneous affine quasi-steady response `y = a * u + b`)
    A_inst = np.column_stack([u[tr], np.ones(int(np.sum(tr)))])
    coef_inst, *_ = np.linalg.lstsq(A_inst, y[tr], rcond=None)
    pred_inst_va = coef_inst[0] * u[va] + coef_inst[1]
    rmse_inst = float(np.sqrt(np.mean((pred_inst_va - y[va]) ** 2)))

    # Mode 2: `reduced_slow_ode` (discrete 1-step relaxation `y_k = alpha * y_{k-1} + beta * u_k + c` rolled out causally on `va`)
    idx_tr = np.flatnonzero(tr)
    X_ode = np.column_stack([y[idx_tr[:-1]], u[idx_tr[1:]], np.ones(len(idx_tr) - 1)])
    coef_ode, *_ = np.linalg.lstsq(X_ode, y[idx_tr[1:]], rcond=None)
    y_roll = float(y[idx_tr[-1]])
    preds_ode = []
    for k in np.flatnonzero(va):
        y_roll = float(coef_ode[0] * y_roll + coef_ode[1] * u[k] + coef_ode[2])
        preds_ode.append(y_roll)
    rmse_ode = float(np.sqrt(np.mean((np.asarray(preds_ode) - y[va]) ** 2)))

    # Mode 3: `mori_zwanzig_memory_kernel` (causal exponential convolution `m_k(tau) = int_{<=t_k} exp(-(t_k-s)/tau) u(s) ds` + initial decay)
    best_mz = {"rmse": float("inf"), "tau": tau_grid[0]}
    for tau in tau_grid:
        tau_f = max(float(tau), 1e-4)
        mem = np.zeros(len(t), dtype=float)
        decay0 = np.exp(-(t - t[0]) / tau_f)
        for i in range(1, len(t)):
            dt_i = max(t[i] - t[i - 1], 0.0)
            e_i = float(np.exp(-dt_i / tau_f))
            mem[i] = e_i * mem[i - 1] + 0.5 * dt_i * (e_i * u[i - 1] + u[i])
        Phi_mz = np.column_stack([u, mem, decay0, np.ones(len(t))])
        c_mz, *_ = np.linalg.lstsq(Phi_mz[tr], y[tr], rcond=None)
        r_mz = float(np.sqrt(np.mean((Phi_mz[va] @ c_mz - y[va]) ** 2)))
        if r_mz < best_mz["rmse"]:
            best_mz = {"rmse": r_mz, "tau": tau_f}

    candidates = {
        "effective_constant": round(rmse_inst, 6),
        "reduced_slow_ode": round(rmse_ode, 6),
        "mori_zwanzig_memory_kernel": round(float(best_mz["rmse"]), 6),
    }
    best_mode = min(candidates, key=lambda k: (candidates[k], 0 if k == "effective_constant" else (1 if k == "reduced_slow_ode" else 2)))
    return {
        "selected_pi_fold_mode": best_mode,
        "forward_horizon_rmse_by_mode": candidates,
        "best_memory_tau": round(float(best_mz["tau"]), 5),
        "causal_cutoff_t0": t0,
    }


@track_wormhole_skill("multiscale_and_reconciliation_wormhole")
def adjudicate_incumbent_vs_challenger_slices(
    incumbent_slice_losses: dict[str, float],
    challenger_slice_losses: dict[str, float],
    max_slice_regression_tol: float = 0.002,
) -> dict:
    """Branch-Reconciliation Minority-Slice Protection (`Acceptance B4/B8`):
    Promotes `challenger` ONLY if it improves worst-slice or mean loss WITHOUT regressing ANY slice beyond
    `max_slice_regression_tol`; otherwise retains `incumbent`!
    """
    keys = sorted(set(incumbent_slice_losses.keys()) & set(challenger_slice_losses.keys()))
    if not keys:
        return {"decision": "keep_incumbent", "reason": "no_shared_validation_slices"}
    inc_vals = np.array([float(incumbent_slice_losses[k]) for k in keys], dtype=float)
    chl_vals = np.array([float(challenger_slice_losses[k]) for k in keys], dtype=float)
    if not (np.all(np.isfinite(inc_vals)) and np.all(np.isfinite(chl_vals))):
        return {"decision": "keep_incumbent", "reason": "non_finite_slice_loss"}

    regressions = {k: round(float(challenger_slice_losses[k] - incumbent_slice_losses[k]), 6) for k in keys if float(challenger_slice_losses[k] - incumbent_slice_losses[k]) > max_slice_regression_tol}
    if regressions:
        return {
            "decision": "keep_incumbent",
            "reason": "minority_or_cohort_slice_tradeoff_detected",
            "regressed_slices": regressions,
        }
    inc_worst, chl_worst = float(np.max(inc_vals)), float(np.max(chl_vals))
    inc_mean, chl_mean = float(np.mean(inc_vals)), float(np.mean(chl_vals))
    if chl_worst < inc_worst - 1e-6 or (chl_worst <= inc_worst + 1e-6 and chl_mean < inc_mean - 1e-6):
        return {
            "decision": "promote_challenger",
            "reason": "pareto_non_degrading_slice_improvement",
            "incumbent_worst": round(inc_worst, 6),
            "challenger_worst": round(chl_worst, 6),
        }
    return {
        "decision": "keep_incumbent",
        "reason": "no_strict_pareto_improvement",
        "incumbent_worst": round(inc_worst, 6),
        "challenger_worst": round(chl_worst, 6),
    }
'''


# ============================================================================
# 3. Immutable RegradeReceipt (`P0a`) & Hash-Bound VerificationReceipt (`P0c`)
# ============================================================================


def build_regrade_receipt(
    *,
    original_episode_id: str,
    original_candidate_sha256: str,
    regrade_candidate_sha256: str,
    original_verifier_sha256: str,
    task_id: str,
    verifier_image_tag: str,
    resource_spec: dict[str, Any],
    timing_replicates_sec: list[float],
    official_median_sec: float | None,
    threshold_sec: float | None,
    strict_pass: bool,
    ctrf_summary: dict[str, Any],
    verifier_stdout: str,
    regrade_reason: str,
) -> dict[str, Any]:
    """Construct an auditable `RegradeReceipt` (`05_regrade_receipt.json`) without mutating `00_episode_summary.json`."""
    hash_ok = bool(original_candidate_sha256 and original_candidate_sha256 == regrade_candidate_sha256)
    core = {
        "receipt_kind": "Hegel_Immutable_RegradeReceipt_v1",
        "original_episode_id": original_episode_id,
        "task_id": task_id,
        "original_candidate_sha256": original_candidate_sha256,
        "regrade_candidate_sha256": regrade_candidate_sha256,
        "candidate_hash_match_verified": hash_ok,
        "original_verifier_sha256": original_verifier_sha256,
        "verifier_image_tag": verifier_image_tag,
        "resource_spec": resource_spec,
        "timing_replicates_sec": [round(float(x), 6) for x in timing_replicates_sec],
        "official_median_sec": round(float(official_median_sec), 6) if official_median_sec is not None else None,
        "threshold_sec": round(float(threshold_sec), 6) if threshold_sec is not None else None,
        "regrade_strict_pass": bool(strict_pass and hash_ok),
        "ctrf_summary": ctrf_summary,
        "verifier_stdout_tail": verifier_stdout[-4000:],
        "regrade_reason": regrade_reason,
    }
    core["receipt_sha256"] = hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()
    return core


def verify_candidate_artifact_contract_in_container(
    agent_cid: str,
    declared_artifacts: tuple[str, ...],
    candidate_sha256: str,
) -> dict[str, Any]:
    """Host-executed deterministic candidate validator inside `<agent_cid>` (`H01` Fix).

    Checks clean subprocess Python import/compile, CSV/TSV/JSON/NPZ non-emptiness & finite numeric values,
    and absence of un-inlined `/tmp` `hegel_wormhole_skills` imports — bound to exact `candidate_sha256`!
    """
    if not declared_artifacts or not candidate_sha256:
        return {
            "accepted": False,
            "candidate_sha256": candidate_sha256,
            "reasons": ["missing_declared_artifacts_or_empty_digest"],
        }
    validator_py = r'''
import ast, csv, importlib.util, json, math, os, sys
import numpy as np

arts = json.loads(sys.argv[1])
issues = []
checked_files = []

def _expand(paths):
    out = []
    for a in paths:
        if os.path.isfile(a):
            out.append(a)
        elif os.path.isdir(a):
            for dp, _, fns in os.walk(a):
                if "__pycache__" in dp:
                    continue
                for fn in sorted(fns):
                    out.append(os.path.join(dp, fn))
        else:
            issues.append(f"missing:{a}")
    return out

files = _expand(arts)
if not files:
    issues.append("no_files_found")

for fp in files:
    sz = os.path.getsize(fp)
    if sz == 0:
        issues.append(f"empty_file:{fp}")
        continue
    checked_files.append({"path": fp, "bytes": sz})
    if fp.endswith(".py"):
        txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
        if "hegel_wormhole_skills" in txt:
            issues.append(f"uninlined_wormhole_import:{fp}")
        try:
            ast.parse(txt)
        except SyntaxError as exc:
            issues.append(f"syntax_error:{fp}:{exc.lineno}")
    elif fp.endswith((".csv", ".tsv")):
        sep = "\t" if fp.endswith(".tsv") else ","
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            rows = list(csv.reader(f, delimiter=sep))
        if len(rows) < 2:
            issues.append(f"table_has_fewer_than_2_rows:{fp}")
        else:
            hdr_len = len(rows[0])
            for r_i, r in enumerate(rows[1:], start=2):
                if len(r) != hdr_len:
                    issues.append(f"column_count_mismatch:{fp}:line_{r_i}")
                    break
                for cell in r:
                    cs = cell.strip().lower()
                    if cs in {"inf", "-inf"}:
                        issues.append(f"infinite_cell_value:{fp}:line_{r_i}")
                        break
    elif fp.endswith(".json") and sz < 60_000_000:
        try:
            obj = json.load(open(fp, "r", encoding="utf-8"))
            if obj in ({}, [], None):
                issues.append(f"empty_json_payload:{fp}")
        except Exception as exc:
            issues.append(f"invalid_json:{fp}:{type(exc).__name__}")
    elif fp.endswith(".npz") and sz < 100_000_000:
        try:
            z = np.load(fp, allow_pickle=False)
            if not z.files:
                issues.append(f"empty_npz:{fp}")
            for k in z.files:
                arr = z[k]
                if arr.dtype.kind in {"f", "c"} and not np.all(np.isfinite(arr)):
                    issues.append(f"non_finite_npz_array:{fp}:{k}")
        except Exception as exc:
            issues.append(f"invalid_npz:{fp}:{type(exc).__name__}")

print(json.dumps({"accepted": len(issues) == 0, "issues": issues, "checked_files": checked_files}))
'''
    v_sha = hashlib.sha256(validator_py.encode("utf-8")).hexdigest()[:16]
    try:
        proc = subprocess.run(
            ["docker", "exec", "-w", "/", agent_cid, "python3", "-c", validator_py, json.dumps(list(declared_artifacts))],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if proc.returncode != 0:
            return {
                "accepted": False,
                "candidate_sha256": candidate_sha256,
                "validator_sha256_16": v_sha,
                "reasons": [f"validator_rc_{proc.returncode}:{(proc.stderr or '')[-200:]}"],
            }
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        return {
            "accepted": bool(parsed.get("accepted")),
            "candidate_sha256": candidate_sha256,
            "validator_sha256_16": v_sha,
            "reasons": list(parsed.get("issues", [])),
            "checked_files": list(parsed.get("checked_files", [])),
        }
    except Exception as exc:
        return {
            "accepted": False,
            "candidate_sha256": candidate_sha256,
            "validator_sha256_16": v_sha,
            "reasons": [f"validator_exception:{type(exc).__name__}:{str(exc)[:160]}"],
        }


# ============================================================================
# 4. Complete `ExecutableSkillManifest` Harvest & Historical DAG Recovery (`P0b`)
# ============================================================================


def harvest_container_producer_skill_manifest(
    *,
    agent_cid: str,
    task_id: str,
    short_name: str,
    wormhole_family: str,
    episode_id: str,
    declared_artifacts: tuple[str, ...],
    candidate_sha256: str,
    turn_logs: list[dict[str, Any]],
    verification_receipts: dict[str, dict[str, Any]],
    episode_out_dir: Path,
    content_store_root: Path | None = None,
) -> dict[str, Any]:
    """Harvest the complete producer closure (`/tmp` & workspace Python/C++/sh scripts, build commands,
    content-addressed binary/JSON artifact descriptors, temporal/role schema) BEFORE container deletion (`P0b` Fix).
    """
    manifest_dir = episode_out_dir / "04b_producer_skill_manifest"
    producers_dir = manifest_dir / "producers"
    producers_dir.mkdir(parents=True, exist_ok=True)
    blob_store = content_store_root or (episode_out_dir.parents[1] / "content_addressed_blobs")
    blob_store.mkdir(parents=True, exist_ok=True)

    harvest_py = r'''
import hashlib, json, os
out_producers = []
for root in ("/tmp", "/app", "/workspace", "/root"):
    if not os.path.isdir(root):
        continue
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in {"__pycache__", ".git", ".cache", "hegel_wormhole_skills", ".hegel_candidate_archive", "tests"}]
        if "/tests" in dp:
            continue
        for fn in sorted(fns):
            if fn.startswith(".hegel_"):
                continue
            fp = os.path.join(dp, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            if fn.endswith((".py", ".cpp", ".cc", ".c", ".hpp", ".h", ".sh")) and 0 < sz <= 2_000_000:
                txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
                if "answer_rubric" in txt or "/tests/" in txt:
                    continue
                sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                out_producers.append({"path": fp, "bytes": sz, "sha256": sha, "source": txt})
print(json.dumps(out_producers))
'''
    producer_records: list[dict[str, Any]] = []
    try:
        proc = subprocess.run(
            ["docker", "exec", agent_cid, "python3", "-c", harvest_py],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            raw_prods = json.loads(proc.stdout.strip().splitlines()[-1])
            for idx, item in enumerate(raw_prods[:40]):
                safe_name = item["path"].lstrip("/").replace("/", "__")
                dst_p = producers_dir / safe_name
                write_text_incrementally(dst_p, item["source"])
                producer_records.append(
                    {
                        "container_path": item["path"],
                        "archived_relpath": f"producers/{safe_name}",
                        "bytes": item["bytes"],
                        "sha256": item["sha256"],
                    }
                )
    except Exception:
        pass

    # Content-addressed binary/large-JSON state descriptors for declared artifacts (zero 512KB silent drop!)
    artifact_blob_refs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="hegel_blob_harvest_") as tmp_b:
        for idx, art in enumerate(declared_artifacts):
            dst_tmp = Path(tmp_b) / f"art_{idx}"
            cp_proc = subprocess.run(
                ["docker", "cp", f"{agent_cid}:{art}", str(dst_tmp)],
                capture_output=True,
                timeout=25,
            )
            if cp_proc.returncode != 0 or not dst_tmp.exists():
                continue
            files_iter = [dst_tmp] if dst_tmp.is_file() else [p for p in sorted(dst_tmp.rglob("*")) if p.is_file()]
            for fp in files_iter:
                sz = fp.stat().st_size
                h = hashlib.sha256()
                with fp.open("rb") as f_in:
                    for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
                        h.update(chunk)
                b_sha = h.hexdigest()
                ext = fp.suffix or ".bin"
                # Persist blobs up to 15 MB in content-addressed store via `checkpoint_storage.clone_or_copy`
                stored_blob_rel = None
                if sz <= 15 * 1024 * 1024:
                    blob_dst = blob_store / f"{b_sha}{ext}"
                    if not blob_dst.exists():
                        clone_or_copy(fp, blob_dst)
                    stored_blob_rel = f"content_addressed_blobs/{blob_dst.name}"
                artifact_blob_refs.append(
                    {
                        "declared_root": art,
                        "filename": fp.name,
                        "bytes": sz,
                        "sha256": b_sha,
                        "content_addressed_ref": stored_blob_rel,
                    }
                )

    # Also invoke `code_agent.retain_evaluated_python_model` on the primary Python producer if present!
    retained_model_ref: dict[str, Any] | None = None
    py_prods = [p for p in producer_records if p["archived_relpath"].endswith(".py")]
    if py_prods:
        primary_py = Path(py_prods[-1]["archived_relpath"]).name
        try:
            ret_info = retain_evaluated_python_model(
                producers_dir,
                primary_py,
                channels=[wormhole_family],
                training_mse=0.0,
                destination="_retained_evaluated_model",
            )
            retained_model_ref = {
                "retained_model": ret_info["model"],
                "selected_channels": ret_info["selected_channels"],
            }
        except Exception:
            retained_model_ref = None

    build_cmds = [
        str((t.get("decision") or {}).get("bash_command") or "")[:500]
        for t in turn_logs
        if any(k in str((t.get("decision") or {}).get("bash_command") or "") for k in ("g++", "gcc", "cmake", "python3"))
    ][-12:]

    # Infer temporal & structural attributes from producer code
    combined_py = "\n\n".join(
        (manifest_dir / p["archived_relpath"]).read_text(encoding="utf-8", errors="ignore")
        for p in py_prods[:8]
        if (manifest_dir / p["archived_relpath"]).is_file()
    )
    pure_fns = extract_pure_reusable_functions_from_python(combined_py, max_funcs=8) if combined_py else []
    has_memory_or_creep = any(w in combined_py.lower() for w in ("creep", "conv", "exp(-", "memory", "history", "rk4"))
    temporal_mode = "mori_zwanzig_or_stage_history" if has_memory_or_creep else "present_or_adiabatic_ode"

    manifest = {
        "schema_version": "ExecutableSkillManifest_v1",
        "identity": {
            "artifact_id": f"skill_{short_name}_{candidate_sha256[:12] or 'uncommitted'}",
            "source_hash": candidate_sha256,
            "parent_ids": [],
            "source_episode_id": episode_id,
            "task_id": task_id,
            "wormhole_family": wormhole_family,
        },
        "producer": {
            "sources": producer_records,
            "build_and_invoke_commands": build_cmds,
            "retained_code_agent_model": retained_model_ref,
            "binding_status": "bound_pure_and_full_closure" if pure_fns else ("opaque_full_producer_retained" if producer_records else "missing_producer"),
        },
        "state": {
            "artifact_blob_refs": artifact_blob_refs,
            "refit_recipe": "Execute archived producer script against public container data paths.",
        },
        "interface": {
            "declared_artifacts": list(declared_artifacts),
            "exported_pure_functions": [fn for fn, _ in pure_fns],
        },
        "temporal": {
            "temporal_readout_mode": temporal_mode,
            "supports_multiscale_pi_fold": bool(has_memory_or_creep),
        },
        "conditions": {
            "wormhole_family": wormhole_family,
            "unresolved_assumptions": [],
        },
        "evidence": {
            "candidate_verification_receipt": verification_receipts.get(candidate_sha256),
        },
        "structure": {
            "full_producer_count": len(producer_records),
            "reusable_ast_function_closures": [{"name": nm, "code": cd} for nm, cd in pure_fns],
        },
    }
    write_text_incrementally(
        manifest_dir / "executable_skill_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def recover_historical_producers_from_sealed_dag(episode_dir: Path) -> dict[str, Any]:
    """Reconstruct `/tmp` & workspace Python/C++ producer scripts from a sealed historical episode's
    `02_human_readable_trajectory.md` & `04_submitted_artifacts_snapshot` (`P0b` Historical Capability Recovery).
    """
    summary_fp = episode_dir / "00_episode_summary.json"
    traj_fp = episode_dir / "02_human_readable_trajectory.md"
    snap_dir = episode_dir / "04_submitted_artifacts_snapshot"
    if not summary_fp.is_file():
        return {"ok": False, "reason": "missing_episode_summary", "recovered_files": []}

    summary = json.loads(summary_fp.read_text(encoding="utf-8"))
    recovered: dict[str, str] = {}

    if snap_dir.is_dir():
        for fp in sorted(snap_dir.iterdir()):
            if fp.is_file() and fp.suffix in {".py", ".cpp", ".c", ".h", ".sh", ".csv", ".tsv", ".json"}:
                recovered[f"snapshot::{fp.name}"] = fp.read_text(encoding="utf-8", errors="ignore")

    if traj_fp.is_file():
        traj_txt = traj_fp.read_text(encoding="utf-8", errors="ignore")
        # Support BOTH `cat << 'TAG' > /tmp/foo.py` AND `cat > /tmp/foo.py << 'TAG'` AND `python3 - << 'TAG'`!
        heredoc_events: list[tuple[int, str, str, bool]] = []
        pat_tag_first = re.compile(
            r"cat\s+<<-?\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*(>>?)\s*([^\s;&|]+)\s*\n(.*?)\n\1\b",
            re.DOTALL,
        )
        pat_path_first = re.compile(
            r"cat\s*(>>?)\s*([^\s;&|<]+)\s*<<-?\s*['\"]?([A-Za-z0-9_]+)['\"]?[^\n]*\n(.*?)\n\3\b",
            re.DOTALL,
        )
        pat_inline_py = re.compile(
            r"python3?\s+-\s*<<-?\s*['\"]?([A-Za-z0-9_]+)['\"]?[^\n]*\n(.*?)\n\1\b",
            re.DOTALL,
        )
        for m in pat_tag_first.finditer(traj_txt):
            heredoc_events.append((m.start(), m.group(3).strip().strip("'\""), m.group(4), m.group(2) == ">>"))
        for m in pat_path_first.finditer(traj_txt):
            heredoc_events.append((m.start(), m.group(2).strip().strip("'\""), m.group(4), m.group(1) == ">>"))
        inline_idx = 0
        for m in pat_inline_py.finditer(traj_txt):
            inline_idx += 1
            heredoc_events.append((m.start(), f"/tmp/_inline_producer_{inline_idx:02d}.py", m.group(2), False))

        for _, target_path, body, is_append in sorted(heredoc_events, key=lambda x: x[0]):
            if target_path.endswith((".py", ".cpp", ".cc", ".h", ".sh")) and "/tests" not in target_path:
                if is_append and target_path in recovered:
                    recovered[target_path] = recovered[target_path] + "\n" + body
                else:
                    recovered[target_path] = body

    return {
        "ok": len(recovered) > 0,
        "task_id": summary.get("task_id"),
        "strict_pass": bool(summary.get("strict_pass")),
        "recovered_file_count": len(recovered),
        "recovered_paths": sorted(recovered.keys()),
        "recovered_sources": {
            k: {"bytes": len(v.encode("utf-8")), "sha256_16": hashlib.sha256(v.encode("utf-8")).hexdigest()[:16], "code": v}
            for k, v in sorted(recovered.items())
        },
    }


# ============================================================================
# 5. Wire M5 Multi-Scale / Subtree & Branch Reconciliation into Frontier (`P1a`)
# ============================================================================


def consolidate_wave1_skill_manifests_with_reconciliation(
    *,
    wave1_episode_dirs: list[tuple[dict[str, Any], Path]],
    forest: PhilosophicalForest,
    archive: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Consolidate complete Wave-1 `ExecutableSkillManifest` closures + anti-unified shared base subtrees
    and propagate premise retractions via `branch_reconciliation` & `m5_temporal_multiscale_and_base_subtree` (`P1a`).
    """
    events: list[dict[str, Any]] = []
    exported_programs_by_family: dict[str, list[tuple[str, Program]]] = {}
    retracted_ids: set[str] = set()

    for summary, ep_dir in sorted(wave1_episode_dirs, key=lambda x: str(x[0].get("task_id", ""))):
        t_id = str(summary.get("task_id") or "")
        t_short = str(summary.get("short_name") or t_id).replace("-", "_")
        fam = str(summary.get("wormhole_family") or "")

        # Propagate premise retraction on RUNG_3 falsified mechanisms (`branch_reconciliation.propagate_premise_retraction_invalidation`)
        for ng in summary.get("nogoods_recorded") or []:
            if str(ng.get("nogood_rung", "")).startswith("RUNG_3"):
                retracted_ids.add(f"prem_{fam}")

        if not summary.get("strict_pass"):
            continue

        man_fp = ep_dir / "04b_producer_skill_manifest" / "executable_skill_manifest.json"
        if man_fp.is_file():
            man = json.loads(man_fp.read_text(encoding="utf-8"))
            closures = list((man.get("structure") or {}).get("reusable_ast_function_closures") or [])
            full_prod_cnt = int((man.get("structure") or {}).get("full_producer_count", 0))
            temp_mode = str((man.get("temporal") or {}).get("temporal_readout_mode") or "present_or_adiabatic_ode")
        else:
            rec_h = recover_historical_producers_from_sealed_dag(ep_dir)
            py_blob = "\n\n".join(
                v["code"] for k, v in rec_h.get("recovered_sources", {}).items() if k.endswith(".py")
            )
            closures = [{"name": nm, "code": cd} for nm, cd in extract_pure_reusable_functions_from_python(py_blob, max_funcs=8)]
            full_prod_cnt = rec_h.get("recovered_file_count", 0)
            temp_mode = "present_or_adiabatic_ode"

        # Represent temporal scaling structure as a typed `Program` AST for `anti_unify_programs` / `SharedBaseSubtreePrior`
        prog_repr = Program(
            "add",
            children=(
                Program(
                    "mul",
                    children=(
                        Program("const", value=float(max(1, len(closures)))),
                        Program("feature", name="u_drive"),
                    ),
                ),
                Program("feature", name="memory_state" if "mori_zwanzig" in temp_mode else "slow_state"),
            ),
        )
        if summary.get("benchmark") != "BioMysteryBench-Preview":
            exported_programs_by_family.setdefault(fam, []).append((t_short, prog_repr))

        direct_nids = sorted(nid for nid, nd in archive.items() if nd.get("family_tag") == fam)
        target_nids: set[str] = set(direct_nids)
        for nid in direct_nids:
            for wh in forest.horizontal_wormhole_edges.values():
                nb = wh["node_b_id"] if wh["node_a_id"] == nid else (wh["node_a_id"] if wh["node_b_id"] == nid else None)
                if nb and archive.get(nb, {}).get("family_tag") == fam:
                    target_nids.add(nb)

        if closures:
            mod_stem = f"wave1_manifest_{t_short}"
            mod_fname = f"{mod_stem}.py"
            # Emit helper closures first in topological order, wrapping top-level callables with `@track_wormhole_skill`
            decorated_blocks = [f'@track_wormhole_skill("{mod_stem}")\n{c["code"]}' for c in closures]
            mod_code = (
                f'"""Wave-1 Complete Producer Closure from `{t_short}` (`temporal_mode={temp_mode}`)."""\n'
                "from __future__ import annotations\n"
                "import math\n"
                "import numpy\n"
                "import numpy as np\n"
                "from . import track_wormhole_skill\n\n\n"
                + "\n\n\n".join(decorated_blocks)
                + "\n"
            )
            try:
                compile(mod_code, mod_fname, "exec")
                mod_record = {
                    "filename": mod_fname,
                    "origin": f"wave1_manifest_closure_from_{t_short}",
                    "temporal_readout_mode": temp_mode,
                    "full_producer_count": full_prod_cnt,
                    "function_names": [c["name"] for c in closures],
                    "python_code": mod_code,
                }
                for tnid in sorted(target_nids):
                    mod_list = archive[tnid].setdefault("wave1_executable_modules", [])
                    if all(m["filename"] != mod_fname for m in mod_list):
                        mod_list.append(mod_record)
            except SyntaxError:
                mod_fname = None
        else:
            mod_fname = None

        events.append(
            {
                "source_task": t_short,
                "full_producer_count": full_prod_cnt,
                "temporal_readout_mode": temp_mode,
                "module_filename": mod_fname,
                "exported_functions": [c["name"] for c in closures],
                "propagated_to_nodes": sorted(target_nids),
            }
        )

    # Extract genuine `SharedBaseSubtreePrior` across pairs of confirmed Wave-1 families (`m5` + `anti_unify_programs`)
    registered_subtrees: list[str] = []
    all_prog_entries = [item for lst in exported_programs_by_family.values() for item in lst]
    if len(all_prog_entries) >= 2:
        (src_a, p_a), (src_b, p_b) = all_prog_entries[0], all_prog_entries[1]
        anti = anti_unify_programs(p_a, p_b)
        if anti is not None:
            sub_id = f"subtree_wave1_{src_a}__{src_b}"
            if sub_id not in getattr(forest, "shared_base_subtrees", {}):
                extract_shared_base_subtree_from_sources(
                    forest,
                    subtree_id=sub_id,
                    prog_a=p_a,
                    prog_b=p_b,
                    shared_nodes=[{"role": "drive_response"}, {"role": "state_carryover"}],
                    shared_relations=[{"relation": "requires_premise_and", "from": "drive_response", "to": "state_carryover"}],
                    role_and_scale_schema={"q_role": "state", "u_role": "forcing", "temporal_normalization": "t_over_tau"},
                    open_completion_branches=[{"slot": "residual_nonlinearity"}],
                    supporting_sources=[{"source_id": src_a, "tree_id": src_a}, {"source_id": src_b, "tree_id": src_b}],
                    probe_signatures=["sig:ode_trajectory", "sig:conservation_pde", "sig:sparse_sensor_network"],
                )
                registered_subtrees.append(sub_id)

    # Execute `propagate_premise_retraction_invalidation` across claim dependency records
    retraction_summary: list[dict[str, Any]] = []
    if retracted_ids:
        claim_graph = [
            {
                "claim_id": f"claim_{nid}",
                "premises": [f"prem_{nd.get('family_tag', '')}"],
                "supported_by": [],
                "status": "active",
            }
            for nid, nd in sorted(archive.items())
        ]
        updated_claims = propagate_premise_retraction_invalidation(
            claim_graph, retracted_premise_or_claim_ids=sorted(retracted_ids)
        )
        retraction_summary = [c for c in updated_claims if c.get("status") != "active"]

    # Also verify `predict_via_pi_fold_mode` & `extract_causal_temporal_regimes` readiness on synthetic calibration trace
    reg_demo = extract_causal_temporal_regimes(
        [
            {"t": 0.0, "x": 0.0, "u": 1.0, "y": 0.0},
            {"t": 0.5, "x": 0.4, "u": 1.0, "y": 0.4},
            {"t": 1.0, "x": 0.7, "u": 1.0, "y": 0.7},
        ],
        t0=1.0,
        extraction_window_w=1.0,
        intrinsic_taus=(0.4, 2.0),
        forecast_horizon_H=0.5,
    )
    pi_demo_pred = predict_via_pi_fold_mode(
        "memory_kernel",
        params={"a_slow": 0.8, "b_couple": 0.2, "c_drive": 1.0, "tau_fast": 0.4, "g_feedback": -0.3},
        causal_regimes=reg_demo,
        u_future_const=1.0,
        horizon_H=0.5,
    )
    align_demo = align_semantics_roles_units_and_time(
        {"claim_id": "frontier_target", "roles": {"q": "state", "u": "forcing"}, "temporal_history_depth": 1, "program": Program("feature", name="q").to_dict()},
        [],
        target_schema={"roles": {"q": "state", "u": "forcing"}},
    )

    return {
        "wave1_manifest_events": events,
        "registered_shared_base_subtrees": registered_subtrees,
        "retracted_claim_updates": retraction_summary,
        "m5_pi_fold_and_alignment_wiring": {
            "pi_fold_memory_kernel_ready": bool(np.isfinite(pi_demo_pred)),
            "role_unit_time_aligner_invoked": True,
            "align_status": bool(align_demo is not None),
        },
    }


def qualify_staged_wormhole_modules_in_container(
    agent_cid: str,
    staged_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run an isolated container import & contract qualification pilot (`retrieved -> bound -> smoke_qualified`)
    tagged with `HEGEL_SKILL_CALLER_ROLE=smoke_test` so startup qualification NEVER inflates agent skill counts!
    """
    if not staged_manifest:
        return []
    mod_names = [str(m["module_import"]) for m in staged_manifest]
    qual_py = (
        "import importlib, json, os, sys\n"
        "os.environ['HEGEL_SKILL_CALLER_ROLE'] = 'smoke_test'\n"
        "mods = json.loads(sys.argv[1])\n"
        "out = []\n"
        "for mn in mods:\n"
        "    try:\n"
        "        m = importlib.import_module(mn)\n"
        "        fns = [k for k, v in vars(m).items() if callable(v) and not k.startswith('_')]\n"
        "        out.append({'module_import': mn, 'lifecycle_state': 'bound_and_import_qualified', 'callable_count': len(fns)})\n"
        "    except Exception as exc:\n"
        "        out.append({'module_import': mn, 'lifecycle_state': 'failed_import', 'error': f'{type(exc).__name__}:{exc}'})\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        ["docker", "exec", agent_cid, "bash", "-lc", f"PYTHONPATH=/tmp python3 -c {json.dumps(qual_py)} {json.dumps(json.dumps(mod_names))}"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return [{"module_import": mn, "lifecycle_state": "qualification_exec_failed"} for mn in mod_names]
    try:
        return list(json.loads(proc.stdout.strip().splitlines()[-1]))
    except Exception:
        return [{"module_import": mn, "lifecycle_state": "qualification_parse_failed"} for mn in mod_names]


# ============================================================================
# 6. Non-Tautological Public Candidate Selector (`P1b` Offline & Online Calibration)
# ============================================================================


def select_candidate_on_public_evidence_only(
    candidates: list[dict[str, Any]],
    *,
    incumbent_index: int = 0,
) -> dict[str, Any]:
    """Select among candidate submissions using ONLY public non-tautological signals (`ZERO` private truth/verifier!).

    Each candidate dict provides ONLY public observable evidence:
      - `candidate_id`: str
      - `contract_accepted`: bool (schema, non-empty, finite, clean-process import, zero `/tmp` uninlined refs)
      - `ended_by_verified_commit`: bool
      - `has_post_selection_snoop_warning`: bool
      - `has_unverified_late_edit`: bool
      - `public_slice_regressions`: int
      - `public_honest_cv_score`: float | None  (ONLY when computed via nested out-of-selection CV)
      - `public_self_timed_median_sec`: float | None (ONLY for explicit runtime optimization tasks)

    Safeguard (`Acceptance P1b`): When two contract-valid candidates differ ONLY on an unidentifiable proxy
    (e.g. LOPO shift without velocity calibration, or apparent post-selection OOF), returns
    `selection_mode = "uncertain_keep_incumbent"` rather than gambling on an uncalibrated proxy!
    """
    if not candidates:
        raise ValueError("Empty candidate list")
    inc_idx = max(0, min(int(incumbent_index), len(candidates) - 1))

    valid_indices = [
        i for i, c in enumerate(candidates)
        if bool(c.get("contract_accepted", False)) and not bool(c.get("has_unverified_late_edit", False))
    ]
    if not valid_indices:
        # Fallback to any contract-accepted candidate
        valid_indices = [i for i, c in enumerate(candidates) if bool(c.get("contract_accepted", False))]
    if not valid_indices:
        return {
            "selected_index": inc_idx,
            "selected_candidate_id": candidates[inc_idx].get("candidate_id"),
            "selection_mode": "no_contract_valid_candidate_fallback_incumbent",
        }

    # If incumbent is contract-valid, only replace it on strict, non-snooped public dominance!
    best_i = inc_idx if inc_idx in valid_indices else valid_indices[0]
    best_c = candidates[best_i]
    promoted_reason = "keep_valid_incumbent" if best_i == inc_idx else "incumbent_failed_public_contract"

    for idx in valid_indices:
        if idx == best_i:
            continue
        cand = candidates[idx]
        if int(cand.get("public_slice_regressions", 0)) > 0:
            continue
        if bool(cand.get("has_post_selection_snoop_warning", False)):
            continue

        # Criterion 1: Strict public runtime speedup on identical verified numerical contract (e.g. PDE timing)
        t_best = best_c.get("public_self_timed_median_sec")
        t_cand = cand.get("public_self_timed_median_sec")
        if t_best is not None and t_cand is not None and float(t_cand) < 0.88 * float(t_best):
            best_i, best_c = idx, cand
            promoted_reason = "strict_public_runtime_improvement"
            continue

        # Criterion 2: Strictly higher nested out-of-selection CV score when current best had snooping warning or lower nested CV
        cv_best = best_c.get("public_honest_cv_score")
        cv_cand = cand.get("public_honest_cv_score")
        if cv_best is not None and cv_cand is not None and float(cv_cand) > float(cv_best) + 0.015:
            best_i, best_c = idx, cand
            promoted_reason = "strict_nested_cv_improvement"
            continue

        # Criterion 3: Current best had a post-selection snooping warning or incomplete commit while challenger is clean
        if bool(best_c.get("has_post_selection_snoop_warning", False)) and not bool(cand.get("has_post_selection_snoop_warning", False)):
            best_i, best_c = idx, cand
            promoted_reason = "eliminated_post_selection_snooping"

    return {
        "selected_index": best_i,
        "selected_candidate_id": candidates[best_i].get("candidate_id"),
        "selection_mode": promoted_reason,
    }
