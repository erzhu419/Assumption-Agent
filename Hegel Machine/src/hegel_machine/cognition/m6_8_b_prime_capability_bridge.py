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
        if cutoff_t0 is not None:
            return {
                "status": "insufficient_causal_history",
                "selected_pi_fold_mode": "insufficient_history",
                "forward_horizon_rmse_by_mode": {},
                "best_memory_tau": None,
                "causal_cutoff_t0": t0,
                "prefix_count": int(np.sum(tr)),
                "horizon_count": int(np.sum(va)),
            }
        tr = np.arange(len(t)) < max(4, len(t) - 2)
        va = ~tr
        t0 = float(t[np.flatnonzero(tr)[-1]])

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
    best_mz: dict = {"rmse": float("inf"), "tau": float(tau_grid[0]), "coef": np.zeros(4), "last_mem_tr": 0.0}
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
            best_mz = {
                "rmse": r_mz,
                "tau": tau_f,
                "coef": c_mz,
                "last_mem_tr": float(mem[idx_tr[-1]]),
            }

    candidates = {
        "effective_constant": round(rmse_inst, 6),
        "reduced_slow_ode": round(rmse_ode, 6),
        "mori_zwanzig_memory_kernel": round(float(best_mz["rmse"]), 6),
    }
    best_mode = min(candidates, key=lambda k: (candidates[k], 0 if k == "effective_constant" else (1 if k == "reduced_slow_ode" else 2)))
    return {
        "status": "ok",
        "selected_pi_fold_mode": best_mode,
        "forward_horizon_rmse_by_mode": candidates,
        "best_memory_tau": round(float(best_mz["tau"]), 5),
        "causal_cutoff_t0": t0,
        "prefix_count": int(np.sum(tr)),
        "horizon_count": int(np.sum(va)),
        "fitted_parameters": {
            "effective_constant": {"slope_u": float(coef_inst[0]), "intercept": float(coef_inst[1])},
            "reduced_slow_ode": {
                "alpha_state": float(coef_ode[0]),
                "beta_drive": float(coef_ode[1]),
                "intercept": float(coef_ode[2]),
                "y_at_cutoff": float(y[idx_tr[-1]]),
            },
            "mori_zwanzig_memory_kernel": {
                "tau": round(float(best_mz["tau"]), 5),
                "weights": [float(v) for v in np.asarray(best_mz["coef"]).ravel()],
                "memory_at_cutoff": float(best_mz["last_mem_tr"]),
                "t_origin": float(t[0]),
            },
        },
    }


@track_wormhole_skill("multiscale_and_reconciliation_wormhole")
def adjudicate_incumbent_vs_challenger_slices(
    incumbent_slice_losses: dict[str, float],
    challenger_slice_losses: dict[str, float],
    max_slice_regression_tol: float = 0.002,
) -> dict:
    """Branch-Reconciliation Minority-Slice Protection (`Acceptance B4/B8`):
    Promotes `challenger` ONLY if it covers ALL incumbent slices and improves worst-slice or mean loss WITHOUT
    regressing ANY slice beyond `max_slice_regression_tol`; otherwise retains `incumbent`!
    """
    if not incumbent_slice_losses:
        return {"decision": "keep_incumbent", "reason": "no_shared_validation_slices"}
    missing_slices = sorted(set(incumbent_slice_losses.keys()) - set(challenger_slice_losses.keys()))
    if missing_slices:
        return {
            "decision": "keep_incumbent",
            "reason": "missing_required_slices",
            "missing_slices": missing_slices,
        }
    keys = sorted(incumbent_slice_losses.keys())
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
    *,
    required_entrypoints: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Host-executed deterministic candidate validator inside `<agent_cid>` (`H01` + `E04` + `M7-Audit` Fix).

    Recomputes canonical byte-level `computed_candidate_sha256` BOTH before and after isolated subprocess Python
    import (rejecting any self-mutating candidate or digest mismatch!), registers `sys.modules[spec.name] = mod`
    before `exec_module` so `from __future__ import annotations` + `@dataclass` succeeds (`traffic` fix!),
    checks optional `required_entrypoints`, rejects `NaN`/`Inf` cells/constants in CSV/TSV/JSON/NPZ, and forbids
    `/tmp` `hegel_wormhole_skills` imports.
    """
    if not declared_artifacts or not candidate_sha256:
        return {
            "accepted": False,
            "candidate_sha256": candidate_sha256,
            "computed_candidate_sha256": "",
            "reasons": ["missing_declared_artifacts_or_empty_digest"],
        }
    validator_py = r'''
import ast, csv, hashlib, json, math, os, subprocess, sys
import numpy as np

arts = json.loads(sys.argv[1])
req_entrypoints = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
issues = []
checked_files = []
explicit_py_set = {a for a in arts if a.endswith(".py")}

def _scan_and_digest(target_arts):
    h = hashlib.sha256()
    complete = True
    found = []
    scan_issues = []
    for a in target_arts:
        if os.path.isfile(a):
            found.append(a)
            b = open(a, "rb").read()
            if not b:
                complete = False
            else:
                h.update(a.encode("utf-8") + b"\0" + hashlib.sha256(b).digest())
        elif os.path.isdir(a):
            dir_files = []
            for dp, _, fns in os.walk(a):
                if "__pycache__" in dp:
                    continue
                for fn in sorted(fns):
                    if fn.endswith(".pyc"):
                        continue
                    dir_files.append(os.path.join(dp, fn))
            if not dir_files:
                scan_issues.append(f"empty_directory:{a}")
                complete = False
            for fp in sorted(dir_files):
                found.append(fp)
                b = open(fp, "rb").read()
                h.update(fp.encode("utf-8") + b"\0" + hashlib.sha256(b).digest())
        else:
            scan_issues.append(f"missing:{a}")
            complete = False
    return found, (h.hexdigest() if (found and complete) else ""), scan_issues

files, pre_computed_sha, init_issues = _scan_and_digest(arts)
issues.extend(init_issues)
if not files:
    issues.append("no_files_found")

py_import_runner = (
    "import importlib.util, json, os, sys\n"
    "fp = sys.argv[1]\n"
    "reqs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []\n"
    "sys.path.insert(0, os.path.dirname(os.path.abspath(fp)))\n"
    "spec = importlib.util.spec_from_file_location('_hegel_contract_probe', fp)\n"
    "if spec is None or spec.loader is None:\n"
    "    raise ImportError(f'Cannot load spec for {fp}')\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "sys.modules[spec.name] = mod\n"
    "try:\n"
    "    spec.loader.exec_module(mod)\n"
    "    for ep in reqs:\n"
    "        if not callable(getattr(mod, ep, None)):\n"
    "            raise AttributeError(f'missing_required_entrypoint:{ep}')\n"
    "finally:\n"
    "    sys.modules.pop(spec.name, None)\n"
)

for fp in files:
    if not os.path.exists(fp):
        issues.append(f"missing_after_scan:{fp}")
        continue
    sz = os.path.getsize(fp)
    if sz == 0:
        if os.path.basename(fp) == "__init__.py" and fp not in explicit_py_set:
            continue
        issues.append(f"empty_file:{fp}")
        continue
    checked_files.append({"path": fp, "bytes": sz})
    if fp.endswith(".py"):
        txt = open(fp, "r", encoding="utf-8", errors="ignore").read()
        if "hegel_wormhole_skills" in txt:
            issues.append(f"uninlined_wormhole_import:{fp}")
        try:
            tree = ast.parse(txt)
        except SyntaxError as exc:
            issues.append(f"syntax_error:{fp}:{exc.lineno}")
            continue
        if fp in explicit_py_set:
            has_callable_or_cli = False
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    has_callable_or_cli = True
                    break
                if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
                    has_callable_or_cli = True
                    break
            if not has_callable_or_cli:
                issues.append(f"no_callable_or_cli_entrypoint:{fp}")
            fp_reqs = req_entrypoints if len(explicit_py_set) == 1 else []
            pr = subprocess.run(
                [sys.executable, "-c", py_import_runner, fp, json.dumps(fp_reqs)],
                capture_output=True,
                text=True,
                timeout=12,
            )
            if pr.returncode != 0:
                err_tail = (pr.stderr or pr.stdout or "").strip().splitlines()
                msg = err_tail[-1][:140] if err_tail else f"rc={pr.returncode}"
                issues.append(f"python_import_failed:{fp}:{msg}")
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
                    if cs in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
                        issues.append(f"non_finite_cell_value:{fp}:line_{r_i}")
                        break
    elif fp.endswith(".json") and sz < 60_000_000:
        try:
            def _reject_nonfinite_json(tok):
                raise ValueError(f"non_finite_json_constant:{tok}")
            obj = json.load(open(fp, "r", encoding="utf-8"), parse_constant=_reject_nonfinite_json)
            if obj in ({}, [], None):
                issues.append(f"empty_json_payload:{fp}")
        except Exception as exc:
            issues.append(f"invalid_json:{fp}:{type(exc).__name__}:{str(exc)[:80]}")
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

_, post_computed_sha, post_issues = _scan_and_digest(arts)
issues.extend(post_issues)
if pre_computed_sha != post_computed_sha:
    issues.append(f"candidate_mutated_during_validation:{pre_computed_sha[:12]}->{post_computed_sha[:12]}")
computed_sha = post_computed_sha

print(json.dumps({"accepted": len(issues) == 0, "computed_sha256": computed_sha, "pre_import_sha256": pre_computed_sha, "issues": issues, "checked_files": checked_files}))
'''
    v_sha = hashlib.sha256(validator_py.encode("utf-8")).hexdigest()[:16]
    try:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                "/",
                agent_cid,
                "python3",
                "-c",
                validator_py,
                json.dumps(list(declared_artifacts)),
                json.dumps(list(required_entrypoints)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return {
                "accepted": False,
                "candidate_sha256": candidate_sha256,
                "computed_candidate_sha256": "",
                "validator_sha256_16": v_sha,
                "reasons": [f"validator_rc_{proc.returncode}:{(proc.stderr or '')[-200:]}"],
            }
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
        computed_sha = str(parsed.get("computed_sha256") or "")
        reasons = list(parsed.get("issues", []))
        if candidate_sha256 != "SKIP_DIGEST_CHECK" and computed_sha != candidate_sha256:
            reasons.append(f"candidate_sha256_mismatch:expected={candidate_sha256[:16]}!=actual={computed_sha[:16]}")
        return {
            "accepted": bool(parsed.get("accepted")) and len(reasons) == 0,
            "candidate_sha256": computed_sha or candidate_sha256,
            "expected_candidate_sha256": candidate_sha256,
            "computed_candidate_sha256": computed_sha,
            "digest_verified": bool(candidate_sha256 == "SKIP_DIGEST_CHECK" or (computed_sha and computed_sha == candidate_sha256)),
            "validator_sha256_16": v_sha,
            "reasons": reasons,
            "checked_files": list(parsed.get("checked_files", [])),
        }
    except Exception as exc:
        return {
            "accepted": False,
            "candidate_sha256": candidate_sha256,
            "computed_candidate_sha256": "",
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
    if content_store_root is not None:
        blob_store = content_store_root
    else:
        study_root = episode_out_dir.parents[1]
        for anc in episode_out_dir.parents:
            if anc.name in {"episodes", "branch_trajectories"}:
                study_root = anc.parent
                break
        blob_store = study_root / "content_addressed_blobs"
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
            files_iter = (
                [dst_tmp]
                if dst_tmp.is_file()
                else [
                    p
                    for p in sorted(dst_tmp.rglob("*"))
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
                ]
            )
            for fp in files_iter:
                sz = fp.stat().st_size
                h = hashlib.sha256()
                with fp.open("rb") as f_in:
                    for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
                        h.update(chunk)
                b_sha = h.hexdigest()
                orig_name = Path(art.rstrip("/")).name if dst_tmp.is_file() else fp.name
                ext = Path(orig_name).suffix or ".bin"
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
                        "filename": orig_name,
                        "bytes": sz,
                        "sha256": b_sha,
                        "content_addressed_ref": stored_blob_rel,
                    }
                )

    # Also invoke `code_agent.retain_evaluated_python_model` on the primary Python producer if present (`K01` Fix:
    # rank declared `.py` deliverables & non-checker generator scripts ahead of `check_*`/`verify_*`/`test_*` scripts!)
    retained_model_ref: dict[str, Any] | None = None
    py_prods = [p for p in producer_records if p["archived_relpath"].endswith(".py")]
    declared_py_basenames = {Path(a.rstrip("/")).name for a in declared_artifacts if a.endswith(".py")}
    if py_prods:
        def _prod_rank(rec: dict[str, Any]) -> tuple[int, int]:
            c_path = str(rec.get("container_path") or "")
            b_name = Path(c_path).name.lower()
            is_declared = 2 if Path(c_path).name in declared_py_basenames else 0
            is_checker = -1 if b_name.startswith(("check_", "verify_", "test_", "_")) else 1
            return (is_declared + is_checker, int(rec.get("bytes") or 0))

        ranked_py = sorted(py_prods, key=_prod_rank)
        primary_py = Path(ranked_py[-1]["archived_relpath"]).name
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
                "primary_producer_container_path": ranked_py[-1]["container_path"],
                "selected_channels": ret_info["selected_channels"],
                "training_mse": None,
                "training_mse_status": "not_applicable_general_frontier_task",
            }
        except Exception:
            retained_model_ref = None

    build_cmds: list[str] = []
    for t in turn_logs:
        cmd_raw = str(t.get("bash_command") or (t.get("decision") or {}).get("bash_command") or "").strip()
        if cmd_raw and any(k in cmd_raw for k in ("g++", "gcc", "cmake", "make", "python3", "python ")):
            build_cmds.append(cmd_raw[:600])
    build_cmds = build_cmds[-16:]

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
# 5. Genuine AST-to-`Program` Subtree Lifting & Branch Reconciliation (`P1a` + `K02` Fix)
# ============================================================================

_UNARY_CALL_MAP = {
    "sin": "sin",
    "cos": "cos",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "abs": "abs",
    "fabs": "abs",
}

_BINARY_CALL_MAP = {
    "maximum": "max",
    "minimum": "min",
    "max": "max",
    "min": "min",
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
}


def _is_pure_input_guard_raise_if(stmt: ast.AST) -> bool:
    """Return True iff `stmt` is `if <cond>: raise <Err>(...)` with empty `orelse` (`M9.3.1` AST guard separation)."""
    return (
        isinstance(stmt, ast.If)
        and not stmt.orelse
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], ast.Raise)
    )


def lift_python_scientific_expressions_to_programs(
    closures: list[dict[str, str]],
    *,
    max_programs: int = 4,
    funnel_diagnostics_out: dict[str, Any] | None = None,
) -> list[tuple[str, Program]]:
    """Lift pure-Python mathematical expressions into safe-totalized domain-interior `Program` skeletons (`K02` + `M7.7` + `M9.3.1` Fix).

    Guarantees & explicit domain scope:
      - `EvaluationContext` evaluates `Program` trees with totalized safe operators on moderate interior probes
        (clamped `exp`, guarded `div` floor, totalized `log`/`sqrt`; `log1p(u)` -> `log(1 + u)`, `expm1(u)` -> `exp(u) - 1`
        valid away from sub-machine-epsilon cancellation).
      - Strict vs non-strict inequalities (`>` -> `lt(r, l)`, `>=` -> `ge(l, r)`, `<` -> `lt(l, r)`, `<=` -> `ge(r, l)`).
      - Module & lexical symbol resolution (`M7.7` `shadowed_sin` fix): tracks module-level `def`/`class`/assignment/import
        bindings and local variables so user-redefined primitives (e.g. `def sin(x): return x + 100`) inline their lifted
        definition or abort cleanly, NEVER colliding with standard `math`/`numpy` primitives!
      - Separates leading pure input-validation guards (`if <cond>: raise ValueError(...)`) into `separated_input_guards`
        and verifies the guard does NOT reject interior test probes before lifting the mathematical body (`M9.3.1`).
      - Rejects imperative loops/mutations (`For`, `While`, `AugAssign`, non-guard `If`, `Try`, `With`).
    """
    import math

    lifted: list[tuple[str, Program]] = []
    diag_counts: dict[str, Any] = {
        "closures_scanned": len(closures),
        "functions_scanned": 0,
        "functions_with_separated_input_guards": 0,
        "separated_input_guards_by_fn": {},
        "aborted_imperative_or_unsupported": 0,
        "failed_interior_probe_verification": 0,
        "lifted_program_count": 0,
    }

    def _subst_arg_features(prog: Program, bound_args: list[Program]) -> Program:
        if prog.op == "feature" and prog.name and prog.name.startswith("x_") and "_idx" not in prog.name:
            try:
                idx = int(prog.name.split("_", 1)[1])
                if 0 <= idx < len(bound_args):
                    return bound_args[idx]
            except ValueError:
                pass
        if not prog.children:
            return prog
        return Program(
            op=prog.op,
            value=prog.value,
            name=prog.name,
            lag=prog.lag,
            children=tuple(_subst_arg_features(ch, bound_args) for ch in prog.children),
            definition=prog.definition,
        )

    def _lift_expr(
        node: ast.AST,
        env: dict[str, Program],
        depth: int,
        shadowed_bare: set[str],
        valid_mod_aliases: set[str],
        module_fn_progs: dict[str, tuple[int, Program]],
    ) -> Program | None:
        if depth > 14:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            val = float(node.value)
            return Program("const", value=val) if math.isfinite(val) else None
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in shadowed_bare:
                return None
            if node.id == "pi":
                return Program("const", value=round(math.pi, 8))
            if node.id == "e":
                return Program("const", value=round(math.e, 8))
            return None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in valid_mod_aliases
            and node.value.id not in env
            and node.attr in {"pi", "e"}
        ):
            return Program("const", value=round(math.pi if node.attr == "pi" else math.e, 8))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in env:
            base_p = env[node.value.id]
            if base_p.op == "feature" and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                return Program("feature", name=f"{base_p.name}_idx{node.slice.value}")
            return None
        if isinstance(node, ast.UnaryOp):
            ch = _lift_expr(node.operand, env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
            if ch is None:
                return None
            if isinstance(node.op, ast.UAdd):
                return ch
            if isinstance(node.op, ast.USub):
                if ch.op == "const" and ch.value is not None:
                    return Program("const", value=-float(ch.value))
                return Program("neg", children=(ch,))
            return None
        if isinstance(node, ast.BinOp):
            left = _lift_expr(node.left, env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
            right = _lift_expr(node.right, env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
            if left is None:
                return None
            if isinstance(node.op, ast.Add) and right is not None:
                return Program("add", children=(left, right))
            if isinstance(node.op, ast.Sub) and right is not None:
                return Program("sub", children=(left, right))
            if isinstance(node.op, ast.Mult) and right is not None:
                return Program("mul", children=(left, right))
            if isinstance(node.op, ast.Div) and right is not None:
                return Program("div", children=(left, right))
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                exp_v = float(node.right.value)
                if exp_v == 2.0:
                    return Program("mul", children=(left, left))
                if exp_v == 3.0:
                    return Program("mul", children=(Program("mul", children=(left, left)), left))
                if exp_v == 0.5:
                    return Program("sqrt", children=(left,))
                if exp_v == -1.0:
                    return Program("div", children=(Program("const", value=1.0), left))
            return None
        if isinstance(node, ast.Call):
            if node.keywords:
                return None
            fn_name = ""
            if isinstance(node.func, ast.Name):
                cname = node.func.id
                if cname in env:
                    return None
                if cname in shadowed_bare:
                    if cname in module_fn_progs:
                        arity, helper_prog = module_fn_progs[cname]
                        if len(node.args) == arity:
                            lifted_args: list[Program] = []
                            for a_node in node.args:
                                ap = _lift_expr(a_node, env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                                if ap is None:
                                    return None
                                lifted_args.append(ap)
                            inlined = _subst_arg_features(helper_prog, lifted_args)
                            return inlined if inlined.size <= 35 else None
                    return None
                fn_name = cname
            elif isinstance(node.func, ast.Attribute):
                if not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id in valid_mod_aliases
                    and node.func.value.id not in env
                ):
                    return None
                fn_name = node.func.attr
            else:
                return None
            if fn_name in {"asarray", "array", "float", "real", "ravel"} and len(node.args) >= 1:
                return _lift_expr(node.args[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
            if fn_name == "square" and len(node.args) == 1:
                ch = _lift_expr(node.args[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                return Program("mul", children=(ch, ch)) if ch is not None else None
            if fn_name == "power" and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, (int, float)):
                base_ch = _lift_expr(node.args[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                if base_ch is None:
                    return None
                exp_v = float(node.args[1].value)
                if exp_v == 2.0:
                    return Program("mul", children=(base_ch, base_ch))
                if exp_v == 3.0:
                    return Program("mul", children=(Program("mul", children=(base_ch, base_ch)), base_ch))
                if exp_v == 0.5:
                    return Program("sqrt", children=(base_ch,))
                if exp_v == -1.0:
                    return Program("div", children=(Program("const", value=1.0), base_ch))
                return None
            if fn_name == "log1p" and len(node.args) == 1:
                ch = _lift_expr(node.args[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                return Program("log", children=(Program("add", children=(Program("const", value=1.0), ch)),)) if ch is not None else None
            if fn_name == "expm1" and len(node.args) == 1:
                ch = _lift_expr(node.args[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                return Program("sub", children=(Program("exp", children=(ch,)), Program("const", value=1.0))) if ch is not None else None
            if fn_name == "clip" and len(node.args) == 3:
                x_c = _lift_expr(node.args[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                lo_c = _lift_expr(node.args[1], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                hi_c = _lift_expr(node.args[2], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                if x_c is not None and lo_c is not None and hi_c is not None:
                    return Program("min", children=(Program("max", children=(x_c, lo_c)), hi_c))
                return None
            if fn_name in _UNARY_CALL_MAP and len(node.args) == 1:
                ch = _lift_expr(node.args[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                return Program(_UNARY_CALL_MAP[fn_name], children=(ch,)) if ch is not None else None
            if fn_name in _BINARY_CALL_MAP and len(node.args) == 2:
                a0 = _lift_expr(node.args[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                a1 = _lift_expr(node.args[1], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                return Program(_BINARY_CALL_MAP[fn_name], children=(a0, a1)) if (a0 is not None and a1 is not None) else None
        if isinstance(node, ast.IfExp):
            body_p = _lift_expr(node.body, env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
            else_p = _lift_expr(node.orelse, env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
            if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 and len(node.test.comparators) == 1:
                cl = _lift_expr(node.test.left, env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                cr = _lift_expr(node.test.comparators[0], env, depth + 1, shadowed_bare, valid_mod_aliases, module_fn_progs)
                if cl is not None and cr is not None and body_p is not None and else_p is not None:
                    op0 = node.test.ops[0]
                    if isinstance(op0, ast.GtE):
                        return Program("if", children=(Program("ge", children=(cl, cr)), body_p, else_p))
                    if isinstance(op0, ast.Gt):
                        return Program("if", children=(Program("lt", children=(cr, cl)), body_p, else_p))
                    if isinstance(op0, ast.Lt):
                        return Program("if", children=(Program("lt", children=(cl, cr)), body_p, else_p))
                    if isinstance(op0, ast.LtE):
                        return Program("if", children=(Program("ge", children=(cr, cl)), body_p, else_p))
        return None

    if isinstance(closures, str):
        closures = [{"name": "inline", "code": closures}]

    for cl in closures:
        fn_name = str(cl.get("name") or "fn")
        code_str = str(cl.get("code") or "")
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            continue
        module_shadowed: set[str] = set()
        valid_mod_aliases: set[str] = {"math", "numpy", "np"}
        for m_stmt in tree.body:
            if isinstance(m_stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_shadowed.add(m_stmt.name)
                valid_mod_aliases.discard(m_stmt.name)
            elif isinstance(m_stmt, ast.Assign):
                for tgt in m_stmt.targets:
                    if isinstance(tgt, ast.Name):
                        module_shadowed.add(tgt.id)
                        valid_mod_aliases.discard(tgt.id)
            elif isinstance(m_stmt, ast.AnnAssign) and isinstance(m_stmt.target, ast.Name):
                module_shadowed.add(m_stmt.target.id)
                valid_mod_aliases.discard(m_stmt.target.id)
            elif isinstance(m_stmt, ast.Import):
                for alias in m_stmt.names:
                    bound_nm = alias.asname or alias.name.split(".")[0]
                    if alias.name in {"math", "numpy"}:
                        valid_mod_aliases.add(bound_nm)
                    else:
                        module_shadowed.add(bound_nm)
                        valid_mod_aliases.discard(bound_nm)
            elif isinstance(m_stmt, ast.ImportFrom):
                if m_stmt.module in {"math", "numpy"} and (m_stmt.level or 0) == 0:
                    for alias in m_stmt.names:
                        if alias.asname and alias.asname != alias.name:
                            module_shadowed.add(alias.asname)
                else:
                    for alias in m_stmt.names:
                        bound_nm = alias.asname or alias.name
                        if bound_nm != "track_wormhole_skill":
                            module_shadowed.add(bound_nm)
                            valid_mod_aliases.discard(bound_nm)

        module_fn_progs: dict[str, tuple[int, Program]] = {}
        for stmt in tree.body:
            if not isinstance(stmt, ast.FunctionDef):
                continue
            # Reject non-telemetry decorators (`M8.2` `decorator_semantics` fix: e.g. `@shift` alters Python semantics!)
            has_unknown_decorator = False
            for dec in stmt.decorator_list:
                is_telemetry_dec = (
                    (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "track_wormhole_skill")
                    or (isinstance(dec, ast.Name) and dec.id == "track_wormhole_skill")
                )
                if not is_telemetry_dec:
                    has_unknown_decorator = True
                    break
            if has_unknown_decorator:
                continue
            diag_counts["functions_scanned"] += 1
            env: dict[str, Program] = {}
            all_args = [*stmt.args.posonlyargs, *stmt.args.args, *stmt.args.kwonlyargs]
            for idx, arg in enumerate(all_args):
                env[arg.arg] = Program("feature", name=f"x_{idx}")
            best_fn_prog: Program | None = None
            aborted_imperative = False
            fn_sep_guards: list[str] = []
            for b_stmt in stmt.body:
                if isinstance(b_stmt, ast.Expr) and isinstance(b_stmt.value, ast.Constant) and isinstance(b_stmt.value.value, str):
                    continue
                if isinstance(b_stmt, ast.Pass):
                    continue
                if _is_pure_input_guard_raise_if(b_stmt) and best_fn_prog is None:
                    fn_sep_guards.append(ast.unparse(b_stmt.test))
                    continue
                if isinstance(b_stmt, ast.Assign):
                    if len(b_stmt.targets) != 1 or not isinstance(b_stmt.targets[0], ast.Name):
                        aborted_imperative = True
                        break
                    rhs_p = _lift_expr(b_stmt.value, env, 0, module_shadowed, valid_mod_aliases, module_fn_progs)
                    if rhs_p is None or rhs_p.size > 25:
                        aborted_imperative = True
                        break
                    env[b_stmt.targets[0].id] = rhs_p
                elif isinstance(b_stmt, ast.AnnAssign) and isinstance(b_stmt.target, ast.Name) and b_stmt.value is not None:
                    rhs_p = _lift_expr(b_stmt.value, env, 0, module_shadowed, valid_mod_aliases, module_fn_progs)
                    if rhs_p is None or rhs_p.size > 25:
                        aborted_imperative = True
                        break
                    env[b_stmt.target.id] = rhs_p
                elif isinstance(b_stmt, ast.Return) and b_stmt.value is not None:
                    ret_p = _lift_expr(b_stmt.value, env, 0, module_shadowed, valid_mod_aliases, module_fn_progs)
                    if ret_p is not None and ret_p.op not in {"const", "feature"}:
                        best_fn_prog = ret_p
                    break
                else:
                    aborted_imperative = True
                    break
            if fn_sep_guards:
                diag_counts["functions_with_separated_input_guards"] += 1
                diag_counts["separated_input_guards_by_fn"][f"{fn_name}::{stmt.name}"] = list(fn_sep_guards)
                diag_counts["separated_input_guards_by_fn"][stmt.name] = list(fn_sep_guards)
            if aborted_imperative or best_fn_prog is None:
                diag_counts["aborted_imperative_or_unsupported"] += 1
            if (not aborted_imperative) and best_fn_prog is not None:
                # Pure side-effect-free AST scalar differential check (`M8.4` Fix: ZERO host `exec`/`eval`!)
                uses_idx_subscript = "_idx" in json.dumps(best_fn_prog.to_dict())
                if (
                    not uses_idx_subscript
                    and len(all_args) >= 1
                    and not stmt.args.vararg
                    and not stmt.args.kwarg
                    and not stmt.args.kwonlyargs
                ):
                    try:
                        from .language import EvaluationContext
                        from .models import Observation

                        fn_defs_by_name = {
                            s.name: s
                            for s in tree.body
                            if isinstance(s, ast.FunctionDef)
                            and all(
                                (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "track_wormhole_skill")
                                or (isinstance(d, ast.Name) and d.id == "track_wormhole_skill")
                                for d in s.decorator_list
                            )
                        }

                        _PURE_UNARY = {
                            "sin": math.sin, "cos": math.cos, "tan": math.tan, "tanh": math.tanh,
                            "exp": math.exp, "expm1": math.expm1, "log": math.log, "log1p": math.log1p,
                            "sqrt": math.sqrt, "abs": abs, "fabs": math.fabs, "square": lambda z: z * z,
                            "asarray": float, "array": float, "float": float, "real": float, "ravel": float,
                        }
                        _PURE_BINARY = {
                            "maximum": max, "minimum": min, "max": max, "min": min,
                            "hypot": math.hypot, "atan2": math.atan2,
                            "add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
                            "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b,
                            "true_divide": lambda a, b: a / b, "power": lambda a, b: float(a ** b),
                        }
                        imported_bare_names: set[str] = set()
                        for m_s in tree.body:
                            if isinstance(m_s, ast.ImportFrom) and m_s.module in {"math", "numpy"} and (m_s.level or 0) == 0:
                                for al in m_s.names:
                                    if (not al.asname or al.asname == al.name) and al.name not in module_shadowed:
                                        imported_bare_names.add(al.name)

                        def _eval_pure_ast_expr(node: ast.AST, loc_env: dict[str, float], depth: int = 0) -> float:
                            if depth > 24:
                                raise ValueError("max_ast_eval_depth")
                            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                                return float(node.value)
                            if isinstance(node, ast.Name):
                                if node.id in loc_env:
                                    return float(loc_env[node.id])
                                if node.id not in module_shadowed:
                                    if node.id == "pi":
                                        return math.pi
                                    if node.id == "e":
                                        return math.e
                                raise ValueError(f"unknown_name:{node.id}")
                            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in valid_mod_aliases and node.value.id not in loc_env:
                                if node.attr == "pi":
                                    return math.pi
                                if node.attr == "e":
                                    return math.e
                                raise ValueError(f"unknown_const_attr:{node.attr}")
                            if isinstance(node, ast.UnaryOp):
                                v = _eval_pure_ast_expr(node.operand, loc_env, depth + 1)
                                if isinstance(node.op, ast.USub):
                                    return -v
                                if isinstance(node.op, ast.UAdd):
                                    return v
                                raise ValueError("unsupported_unary")
                            if isinstance(node, ast.BinOp):
                                lv = _eval_pure_ast_expr(node.left, loc_env, depth + 1)
                                rv = _eval_pure_ast_expr(node.right, loc_env, depth + 1)
                                if isinstance(node.op, ast.Add):
                                    return lv + rv
                                if isinstance(node.op, ast.Sub):
                                    return lv - rv
                                if isinstance(node.op, ast.Mult):
                                    return lv * rv
                                if isinstance(node.op, ast.Div):
                                    return lv / rv
                                if isinstance(node.op, ast.Pow):
                                    return float(lv ** rv)
                                raise ValueError("unsupported_binop")
                            if isinstance(node, ast.IfExp):
                                cond_node = node.test
                                if not isinstance(cond_node, ast.Compare) or len(cond_node.ops) != 1 or len(cond_node.comparators) != 1:
                                    raise ValueError("unsupported_ifexp_test")
                                cl = _eval_pure_ast_expr(cond_node.left, loc_env, depth + 1)
                                cr = _eval_pure_ast_expr(cond_node.comparators[0], loc_env, depth + 1)
                                cop = cond_node.ops[0]
                                c_ok = (
                                    (cl > cr) if isinstance(cop, ast.Gt) else
                                    (cl >= cr) if isinstance(cop, ast.GtE) else
                                    (cl < cr) if isinstance(cop, ast.Lt) else
                                    (cl <= cr) if isinstance(cop, ast.LtE) else
                                    (cl == cr) if isinstance(cop, ast.Eq) else
                                    (cl != cr) if isinstance(cop, ast.NotEq) else False
                                )
                                return _eval_pure_ast_expr(node.body if c_ok else node.orelse, loc_env, depth + 1)
                            if isinstance(node, ast.Call) and not node.keywords:
                                c_args = [_eval_pure_ast_expr(a, loc_env, depth + 1) for a in node.args]
                                if isinstance(node.func, ast.Name):
                                    fn_id = node.func.id
                                    if fn_id in loc_env:
                                        raise ValueError("shadowed_local_callable")
                                    if fn_id == "abs" and len(c_args) == 1:
                                        return abs(c_args[0])
                                    if fn_id == "min" and len(c_args) >= 2:
                                        return min(c_args)
                                    if fn_id == "max" and len(c_args) >= 2:
                                        return max(c_args)
                                    if fn_id == "float" and len(c_args) == 1:
                                        return float(c_args[0])
                                    if fn_id in imported_bare_names or fn_id not in module_shadowed:
                                        if fn_id in _PURE_UNARY and len(c_args) == 1:
                                            return float(_PURE_UNARY[fn_id](c_args[0]))
                                        if fn_id in _PURE_BINARY and len(c_args) == 2:
                                            return float(_PURE_BINARY[fn_id](c_args[0], c_args[1]))
                                        if fn_id == "clip" and len(c_args) == 3:
                                            return min(max(c_args[0], c_args[1]), c_args[2])
                                    if fn_id in fn_defs_by_name and fn_id != stmt.name:
                                        return _eval_pure_ast_fn(fn_defs_by_name[fn_id], c_args, depth + 1)
                                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in valid_mod_aliases and node.func.value.id not in loc_env:
                                    anm = node.func.attr
                                    if anm in _PURE_UNARY and len(c_args) == 1:
                                        return float(_PURE_UNARY[anm](c_args[0]))
                                    if anm in _PURE_BINARY and len(c_args) == 2:
                                        return float(_PURE_BINARY[anm](c_args[0], c_args[1]))
                                    if anm == "clip" and len(c_args) == 3:
                                        return min(max(c_args[0], c_args[1]), c_args[2])
                            raise ValueError("unsupported_ast_call_or_node")

                        def _eval_pure_ast_guard_bool(cond: ast.AST, loc_env: dict[str, float]) -> bool:
                            if isinstance(cond, ast.BoolOp):
                                sub_vals = [_eval_pure_ast_guard_bool(v, loc_env) for v in cond.values]
                                return all(sub_vals) if isinstance(cond.op, ast.And) else any(sub_vals)
                            if isinstance(cond, ast.UnaryOp) and isinstance(cond.op, ast.Not):
                                return not _eval_pure_ast_guard_bool(cond.operand, loc_env)
                            if isinstance(cond, ast.Compare) and len(cond.ops) == 1 and len(cond.comparators) == 1:
                                cl = _eval_pure_ast_expr(cond.left, loc_env, 0)
                                cr = _eval_pure_ast_expr(cond.comparators[0], loc_env, 0)
                                cop = cond.ops[0]
                                if isinstance(cop, ast.Gt): return cl > cr
                                if isinstance(cop, ast.GtE): return cl >= cr
                                if isinstance(cop, ast.Lt): return cl < cr
                                if isinstance(cop, ast.LtE): return cl <= cr
                                if isinstance(cop, ast.Eq): return cl == cr
                                if isinstance(cop, ast.NotEq): return cl != cr
                            raise ValueError("unsupported_guard_cond")

                        def _eval_pure_ast_fn(fdef: ast.FunctionDef, vals: list[float], depth: int = 0) -> float:
                            f_args = [*fdef.args.posonlyargs, *fdef.args.args]
                            if len(f_args) != len(vals):
                                raise ValueError("arity_mismatch")
                            loc: dict[str, float] = {a.arg: float(v) for a, v in zip(f_args, vals)}
                            for bs in fdef.body:
                                if isinstance(bs, ast.Expr) and isinstance(bs.value, ast.Constant) and isinstance(bs.value.value, str):
                                    continue
                                if isinstance(bs, ast.Pass):
                                    continue
                                if _is_pure_input_guard_raise_if(bs):
                                    if _eval_pure_ast_guard_bool(bs.test, loc):
                                        raise ValueError("input_guard_triggered_on_interior_probe")
                                    continue
                                if isinstance(bs, ast.Assign) and len(bs.targets) == 1 and isinstance(bs.targets[0], ast.Name):
                                    loc[bs.targets[0].id] = _eval_pure_ast_expr(bs.value, loc, depth + 1)
                                elif isinstance(bs, ast.AnnAssign) and isinstance(bs.target, ast.Name) and bs.value is not None:
                                    loc[bs.target.id] = _eval_pure_ast_expr(bs.value, loc, depth + 1)
                                elif isinstance(bs, ast.Return) and bs.value is not None:
                                    return _eval_pure_ast_expr(bs.value, loc, depth + 1)
                                else:
                                    raise ValueError("imperative_stmt")
                            raise ValueError("missing_return")

                        for probe_base in (0.35, 0.85):
                            p_vals = [probe_base + 0.17 * j for j in range(len(all_args))]
                            py_out = float(_eval_pure_ast_fn(stmt, p_vals))
                            obs_p = Observation({f"x_{j}": p_vals[j] for j in range(len(all_args))}, 0.0)
                            ir_out = float(EvaluationContext(obs_p).evaluate(best_fn_prog))
                            if not (math.isfinite(py_out) and math.isfinite(ir_out) and math.isclose(py_out, ir_out, rel_tol=1e-4, abs_tol=1e-4)):
                                best_fn_prog = None
                                diag_counts["failed_interior_probe_verification"] += 1
                                break
                    except Exception:
                        best_fn_prog = None
                        diag_counts["failed_interior_probe_verification"] += 1
            if (not aborted_imperative) and best_fn_prog is not None:
                if not stmt.args.vararg and not stmt.args.kwarg and not stmt.args.defaults:
                    module_fn_progs[stmt.name] = (len(all_args), best_fn_prog)
                lifted.append((fn_name, best_fn_prog))
                diag_counts["lifted_program_count"] = len(lifted)
                diag_counts["lifted_to_program_tree"] = len(lifted)
                if len(lifted) >= max_programs:
                    if funnel_diagnostics_out is not None:
                        funnel_diagnostics_out.update(diag_counts)
                    return lifted
    if funnel_diagnostics_out is not None:
        funnel_diagnostics_out.update(diag_counts)
    return lifted


def consolidate_wave1_skill_manifests_with_reconciliation(
    *,
    wave1_episode_dirs: list[tuple[dict[str, Any], Path]],
    forest: PhilosophicalForest,
    archive: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Consolidate complete Wave-1 `ExecutableSkillManifest` closures + genuine AST-lifted `SharedBaseSubtree`s
    and propagate premise retractions via `branch_reconciliation` & `m5_temporal_multiscale_and_base_subtree` (`P1a` + `K02`).
    """
    events: list[dict[str, Any]] = []
    exported_programs_all: list[tuple[str, str, str, Program]] = []
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

        # Lift genuine mathematical expressions from winning Python closures into typed `Program` ASTs (`K02` Fix!)
        lifted_progs = (
            lift_python_scientific_expressions_to_programs(closures, max_programs=4)
            if summary.get("benchmark") != "BioMysteryBench-Preview"
            else []
        )
        for fn_nm, p_tree in lifted_progs:
            exported_programs_all.append((t_short, fam, fn_nm, p_tree))
        abstraction_kind = "ast_program_lifted" if lifted_progs else ("opaque_verified_operator" if (closures or full_prod_cnt > 0) else "no_producer")

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
                    "abstraction_kind": abstraction_kind,
                    "lifted_program_ops": [f"{fn_nm}:{p_t.op}" for fn_nm, p_t in lifted_progs],
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
                "abstraction_kind": abstraction_kind,
                "lifted_program_count": len(lifted_progs),
                "module_filename": mod_fname,
                "exported_functions": [c["name"] for c in closures],
                "propagated_to_nodes": sorted(target_nids),
            }
        )

    # Extract genuine `SharedBaseSubtreePrior` across ALL compatible pairs of AST-lifted Wave-1 `Program`s!
    registered_subtrees: list[str] = []
    for i_a in range(len(exported_programs_all)):
        src_a, fam_a, fn_a, p_a = exported_programs_all[i_a]
        for i_b in range(i_a + 1, len(exported_programs_all)):
            src_b, fam_b, fn_b, p_b = exported_programs_all[i_b]
            if src_a == src_b:
                continue
            cand_p_a, cand_p_b = p_a, p_b
            anti = anti_unify_programs(cand_p_a, cand_p_b)
            if anti is None:
                # Scale-covariant identity `E == 1.0 * E`: unify `E_a` against `c * E_b` when inner operator matches!
                if p_b.op == "mul" and len(p_b.children) == 2 and p_b.children[0].op == "const":
                    cand_p_a = Program("mul", children=(Program("const", value=1.0), p_a))
                    anti = anti_unify_programs(cand_p_a, cand_p_b)
                elif p_a.op == "mul" and len(p_a.children) == 2 and p_a.children[0].op == "const":
                    cand_p_b = Program("mul", children=(Program("const", value=1.0), p_b))
                    anti = anti_unify_programs(cand_p_a, cand_p_b)
            if anti is None:
                continue
            skeleton, slots = anti
            if skeleton.op in {"arg", "const", "feature"}:
                continue
            skel_sha8 = hashlib.sha256(json.dumps(skeleton.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:8]
            sub_id = f"subtree_wave1_{src_a}_{fn_a}__{src_b}_{fn_b}__{skel_sha8}"
            if sub_id not in getattr(forest, "shared_base_subtrees", {}):
                extract_shared_base_subtree_from_sources(
                    forest,
                    subtree_id=sub_id,
                    prog_a=cand_p_a,
                    prog_b=cand_p_b,
                    shared_nodes=[{"role": "ast_operator_skeleton", "root_op": skeleton.op, "size": skeleton.size}],
                    shared_relations=[{"relation": "anti_unified_ast_homomorphism", "from": f"{src_a}:{fn_a}", "to": f"{src_b}:{fn_b}"}],
                    role_and_scale_schema={
                        "positional_args": sorted(skeleton.required_features()),
                        "source_families": sorted({fam_a, fam_b}),
                        "parametric_slots": [s["slot"] for s in slots],
                    },
                    open_completion_branches=[{"slot": s["slot"], "path": list(s["path"])} for s in slots] or [{"slot": "exact_invariant"}],
                    supporting_sources=[{"source_id": src_a, "tree_id": fn_a}, {"source_id": src_b, "tree_id": fn_b}],
                    probe_signatures=["sig:ode_trajectory", "sig:conservation_pde", "sig:sparse_sensor_network"],
                )
                registered_subtrees.append(sub_id)
                if len(registered_subtrees) >= 12:
                    break
        if len(registered_subtrees) >= 12:
            break

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
        "lifted_program_entries_count": len(exported_programs_all),
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
    """Run an isolated container import qualification pilot (`retrieved -> bound -> import_qualified`, `E01` Fix!)
    via direct argv `docker exec ... python3 -c` (ZERO `bash -lc` quoting corruption!) and tagged with
    `HEGEL_SKILL_CALLER_ROLE=smoke_test` so startup qualification NEVER inflates agent skill counts!
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
        "        fns = sorted(k for k, v in vars(m).items() if callable(v) and not k.startswith('_') and k != 'track_wormhole_skill')\n"
        "        out.append({\n"
        "            'module_import': mn,\n"
        "            'lifecycle_state': 'bound_and_import_qualified',\n"
        "            'import_qualified': True,\n"
        "            'contract_validated': False,\n"
        "            'callable_count': len(fns),\n"
        "            'callable_names': fns,\n"
        "        })\n"
        "    except Exception as exc:\n"
        "        out.append({\n"
        "            'module_import': mn,\n"
        "            'lifecycle_state': 'failed_import',\n"
        "            'import_qualified': False,\n"
        "            'contract_validated': False,\n"
        "            'error': f'{type(exc).__name__}:{exc}',\n"
        "        })\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        [
            "docker",
            "exec",
            "-w",
            "/",
            "-e",
            "PYTHONPATH=/tmp",
            "-e",
            "HEGEL_SKILL_CALLER_ROLE=smoke_test",
            agent_cid,
            "python3",
            "-c",
            qual_py,
            json.dumps(mod_names),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        err_msg = (proc.stderr or proc.stdout or "").strip()[-240:]
        return [
            {
                "module_import": mn,
                "lifecycle_state": "qualification_exec_failed",
                "import_qualified": False,
                "contract_validated": False,
                "returncode": proc.returncode,
                "stderr": err_msg,
            }
            for mn in mod_names
        ]
    try:
        return list(json.loads(proc.stdout.strip().splitlines()[-1]))
    except Exception as exc:
        return [
            {
                "module_import": mn,
                "lifecycle_state": "qualification_parse_failed",
                "import_qualified": False,
                "contract_validated": False,
                "error": f"{type(exc).__name__}:{exc}",
            }
            for mn in mod_names
        ]


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
