"""Read-only deterministic controls for M6.7 library contract claims.

No repository edits, no model calls, no Docker, no hidden benchmark data.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

BASE = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("HEGEL_AUDIT_REPO", os.environ.get("HEGEL_M67_REPO", BASE / "repo_m691_4001bef56")))
HM = REPO if REPO.name == "Hegel Machine" else REPO / "Hegel Machine"
SRC = HM / "src"
sys.path.insert(0, str(SRC))
from hegel_machine.cognition import m6_4_executable_wormhole_library as lib


def module_namespace(code: str) -> dict:
    code = code.replace("from . import track_wormhole_skill", "")
    ns = {"track_wormhole_skill": lambda _: lambda fn: fn}
    exec(compile(code, "<saved-library-module>", "exec"), ns)
    return ns


out = {}
extract = lib.extract_pure_reusable_functions_from_python
cases = {
    "helper_call_graph": '''def helper(x):
    """A pure reusable mathematical primitive."""
    return np.asarray(x, dtype=float) * 2.0
def complete_operator(x):
    """The complete pure operator consumes that primitive."""
    return helper(x) + 1.0
''',
    "nested_pure_function": '''def outer_operator(x):
    """A pure composite operator containing a local function."""
    def inner_scale(y):
        return y * 2.0
    return inner_scale(x) + 1.0
''',
    "nested_scope_unbound_accepted": '''def broken_operator(x):
    """Inner bindings must not authorize outer free names."""
    def inner():
        hidden = 1.0
        return hidden
    return x + hidden
''',
    "file_io_accepted": '''def numeric_reader(path):
    """This is ordinary file I/O, despite the pure-function claim."""
    return np.loadtxt(path, delimiter=',', ndmin=2)
''',
}
for label, code in cases.items():
    funcs = extract(code)
    out[label] = {"extracted": [name for name, _ in funcs]}
    if label == "nested_scope_unbound_accepted" and funcs:
        ns = {"np": np}
        exec(funcs[0][1], ns)
        try:
            ns[funcs[0][0]](2.0)
        except Exception as exc:
            out[label]["runtime_exception"] = type(exc).__name__ + ": " + str(exc)

relay = module_namespace(lib.SKILL_02_CONSERVATION_RELAY)
out["nan_parity"] = relay["verify_vectorized_vs_scalar_recurrence"](
    lambda x: np.array([np.nan]), lambda x: np.array([1.0]), [(0,)]
)
profiles = np.array([[-5.0], [0.0], [1.0], [10.0]])
sel = relay["minimax_consensus_candidate_selector"](list(range(4)), lambda _: profiles)
worst_pair_disagreement = np.max(np.abs(profiles[:, None, :] - profiles[None, :, :]), axis=(1, 2))
out["consensus_not_minimax"] = {
    "profiles": profiles.ravel().tolist(),
    "library_selected_index": sel["selected_index"],
    "worst_disagreement_by_candidate": worst_pair_disagreement.tolist(),
    "actual_minimax_index": int(np.argmin(worst_pair_disagreement)),
}

cohort = module_namespace(lib.SKILL_04_COHORT_TRANSFER)
callback_seen = []
def check_y(Xtr, ytr, Xte, ctr, cte):
    callback_seen.append(sorted(set(int(v) for v in ytr)))
    return np.full(len(Xte), 0.5)
cohort["leave_one_cohort_out_calibrated_cv"](
    np.arange(24.0).reshape(12, 2),
    np.array([0, 1, 0, 1, 0, 1, 0, 1, -1, -1, -1, -1]),
    np.repeat(["a", "b", "unlabeled"], 4),
    check_y,
)
out["loco_unlabeled_rows_in_training"] = callback_seen

fourier = module_namespace(lib.SKILL_03_ASSIMILATION_CERT)
B, _ = fourier["real_fourier_smoothness_basis"](16, 8)
out["fourier_nyquist_no_guard"] = {
    "shape": list(B.shape), "rank": int(np.linalg.matrix_rank(B)),
    "max_gram_identity_error": float(np.max(np.abs(B.T @ B - np.eye(B.shape[1])))),
}

wave = module_namespace(lib.SKILL_05_WAVE_IMAGING)
bad_common_peak = np.tile([0.0, 1.0], (8, 1))
out["lopo_common_bias_invisible"] = wave["leave_one_pair_out_grid_stability"](
    bad_common_peak, np.array([[0., 0.], [10., 10.]])
)
out["lopo_common_bias_invisible"]["note"] = "All eight rays share a biased peak; stability is zero despite a hypothetically wrong location. This is an identifiability counterexample, not a new benchmark result."

with tempfile.TemporaryDirectory(prefix="m67_telemetry_controls_") as td:
    td = Path(td)
    log = td / "calls.jsonl"
    src = lib.SKILL_INIT_PY.replace('"/tmp/.hegel_skill_activations.jsonl"', repr(str(log)))
    ns = {}
    exec(src, ns)
    def always_fails():
        raise ValueError("intended control failure")
    wrapped = ns["track_wormhole_skill"]("control")(always_fails)
    try:
        wrapped()
    except ValueError:
        pass
    recs = [json.loads(s) for s in log.read_text().splitlines()]
    out["failed_function_counted_activation"] = {
        "records": recs,
        "count": lib.sum_skill_activation_calls(recs),
        "has_return_or_error_status": any("status" in r or "exception" in r for r in recs),
    }
    log.unlink()
    child = src + '''
import os
@track_wormhole_skill("control")
def identity(x): return x
for i in range(100): identity(i)
os._exit(0)
'''
    subprocess.run([sys.executable, "-c", child], check=True)
    recs = [json.loads(s) for s in log.read_text().splitlines()]
    out["abrupt_process_exit_undercount"] = {
        "actual_calls": 100,
        "persisted_calls": lib.sum_skill_activation_calls(recs),
    }


target = Path(__file__).with_name("library_regression_controls.json")
target.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")
print(json.dumps(out,indent=2,ensure_ascii=False))
