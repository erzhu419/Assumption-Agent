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
REPO = Path(os.environ.get("HEGEL_M67_REPO", BASE / "repo_m67_948e2d57a"))
SRC = REPO / "Hegel Machine/src"
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

E = REPO / "Hegel Machine/docs/evidence"
stats = {}
for epack in sorted(E.glob("m6_[4-7]*")):
    if not epack.is_dir():
        continue
    rows = []
    for summary_file in sorted(epack.glob("episodes/*Hegel*/00_episode_summary.json")):
        summary = json.loads(summary_file.read_text())
        if summary.get("benchmark") != "Terminal-Bench-Science-0.1":
            continue
        dag = json.loads((summary_file.parent / "01_decision_provenance_dag.json").read_text())
        harvest = [r for r in dag["records"] if r["decision_type"] == "D_SKILL_ACTIVATION_HARVEST"]
        activated = harvest[-1]["candidates_evaluated"][0]["activated_functions"] if harvest else []
        rows.append({
            "task_id": summary["task_id"], "strict_pass": summary["strict_pass"],
            "skill_activation_count": summary.get("skill_activation_count", 0),
            "saved_sample_call_count": lib.sum_skill_activation_calls(summary.get("skill_activations", [])),
            "saved_sample_record_count": len(summary.get("skill_activations", [])),
            "full_log_record_count_in_dag": harvest[-1]["chosen_option"].get("log_record_count") if harvest else None,
            "activated_functions": activated,
            "gate_executed": summary.get("host_wormhole_gate_executed"),
            "gate_reports": summary.get("host_wormhole_gate_reports"),
        })
    consolidation_files = list(epack.glob("wave1_to_wave2_consolidation*.json"))
    learned_events = []
    for fp in consolidation_files:
        learned_events.extend(json.loads(fp.read_text()).get("wave1_ast_executable_sublation_events", []))
    learned_names = {name for event in learned_events for name in event["exported_functions"]}
    activated_names = {name for row in rows for name in row["activated_functions"]}
    stats[epack.name] = {
        "episodes": rows, "learned_export_events": learned_events,
        "learned_export_names_seen_in_any_harvest": sorted(learned_names & activated_names),
    }
out["saved_evidence"] = stats

target = Path(__file__).with_name("library_controls_results.json")
target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "saved_evidence"}, indent=2, ensure_ascii=False))
print("Saved evidence analysis:", str(target))
