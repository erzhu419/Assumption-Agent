#!/usr/bin/env python3
"""Read-only M6.3 audit controls; no model calls and no repository mutations.

HEGEL_AUDIT_REPO selects repository root or Hegel Machine directory.
HEGEL_M67_REPO remains supported for legacy callers. Output is confined here.
The isolation test reads ONLY a numeric marker created by this script.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("HEGEL_AUDIT_REPO", os.environ.get("HEGEL_M67_REPO", str(HERE.parents[1] / "repo_m691_4001bef56"))))
HM = ROOT if ROOT.name == "Hegel Machine" else ROOT / "Hegel Machine"
sys.path.insert(0, str(HM / "src"))
from hegel_machine.cognition import m6_3_sovereign_causal_wormhole_engine as m
from hegel_machine.cognition.m5_live_gpt_and_provenance import (
    _execute_authored_python_in_isolated_subprocess as execute,
    verify_decision_provenance_chain,
)
from hegel_machine.cognition.m6_live_gpt_causal_aufhebung_benchmark import build_m6_causal_24_waves

EV = HM / "docs/evidence/m6_3_sovereign_ast_causal_96ep_20261121"
RECS = [json.loads(line) for line in (EV / "all_episode_receipts.jsonl").read_text().splitlines()]
WAVES = build_m6_causal_24_waves(seed=20261121)
DS = {d.task_id: d for w in WAVES.values() for d in w}

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def execute_small(code, rows=None):
    if rows is None:
        rows = [{"x": float(i), "y": 2.0 * i} for i in range(1, 5)]
    with tempfile.TemporaryDirectory(prefix="tracka_control_") as td:
        r = execute(Path(td), step_idx=1, python_code=code, train_rows=rows, confirm_rows=rows, heldout_rows=rows)
    return {k: v for k, v in r.items() if k not in {"model_path", "exec_wall_ms"}}

out = {}

# Real successful programs discarded by the literal-key extractor.
dropped = []
for r in RECS:
    if r["wave"] != "wave_1" or r["is_out_of_scope"] or not r["strict_pass"]:
        continue
    a = m.extract_and_translate_authored_ast_v2(r["winning_python_code"])
    if not a["used_keys"]:
        dropped.append({"model": r["model"], "arm": r["arm"], "task_id": r["task_id"], "heldout_rmse": r["heldout_rmse"], "extracted_keys": a["used_keys"], "code_sha256": hashlib.sha256(r["winning_python_code"].encode()).hexdigest()})
out["real_successful_source_code_silently_discarded"] = dropped

# Shadowing and dynamic key cases stay inside accepted Python, without custom dependencies.
cases = {
    "constant_tuple_keys": "KEYS=('x','z')\ndef predict(row): return sum(row[k] for k in KEYS)\n",
    "row_alias_rebound_to_parameter_dict": "def predict(row):\n r = row\n r = {'x': 7.0}\n return r['x'] + row['z']\n",
    "keyword_helper_argument": "def helper(q): return q['x']\ndef predict(row): return helper(q=row) + row['z']\n",
    "nested_helper_function": "def predict(row):\n def helper(q): return q['x']\n return helper(row) + row['z']\n",
}
out["ast_contract_controls"] = {k: m.extract_and_translate_authored_ast_v2(v, {"x": "x_target", "z": "z_target"}) for k, v in cases.items()}

# Current actual accepted content with Bayes-ball disabled. Same candidates, same graph, no model calls.
solw1 = [dict(r) for r in RECS if r["model"] == "gpt-6.2-sol" and r["arm"] == "Hegel_M6_3_Sovereign_LiveGPT" and r["wave"] == "wave_1"]
f1, a1 = m.build_m6_1_clean_k40_forest()
normal = m.consolidate_wave1_sovereign_m6_3(wave1_receipts=solw1, arm="Hegel_M6_3_Sovereign_LiveGPT", forest=f1, node_archive=a1)
f2, a2 = m.build_m6_1_clean_k40_forest()
original_dsep = m.is_d_separated_dag
try:
    m.is_d_separated_dag = lambda *args, **kwargs: False
    nogate = m.consolidate_wave1_sovereign_m6_3(wave1_receipts=solw1, arm="Hegel_M6_3_Sovereign_LiveGPT", forest=f2, node_archive=a2)
finally:
    m.is_d_separated_dag = original_dsep
out["actual_sol_wave1_dsep_ablation"] = {"archive_equal": a1 == a2, "normal_archive_sha256": digest(a1), "disabled_archive_sha256": digest(a2), "normal_writeback_C": normal["writeback_cost_C"], "disabled_writeback_C": nogate["writeback_cost_C"], "normal_blocked_target_count": sum(len(e["collider_shielded_blocked_targets"]) for e in normal["events"]), "disabled_blocked_target_count": sum(len(e["collider_shielded_blocked_targets"]) for e in nogate["events"])}

# Actual model receipt, only the old association's note is replaced; no source candidate mutation.
r = next(copy.deepcopy(x) for x in solw1 if x["strict_pass"] and "C01" in x["task_id"])
r["pilot_rows_snapshot"] = DS[r["task_id"]].pilot_rows
retro = []
for shadow in ("old law y = 1000000.0", "old law y = -1000000.0"):
    f, a = m.build_m6_1_clean_k40_forest()
    for v in a.values():
        v["verified_mechanism_note"] = shadow
    c = m.consolidate_wave1_sovereign_m6_3(wave1_receipts=[r], arm="Hegel_M6_3_Sovereign_LiveGPT", forest=f, node_archive=a)
    retro.append({"old_shadow_note": shadow, "receipt": c["events"][0]["empirical_retroactive_projection"]})
out["retroactive_projection_old_shadow_invariance"] = retro

# A declared 'verified' owner is accepted even when most active input keys are uncovered.
f, a = m.build_m6_1_clean_k40_forest()
own = m.resolve_verified_writeback_owner(recalled_node_ids=["node_C02_TreeM1_surface"], observed_signatures=["sig:c02_treeM1_extract"], active_ast_keys=["m_2", "foreign_required_1", "foreign_required_2"], violated_root_ids=[], forest=f, node_archive=a)
out["owner_partial_coverage_control"] = own

# Archive dedup key conflates different structure/conditions with the same used variables.
e = {}
first = m._append_pareto_candidate(e, used_keys=["x"], python_code="def predict(row): return row['x']", confirm_rmse=0.0, provenance_tag="linear")
second = m._append_pareto_candidate(e, used_keys=["x"], python_code="def predict(row): return row['x']**2", confirm_rmse=0.0, provenance_tag="quadratic")
out["same_variables_distinct_mechanisms"] = {"first_added": first, "second_added": second, "archive": e}

# Strictly benign isolation control: only a synthetic marker outside the sandbox is read.
with tempfile.TemporaryDirectory(prefix="m63_marker_") as td:
    marker = Path(td) / "public_audit_marker.txt"
    marker.write_text("37.0")
    direct = f"v=float(open({str(marker)!r}).read())\ndef predict(row): return v\n"
    ffi = f'''import ctypes
libc = ctypes.CDLL(None)
libc.open.argtypes = [ctypes.c_char_p, ctypes.c_int]
libc.open.restype = ctypes.c_int
libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.read.restype = ctypes.c_ssize_t
libc.close.argtypes = [ctypes.c_int]
buf = ctypes.create_string_buffer(64)
fd = libc.open({str(marker).encode()!r}, 0)
n = libc.read(fd, buf, 63)
libc.close(fd)
v = float(buf.value.decode())
def predict(row): return v
'''
    rows = [{"x": 1.0, "y": 37.0}]
    out["audit_hook_isolation_benign_marker"] = {"python_open": execute_small(direct, rows), "ctypes_libc_read": execute_small(ffi, rows), "scope": "Read only a numeric marker created by this audit. Does not imply original agents exploited the boundary."}

out["abi_state_optional_parameter"] = execute_small("def fit(rows): return 2.0\ndef predict(row, state=None): return row['x'] * (state if state is not None else 99.0)\n")
out["abi_state_required_parameter"] = execute_small("def fit(rows): return 2.0\ndef predict(row, state): return row['x'] * state\n")

(HERE / "tracka_controls_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k:v for k,v in out.items() if k in ('real_successful_source_code_silently_discarded','actual_sol_wave1_dsep_ablation','owner_partial_coverage_control','same_variables_distinct_mechanisms','retroactive_projection_old_shadow_invariance','abi_state_optional_parameter','audit_hook_isolation_benign_marker')},indent=2,ensure_ascii=False))
