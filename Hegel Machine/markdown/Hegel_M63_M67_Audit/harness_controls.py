"""Small offline counterexamples for the pinned M6.7 harness; no Docker/LLM/network calls.

HEGEL_M67_REPO may point to a checkout of commit 948e2d57a.
These controls do not alter historical benchmark outcomes.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("HEGEL_M67_REPO", str(HERE.parent / "repo_m67_948e2d57a")))
sys.path.insert(0, str(REPO / "Hegel Machine/src"))
from hegel_machine.cognition import m6_frontier_real_benchmarks as f

result = {"commit": "948e2d57a48b74369158ba2fbdee87bd32102403", "no_model_or_docker_calls": True}
commands = [
    "true",
    "python3 -c \"print('solver.py')\"",
    "pytest -q /tmp/test_unrelated.py",
    "python3 /tmp/validate_saved_candidate.py",
]
result["verification_string_checks"] = [
    {"command": s, "accepted": f.is_substantive_verification_command(s, ("/app/solver.py",))}
    for s in commands
]

task = f.FrontierBenchmarkTask(
    task_id="audit_offline_mock", short_name="audit_offline_mock", benchmark="Terminal-Bench-Science-0.1",
    wave="wave_1", scientific_domain="audit", wormhole_family="audit", instruction_text="Write a solver.",
    agent_image_tag="mock", verifier_image_tag="mock", declared_artifacts=("/app/solver.py",),
)

class FakeClient:
    responses = []
    def __init__(self, **kwargs):
        self.remaining = iter(self.responses)
    def call_structured(self, **kwargs):
        action, command = next(self.remaining)
        return {
            "ok": True, "wall_ms": 0.0, "prompt_tokens": 0, "cached_tokens": 0,
            "completion_tokens": 0, "returned_model": "offline_mock",
            "parsed": {"action_type": action, "bash_command": command, "final_answer": "done",
                       "rationale": "Offline control", "considered_alternatives": [],
                       "active_hypothesis_and_worst_slice_metric": "", "nogood_update": {}},
        }

class FakeIndex:
    def retrieve(self, *args, **kwargs):
        return {"candidates": [], "query_compute_cost": 0.0}

def exec_noop(*args, **kwargs):
    return {"stdout": "", "stderr": "", "exit_code": 0, "timed_out": False, "wall_ms": 0.0}

def verifier_noop(*args, **kwargs):
    return {"strict_pass": False, "reward": 0.0, "verifier_wall_ms": 0.0,
            "artifact_audit": {"all_present": True}, "ctrf_summary": {},
            "verifier_stdout": "", "verifier_stderr": "", "verifier_timed_out": False}

result["run_frontier_episode_controls"] = []
for label, cap, steps in [
    ("repeat_submit_without_verification", 5, [("submit_completed_task", None), ("submit_completed_task", None)]),
    ("unrelated_print_clears_verification", 5, [("execute_in_remote_docker", commands[1]), ("submit_completed_task", None)]),
    ("last_turn_bypasses_gate", 1, [("submit_completed_task", None)]),
]:
    FakeClient.responses = steps
    forest, archive, _ = f.build_m6_frontier_general_forest()
    with tempfile.TemporaryDirectory(prefix="m67_offline_harness_") as tmp:
        with patch.multiple(
            f, FrontierCodexClient=FakeClient,
            start_detached_container=lambda *a, **k: "mock",
            stop_container_safely=lambda *a, **k: None,
            exec_container_bash=exec_noop,
            snapshot_container_files=lambda *a, **k: {},
            compute_declared_artifacts_digest=lambda *a, **k: "a" * 64,
            check_declared_artifacts_in_container=lambda *a, **k: {
                "all_present": True, "missing_or_empty": [], "artifact_status": {"/app/solver.py": "FILE:40"}},
            find_uninlined_wormhole_imports_in_container=lambda *a, **k: [],
            run_host_mechanical_wormhole_gate_probes=lambda *a, **k: [],
            harvest_container_skill_activations=lambda *a, **k: [],
            save_submitted_artifacts_snapshot=lambda *a, **k: [],
            run_tb_separate_verifier=verifier_noop,
        ):
            summary = f.run_frontier_episode(
                task, "Hegel_M6_7_Frontier_LiveGPT", model="offline_mock",
                codex_info={"provider_backend": "offline_mock", "auth_mode": "offline_mock"},
                forest=forest, forest_archive=archive, hippo_idx=FakeIndex(),
                episode_out_dir=Path(tmp), max_turns=cap,
            )
        result["run_frontier_episode_controls"].append({
            "case": label, "max_turns": cap, "steps": steps,
            **{k: summary[k] for k in ["ended_by_finish", "turns_used", "pullback_gate_triggered", "strict_pass", "provenance_verified"]},
        })

# Extract the exact committed lock class, avoiding a study-script entry point.
lock_path = REPO / "Hegel Machine/scripts/run_m6_5_frontier_28turn_study.py"
tree = ast.parse(lock_path.read_text())
cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "_HeavyTimingRWLock")
ns = {"threading": threading}
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(lock_path), "exec"), ns)
lock = ns["_HeavyTimingRWLock"]()
outer_held = threading.Event()
request_nested = threading.Event()
nested_acquired = threading.Event()
writer_acquired = threading.Event()

def reader():
    lock.acquire_read()
    outer_held.set()
    request_nested.wait(2)
    lock.acquire_read()
    nested_acquired.set()
    lock.release_read()
    lock.release_read()

def writer():
    lock.acquire_write()
    writer_acquired.set()
    lock.release_write()

tr = threading.Thread(target=reader, daemon=True)
tw = threading.Thread(target=writer, daemon=True)
tr.start()
assert outer_held.wait(2)
tw.start()
# Condition waiting gives a deterministic writer-waiting state, with a bounded total wait.
import time
deadline = time.monotonic() + 2
while time.monotonic() < deadline:
    with lock._cond:
        waiting = lock._writers_waiting
    if waiting:
        break
    threading.Event().wait(0.002)
assert waiting == 1
request_nested.set()
nested_acquired.wait(0.2)
with lock._cond:
    result["nested_reader_lock_control"] = {
        "outer_reader_count": lock._readers,
        "writers_waiting": lock._writers_waiting,
        "nested_read_acquired": nested_acquired.is_set(),
        "writer_acquired": writer_acquired.is_set(),
        "deadlock_condition": lock._readers == 1 and lock._writers_waiting == 1 and not nested_acquired.is_set() and not writer_acquired.is_set(),
        "scope": "latent deterministic interleaving; no claim it occurred in saved completed studies",
    }
    # Controlled teardown, not a proposed production fix: unblock our daemon threads.
    lock._writers_waiting = 0
    lock._cond.notify_all()
tr.join(1)
tw.join(1)

(HERE / "harness_controls_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps(result, indent=2, ensure_ascii=False))
