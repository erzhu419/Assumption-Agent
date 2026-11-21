"""Offline controls on pinned frontier harness; no model or Docker calls.

Run from any directory. An optional first argument selects the M6 checkout.
The Docker copy is mocked only to reproduce standard file-versus-directory
copy semantics; lesson selection uses real committed episode receipts.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("HEGEL_M6_REPO", str(HERE.parent / "repo_m6_b1d0c3e99")))
sys.path.insert(0, str(REPO / "Hegel Machine/src"))
from hegel_machine.cognition import m6_frontier_real_benchmarks as frontier

bundle = REPO / "Hegel Machine/docs/evidence/m6_2_frontier_28turn_40ep_20261121"
all_eps = json.loads((bundle / "all_episode_summaries.json").read_text())
eps = [e for e in all_eps if e["arm"] == "Hegel_M6_2_Frontier_LiveGPT"]
result = {"commit": "b1d0c3e99f1e6b12cbc62ca8311f1a25d66f2dbf", "no_model_or_docker_calls": True}

with tempfile.TemporaryDirectory(prefix="audit_frontier_copy_") as tmp:
    td = Path(tmp)
    origin = td / "original"
    origin.mkdir()
    (origin / "model.py").write_text("def predict(x):\n    return x\n")
    (origin / "app").mkdir()
    shutil.copy2(origin / "model.py", origin / "app/model.py")

    def fake_copy(command, **kwargs):
        assert command[:2] == ["docker", "cp"]
        source = origin / command[2].split(":", 1)[1].lstrip("/")
        target = Path(command[3])
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    with patch.object(frontier.subprocess, "run", fake_copy):
        single = frontier.save_submitted_artifacts_snapshot("mock", ("/model.py",), td / "single")
        directory = frontier.save_submitted_artifacts_snapshot("mock", ("/app/",), td / "directory")
    result["snapshot_file_extension_control"] = {
        "original_file_bytes": (origin / "model.py").stat().st_size,
        "declared_single_file": "/model.py",
        "single_file_saved": single,
        "same_file_in_declared_directory_saved": directory,
        "explanation": "docker cp stages single files as art_0, losing the suffix used by the whitelist",
    }

consolidation_file = bundle / "wave1_to_wave2_consolidation_gpt-6.2-sol.json"
original_receipt = json.loads(consolidation_file.read_text())
order = [x["source_task"] for x in original_receipt["events"]]
lookup = {e["short_name"]: e for e in eps if e["wave"] == "wave_1"}
out = {}
for label, task_order in [("recorded_completion_order", order), ("reverse_completion_order", list(reversed(order)))]:
    forest, archive, graph = frontier.build_m6_frontier_general_forest()
    receipt = frontier.consolidate_wave1_into_m6_forest(
        wave1_m6_summaries=[lookup[t] for t in task_order],
        forest=forest, archive=archive, meta_dag=graph,
    )
    kept = {nid: x["wave1_consolidated_nogoods"] for nid, x in archive.items()}
    actual_events = []
    for event in receipt["events"]:
        actual = [nid for nid in event["updated_nodes"] if event["lesson"] in kept[nid]]
        actual_events.append({"source_task": event["source_task"], "claimed_targets": len(event["updated_nodes"]), "retained_targets": len(actual)})
    out[label] = {"order": task_order, "lessons_by_node": kept, "events": actual_events}
left, right = out["recorded_completion_order"], out["reverse_completion_order"]
out["changed_node_membership"] = [nid for nid in left["lessons_by_node"] if set(left["lessons_by_node"][nid]) != set(right["lessons_by_node"][nid])]
out["changed_node_order_or_membership"] = [nid for nid in left["lessons_by_node"] if left["lessons_by_node"][nid] != right["lessons_by_node"][nid]]
result["completion_order_controls"] = out
result["stress_gate_attribution_check"] = [{"task": e["short_name"], "strict_pass": e["strict_pass"], "turns": e["turns_used"], "stress_gate_triggered": e.get("stress_challenge_gate_triggered", False)} for e in eps]

dest = HERE / "frontier_controls_results.json"
dest.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(json.dumps({
    "snapshot": result["snapshot_file_extension_control"],
    "changed_lesson_membership_nodes": out["changed_node_membership"],
    "actual_vs_reported_updates": left["events"],
    "stress_gate_triggered_tasks": [x["task"] for x in result["stress_gate_attribution_check"] if x["stress_gate_triggered"]]
}, indent=2, ensure_ascii=False))
