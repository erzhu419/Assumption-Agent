#!/usr/bin/env python3
"""Replay paid final observations, then check six complete controller trajectories."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from scripts import run_response_revision_successors as previous

SOURCE = ROOT / "artifacts/response_exhausted_stage_20261011"
OUTPUT = ROOT / "artifacts/response_point_finalization_20261011"
METHOD = previous.original.METHOD
read, write = previous.read, previous.write
JOBS = (
    ("retained", "square_original__gain_higher__adapted_law"),
    ("retained", "square_shifted__gain_lower__original_law_return"),
    ("revised_only", "square_shifted__gain_higher__adapted_law"),
    ("retained", "square_original__gain_lower__original_law_return"),
    ("retained", "cubic_shifted_1__gain_lower__original_law_return"),
    ("revised_only", "cubic_shifted_4__gain_higher__original_law_return"),
)


def location(root, arm, case):
    return root / "after" / arm / case / "frozen_numeric"


def prepare(output):
    from scripts.run_response_matched_adoption import prepare_bundle
    from scripts.run_response_composition_transfer import declarations

    old = read(SOURCE / "protocol.json")
    rows = read(SOURCE / "brief.json")
    opened = [{"arm": row["arm"], "case": row["case"], "question_id": point["question_id"]}
              for row in rows for point in row["point_confirmations"]
              if point["question_status"] == "open"]
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "protocol.json").exists(), "Preserve the prepared protocol"
    protocol = {"baseline": str(SOURCE), "jobs": [{"arm": a, "case": c} for a, c in JOBS],
        "open_point_replays": opened, "limits": old["limits"], "cases": old["cases"],
        "expected_replay_supported": 8, "expected_replay_unconfirmed": 14,
        "intervention": "Attach existing point evaluation to the settled final tuning step; preserve original scientific answers and physical actions.",
        "baseline_reuse": "Existing completed trajectories are controls; no baseline reruns and no replacement of the prior two time-limited failures.",
        "replay": "Replay recorded actions in the unchanged public fixture, match every archived public observation, then compare old/new select_actions before the last action and execute the old answer only if scheduled.",
        "selection": "Three distinct positive final-point patterns, two incidental-activation controls, and the previous full-research confirmation gain.",
        "scope": "Consumed development conditions, not new tasks or model adoption.",
        "model_calls": 0, "maximum_workers": 2}
    write(output / "protocol.json", protocol)
    for module in previous.MODULES:
        target = output / "runtime" / (module + ".py")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "src/hegel_machine/cognition" / target.name, target)
    changes = []
    for arm, case in JOBS:
        config = old["cases"][case]
        source = SOURCE / "after" / arm / "sources" / config["source_id"] / "numeric_bundle.json"
        old_bundle = read(source)
        new_bundle = prepare_bundle(old_bundle)
        before, after = old_bundle["files"][METHOD], new_bundle["files"][METHOD]
        assert declarations(before) == declarations(after)
        assert {k: v for k, v in old_bundle.items() if k != "files"} == {
            k: v for k, v in new_bundle.items() if k != "files"}
        assert {k: v for k, v in old_bundle["files"].items() if k != METHOD} == {
            k: v for k, v in new_bundle["files"].items() if k != METHOD}
        functions = lambda code: {node.name: ast.dump(node) for node in ast.parse(code).body
                                  if isinstance(node, ast.FunctionDef)}
        first, second = functions(before), functions(after)
        changed = sorted(k for k in first.keys() | second.keys() if first.get(k) != second.get(k))
        assert set(changed) <= {"_tune", "select_actions"}, changed
        assert [ast.dump(node) for node in ast.parse(before).body if not isinstance(node, ast.FunctionDef)] == [
            ast.dump(node) for node in ast.parse(after).body if not isinstance(node, ast.FunctionDef)]
        prior_method = output / "before_controllers" / arm / (config["source_id"] + ".py")
        prior_method.parent.mkdir(parents=True, exist_ok=True)
        prior_method.write_text(before)
        write(output / "after" / arm / "sources" / config["source_id"] / "numeric_bundle.json", new_bundle)
        write(output / "after" / arm / "protocol.json", {
            "cases": old["cases"], "limits": old["limits"], "arms": ["frozen_numeric"]})
        changes.append({"arm": arm, "case": case, "changed_functions": changed,
                        "knowledge_declarations_methods_preserved": True})
    write(output / "bundle_refresh.json", changes)
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    print(json.dumps({"status": "prepared", "complete_runs": len(JOBS), "public_replays": len(opened)}), flush=True)


def check_runtime(output):
    for name in previous.MODULES:
        assert (output / "runtime" / (name + ".py")).read_text() == (
            ROOT / "src/hegel_machine/cognition" / (name + ".py")).read_text(), name


def replay(output):
    previous.load_runtime(output / "runtime")
    from scripts import run_response_matched_adoption as matched
    from scripts.run_response_composition_transfer import declarations

    protocol = read(output / "protocol.json")
    matched.study.CASES = protocol["cases"]
    rows = []
    for item in protocol["open_point_replays"]:
        arm, case = item["arm"], item["case"]
        source = location(SOURCE, arm, case)
        trace = read(source / "research_trace.json")
        actions = read(source / "action_trace.json")
        lab = matched.study.TransferSignalLab(case)
        observations = [{"index": action["index"], **deepcopy(lab.act(action["action"]))}
                        for action in actions]
        assert [{k: record[k] for k in ("index", "action", "feedback")} for record in observations] == actions
        assert all(observations[record["index"]] == record
                   for record in read(source / "evidence_observations.json"))
        old_source = (source / "final_controller.py").read_text()
        new_source = matched.prepare_bundle({"files": {METHOD: old_source}, "methods": []})["files"][METHOD]
        assert declarations(old_source) == declarations(new_source)
        state = trace["state"]
        assert state["test"]["question"] == item["question_id"]
        trial = state["test"]
        memory = {"program_path": METHOD, "questions": trace["questions"]}
        row = {**item, "prediction": trial["prediction"], "observation_index": observations[-1]["index"],
               "archived_public_observations_preserved": True}
        for phase, code in (("before", old_source), ("after", new_source)):
            namespace = {"__name__": "paid_point_replay"}
            exec(compile(code, METHOD, "exec"), namespace)
            with TemporaryDirectory(prefix="hegel-point-final-") as temporary:
                namespace["STATE"] = Path(temporary) / "state.json"
                namespace["STATE"].write_text(json.dumps(state))
                proposal = namespace["select_actions"]({"current_observation": observations[-2]["observation"]},
                                                       observations[:-1], deepcopy(memory))
                assert proposal["actions"] == [actions[-1]["action"]]
                followups = proposal.get("after_observation", [])
                # Deliberately call the archived answer, not a new interpretation.
                old_namespace = {"__name__": "paid_point_answer"}
                exec(compile(old_source, METHOD, "exec"), old_namespace)
                old_namespace["STATE"] = namespace["STATE"]
                answer = None
                for step in followups:
                    assert step == {"action": "evaluate", "arguments": {
                        "path": METHOD, "question_id": trial["question"]}}
                    answer = old_namespace["answer"]({"current_observation": observations[-1]["observation"]},
                                                      observations, deepcopy(memory))
                questions = deepcopy(trace["questions"])
                if answer:
                    next(q for q in questions if q["id"] == trial["question"]).update(answer)
                points = matched.point_confirmations({"reference_state": state,
                    "research": {"questions": questions}, "observations": observations})
                point = next(point for point in points if point["question_id"] == trial["question"])
                row[phase] = {"proposal": proposal, "answer": answer, "point": point}
        _, frequency = matched.recipe._dialog(observations[-1]["observation"])
        row["actual_frequency"] = frequency
        row["absolute_error"] = abs(frequency - trial["prediction"])
        row["settled"] = matched.recipe._tune(observations[-1]["observation"], trial["prediction"]) is None
        rows.append(row)
    supported = sum(row["after"]["point"]["physically_confirmed"] for row in rows)
    assert supported == protocol["expected_replay_supported"]
    assert len(rows) - supported == protocol["expected_replay_unconfirmed"]
    assert not any(row["before"]["point"]["physically_confirmed"] for row in rows)
    assert all(row["settled"] and row["absolute_error"] <= .51 for row in rows
               if row["after"]["point"]["physically_confirmed"])
    write(output / "public_replay.json", {"status": "finished", "rows": rows,
        "supported": supported, "unconfirmed": len(rows) - supported,
        "new_observations": 0, "model_calls": 0,
        "interpretation": "Replay of already paid observations; not new confirmations or new task successes."})
    print(json.dumps({"public_replays": len(rows), "supported": supported, "unconfirmed": len(rows)-supported}), flush=True)


def summarize(output):
    protocol = read(output / "protocol.json")
    pairs = []
    for item in protocol["jobs"]:
        before = location(SOURCE, **item)
        after = location(output, **item)
        if not (after / "brief.json").exists():
            continue
        old, new = read(before / "brief.json"), read(after / "brief.json")
        old_knowledge, new_knowledge = (read(path / "learned_response_knowledge.json")
                                        for path in (before, after))
        pairs.append({**item, "before": old, "after": new,
            "complete_actions_preserved": read(before / "action_trace.json") == read(after / "action_trace.json"),
            "charged_compute_preserved": old["costs"]["research_compute_units"] == new["costs"]["research_compute_units"],
            "exported_knowledge_preserved_except_artifact_path": {
                k: v for k, v in old_knowledge.items() if k != "source_artifact"} == {
                k: v for k, v in new_knowledge.items() if k != "source_artifact"},
            "task_outcome_preserved": all(old[k] == new[k] for k in ("success", "status")),
            "confirmation_gain": sum(new["requested_point_confirmed"].values()) - sum(old["requested_point_confirmed"].values())})
    aggregate = {"full_runs": len(pairs),
        "task_success_before": sum(row["before"]["success"] for row in pairs),
        "task_success_after": sum(row["after"]["success"] for row in pairs),
        "actions_preserved": sum(row["complete_actions_preserved"] for row in pairs),
        "compute_preserved": sum(row["charged_compute_preserved"] for row in pairs),
        "knowledge_preserved": sum(row["exported_knowledge_preserved_except_artifact_path"] for row in pairs),
        "requested_confirmation_gain": sum(row["confirmation_gain"] for row in pairs),
        "unsupported_positive_verdicts": sum(row["after"]["unsupported_positive_verdicts"] for row in pairs),
        "python_seconds_before": sum(row["before"]["costs"]["python_execution_seconds"] for row in pairs),
        "python_seconds_after": sum(row["after"]["costs"]["python_execution_seconds"] for row in pairs),
        "environment_actions": sum(row["after"]["costs"]["environment_steps"] for row in pairs),
        "research_compute": sum(row["after"]["costs"]["research_compute_units"] for row in pairs),
        "model_calls": 0,
        "python_timing_scope": "Different host load from historical controls; reported accounting, not a speedup claim",
        "after_observation_error_scope": "Detailed transient event logs are not retained by the reused runner; check execution status and each expected terminal confirmation."}
    summary = {"status": "finished" if len(pairs) == len(protocol["jobs"]) else "running",
               "completed": len(pairs), "planned": len(protocol["jobs"]), "pairs": pairs, "aggregate": aggregate}
    write(output / "summary.json", summary)
    write(output / "brief.json", aggregate)
    return summary


def run(output):
    protocol = read(output / "protocol.json")
    def launch(item):
        log = output / "logs" / (item["arm"] + "__" + item["case"] + ".log")
        log.parent.mkdir(exist_ok=True)
        with log.open("w") as stream:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                "--case", item["case"], "--arm", item["arm"]], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        return {**item, "returncode": result.returncode, "log": str(log)}
    assert not any(location(output, **item).exists() for item in protocol["jobs"])
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(launch, item) for item in protocol["jobs"]]):
            record = future.result()
            records.append(record)
            write(output / "execution_status.json", records)
            summary = summarize(output)
            print(json.dumps({**record, "completed": summary["completed"]}), flush=True)
    if any(row["returncode"] for row in records):
        raise SystemExit("Retain failed executions and inspect their logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=previous.ARMS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output)
    else:
        check_runtime(output)
        if args.case and args.arm:
            previous.load_runtime(output / "runtime")
            previous.original.run_case(output / "after" / args.arm, args.case, "frozen_numeric")
        elif args.replay:
            replay(output)
        elif args.run:
            run(output)
        elif args.summarize:
            print(json.dumps(summarize(output), indent=2))
        else:
            parser.error("Choose prepare, replay, run, or summarize")
