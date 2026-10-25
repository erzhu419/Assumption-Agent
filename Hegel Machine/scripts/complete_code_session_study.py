#!/usr/bin/env python3
"""Execute the fixed completion extension and preserve both comparison reports."""

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_text_incrementally
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.code_agent import _copy_model_file, CodexCodeProvider, retain_evaluated_python_model
from hegel_machine.cognition.code_session import CodeComparisonSession

PRIMARY = ROOT / "artifacts/code_session_v026_20260909.json"
OUTPUT = ROOT / "artifacts/code_session_v026_completion_extension_20260909.json"
WORKSPACE = ROOT / "artifacts/code_session_v026_completion_extension_workspace"
HEGEL = ROOT / "artifacts/code_session_v026_hegel_learning_20260909.json"


def usage_summary(code):
    events = [event for result in code["results"] for event in result["events"]]
    usage = {key: sum(event.get("usage", {}).get(key) or 0 for event in events)
             for key in ("input_tokens", "output_tokens", "cached_input_tokens", "wall_time_seconds")}
    usage["missing_usage_calls"] = sum(event.get("usage", {}).get("total_tokens") is None for event in events)
    return usage


def reuse_summary(result):
    inherited = result["inherited_files"]
    sources = [event["arguments"].get("code", "") for event in result["events"]
               if event["action"] == "run" and event["feedback"].get("returncode") == 0]
    if result["model"]:
        sources.append(result["model_source"] or "")
    imports = [path for path in inherited if path.endswith(".py") and any(
        re.search(r"(?:from|import)\s+" + re.escape(path[:-3].replace("/", ".")) + r"\b", source)
        for source in sources)]
    return {"task_position": result["task_position"], "inherited_files": inherited,
            "declared_reused_inherited_files": sorted(set(inherited).intersection(result["reused_files"])),
            "shared_imports_in_successful_programs": imports}


def main():
    primary = json.loads(PRIMARY.read_text())
    primary["hegel"] = json.loads(HEGEL.read_text())
    before = primary["code"]["costs"]
    if OUTPUT.exists():
        state = json.loads(OUTPUT.read_text())
    else:
        state = deepcopy(primary)
        state["evaluation"] = None
        state["code"]["config"].update(max_model_calls=12, max_steps=20, token_budget=200000)
        state["code"]["totals"].update(token_budget=650000, max_model_calls=42)
        for index in (2, 5):
            result = state["code"]["results"][index]
            destination = WORKSPACE / f"task_{index:03d}"
            shutil.copytree(result["workspace"], destination, copy_function=_copy_model_file)
            result["workspace"] = str(destination)
        result = state["code"]["results"][2]
        last = result["events"][-1]
        if last["action"] != "evaluate" or last["feedback"].get("returncode") != 0:
            raise ValueError("The preserved actual successful evaluation is required")
        retained = retain_evaluated_python_model(result["workspace"], last["arguments"]["model"],
                                                 result["selected_channels"], last["feedback"]["mse"])
        result.update(retained, model_origin="last_successful_evaluation")
        state["code"]["knowledge"]["lessons"][2]["model_source"] = result["model_source"]
    session = CodeComparisonSession(WORKSPACE, state=state,
        provider_factory=lambda path: CodexCodeProvider(path, timeout_seconds=180))

    def checkpoint(_=None):
        report = session.to_dict()
        report["extension"] = {"kind": "completion_extension", "original": str(PRIMARY),
            "original_costs": before, "retained_evaluated_candidate": 2,
            "extra_costs": {key: report["code"]["costs"][key]-value for key, value in before.items()}}
        write_text_incrementally(OUTPUT, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"visited": len(report["code"]["results"]),
            "models": sum(bool(row["model"]) for row in report["code"]["results"]),
            "costs": report["code"]["costs"]}), flush=True)

    checkpoint()
    if not session.code.state["results"][5]["model"]:
        session.code.continue_task(5, on_checkpoint=checkpoint)
    session.code.run(on_checkpoint=checkpoint)
    checkpoint()
    # All learning, including the extension, has stopped before these scores.
    original = CodeComparisonSession(ROOT / "artifacts/code_session_v026_20260909_workspace", state=primary)
    original.evaluate()
    session.evaluate()
    checkpoint()
    primary_report = ROOT / "artifacts/code_session_v026_primary_comparison_20260909.json"
    if primary_report.exists():
        raise FileExistsError(primary_report)
    write_text_incrementally(primary_report, json.dumps(original.to_dict(), ensure_ascii=False, indent=2),
        base=PRIMARY)
    report = session.to_dict()
    summary = {"primary": {"status": original.to_dict()["status"], "code_costs": before,
                    "usable_models": sum(bool(row["model"]) for row in primary["code"]["results"]),
                    "model_usage": usage_summary(primary["code"])},
        "completion_extension": {"status": report["status"], "code_costs": report["code"]["costs"],
            "extra_costs": {key: report["code"]["costs"][key]-value for key, value in before.items()},
            "usable_models": sum(bool(row["model"]) for row in report["code"]["results"]),
            "model_usage": usage_summary(report["code"])}, "hegel_costs": report["hegel"]["costs"],
        "tasks": [{"task_position": row["task_position"], "task_index": row["task_index"],
            "seed": row["seed"], "primary_code": old["code"], "extension_code": row["code"],
            "hegel": row["hegel"]} for old, row in zip(original.evaluation, session.evaluation)],
        "reuse": [reuse_summary(result) for result in report["code"]["results"]]}
    with (ROOT / "artifacts/code_session_v026_summary_20260909.json").open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
