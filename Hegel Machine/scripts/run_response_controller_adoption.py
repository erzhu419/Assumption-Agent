#!/usr/bin/env python3
"""Repeat recorded development failures through the existing live researcher."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.world_reference_hypothesis import _retain_declaration
from scripts import consolidate_response_measurement as study
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_response_family_recovery import declarations


def run_case(case, config, bundle, output, limits):
    study.CASES = {case: config}
    directory = output / case / "live"
    directory.mkdir(parents=True, exist_ok=False)

    def checkpoint(data):
        latest = data["events"][-1] if data["events"] else {}
        write_json(directory / "active.json", {"status": "running", "costs": data["costs"],
            "score": data["score"], "latest": {key: latest.get(key) for key in
                ("action", "inquiry_question_id")},
            "operator": latest.get("arguments", {}).get("operator")})

    row = study.run_transfer_case(case, "live", bundle, output, limits,
        provider_factory=lambda workspace: CodexCodeProvider(workspace,
            model="gpt-6-luna", reasoning_effort="medium",
            timeout_seconds=limits["provider_request_timeout_seconds"]),
        on_checkpoint=checkpoint)
    result = json.loads((directory / "result.json").read_text())
    row["submissions"] = [{"arguments": event["arguments"], "feedback": event["feedback"]}
        for event in result["events"] if event.get("arguments", {}).get("operator")
        in ("reference_hypothesis", "reference_response")]
    final = declarations((directory / "agent" / study.METHOD_PATH).read_text())
    original = declarations(bundle["files"][study.METHOD_PATH])
    row.update(additions=[item for item in final if item not in original],
        removed=[item for item in original if item not in final])
    write_json(directory / "brief.json", row)
    write_json(directory / "active.json", {"status": "finished", **row})
    return row


def run(source, output, cases):
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source / "protocol.json").read_text())
    selected = {case: old["cases"][case] for case in cases}
    auth = verify_codex_subscription()
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((source / "initial_bundle.json").read_text())
    current = Path(recipe.__file__).read_text()
    for hypothesis in declarations(original["files"][study.METHOD_PATH]):
        current = _retain_declaration(current, hypothesis)
    assert declarations(current) == declarations(original["files"][study.METHOD_PATH])
    bundle = {"files": {study.METHOD_PATH: current}, "methods": original["methods"]}
    write_json(output / "initial_bundle.json", bundle)
    write_json(output / "protocol.json", {"study": "paid_conflict_and_controller_live_adoption",
        "historical_source": str(source), "cases": selected, "limits": old["limits"],
        "model": old["model"], "auth_mode": auth,
        "scope": "Repeat preselected development failures from zero observations with the same "
                 "nine declarations and resource ceilings. Existing parent chooses inquiry; the "
                 "model chooses structures, experiments and stopping. Host supplies neither "
                 "new scientific candidates nor target settings and does not intervene in flight. "
                 "Historical live runs are diagnostic context, not replicated paired estimates."})
    rows = []
    with ProcessPoolExecutor(max_workers=len(selected)) as pool:
        futures = {pool.submit(run_case, case, entry["config"], bundle, output, old["limits"]): case
                   for case, entry in selected.items()}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(output / "brief.json", rows)
            print(json.dumps({key: row[key] for key in ("case", "status", "success", "costs")}), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
        default=ROOT / "artifacts/response_stagnation_live_20261009")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["task_01", "task_02"])
    args = parser.parse_args()
    run(args.source, args.output, args.cases)
