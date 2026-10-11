#!/usr/bin/env python3
"""Run revised consequence priorities once per case, reusing the recorded baseline."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from scripts import consolidate_response_measurement as study
from hegel_machine.cognition import world_response_consequences as consequences


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output / "brief.json").exists():
        raise FileExistsError(f"Completed or partial results already exist: {output / 'brief.json'}")
    output.mkdir(parents=True, exist_ok=True)
    bundle_text = (source / "method_bundle.json").read_text()
    bundle = json.loads(bundle_text)
    previous_protocol = json.loads((source / "protocol.json").read_text())
    limits = {**previous_protocol["limits"], "max_model_calls": 0}
    recorded = [row for row in json.loads((source / "brief.json").read_text())
                if row["arm"] == "consolidated"]
    if len(recorded) != len(study.CASES) or {row["case"] for row in recorded} != set(study.CASES):
        raise ValueError("The recorded consolidated arm must contain all four transfer cases")
    study.write_json(output / "historical_baseline.json", recorded)
    (output / "method_bundle.json").write_text(bundle_text)
    (output / "world_response_consequences.py").write_text(Path(consequences.__file__).read_text())
    study.write_json(output / "protocol.json", {
        "source": str(source), "recorded_baseline": str(source / "brief.json"),
        "cases": deepcopy(study.CASES), "limits": limits, "procedure_max_steps": 256,
        "model_calls": 0,
        "arms": {
            "historical_baseline": "Recorded consolidated arm with previous consequence priorities",
            "consolidated": "Same frozen program with current consequence panel and comparison priorities"},
        "source_change": "None: the previous method bundle, source and scientific declarations are copied unchanged.",
        "scope": "Same four development cases and effective limits; one new zero-LLM execution per case. "
                 "The completed prior control is reused, not rerun."})
    rows = []
    order = {case: index for index, case in enumerate(study.CASES)}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(study.run_transfer_case, case, "consolidated", bundle, output, limits): case
                   for case in study.CASES}
        for future in as_completed(futures):
            rows.append(future.result())
            rows.sort(key=lambda row: order[row["case"]])
            study.write_json(output / "brief.json", rows)
            print(json.dumps({"finished": futures[future], "completed_cases": len(rows)}), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
