"""Paired current-runtime ablation of the new stalled-support revision only."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from scripts import run_response_structure_transfer as transfer


def run_case(case, config, package, output, limits):
    from hegel_machine.cognition import world_response_hypotheses
    world_response_hypotheses._revise_stalled_support = lambda *args: (None, None)
    return transfer.run_case(case, config, "retain_cubic", package, output, limits)


if __name__ == "__main__":
    directory = Path(__file__).parent
    baseline = directory / "retained_controls"
    output = directory / "retained_controls_without_joint_revision"
    protocol = json.loads((baseline / "protocol.json").read_text())
    packages, _ = transfer.frozen_packages(Path(protocol["source"]))
    cases = {case: transfer.successor_cases()[case] for case in protocol["cases"]}
    assert json.loads(json.dumps(cases)) == protocol["cases"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps({**protocol,
        "paired_current_runtime_directory": str(baseline),
        "intervention": "Only an in-process _revise_stalled_support=lambda *args: (None,None). No production edits or old-version restoration.",
        "scope": "Same three frozen-source cases, same current controller and hypothesis code, same original limits, max_model_calls=0. Only disable the new stalled-support revision. Each case starts fresh in its own worker; no source state or target setting is injected."}, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_case, case, config, packages["retain_cubic"],
                              output, protocol["limits"]): case for case, config in cases.items()}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda value: list(cases).index(value["case"]))
            (output / "summary.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({"case": row["case"], "success": row["success"],
                              "status": row["status"], "actions": row["costs"]["environment_steps"],
                              "compute": row["costs"]["research_compute_units"]}), flush=True)
