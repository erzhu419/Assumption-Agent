"""Run frozen independent apparatus cases; save compact outputs only."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime/scripts"))
from run_composed_local_cycle import protocol, run_cycle
import hegel_machine.cognition.unified_research as research


def execute(case):
    destination = ROOT / f"case_{case['apparatus_seed']}_{case['profile']}.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = run_cycle(protocol(**case))
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return {**case, "cycle_passed": result["cycle_passed"],
        "failed_checks": [key for key, value in result["cycle_checks"].items() if not value],
        "source_deletion_effect": result["source_deletion_effect"],
        "arms": {arm: {key: value[key] for key in ("solved", "mean_mse", "costs")}
                 for arm, value in result["results"].items()}}


def main():
    assert Path(research.__file__).is_relative_to(ROOT / "runtime")
    plan = json.loads((ROOT / "protocol.json").read_text())
    started = monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(execute, case) for case in plan["cases"]]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items()
                              if key not in ("arms", "source_deletion_effect")}), flush=True)
    rows.sort(key=lambda row: row["apparatus_seed"])
    summary = {"protocol": plan, "case_count": len(rows),
        "cycle_pass_count": sum(row["cycle_passed"] for row in rows),
        "same_quality_pareto_saving_count": sum(row["source_deletion_effect"]["same_quality_pareto_saving"]
                                                for row in rows),
        "rows": rows, "elapsed_seconds": monotonic() - started, "external_model_calls": 0}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("protocol", "rows")}), flush=True)


if __name__ == "__main__":
    main()
