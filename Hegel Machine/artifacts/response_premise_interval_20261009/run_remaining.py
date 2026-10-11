"""Complete the previously specified paired study after its first execution."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from scripts.run_response_structure_transfer import run_case, successor_cases, summarize
from scripts.run_online_inquiry_llm import write_json

if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    protocol = json.loads((output / "protocol.json").read_text())
    packages = {arm: json.loads((output / (arm + "_bundle.json")).read_text())
                for arm in ("without_cubic", "retain_cubic")}
    rows = json.loads((output / "brief.json").read_text())
    completed = {(row["case"], row["arm"]) for row in rows}
    cases = successor_cases()
    order = {(case, arm): index for index, (case, arm) in enumerate(
        (case, arm) for case in cases for arm in packages)}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_case, case, config, arm, package, output, protocol["limits"]): (case, arm)
                   for case, config in cases.items() for arm, package in packages.items()
                   if (case, arm) not in completed}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda value: order[value["case"], value["arm"]])
            write_json(output / "brief.json", rows)
            write_json(output / "summary.json", summarize(rows))
            print(json.dumps({"finished": futures[future], "completed": len(rows),
                              "success": row["success"], "actions": row["costs"]["environment_steps"]}), flush=True)
