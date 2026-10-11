"""Run the specified fresh tasks through the existing parent and inquiry loop."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from scripts import consolidate_response_measurement as study
from scripts.run_response_family_recovery import declarations
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from hegel_machine.cognition.code_agent import CodexCodeProvider


def run_case(case, config, package, output, limits):
    study.CASES = {case: config}
    directory = output / case / "live"
    directory.mkdir(parents=True, exist_ok=False)

    def checkpoint(data):
        latest = data["events"][-1] if data["events"] else {}
        write_json(directory / "active.json", {"status": "running", "costs": data["costs"],
            "score": data["score"], "latest_event": {
                "action": latest.get("action"), "operator": latest.get("arguments", {}).get("operator"),
                "status": latest.get("feedback", {}).get("status"),
                "inquiry_question_id": latest.get("inquiry_question_id")}})

    provider_factory = lambda workspace: CodexCodeProvider(workspace, model="gpt-6-luna",
        reasoning_effort="medium", timeout_seconds=limits["provider_request_timeout_seconds"])
    row = study.run_transfer_case(case, "live", package, output, limits,
        provider_factory=provider_factory, on_checkpoint=checkpoint)
    result = json.loads((directory / "result.json").read_text())
    source = (directory / "agent" / study.METHOD_PATH).read_text()
    initial = declarations(package["files"][study.METHOD_PATH])
    final = declarations(source)
    row.update(declarations_before=initial, declarations_after=final,
        additions=[item for item in final if item not in initial],
        removed=[item for item in initial if item not in final],
        inquiries=result["research"]["llm_inquiries"], questions=result["research"]["questions"])
    write_json(directory / "brief.json", row)
    write_json(directory / "active.json", {"status": "finished", "execution": row["status"],
        "success": row["success"], "costs": row["costs"]})
    return row


if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    protocol = json.loads((output / "protocol.json").read_text())
    assert verify_codex_subscription() == protocol["auth_mode"]
    package = json.loads((output / "initial_bundle.json").read_text())
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_case, case, entry["config"], package, output, protocol["limits"]): case
                   for case, entry in protocol["cases"].items()}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda value: value["case"])
            write_json(output / "brief.json", rows)
            print(json.dumps({"case": row["case"], "completed": len(rows), "success": row["success"],
                "status": row["status"], "model_calls": row["costs"]["model_calls"],
                "actions": row["costs"]["environment_steps"],
                "added": [item["id"] for item in row["additions"]]}), flush=True)
