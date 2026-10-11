"""Run the existing regression and retain its actual failure-local evidence."""

import json
from pathlib import Path

import pytest


DIRECTORY = Path(__file__).resolve().parent


class Diagnosis:
    def pytest_exception_interact(self, node, call, report):
        traceback = call.excinfo.value.__traceback__
        while traceback is not None:
            values = traceback.tb_frame.f_locals
            changed = values.get("changed")
            if isinstance(changed, dict) and "costs" in changed and "actions" in changed:
                actions = changed["actions"]
                refinements = [action for action in actions
                               if action["action"] == "refine_relation_parameter_space"]
                payload = {"failure": str(call.excinfo.value), "costs": changed["costs"],
                    "status": changed["status"], "confirmation_loss": changed["confirmation_loss"],
                    "last_refinements": refinements[-5:],
                    "acquisitions": [action for action in actions if action["action"] == "acquire"],
                    "decisions": [{key: value for key, value in action.items() if key not in
                        ("program", "plan", "evidence", "graph")} for action in actions
                        if action["action"] in {"validate_frozen_program", "revise_relation_from_confirmation",
                            "refine_relation_parameter_space", "retain_program_experiment_counterexample"}],
                    "returned_costs": values.get("returned", {}).get("costs")}
                (DIRECTORY / "radius_regression_v2_failure.json").write_text(json.dumps(payload, indent=2) + "\n")
                break
            traceback = traceback.tb_next


raise SystemExit(pytest.main(["-q",
    "tests/test_cognition_nested_relation_revisions.py::test_nested_relation_uncertainty_is_researched_before_confirmation_and_old_call_remains_returnable"],
    plugins=[Diagnosis()]))
