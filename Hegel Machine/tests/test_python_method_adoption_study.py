"""Adoption selects existing code; fresh practice still decides publication."""

from copy import deepcopy
import json
from pathlib import Path

from scripts import run_python_method_adoption_study as runner


def study_inputs():
    sources = json.loads(runner.SOURCE_OUTCOMES.read_text())["sources"]
    reference_plan = json.loads((runner.BRIDGE / "protocol.json").read_text())
    reference = json.loads((runner.BRIDGE / "summary.json").read_text())
    methods = {
        "historical_bridge": (runner.BRIDGE / "historical_bridge.py").read_text(),
        "unmodified_candidate": (runner.BRIDGE / "unmodified_candidate.py").read_text(),
    }
    return sources, reference_plan, reference, methods


def test_independent_adopters_rotate_library_order_and_buy_fresh_confirmation_streams():
    sources, reference_plan, _, _ = study_inputs()
    plan = runner.study_protocol(sources)
    replicas = plan["author_replicas"]
    assert [item["id"] for item in replicas] == ["author_0", "author_1", "author_2"]
    assert replicas[0]["library_order"] == replicas[1]["library_order"][::-1]
    for replica in replicas:
        assert set(replica["library_order"]) == {"unmodified_candidate", "historical_bridge"}
        assert set(replica["practice_cases"]) == set(reference_plan["practice_cases"])
    for name, original in reference_plan["practice_cases"].items():
        specs = [replica["practice_cases"][name] for replica in replicas]
        assert len({spec["confirmation_seed"] for spec in specs}) == 3
        assert all(spec["confirmation_seed"] != original["confirmation_seed"] for spec in specs)
        assert all({key: value for key, value in spec.items() if key != "confirmation_seed"}
                   == {key: value for key, value in original.items() if key != "confirmation_seed"}
                   for spec in specs)
    for name in ("transfer_cases", "limits", "procedure_step_limit", "evaluation"):
        assert plan[name] == reference_plan[name]


def test_only_identical_published_source_and_target_contract_can_reuse_existing_outcomes():
    sources, reference_plan, reference, methods = study_inputs()
    plan = runner.study_protocol(sources)
    saved = deepcopy(reference)
    reused = runner.reusable_targets(methods["historical_bridge"], plan,
        reference_plan, reference, methods)
    assert len(reused) == 10
    assert {item["target_id"] for item in reused} == {
        spec["id"] for spec in plan["transfer_cases"]}
    assert all(item["confirmed_and_correct"] for item in reused)
    reused[0]["costs"]["observation_cost"] = -1
    assert reference == saved

    # A source edit requires target execution even if it only changes a comment.
    changed = methods["historical_bridge"] + "\n# changed source\n"
    assert runner.reusable_targets(changed, plan, reference_plan, reference, methods) is None
    for field in ("transfer_cases", "limits", "procedure_step_limit", "evaluation"):
        changed_plan = deepcopy(plan)
        if field == "transfer_cases":
            changed_plan[field][0]["confirmation_seed"] += 1
        elif field == "limits":
            changed_plan[field]["observation_budget"] += 1
        elif field == "procedure_step_limit":
            changed_plan[field] += 1
        else:
            changed_plan[field] = {**changed_plan[field], "different_evaluation": True}
        assert runner.reusable_targets(methods["historical_bridge"], changed_plan,
            reference_plan, reference, methods) is None


def test_explicit_library_copy_runs_real_five_case_practice_and_failed_authors_are_not_adopted(tmp_path):
    _, _, _, methods = study_inputs()
    seen = {}

    class ScriptedProvider:
        """Return ordinary actions only; the real learner executes every practice."""

        def __init__(self, workspace):
            self.workspace = Path(workspace)
            self.calls = 0
            self.name = next(part for part in self.workspace.parts if part.startswith("author_"))
            seen[self.name] = self

        def next_action(self, request):
            self.calls += 1
            usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0,
                     "cached_input_tokens": 0}
            if self.calls == 1:
                # Merely supplying a library never replaces the inherited bad choice.
                assert (self.workspace / runner.METHOD_PATH).read_text() == methods["unmodified_candidate"]
                assert request["files"][runner.METHOD_PATH] == methods["unmodified_candidate"]
                assert not list((self.workspace / "trials").glob("trial_*"))
                if self.name != "author_0":
                    return {"action": "provider_error", "arguments": {}, "usage": usage,
                            "error": {"message": "offline fixture stops before any adoption"}}
                candidates = [name for name, source in request["files"].items()
                              if source == methods["historical_bridge"]]
                assert len(candidates) == 1
                self.chosen = candidates[0]
                code = ("from pathlib import Path\n"
                        f"Path({runner.METHOD_PATH!r}).write_text(Path({self.chosen!r}).read_text())\n")
                return {"action": "run", "arguments": {"code": code}, "usage": usage}
            assert self.calls == 2
            assert (self.workspace / runner.METHOD_PATH).read_text() == methods["historical_bridge"]
            assert request["files"][runner.METHOD_PATH] == methods["historical_bridge"]
            assert request["events"][-1]["feedback"]["returncode"] == 0
            assert set(request["events"][-1]["feedback"]["current_source_coverage"]["missing_cases"]) == {
                "case_0", "case_1", "case_2", "case_3", "case_4"}
            return {"action": "finish", "arguments": {"path": runner.METHOD_PATH}, "usage": usage}

    output = tmp_path / "adoption"
    result = runner.run_study(output, provider_factory=ScriptedProvider,
                              auth_mode="offline_scripted_provider")
    assert len(seen) == 3
    assert seen["author_0"].calls == 2
    assert seen["author_1"].calls == seen["author_2"].calls == 1

    # Read actual learner records, not a mocked numerical-success summary.
    adopted = json.loads((seen["author_0"].workspace / "result.json").read_text())
    assert adopted["status"] == "completed"
    assert adopted["method_source"] == methods["historical_bridge"]
    assert [event["action"] for event in adopted["events"]] == ["run", "finish"]
    trials = adopted["practice_trials"]
    assert len(trials) == 5 and all(trial["status"] == "completed" for trial in trials)
    coverage = adopted["events"][-1]["feedback"]["current_source_coverage"]
    assert coverage["missing_cases"] == []
    assert coverage["scoped_revision"] == {
        "zero_meter_scoped_recovery_cases": ["case_3"],
        "measured_recovery_cases": ["case_4"], "distinct_cases": True}
    assert all(trial["method_source"] == methods["historical_bridge"] for trial in trials)
    assert all(trial["costs"]["model_calls"] == 0 for trial in trials)
    for name in ("author_1", "author_2"):
        failed = json.loads((seen[name].workspace / "result.json").read_text())
        assert failed["status"] == "provider_failed"
        assert failed["method_source"] is None and failed["practice_trials"] == []
        assert (seen[name].workspace / runner.METHOD_PATH).read_text() == methods["unmodified_candidate"]

    assert result["published_authors"] == result["exact_supplied_adoptions"] == 1
    assert sum(len(author["results"]) for author in result["authors"]) == 10
    assert result["skipped_target_count"] == 20
    assert [len(author["skipped"]) for author in result["authors"]] == [0, 10, 10]
    assert result["new_target_count"] == 0
    assert result["new_target_costs"] == runner.total_costs([])
    assert result["reused_target_records"] == result["unique_reused_target_executions"] == 10
    assert result["new_work_costs"]["reported_tokens"] == 0
