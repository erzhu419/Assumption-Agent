"""Paid-prefix isolation, terminal timing and negative outcomes for the bridge study."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path

from hegel_machine.cognition.models import ResearchConfig
from test_scoped_retention_comparison import LinearLab


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_workflow_bridge_study.py"
    spec = importlib.util.spec_from_file_location("workflow_bridge_study",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_method_intervention_preserves_all_other_session_state():
    study = runner()
    source = {"knowledge":{"research_workflows":{"methods":[{"method_id":"acquired"}],"next_method_index":4},
        "concepts":[{"concept_id":"existing"}],"instrument_models":[{"model_id":"sensor"}],
        "program_experiment_methods":[{"method_id":"legacy"}],"action_experience":{"paid":3}},
        "results":[{"research":{"costs":{"compute_cost":19}}}],"pending_revisits":[{"paid":True}]}
    retained,removed = (study._fork_state(source,arm) for arm in study.ARMS)
    assert retained == source
    expected = deepcopy(source)
    expected["knowledge"]["research_workflows"]["methods"] = []
    assert removed == expected
    removed["knowledge"]["concepts"].clear()
    assert retained == source


def test_real_learning_prefix_is_shared_and_all_learning_precedes_terminal_scoring():
    study = runner()
    events,evaluations,purchases = [],[],[]

    class PurchasedLab(LinearLab):
        def acquire(self,probe,channels=()):
            batch = super().acquire(probe,channels)
            purchases.append((self.seed,batch.observation_cost,batch.measurement_cost))
            return batch

    plan = study.protocol(62001)
    plan["tasks"] = [{"coefficients":coefficients,"seed":62001+10007*i}
        for i,coefficients in enumerate(((1.,0.),(0.,1.),(1.,1.),(1.,2.),(1.,1.),(1.,2.)))]
    plan["research_config"] = asdict(ResearchConfig(observation_budget=96,compute_budget=24000,
        minimum_improvement=1e-7,max_rounds=1,seed=62001))
    plan["measurement_budget"] = 0
    plan["evaluation"] = {"length":8,"trajectories":1,"seed_offset":1000003}
    plan["impulse_amplitudes"] = []
    result = study.run_study(plan,lab_factory=lambda task:
        PurchasedLab(**task,events=events,evaluations=evaluations))
    assert result["intervention_active"]
    assert result["prefix_workflow_ids"]
    retained,removed = (result["results"][arm] for arm in study.ARMS)
    assert retained["initial_workflows"]["methods"]
    assert removed["initial_workflows"]["methods"] == []
    assert retained["initial_workflows"]["next_method_index"] == removed["initial_workflows"]["next_method_index"]
    assert removed["final_workflows"]["methods"]  # It can learn naturally after deletion.
    assert not set(result["prefix_workflow_ids"]) & {
        method["method_id"] for method in removed["final_workflows"]["methods"]}
    assert retained["tasks"][2]["available_concepts"] == removed["tasks"][2]["available_concepts"]
    for report in result["results"].values():
        assert report["solved"] == report["fresh_confirmed"] == report["confirmed_and_solved"] == 6
        for key in study.COST_KEYS:
            total = sum(task["costs"][key] for task in report["tasks"])
            total += report["screening_compute_cost"] if key == "compute_cost" else 0
            assert report["costs"][key] == total
            assert report["suffix_costs"][key]+result["common_prefix_costs"][key] == total
            assert report["evaluation_costs"][key] == sum(task["evaluation_costs"][key]
                for task in report["tasks"])
        assert report["costs"]["observation_cost"] <= 6*96
    assert sum(rows for _,rows,_ in purchases) == result["common_prefix_costs"]["observation_cost"] + sum(
        report["suffix_costs"]["observation_cost"] for report in result["results"].values())
    first = events.index("evaluation")
    assert set(events[:first]) == {"learning"}
    assert events[first:] == ["evaluation"]*12
    assert Counter(evaluations) == Counter({(task["seed"],task["seed"]+1000003,8,1):2
        for task in plan["tasks"]})
    assert result["external_model_calls"] == 0

    # The reporting path keeps a poor arm and execution failures in the roster.
    negative = deepcopy(result)
    poor = negative["results"]["retained_workflows"]["tasks"][3]
    poor.update(solved=False,confirmed_and_solved=False,mse=1.,status="unresolved")
    negative["results"]["retained_workflows"]["solved"] -= 1
    negative["paired_retained_minus_removed"]["solved"] = -1
    summary = study.summarize([negative],[{"seed":99001,"exception":"test failure"}],bootstrap_samples=20)
    assert summary["planned_cases"] == 2 and summary["completed_cases"] == 1
    assert summary["paired_retained_minus_removed"]["solved"]["interval_95"] == [-1.,-1.]
    assert summary["totals"]["retained_workflows"]["failed_tasks"][0]["task_position"] == 3
    assert len(summary["execution_failures"]) == 1


def test_registered_roster_balances_the_two_existing_instrument_modes():
    plan = runner().cohort_protocol()
    assert len({case["seed"] for case in plan["cases"]}) == 12
    assert Counter(case["instrument_mode"] for case in plan["cases"]) == {"finite":6,"recurrent":6}


def test_existing_paid_unresolved_return_reaches_workflow_feedback():
    result = runner().run_return_regression()
    assert result["initial"]["status"] == "unresolved"
    assert result["queued_offers"] and result["revisit_count"] == 1
    assert any(event["action"] == "choose_research_portfolio" for event in result["scheduler_events"])
    returned = result["results"][0]
    assert returned["status"] == "completed" and returned["confirmation_loss"] <= 1e-7
    assert returned["knowledge_return"]
    events = returned["continuation_actions"]
    assert any(event["action"] == "compare_continuation_candidates" for event in events)
    assert any(event["action"] == "validate_continued_program" for event in events)
    assert any(item["production_methods"] for item in result["produced_concepts"])
    assert any(method.get("knowledge_outcomes") for method in result["workflows"]["methods"])
    assert all(score["mse"] < 1e-6 for score in result["evaluation"])
