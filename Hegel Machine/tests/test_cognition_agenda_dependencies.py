"""Declared prerequisites must become measured, actually consumed knowledge."""

from dataclasses import replace
import json

from hegel_machine.cognition.agenda import ResearchAgenda
from hegel_machine.cognition.agenda_dependency_demo import dependency_agenda, run_dependency_demo
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.models import Probe, ResearchConfig


SOURCE = {"kind": "suite", "seed": 71, "task_index": 0}
CONFIG = ResearchConfig(seed=71, observation_budget=48, compute_budget=9000, max_rounds=2)


def test_one_real_prerequisite_supplies_two_successors_after_json_resume(tmp_path):
    report = run_dependency_demo(tmp_path)
    assert report["initial_offers"] == [{"question_id": "shared_mechanism", "demand_count": 2, "expected_gain": None}]
    assert report["initial_blocked"]["parameter_transfer"] == ["shared_mechanism"]
    assert report["initial_blocked"]["renamed_transfer"] == ["shared_mechanism"]
    assert report["shared"]["status"] == "completed"
    visits = report["shared"]["visits"]
    assert [visit["question_id"] for visit in visits].count("shared_mechanism") == 1
    concept = report["after_source_before_resume"]["answer_concepts"][0]
    for visit in visits[1:]:
        assert visit["answer"]["selected_operation"] == "abstract_reuse"
        assert visit["answer"]["dependency_consumption"] == {"shared_mechanism": [concept]}
    assert all(item["loss"] < 1e-12 for item in report["shared"]["evaluation"].values())
    assert report["shared_minus_separate_costs"]["observation_cost"] < 0
    assert report["shared_minus_separate_costs"]["compute_cost"] < 0


def test_changed_concrete_demand_reorders_actual_prerequisite_execution(tmp_path):
    agenda = dependency_agenda(tmp_path)
    agenda.add_task("other_source", source={**SOURCE, "seed": 74},
                    config=replace(CONFIG, compute_budget=12000))
    assert agenda.offers()[0]["question_id"] == "shared_mechanism"
    agenda.set_dependencies("parameter_transfer", ["other_source"])
    agenda.set_dependencies("renamed_transfer", ["other_source"])
    offers = agenda.offers()
    assert offers[0]["question_id"] == "other_source" and offers[0]["demand_count"] == 2
    assert all(offer["expected_gain"] is None for offer in offers)
    report = agenda.run(max_steps=1)
    assert report["visits"][0]["question_id"] == "other_source"
    assert report["questions"]["shared_mechanism"]["visits"] == 0
    assert agenda.graph.nodes["other_source"].answer is not None


def test_failed_prerequisite_does_not_satisfy_a_downstream_request(tmp_path):
    agenda = ResearchAgenda(directory=tmp_path)
    agenda.add_task("noisy_source", source={**SOURCE, "task_index": 3}, config=CONFIG, max_visits=1)
    agenda.add_task("blocked_successor", source={**SOURCE, "task_index": 8},
                    config=CONFIG, depends_on=("noisy_source",))
    report = agenda.run()
    assert report["status"] == "blocked_on_dependencies"
    assert report["questions"]["blocked_successor"]["visits"] == 0
    assert not report["visits"][0]["answer"]["accepted"]
    assert agenda.graph.nodes["noisy_source"].answer is None


def test_new_paid_evidence_reopens_source_and_previously_completed_successors(tmp_path):
    agenda = dependency_agenda(tmp_path)
    agenda.run()
    assert all(node.answer is not None for node in agenda.graph.nodes.values())
    previous = {key: item["visits"] for key, item in agenda.state["questions"].items()}
    probe = Probe("later_support", tuple({"x": x, "z": -x} for x in (-1.2, -.4, .4, 1.2)))
    agenda.add_evidence("shared_mechanism", probe, make_suite(71)[0].observe(probe))
    assert all(node.answer is None for node in agenda.graph.nodes.values())
    assert all(node.alternatives for node in agenda.graph.nodes.values())
    restored = ResearchAgenda(directory=tmp_path, state=json.loads(json.dumps(agenda.to_dict())))
    assert [offer["question_id"] for offer in restored.offers()] == ["shared_mechanism"]
    restored.run()
    assert restored.state["questions"]["shared_mechanism"]["visits"] > previous["shared_mechanism"]
    assert any(restored.state["questions"][key]["visits"] > previous[key]
               for key in ("parameter_transfer", "renamed_transfer"))
    for key, item in restored.state["questions"].items():
        if item["visits"] == previous[key]:
            assert item["status"] == "ready" and restored.graph.nodes[key].answer is None


def test_completed_measurement_reopens_saved_workspace_with_new_paid_budget(tmp_path):
    agenda = ResearchAgenda(directory=tmp_path, observation_budget=400, compute_budget=300000,
                            measurement_budget=4096)
    agenda.add_measurement("measured", source={"kind": "measurement", "task_index": 0, "seed": 313},
        config=ResearchConfig(seed=313, observation_budget=64, compute_budget=60000, max_rounds=2),
        measurement_budget=512)
    first = agenda.run()
    previous = first["questions"]["measured"]
    assert previous["research_state"]["stage"] == "completed"
    old_rows = len(previous["research_state"]["workspace"]["observations"])
    old_episodes = len(previous["research_state"]["episodes"])
    agenda.add_task("new_required_knowledge", source=SOURCE, config=CONFIG)
    agenda.set_dependencies("measured", ["new_required_knowledge"])
    assert agenda.state["questions"]["measured"]["research_state"]["stage"] == "research"
    agenda = ResearchAgenda(directory=tmp_path, state=json.loads(json.dumps(agenda.to_dict())))
    second = agenda.run()
    continued = second["questions"]["measured"]
    assert len(continued["research_state"]["episodes"]) > old_episodes
    assert len(continued["research_state"]["workspace"]["observations"]) > old_rows
    later_visit = [visit for visit in second["visits"] if visit["question_id"] == "measured"][-1]
    assert later_visit["costs"]["observation_cost"] > 0 and later_visit["costs"]["compute_cost"] > 0
    assert continued["costs"]["compute_cost"] > previous["costs"]["compute_cost"]
    for name in ("observation_cost", "compute_cost", "measurement_cost"):
        evaluator = second.get("evaluator_compute_cost", 0) if name == "compute_cost" else 0
        assert sum(visit["costs"][name] for visit in second["visits"]) + evaluator == second["spent"][name]
        assert later_visit["costs"][name] <= later_visit["quoted_costs"][name]


def test_low_error_at_visit_limit_does_not_publish_an_open_research_answer(tmp_path):
    agenda = ResearchAgenda(directory=tmp_path)
    agenda.add_task("unresolved_boundary", source={**SOURCE, "task_index": 2},
        config=replace(CONFIG, max_rounds=1), max_visits=1, answer_tolerance=.2)
    agenda.add_task("consumer", source={**SOURCE, "task_index": 7}, config=CONFIG,
                    depends_on=("unresolved_boundary",))
    report = agenda.run()
    source = report["questions"]["unresolved_boundary"]
    assert source["result"]["validation_loss"] < .2 and source["result"]["concepts_added"]
    assert source["research_state"]["experiment_frontier"]
    assert not report["visits"][0]["answer"]["accepted"]
    assert report["questions"]["consumer"]["visits"] == 0
    assert report["status"] == "blocked_on_dependencies"
