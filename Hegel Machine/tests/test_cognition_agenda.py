"""Real mixed-engine scheduling, saved work and asynchronous evidence delivery."""

from dataclasses import replace
import json
import time

import pytest

from hegel_machine.cognition.agenda import RESOURCES, ResearchAgenda
from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Probe, ResearchConfig
from hegel_machine.cognition.workspace import ResearchWorkspace


SOURCE = {"kind": "suite", "seed": 71, "partition": "development", "task_index": 0}
TASK_CONFIG = ResearchConfig(seed=71, observation_budget=32, compute_budget=6000, max_rounds=2)


def mixed_agenda(directory, **limits):
    agenda = ResearchAgenda(directory=directory, observation_budget=400, compute_budget=350000,
                            measurement_budget=4096, **limits)
    # Deliberately register expensive object work before cheaper instruments.
    agenda.add_structure("objects", source={"kind": "structure", "objects": 2, "readout": "total",
        "variant": "exchange", "seed": 71}, config=ResearchConfig(seed=71,
        observation_budget=80, compute_budget=100000, max_rounds=2), measurement_budget=2048)
    agenda.add_measurement("instruments", source={"kind": "measurement", "task_index": 0,
        "seed": 313, "backend": "local"}, config=ResearchConfig(seed=313,
        observation_budget=64, compute_budget=60000, max_rounds=2), measurement_budget=512)
    return agenda


def test_three_actual_engines_compete_with_shared_knowledge_and_json_resume(tmp_path):
    agenda = mixed_agenda(tmp_path)
    agenda.add_task("temporal", source=SOURCE, config=TASK_CONFIG)
    assert [offer["question_id"] for offer in agenda.offers()] == ["temporal", "instruments", "objects"]
    assert all(offer["expected_gain"] is None for offer in agenda.offers())
    partial = agenda.run(max_steps=2)
    assert [visit["question_id"] for visit in partial["visits"]] == ["temporal", "instruments"]
    measured_before = partial["questions"]["instruments"]["result"]["costs"]
    assert partial["questions"]["instruments"]["research_state"]["stage"] == "research"
    restored = ResearchAgenda(directory=tmp_path, state=json.loads(json.dumps(partial)))
    report = restored.run()
    assert report["status"] == "completed"
    assert {visit["kind"] for visit in report["visits"]} == {"task", "measurement", "structure"}
    assert len(report["knowledge"]["concepts"]) >= partial["visits"][0]["concepts_after"] > 0
    assert report["knowledge"]["structural_priors"]
    measured = report["questions"]["instruments"]
    assert measured["result"]["selected_channels"]
    assert measured["result"]["costs"]["observation_cost"] > measured_before["observation_cost"]
    assert report["questions"]["objects"]["result"]["selected_scale"] >= 1
    for name in RESOURCES:
        evaluator = report.get("evaluator_compute_cost", 0) if name == "compute_cost" else 0
        assert report["spent"][name] == sum(visit["costs"][name] for visit in report["visits"]) + evaluator
        assert report["spent"][name] <= report["limits"][name]
        assert all(visit["costs"][name] <= visit["quoted_costs"][name] for visit in report["visits"])
    assert report["spent"]["measurement_cost"] > 0 and report["spent"]["experiment_cost"] == 0


def test_new_question_can_preempt_existing_work_and_pending_feedback_releases_resources(tmp_path):
    agenda = mixed_agenda(tmp_path)
    agenda.add_task("awaited", source=SOURCE, config=TASK_CONFIG)
    probe = Probe("delayed_new_history", tuple({"x": x, "z": -x} for x in (-.8, -.2, .4, .9)))
    agenda.submit_experiment("awaited", probe, delay_seconds=.5)
    assert "awaited" not in [offer["question_id"] for offer in agenda.offers()]
    first = agenda.run(max_steps=1)
    assert first["visits"][0]["question_id"] == "instruments"
    assert first["pending_experiments"]
    agenda = ResearchAgenda(directory=tmp_path, state=json.loads(json.dumps(first)))
    agenda.add_task("arrived_later", source={**SOURCE, "task_index": 1}, config=TASK_CONFIG)
    assert agenda.offers()[0]["question_id"] == "arrived_later"
    deadline = time.monotonic() + 5.
    while agenda.state["pending_experiments"] and time.monotonic() < deadline:
        agenda.poll_experiments()
        time.sleep(.02)
    assert not agenda.state["pending_experiments"]
    awaited = agenda.state["questions"]["awaited"]
    assert awaited["incoming_evidence"] and len(awaited["research_state"]["observations"]) == probe.cost
    assert "awaited" in [offer["question_id"] for offer in agenda.offers()]
    paid = dict(agenda.state["spent"])
    assert agenda.poll_experiments() == [] and agenda.state["spent"] == paid
    report = agenda.run()
    assert report["spent"]["experiment_cost"] == 1
    assert report["spent"]["observation_cost"] == probe.cost + sum(visit["costs"]["observation_cost"] for visit in report["visits"])
    assert report["questions"]["awaited"]["visits"] >= 1


def test_unaffordable_instrument_work_stays_pending_without_purchasing_anything(tmp_path):
    agenda = ResearchAgenda(directory=tmp_path, observation_budget=300, compute_budget=300000,
                            measurement_budget=10)
    agenda.add_measurement("too_expensive", source={"kind": "measurement", "task_index": 0, "seed": 313},
        config=ResearchConfig(observation_budget=64, compute_budget=60000), measurement_budget=512)
    agenda.add_task("affordable", source=SOURCE, config=TASK_CONFIG)
    report = agenda.run()
    assert report["questions"]["too_expensive"]["visits"] == 0
    assert report["questions"]["too_expensive"]["status"] == "ready"
    assert report["spent"]["measurement_cost"] == 0
    assert report["status"] == "no_payable_research"


def test_actual_return_changes_cross_question_selection_without_a_fabricated_gain(tmp_path):
    environment = make_suite(71, "development")[0]
    rows = [row for probe in environment.spec.initial_probes for row in environment.observe(probe)]
    old = Candidate("existing_bad_explanation", Program("const", value=0.), "refit")
    workspace = ResearchWorkspace(environment.spec.task_id, observations=rows, reference=list(rows),
        incumbent=old, bank=[old], fallback=0., initial_loss=loss(old.program, rows, 0., .5))
    agenda = ResearchAgenda(directory=tmp_path, observation_budget=400, compute_budget=100000)
    agenda.add_task("measured_repair", source=SOURCE, config=TASK_CONFIG,
                    research_state=workspace.to_dict(), max_visits=1)
    first = agenda.run()
    gain = first["visits"][0]["feedback"]["normalized_gain"]
    assert gain > 0. and first["visits"][0]["feedback"]["reference_rows"] > 0
    restored = ResearchAgenda(directory=tmp_path, state=json.loads(json.dumps(first)))
    restored.add_task("new_without_model", source=SOURCE,
        config=replace(TASK_CONFIG, observation_budget=16, compute_budget=3000))
    restored.add_task("supported_repair", source=SOURCE, config=TASK_CONFIG,
        research_state=workspace.to_dict(), max_visits=1)
    offers = restored.offers()
    assert offers[0]["question_id"] == "supported_repair"
    assert offers[0]["expected_gain"] == min(1., gain) and not offers[0]["exploration"]
    assert next(offer for offer in offers if offer["question_id"] == "new_without_model")["expected_gain"] is None


def test_direct_external_delivery_is_paid_and_cannot_exceed_shared_resources(tmp_path):
    environment = make_suite(71, "development")[0]
    probe = Probe("direct_external", tuple({"x": x, "z": -x} for x in (-.8, -.2, .4, .9)))
    rows = environment.observe(probe)
    agenda = ResearchAgenda(directory=tmp_path, observation_budget=5, measurement_budget=6)
    agenda.add_task("external", source=SOURCE, config=TASK_CONFIG)
    agenda.add_evidence("external", probe, rows, measurement_cost=3)
    assert agenda.state["spent"]["observation_cost"] == 4
    assert agenda.state["spent"]["measurement_cost"] == 3
    with pytest.raises(ValueError, match="shared agenda resources"):
        agenda.add_evidence("external", probe, rows)
    assert len(agenda.state["questions"]["external"]["research_state"]["observations"]) == 4


def test_public_cli_resumes_three_processes_with_dynamic_questions_and_actual_feedback(tmp_path):
    from hegel_machine.cognition.cli import main

    checkpoint = tmp_path / "agenda.json"
    assert main(["cognition", "agenda-demo", "--checkpoint", str(checkpoint)]) == 0
    report = json.loads((tmp_path / "agenda_demo_summary.json").read_text())
    first, second, third = report["processes"]
    assert first["selected"] == ["instruments"] and first["pending_experiments"] == 1
    assert second["selected"] == ["instruments", "arrived_later"]
    assert 2 <= third["selected"].index("awaited") < third["selected"].index("objects")
    assert third["pending_experiments"] == 0
    assert report["status"] == "completed"
    assert {item["kind"] for item in report["questions"]} == {"task", "measurement", "structure"}
    assert report["concepts"] >= second["concepts"] > first["concepts"]
    assert report["structural_priors"] > 0
    assert len(report["deliveries"]) == 1 and report["deliveries"][0]["rows"] == 4
    assert report["external_costs"] == {"observation_cost": 4, "measurement_cost": 0, "experiment_cost": 1}
    for name in RESOURCES:
        assert 0 < report["spent"][name] <= report["limits"][name]
        evaluator = report.get("evaluator_compute_cost", 0) if name == "compute_cost" else 0
        assert report["spent"][name] == report["actual_visit_costs"][name] + report["external_costs"].get(name, 0) + evaluator
