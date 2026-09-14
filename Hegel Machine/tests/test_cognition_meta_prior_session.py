"""Public continuation must carry learned priors and actually resume experiments."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from hegel_machine.cli import main
from hegel_machine.cognition.frontier import FrontierSession, STAGES, frontier_summary
from hegel_machine.cognition.meta_prior_session import MetaPriorSession, meta_prior_summary


def test_cli_resume_inside_research_matches_full_sequence_and_retains_calls(tmp_path, capsys):
    path = tmp_path / "prior.json"
    args = ["cognition", "prior", "--checkpoint", str(path)]
    assert main([*args, "--research-steps", "2"]) == 0
    first = json.loads(path.read_text())
    assert first["active_task"]["research"]["research_state"]["stage"] == "test"
    assert first["costs"]["observation_cost"] == 16
    assert not first["knowledge"]["meta_priors"]
    capsys.readouterr()
    memory = tmp_path / "knowledge.json"
    assert main([*args, "--evaluate", "--memory", str(memory)]) == 0
    final = json.loads(path.read_text())
    summary = json.loads(capsys.readouterr().out)
    full = MetaPriorSession().run()
    assert final["costs"] == full["costs"]
    assert final["results"] == json.loads(json.dumps(full["results"]))
    assert len(summary["results"]) == 4 and not summary["pending_tasks"]
    assert all(row["mse"] < 1e-6 for row in final["evaluation"])
    assert final["results"][2]["research"]["kind"] == "residual_extension"
    assert json.loads(memory.read_text()) == final["knowledge"]
    assert main([*args, "--evaluate"]) == 0
    capsys.readouterr()
    assert json.loads(path.read_text()) == final


def test_frontier_continues_new_prior_stage_with_its_existing_library(tmp_path):
    session = FrontierSession(seed=31001, directory=tmp_path)
    # A saved lifecycle can already have completed the older stages. Only the
    # new missing stage should execute, through the actual frontier dispatch.
    session.state["stages"] = {stage: {"status": "completed"}
                               for stage in STAGES if stage != "meta_prior_research"}
    saved = []
    report = session.run(on_checkpoint=saved.append)
    stage = report["stages"]["meta_prior_research"]
    assert report["status"] == "completed" and not report["pending_stages"]
    assert stage["status"] == "completed" and len(stage["results"]) == 4
    assert len(report["knowledge"]["meta_priors"]) == 4
    assert any(checkpoint["stages"]["meta_prior_research"].get("active_task") for checkpoint in saved)
    assert all(row["mse"] < 1e-6 for row in stage["evaluation"])
    summary = frontier_summary(report)
    assert summary["completed_stages"] == len(STAGES)
    assert summary["meta_prior_research"]["costs"] == stage["costs"]
    resumed = FrontierSession(directory=tmp_path, state=json.loads(json.dumps(report)))
    assert resumed.run() == json.loads(json.dumps(report))


def test_public_optimizer_continuation_saves_adoption_and_never_rebuys_completed_tasks(tmp_path, capsys):
    project = Path("/home/erzhu419/mine_code/KG_op/SC-OLH-KG")
    if not project.is_dir():
        pytest.skip("the optional local SC-KG checkout is unavailable")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"source": {"kind": "sc_kg", "project_root": str(project),
        "problems": ["rzdt2", "rzdt2"], "dimension": 5, "seed": 843010020},
        "config": {"seed": 843010020, "observation_budget": 32, "compute_budget": 60000,
                   "minimum_improvement": 1e-7, "max_rounds": 2}}))
    optimizer_request = tmp_path / "optimizer.json"
    settings = json.loads((Path(__file__).resolve().parents[1] / "config/cognition_sc_kg_optimizer.json").read_text())
    optimizer_request.write_text(json.dumps({**settings, "N": 36, "K1": 3,
        "posterior_pool_size": 8, "posterior_keep": 1, "eval_pool_size": 8}))
    checkpoint, memory = tmp_path / "prior.json", tmp_path / "memory.json"
    arguments = ["cognition", "prior", "--checkpoint", str(checkpoint), "--optimize", "--memory", str(memory)]
    assert main([*arguments, "--request", str(request), "--optimizer-config", str(optimizer_request),
                 "--research-steps", "2"]) == 0
    paused = json.loads(checkpoint.read_text())
    assert paused["active_task"]["position"] == 0
    assert paused["active_task"]["research"]["research_state"]["stage"] == "test"
    assert paused["costs"]["observation_cost"] == 8
    assert paused["results"] == paused["optimizations"] == []
    assert paused["pending_tasks"] == [0, 1]
    assert json.loads(capsys.readouterr().out)["optimizer_pending_tasks"] == [0, 1]
    # Saved deployment configuration must also drive a resume that omits the
    # original --optimize flag; it cannot silently research both tasks first.
    resume = [item for item in arguments if item != "--optimize"]
    assert main([*resume, "--optimizer-steps", "1"]) == 0
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(checkpoint.read_text())
    optimization = report["optimizations"][0]
    assert len(report["results"]) == 1 and report["active_task"] is None
    assert report["pending_tasks"] == [1]
    assert report["results"][0]["available_prior_ids"] == []
    assert optimization["total_simulator_calls"] == 36
    assert len(optimization["adoption"]["observations"]) == optimization["new_optimizer_calls"]
    assert optimization["research_calls"] + optimization["new_optimizer_calls"] == 36
    assert len(optimization["hypothesis_revision"]["observations"]) == optimization["new_optimizer_calls"]
    assert optimization["revision_costs"]["recovery_observations"] == optimization["new_optimizer_calls"]
    assert optimization["hypothesis_revision"]["phase"] == "monitoring"
    assert optimization["scope_revisions"] == optimization["revision_prior_updates"] == []
    assert optimization["objective_research"]["compute_cost"] >= 0
    assert optimization["objective_costs"]["observations"] == optimization["new_optimizer_calls"]
    assert summary["optimizer_pending_tasks"] == [1]
    assert summary["status"] == report["status"] == "optimization_paused"
    assert summary["optimizations"][0]["total_simulator_calls"] == 36
    assert summary["optimizer_costs"]["total_optimizer_calls"] == 36
    assert summary["optimizer_costs"]["research_calls_included"] + summary["optimizer_costs"]["new_optimizer_calls"] == 36
    assert summary["optimizations"][0]["revision_research_costs"] == optimization["hypothesis_revision"]["costs"]
    assert any(prior.get("deployment_trials") for prior in report["knowledge"]["meta_priors"])
    assert json.loads(memory.read_text()) == report["knowledge"]
    assert main(resume) == 0
    capsys.readouterr()
    completed = json.loads(checkpoint.read_text())
    assert completed["status"] == "completed" and len(completed["optimizations"]) == 2
    assert completed["optimizations"][0] == optimization
    assert completed["results"][1]["available_prior_ids"] == [
        prior["prior_id"] for prior in report["knowledge"]["meta_priors"]]
    assert main(resume) == 0
    capsys.readouterr()
    assert json.loads(checkpoint.read_text()) == completed


def test_deployment_trials_follow_scope_boundaries_and_exact_prior_records():
    from hegel_machine.cognition.prior_adoption import PriorAdoption

    session = MetaPriorSession()
    session.library.meta_priors = [
        {"prior_id": "earlier_a", "source_concept_id": "a", "task_id": "earlier_task"},
        {"prior_id": "initial_a", "source_concept_id": "a", "task_id": "current_task"},
        {"prior_id": "replacement_b", "source_concept_id": "b", "task_id": "current_task:scope_32"},
        # An unchanged executable can be supported in a different data scope;
        # it must not inherit every trial previously associated with its ID.
        {"prior_id": "returned_a", "source_concept_id": "a", "task_id": "current_task:scope_36"},
    ]
    policy = PriorAdoption()
    for index in range(4):
        policy.observe(0., 1., 1., context={"stage": 33 + index,
            "purpose": "confirmation" if index == 1 else "optimization"})
    assert policy.selected == "learned"
    policy.begin_scope({"stage": 36, "scope_start": 32}, selected="learned")
    for index in range(4):
        policy.observe(0., 2., 0., context={"stage": 37 + index,
            "purpose": "coverage" if index < 2 else "validation"})
    assert policy.selected == "default"
    policy.begin_scope({"stage": 40, "scope_start": 36}, selected="learned")
    for index in range(3):
        policy.observe(0., 1., 1., context={"stage": 41 + index})
    result = {"adoption": policy.to_dict(), "revision_prior_updates": [
        {"prior_id": "replacement_b", "source_concept_id": "b", "scope_start": 32},
        {"prior_id": "returned_a", "source_concept_id": "a", "scope_start": 36}]}
    research = {"task_id": "current_task", "prior_updates": [
        {"prior_id": "initial_a", "source_concept_id": "a"}]}
    segments = session._record_optimizer_trials(research, result)
    assert [(item["observation_start"], item["observation_end"], item["selected"])
            for item in segments] == [(0, 4, "learned"), (4, 8, "default"), (8, 11, "learned")]
    assert "deployment_trials" not in session.library.meta_priors[0]
    trials = [prior["deployment_trials"][0] for prior in session.library.meta_priors[1:]]
    assert [trial["prior_id"] for trial in trials] == ["initial_a", "replacement_b", "returned_a"]
    assert [trial["new_predictions"] for trial in trials] == [4, 4, 3]
    assert [len(trial["decisions"]) for trial in trials] == [1, 1, 0]
    assert [row for trial in trials for row in trial["observations"]] == policy.to_dict()["observations"]
    assert sum(trial["comparison_costs"]["policy_observations"] for trial in trials) == 11
    assert sum(trial["comparison_costs"]["paired_prediction_rows"] for trial in trials) == 22
    assert [trial["observation_calls"] for trial in trials] == [
        {"optimization_calls": 3, "confirmation_calls": 1, "coverage_calls": 0, "validation_calls": 0},
        {"optimization_calls": 0, "confirmation_calls": 0, "coverage_calls": 2, "validation_calls": 2},
        {"optimization_calls": 3, "confirmation_calls": 0, "coverage_calls": 0, "validation_calls": 0}]
    assert [segment["observation_calls"] for segment in segments] == [trial["observation_calls"] for trial in trials]
    assert sum(sum(trial["observation_calls"].values()) for trial in trials) == 11


def test_saved_deployed_revision_is_consumed_before_the_next_tasks_first_proposal(monkeypatch):
    """Replay a real deployment, then test its callable on a controlled response.

    Task A's optimizer observations were bought by the saved experiment. Task
    B's analytic apparatus supplies research labels here; no SC-KG draw is made.
    """
    from hegel_machine.cognition.language import EvaluationContext, Program
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import Observation, ResearchConfig, TaskSpec

    source = Path(__file__).resolve().parents[1] / (
        "artifacts/sc_kg_constraint_knowledge_20260912/revision4/constraint_confirmation")
    initial = json.loads((source / "research/constraint_rewrite_1_initial.json").read_text())
    deployed = json.loads((source / "runs/constraint_rewrite_1_mechanism_change_hegel_constraint_knowledge.json").read_text())
    accepted = deployed["recovery"]["episodes"][0]["accepted_revision"]
    program = Program.from_dict(accepted["program"])
    assert accepted["scope_start"] == 45 and accepted["validation_mse"] < .01
    session = MetaPriorSession(ResearchConfig(observation_budget=32, compute_budget=60000,
        minimum_improvement=1e-7, max_rounds=2, seed=2546016007),
        source={"kind": "sc_kg", "project_root": str(source), "dimension": 3,
                "seed": 2546016007, "problems": ["state_policy_rzdt1", "rzdt2"]},
        library=KnowledgeLibrary.from_dict(initial["library"]))
    assert session.library.meta_priors == []
    session.state["results"] = [{"position": 0, "research": initial["result"],
                                 "available_prior_ids": []}]
    spec = TaskSpec(initial["result"]["task_id"], "Predict a measured constraint", ("x1", "x2", "x3"), (), (),
        feature_bounds={name: (0., 1.) for name in ("x1", "x2", "x3")}, max_probe_length=16)
    events, replacements = [], []

    def replay_deployment(config=None, *, max_tasks=None, on_checkpoint=None):
        assert not session.state["optimizations"] and len(session.state["results"]) == 1
        assert session.state["active_task"] is None
        events.append("deploy_A")
        updates = session._retain_optimizer_revisions(spec, deployed["recovery"], set())
        assert len(updates) == 1
        replacements.extend(updates)
        result = {"position": 0, "task_id": spec.task_id,
            "research_calls": 32, "new_optimizer_calls": 32, "total_simulator_calls": 64,
            "adoption": deepcopy(deployed["adoption_policy"]), "adoption_costs": {},
            "hypothesis_revision": deepcopy(deployed["recovery"]),
            "revision_prior_updates": updates, "scope_revisions": deepcopy(deployed["scope_revisions"]),
            "objective_research": deepcopy(deployed["objective_research"]), "final": deepcopy(deployed["final"])}
        result["deployment_segments"] = session._record_optimizer_trials(initial["result"], result)
        session.state["optimizations"].append(result)
        if on_checkpoint:
            on_checkpoint(session.to_dict())
        return session.to_dict()

    class Successor:
        def __init__(self):
            self.spec = replace(spec, task_id="successor_of_deployed_constraint")
            self.calls = 0

        def observe(self, probe):
            assert replacements and session.library.meta_priors
            events.append("research_B")
            self.calls += probe.cost
            return [Observation(dict(point), .2 + 1.5 * program.predict(Observation(dict(point), 0.)) ** 2)
                    for point in probe.sequence]

    successor = Successor()

    def environment(index):
        assert index == 1 and replacements
        return successor

    monkeypatch.setattr(session, "optimize", replay_deployment)
    monkeypatch.setattr(session, "_environment", environment)
    report = session.run_deploy({"N": 64}, research_steps=2)
    assert events == ["deploy_A", "research_B"]
    prior_id, source_id = replacements[0]["prior_id"], replacements[0]["source_concept_id"]
    active = report["active_task"]
    assert active["position"] == 1 and active["available_prior_ids"] == [prior_id]
    state = active["research"]["research_state"]
    assert state["stage"] == "test" and state["costs"]["observation_cost"] == successor.calls == 16
    incumbent = Program.from_dict(state["incumbent"]["candidate"]["program"])
    context = EvaluationContext(Observation({"x1": .43, "x2": .17, "x3": .81}, 0.))
    assert incumbent.predict(context.observation, context=context) == pytest.approx(
        .2 + 1.5 * program.predict(context.observation) ** 2, abs=1e-7)
    assert context.call_evaluations > 0
    assert any(call.definition.source_concept_id == source_id
               for call in session.library.callable_atoms(successor.spec, structural_transfer=False,
                                                          limit=len(session.library.definitions)))
    assert report["costs"]["observation_cost"] == 48
    assert len(report["optimizations"]) == len(report["results"]) == 1
    prior, = report["knowledge"]["meta_priors"]
    assert prior["source_concept_id"] == source_id and prior["deployment_trials"]


def test_research_task_limit_stops_only_after_that_tasks_deployment(monkeypatch):
    """The lifecycle seam uses real analytic research and no SC-KG draws."""
    from hegel_machine.cognition.meta_prior_lab import make_meta_prior_sequence

    session = MetaPriorSession(source={"kind": "sc_kg", "project_root": "unused_in_lifecycle_test",
                                      "seed": 31001, "problems": ["rzdt2", "rzdt2"]})
    environment = make_meta_prior_sequence(31001)[0]
    events = []

    def task_environment(index):
        assert index == 0, "the next task must wait until a subsequent lifecycle call"
        events.append("research_A")
        return environment

    def deployment(config=None, *, max_tasks=None, on_checkpoint=None):
        assert len(session.state["results"]) == 1 and session.state["active_task"] is None
        assert not session.state["optimizations"]
        assert session.state["results"][0]["research"]["status"] == "completed"
        events.append("deploy_A")
        session.state["optimizations"].append({"position": 0})
        return session.to_dict()

    monkeypatch.setattr(session, "_environment", task_environment)
    monkeypatch.setattr(session, "optimize", deployment)
    report = session.run_deploy({"N": 96}, max_tasks=1)
    assert events == ["research_A", "deploy_A"]
    assert report["status"] == "optimization_paused" and report["pending_tasks"] == [1]
    assert len(report["results"]) == len(report["optimizations"]) == 1
    assert report["costs"]["observation_cost"] == environment.observation_count


def test_repaired_observation_preserves_prior_and_paid_trials_without_superseded_decisions():
    from hegel_machine.cognition.prior_adoption import PriorAdoption

    session = MetaPriorSession()
    session.library.meta_priors = [{"prior_id": "learned_prior", "source_concept_id": "a",
                                   "task_id": "current_task", "status": "supported"}]
    policy = PriorAdoption()
    for stage in range(3):
        policy.observe(1., 0., 0., context={"stage": stage + 33, "purpose": "optimization"})
    before = policy.comparison_snapshot()
    policy.observe(1., 0., 1., context={"stage": 36, "purpose": "optimization"})
    policy.observe(0., 1., 0., context={"stage": 37, "purpose": "confirmation"})
    policy.repair_observation(3, 4, comparison_state=before,
        default_mean=1., learned_mean=0., noise_variance=0.)
    assert policy.selected == "learned"
    result = {"adoption": policy.to_dict(), "revision_prior_updates": []}
    research = {"task_id": "current_task", "prior_updates": [{"prior_id": "learned_prior"}]}
    segments = session._record_optimizer_trials(research, result)
    prior, = session.library.meta_priors
    trial, = prior["deployment_trials"]
    assert prior["status"] == "supported" and len(segments) == 1
    assert trial["new_predictions"] == 5 and trial["comparison_costs"]["paired_prediction_rows"] == 10
    assert trial["observation_calls"] == {"optimization_calls": 4, "confirmation_calls": 1,
                                         "coverage_calls": 0, "validation_calls": 0}
    assert trial["observations"][3]["target"] == 1.
    assert trial["observations"][3]["observation_status"] == "disputed"
    assert trial["observations"][4]["repaired_prediction"]["learned_error"] == 0.
    decision, = trial["decisions"]
    assert decision["observation_indices"] == [0, 1, 2, 4]
    assert decision["selected"] == "learned"
    assert any("superseded_by_observation_repair" in row for row in policy.to_dict()["decisions"])


def test_summary_includes_revision_compilation_and_counts_reused_calls_once():
    report = MetaPriorSession().to_dict()
    report["optimizations"] = [{"position": 0, "research_calls": 32,
        "new_optimizer_calls": 32, "total_simulator_calls": 64,
        "adoption": {"selected": "learned"}, "adoption_costs": {}, "final": {},
        "scope_revisions": [{"scope_start": 44}], "revision_costs": {"scope_model_fits": 3},
        "hypothesis_revision": {"phase": "monitoring", "costs": {
            "compute_cost": 1234, "confirmation_calls": 2, "coverage_calls": 8, "validation_calls": 4}}}]
    summary = meta_prior_summary(report)
    assert summary["optimizer_costs"] == {"total_optimizer_calls": 64,
        "research_calls_included": 32, "new_optimizer_calls": 32, "objective_research_compute": 0,
        "revision_research_costs": {"compute_cost": 1234, "confirmation_calls": 2,
                                    "coverage_calls": 8, "validation_calls": 4}}
    assert summary["optimizations"][0]["revision_research_costs"]["compute_cost"] == 1234
    # Confirmation/coverage/validation are purposes within the 32 new calls, not extra
    # simulator charges to add on top of the complete 64-call optimizer run.
    assert summary["optimizer_costs"]["total_optimizer_calls"] == (
        summary["optimizer_costs"]["research_calls_included"]
        + summary["optimizer_costs"]["new_optimizer_calls"])
