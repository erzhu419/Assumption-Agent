"""The study interventions preserve paid evidence and isolate runtime behavior."""

import importlib.util
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("investment_study",
        ROOT / "scripts/run_research_investment_attribution_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_deletion_drops_dependent_queued_work_but_keeps_own_paid_prefix(runner):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.memory import KnowledgeLibrary
    from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
    from hegel_machine.cognition.object_continuation import encode_entry

    library = KnowledgeLibrary()
    spec = TaskSpec("source", "Measured response", ("x",), (), (),
                    feature_bounds={"x": (-1., 1.)})
    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))
    rows = [Observation({"x": v}, v * v) for v in (-1., -.5, 0., .5, 1.)]
    source, _ = library.learn(Candidate("square", square, "refit"), spec, rows)
    call = next(p for p in library.callable_atoms(spec)
                if p.definition.source_concept_id == source)
    child_program = Program("add", children=(call, Program("const", value=1.)))
    child, _ = library.learn(Candidate("shifted_square", child_program, "refit",
                                     parents=(source,)), spec)
    unrelated, _ = library.learn(Candidate("absolute", Program("abs", children=(x,)), "refit"), spec)
    reduced = library.without({source})
    removed = {c.concept_id for c in library.concepts} - {c.concept_id for c in reduced.concepts}
    assert removed == {source, child} and unrelated not in removed

    def offer(candidate):
        return {"task_position": 1, "source_task_position": 0, "candidate": candidate,
                "relative_gain": .5,
                "minimum_resources": {"observation_cost": 48, "measurement_cost": 0, "compute_cost": 100}}

    by_source = offer({"program": square.to_dict(), "source_concept_ids": [source]})
    by_call = offer({"program": call.to_dict(), "source_concept_ids": []})
    by_child = offer({"program": child_program.expanded().to_dict(), "source_concept_ids": [child]})
    by_draft = offer({"method_opportunity": {"draft_ids": ["removed_draft"]}})
    own = offer({"program": Program("abs", children=(x,)).to_dict(), "source_concept_ids": [unrelated]})
    question = offer({"response_question": {"question_id": "own_unresolved_response",
        "parent": encode_entry({"candidate": Candidate("own_parent", x, "refit")}),
        "candidate": encode_entry({"candidate": Candidate("own_candidate",
            Program("abs", children=(x,)), "refit", parents=(unrelated,))})}})
    attempt = {"task_position": 1, "paid_batches": 2, "draft_id": "removed_draft",
               "deferred_offer": offer({"method_experiment": {"source_draft_ids": ["removed_draft"]}})}
    state = {"pending_revisits": [by_source, own, by_call, question, by_child, by_draft],
        "method_revisit_attempts": [attempt, {"draft_id": "own_draft", "deferred_offer": own}],
        "results": [{"research": {"program": square.to_dict(), "batches": [
            {"observations": [asdict(row) for row in rows]}]}}],
        "costs": {"observation_cost": len(rows), "measurement_cost": 0, "compute_cost": 100},
        "screening_compute_cost": 37, "knowledge": reduced.to_dict()}
    before = deepcopy(state)
    receipts = runner.prune_deleted_offers(state, removed, {"removed_draft"})
    assert len(receipts) == 5
    assert state["pending_revisits"] == [own, question]
    assert state["method_revisit_attempts"][0] == {key: value for key, value in attempt.items()
                                                  if key != "deferred_offer"}
    assert state["method_revisit_attempts"][1] == before["method_revisit_attempts"][1]
    for key in ("results", "costs", "screening_compute_cost", "knowledge"):
        assert state[key] == before[key]


def test_runtime_ablation_survives_json_and_terminal_waits_for_every_chain(runner, tmp_path):
    protocol = {"config": {"observation_budget": 192, "compute_budget": 120000},
        "roots": {"root01": {"tasks": []}, "root02": {"tasks": []}},
        "arms": {arm: {"runtime": "src"} for arm in
            ("baseline_156", "candidate", "delete_consumed_sources", "no_target_withdrawal")}}
    script = '''
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import run_research_investment_attribution_study as runner
settings = json.load(sys.stdin)
protocol, arm = settings['protocol'], settings['arm']
module = runner.runtime(protocol, arm)
import hegel_machine
from hegel_machine.cognition import memory
runtime = (runner.ROOT / protocol['arms'][arm]['runtime']).resolve()
assert Path(module.__file__).is_relative_to(runtime)
assert Path(memory.__file__).is_relative_to(runtime)
if arm in ('candidate', 'no_target_withdrawal'):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Candidate, Observation, TaskSpec
    from hegel_machine.cognition.evaluation import loss
    session = runner.session_for(module, protocol, 'root01')
    source = TaskSpec('source', 'Measured response', ('x',), (), (),
                      feature_bounds={'x': (-1., 1.)})
    target = TaskSpec('target', 'Measured response', ('x',), (), (),
                      feature_bounds={'x': (-1., 1.)})
    x = Program('feature', name='x')
    program = Program('mul', children=(x, x))
    rows = [Observation({'x': v}, v * v) for v in (-1., -.5, 0., .5, 1.)]
    source_id, _ = session.library.learn(Candidate('square', program, 'refit'), source, rows)
    changed = [Observation(row.features, 2. * row.target) for row in rows]
    error = loss(program, changed, 0., .5)
    assert error > 0.
    evidence = {'status': 'counterexample', 'measured_loss': error, 'rows': len(changed)}
    expected = arm == 'candidate'
    for restored in (False, True):
        if restored:
            session = runner.session_for(module, protocol, 'root01',
                json.loads(json.dumps(session.to_dict())))
        session.library.record_transfer_counterexample(target, program, [source_id],
            evidence=evidence, probe_id='measured_changed_response')
        assert session.library.transfer_refuted(target, program) is expected
    assert session.library.retrieve(source)
    session.library.record_counterexample(source_id, target, error, 0., changed[0])
    assert not session.library.retrieve(target)
    assert session.library.retrieve(source)
print(json.dumps({'arm': arm, 'version': hegel_machine.__version__}))
'''
    for arm in ("candidate", "no_target_withdrawal"):
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
            input=json.dumps({"protocol": protocol, "arm": arm}),
            capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)["arm"] == arm

    for arm in protocol["arms"]:
        for root in protocol["roots"]:
            runner.save(runner.checkpoint(tmp_path, arm, root),
                        {"study": {"learning_complete": True}})
    # A different arm's last root still learning blocks even the first arm.
    runner.save(runner.checkpoint(tmp_path, "no_target_withdrawal", "root02"),
                {"study": {"learning_complete": False}})
    with pytest.raises(ValueError, match="all four arms must finish learning"):
        runner.evaluate(tmp_path, protocol, "baseline_156", "root01")
    assert not list(tmp_path.rglob("evaluation.json"))


def test_deleted_source_response_question_cannot_be_requeued_from_paid_report(runner):
    from hegel_machine.cognition.language import Program
    from hegel_machine.cognition.models import Candidate
    from hegel_machine.cognition.object_continuation import encode_entry
    from hegel_machine.cognition.response_research_questions import response_question_offer

    x = Program("feature", name="x")
    square = Program("mul", children=(x, x))

    def entry(name, program, sources=()):
        return encode_entry({"candidate": Candidate(name, program, "mechanism_revision",
            parents=tuple(sources)), "kind": "mechanism_full_refit", "family_id": "T03",
            "fallback": 0., "method_source_ids": list(sources)})

    withdrawn = {"question_id": "response_question_0001", "status": "pending",
        "parent": entry("old_temporal_refit", x, ("deleted_source",)),
        "candidate": entry("old_full_refit", square, ("deleted_source",)),
        "observed_gain": .1, "uncertainty_margin": .2, "reference_loss": 1.,
        "candidate_loss": .9, "rows": 4, "probe_ids": ["paid_pilot"],
        "gains": [], "reference_losses": [], "candidate_losses": []}
    own = {**deepcopy(withdrawn), "question_id": "own_question",
           "parent": entry("own_parent", x), "candidate": entry("own_candidate", square)}
    report = {"status": "unresolved", "program": x.to_dict(), "method_source_ids": [],
        "used_sources": [], "selected_channels": [], "selection_loss": 1.,
        "batches": [{"probe": {"probe_id": "paid_pilot", "sequence": [{"x": .2}, {"x": .5}]},
            "observations": [{"features": {"x": .2}, "target": .04, "history": []},
                {"features": {"x": .5}, "target": .25, "history": [{"x": .2, "__target__": .04}]}]}],
        "response_questions": [withdrawn, own]}
    resources = {"observation_cost": 48, "measurement_cost": 0, "compute_cost": 500}
    limits = {"observation_budget": 192, "measurement_budget": 0, "compute_budget": 12000}

    def requeue(question):
        offer = response_question_offer(report, question, 24, resources, resource_limits=limits)
        if offer is not None:
            offer.update(task_position=2, source_task_position=2)
        return offer

    state = {"results": [{"task_position": 2, "research": report}],
        "pending_revisits": [requeue(withdrawn), requeue(own)], "method_revisit_attempts": [],
        "costs": {"observation_cost": 2, "measurement_cost": 0, "compute_cost": 100}}
    assert all(state["pending_revisits"])
    before = deepcopy(state)
    receipts = runner.prune_deleted_offers(state, {"deleted_source"}, set())
    assert receipts and withdrawn["status"] == "withdrawn_source_deletion"
    assert requeue(withdrawn) is None and requeue(own) is not None
    assert [offer["question_id"] for offer in state["pending_revisits"]] == ["own_question"]
    assert state["costs"] == before["costs"]
    for key in ("program", "batches", "method_source_ids", "used_sources"):
        assert report[key] == before["results"][0]["research"][key]
    assert {key: value for key, value in withdrawn.items() if key != "status"} == {
        key: value for key, value in before["results"][0]["research"]["response_questions"][0].items()
        if key != "status"}
    assert own == before["results"][0]["research"]["response_questions"][1]
