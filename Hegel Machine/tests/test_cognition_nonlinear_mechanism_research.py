"""Measured counterexamples revise a mechanism inside a nonlinear learned call."""

from copy import deepcopy
from dataclasses import replace

import pytest

from hegel_machine.cognition.competing_object_experiments import (
    assess_competing_object_probe, plan_competing_object_probe,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import MeasurementBatch
from hegel_machine.cognition.mechanism_revision import propose_mechanism_revisions
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from test_cognition_mechanism_revision_research import CONFIG, restored, _tested_mechanisms
from test_cognition_nested_relation_revisions import composition
from test_cognition_program_action_followup import NestedResponseLab


class NonlinearMechanismLab(NestedResponseLab):
    def __init__(self, event_gain, episode, outer_gain=1.):
        super().__init__(.6)
        self.event_gain, self.outer_gain = event_gain, outer_gain
        self.spec = replace(self.spec, task_id=episode)

    def acquire(self, probe, channels=()):
        assert not channels
        rows, history, flows, combined = [], [], [], []
        for point in probe.sequence:
            flow = ((point['f0'] - point['f1']) * (point['q1'] - point['q0'])
                    if abs(point['x0'] - point['x1']) < self.radius else 0.)
            event = flow - flows[-1] if flows else 0.
            flows.append(flow)
            combined.append(1.3 * (point['q0'] + point['q1']) + self.event_gain * event)
            target = self.outer_gain * sum(combined[-3:]) ** 2
            rows.append(Observation(dict(point), target, tuple(history)))
            history.append({**point, '__target__': target})
        self.acquisitions.append({'probe': probe, 'observations': rows})
        return MeasurementBatch(rows, len(rows), 0, ())


def contains(program, operator):
    return program.op == operator or any(contains(child, operator) for child in program.children)


def nonlinear_source():
    lab = NonlinearMechanismLab(-.7, 'nonlinear_source')
    paid = lab.sample(31, 64).observations
    inner = composition(.6).expanded()
    program = Program('mul', children=(inner, inner))
    assert loss(program, paid, 0., .5) < 1e-24
    library = KnowledgeLibrary()
    source_id, added = library.learn(Candidate('observed_nonlinear_composition', program,
        'observed_response'), lab.spec, paid)
    assert added
    return restored(library), source_id


def run_nonlinear_episode(library, gain, episode):
    lab = NonlinearMechanismLab(gain, episode)
    report = UnifiedResearchAgent(CONFIG, library=library).run(lab)
    assert report['costs']['observation_cost'] == sum(
        len(batch['observations']) for batch in lab.acquisitions) <= CONFIG.observation_budget
    assert report['costs']['measurement_cost'] == 0
    assert report['costs']['compute_cost'] <= CONFIG.compute_budget
    # These trajectories are unavailable to fitting, probe choice or stopping.
    fresh = [row for seed, length in ((991, 1), (992, 3), (993, 47))
             for row in lab.sample(seed, length).observations]
    error = loss(Program.from_dict(report['program']), fresh, report['fallback'], .5)
    return report, lab, error


@pytest.fixture(scope='module')
def nonlinear_history():
    library, source_id = nonlinear_source()
    before = deepcopy(library.to_dict())
    removed, removed_lab, removed_error = run_nonlinear_episode(library, 0., 'nonlinear_removed')
    after_removal = deepcopy(library.to_dict())
    returned_library = restored(library)
    returned, returned_lab, returned_error = run_nonlinear_episode(
        returned_library, -.7, 'nonlinear_returned')
    return locals()


def test_internal_deletion_preserves_square_and_window_after_real_experiment(nonlinear_history):
    history = nonlinear_history
    report = history['removed']
    tested, outcomes = _tested_mechanisms(report, history['removed_lab'])
    assert report['status'] == 'completed' and history['removed_error'] < CONFIG.minimum_improvement
    assert report['selected_kind'] == 'mechanism_deletion'
    assert outcomes['retained_mechanism'][0][1]['status'] == 'counterexample'
    assert any(outcome['status'] == 'supported_on_probe'
               for _, outcome in outcomes['mechanism_deletion'])
    program = Program.from_dict(report['program']).expanded()
    assert contains(program, 'mul') and contains(program, 'window_sum')
    assert not contains(program, 'neighbor_sum') and not contains(program, 'event_delta')
    assert program.history_depth == 2
    confirmation = next(event for event in report['actions']
                        if event['action'] == 'validate_frozen_program' and event['accepted'])
    index = report['actions'].index(confirmation)
    fresh = next(event for event in reversed(report['actions'][:index])
                 if event['action'] == 'acquire' and event['purpose'] == 'confirm_frozen_program')
    assert report['actions'].index(tested) < report['actions'].index(fresh) < index
    assert fresh['probe_id'] != tested['probe_id']
    assert fresh['observation_cost'] == confirmation['observation_rows']
    assert report['concepts_added'] and history['source_id'] in report['method_source_ids']


def test_json_handoff_keeps_original_nonlinear_mechanism_available(nonlinear_history):
    history = nonlinear_history
    final = history['returned_library'].to_dict()
    for definition in history['before']['definitions']:
        assert definition in history['after_removal']['definitions']
        assert definition in final['definitions']
    returned = history['returned']
    assert returned['status'] == 'completed' and history['returned_error'] < CONFIG.minimum_improvement
    assert history['source_id'] in returned['used_sources']
    assert all(identifier in {item['concept_id'] for item in final['concepts']}
               for identifier in history['removed']['concepts_added'])


def test_real_disagreement_rejects_wrong_internal_deletion():
    library, source_id = nonlinear_source()
    lab = NonlinearMechanismLab(-.7, 'necessary_nonlinear_mechanism', outer_gain=1.7)
    call = library.callable_atoms(lab.spec, limit=1)[0]
    assert call.op == 'call'
    parent = {'candidate': Candidate('retained_source', call, 'callable_transfer'),
              'kind': 'callable_transfer', 'family_id': 'T03'}
    paid = [row for seed in (11, 12) for row in lab.sample(seed, 24).observations]
    budget = SearchBudget(24000)
    entries = propose_mechanism_revisions(lab.spec, paid, budget, parent=parent,
                                          priors=library.meta_priors)
    full = next(item for item in entries if item['kind'] == 'mechanism_full_refit')
    wrong = next(item for item in entries if item['kind'] == 'mechanism_deletion'
        and not contains(item['candidate'].program, 'neighbor_sum')
        and contains(item['candidate'].program, 'window_sum'))
    plan = plan_competing_object_probe(lab.spec, [full, wrong], budget,
                                      length=9, tolerance=CONFIG.minimum_improvement)
    assert plan['probe'] is not None and plan['predicted_separation'] > CONFIG.minimum_improvement
    measured = lab.acquire(plan['probe']).observations
    evidence = assess_competing_object_probe(plan, list(zip(measured, measured)), budget,
                                            tolerance=CONFIG.minimum_improvement)
    assert [item['status'] for item in evidence['hypotheses']] == [
        'supported_on_probe', 'counterexample']
    for entry, outcome in zip((full, wrong), evidence['hypotheses']):
        assert outcome['measured_loss'] == pytest.approx(
            loss(entry['candidate'].program, measured, 0., .5))
        assert source_id in entry['method_source_ids']
    assert evidence['compute_cost'] == 2 * len(measured)
    assert 0 < budget.spent <= budget.limit
    assert sum(len(batch['observations']) for batch in lab.acquisitions) == len(paid) + len(measured)
