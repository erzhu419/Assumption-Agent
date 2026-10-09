"""The acquired response basis can be revised without a private target template."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
from random import Random

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.mechanism_revision import propose_response_refits
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.relation_revisions import _relations
from hegel_machine.cognition.relation_uncertainty import RelationIntervalResearch
from test_cognition_mechanism_revision import contains_relation, source_parent
from test_cognition_mechanism_revision_research import ChangingMechanismLab, source_knowledge


def _static_response():
    rate, stress = Program('feature', name='rate'), Program('feature', name='stress')
    cube = Program('mul', children=(rate, Program('mul', children=(rate, rate))))
    original = Program('add', children=(Program('add', children=(
        Program('mul', children=(Program('const', value=1.1), cube)),
        Program('mul', children=(Program('const', value=-.2), stress)))),
        Program('mul', children=(Program('const', value=.05), rate))))
    spec = TaskSpec('unlabeled_response', 'Predict the measured response',
        ('rate', 'stress'), (), ())

    def observations(seed, count):
        rng = Random(seed)
        rows = []
        for _ in range(count):
            x, z = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
            rows.append(Observation({'rate': x, 'stress': z},
                .7 + 2 * x ** 3 - .4 * z + rng.gauss(0., .002)))
        return rows

    return spec, {'candidate': Candidate('measured_parent', original, 'research_program')}, observations


def test_static_signed_basis_is_refitted_and_weak_terms_are_only_proposed_for_omission():
    spec, parent, observations = _static_response()
    before = parent['candidate'].program.to_dict()
    training, comparison = observations(91, 64), observations(191, 64)
    budget = SearchBudget(4000)
    entries = propose_response_refits(spec, training, budget, parent=parent, max_candidates=2)
    assert len(entries) == 2
    full, smaller = entries
    assert full['kind'] == 'mechanism_full_refit'
    assert smaller['mechanism_revision']['removed_term'] == Program('feature', name='rate').to_dict()
    assert loss(parent['candidate'].program, comparison, 0., .5) > .1
    for entry in entries:
        assert loss(entry['candidate'].program, comparison, 0., .5) < 1e-4
        assert entry['mechanism_revision']['original_program'] == before
        assert entry['mechanism_revision']['status'] == 'proposed'
        assert 'validation_loss' not in entry
    assert parent['candidate'].program.to_dict() == before
    assert 0 < budget.spent <= budget.limit


def test_actual_noisy_learned_window_improves_on_paid_rows_outside_its_fit():
    fixture = Path(__file__).parent / 'fixtures/noisy_window_root01_incoming.json.gz'
    payload = json.loads(gzip.decompress(fixture.read_bytes()))
    original = Program.from_dict(payload['baseline']['program'])
    batches = [[Observation(row['features'], row['target'], tuple(row['history']))
                for row in batch['observations']] for batch in payload['paid_batches']]
    training = batches[0] + batches[1]
    spec = TaskSpec('paid_response', 'Predict the response', ('x', 'z'), (), ())
    parent = {'candidate': Candidate('actual_window', original, 'research_program')}
    budget = SearchBudget(4000)
    entries = propose_response_refits(spec, training, budget, parent=parent)
    assert entries and entries[0]['kind'] == 'mechanism_full_refit'
    full = entries[0]
    assert full['candidate'].program.history_depth == original.history_depth == 2
    # These were acquired by the old run, not generated from its noiseless mean.
    # The single held-out batch gains are not claims of statistical confirmation.
    for rows in batches[2:]:
        assert loss(full['candidate'].program, rows, 0., .5) < loss(original, rows, 0., .5)
    assert full['training_loss'] < loss(original, training, 0., .5)
    assert full['candidate'].parents == ('concept_0001',)
    assert full['method_source_ids'] == ['concept_0001']
    assert full['mechanism_revision']['status'] == 'proposed'
    assert all(item['mechanism_revision']['removed_operator'] is None for item in entries)
    assert all(item['candidate'].program.history_depth == 2 for item in entries)
    assert 0 < budget.spent <= budget.limit


def test_relation_ranges_survive_joint_response_refit_and_json_replay():
    library, source_id = source_knowledge()
    parent = source_parent(library, source_id)
    radius = _relations(parent['candidate'].program.expanded())[0].children[0].value
    prior = {'source_concept_id': source_id,
        'structure': {'fits': [{'radius': radius}], 'radius_interval': [0., radius + .2]}}
    before = deepcopy(prior)
    inherited = RelationIntervalResearch(parent['candidate'].program, [prior], SearchBudget(4000))
    lab = ChangingMechanismLab(-.35, 'changed_response')
    entries = propose_response_refits(lab.spec, lab.sample(71, 48).observations,
        SearchBudget(12000), parent=parent, priors=[prior])
    full = entries[0]
    assert full['kind'] == 'mechanism_full_refit' and contains_relation(full['candidate'].program)
    for entry in entries:
        plan = json.loads(json.dumps(entry.get('research_program')))
        replay = RelationIntervalResearch(entry['candidate'].program, [prior],
            SearchBudget(4000), research_program=plan)
        if contains_relation(entry['candidate'].program):
            assert plan['status'] == 'requires_fresh_validation'
            assert replay.parameters[0]['radius_intervals'] == inherited.parameters[0]['radius_intervals']
        else:
            assert plan is None and not replay.parameters
    assert prior == before


def test_refits_respect_an_unfunded_search_and_single_candidate_limit():
    spec, parent, observations = _static_response()
    rows = observations(91, 64)
    budget = SearchBudget(1)
    assert propose_response_refits(spec, rows, budget, parent=parent) == []
    assert budget.spent <= budget.limit
    budget = SearchBudget(600)
    entries = propose_response_refits(spec, rows, budget, parent=parent, max_candidates=1)
    assert len(entries) == 1 and entries[0]['kind'] == 'mechanism_full_refit'
    assert budget.spent <= budget.limit
