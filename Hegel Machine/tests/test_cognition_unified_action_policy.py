"""Research directions learn from actual paid trajectories and change later work."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget
from hegel_machine.cognition.research_graph import ResearchGraph
from hegel_machine.cognition.structure_lab import ObjectSchema, StructureLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_structure import propose_structures


CONFIG = ResearchConfig(compute_budget=120000, minimum_improvement=1e-7, max_candidates=8)


class ObservedResearch(UnifiedResearchAgent):
    """Record real assessment inputs/results without replacing search or fitting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assessments = []

    def _assess(self, claims, validation, budget, fallback, q):
        start = budget.spent
        checked = super()._assess(claims, validation, budget, fallback, q)
        self.assessments.append((list(checked), validation, budget.spent - start,
                                 fallback, budget.spent))
        return checked


def _rename(spec, rows, transitions, prefix):
    mapping = {name: prefix + name for name in (*spec.feature_names, *spec.feature_bounds)}
    def values(fields):
        return {mapping.get(name, name): value for name, value in fields.items()}
    schema = ObjectSchema.from_dict(json.loads(spec.context['object_schema'])).remap_features(mapping)
    renamed_spec = replace(spec, task_id='another_public_task',
        feature_names=tuple(mapping[name] for name in spec.feature_names),
        feature_bounds=values(spec.feature_bounds),
        context={**spec.context, 'object_schema': json.dumps(schema.to_dict())})
    renamed_rows = [Observation(values(row.features), row.target,
                               tuple(values(past) for past in row.history)) for row in rows]
    renamed_transitions = [{key: values(value) for key, value in item.items()} for item in transitions]
    return renamed_spec, renamed_rows, renamed_transitions


def _research(library, *, training_seed=11, validation_seed=41, renamed=False,
              config=CONFIG, reuse_validation=False):
    laboratory = StructureLab(readout='focus', seed=91)
    training = laboratory.acquire(laboratory.random_probe(training_seed, length=24))
    validation = laboratory.acquire(laboratory.random_probe(validation_seed, length=24))
    spec, rows, transitions = laboratory.spec, training.observations, training.transitions
    checked = validation.observations
    if renamed:
        spec, rows, transitions = _rename(spec, rows, transitions, 'next_')
        _, checked, _ = _rename(laboratory.spec, checked, [], 'next_')
    if reuse_validation:
        rows = [*rows, *checked]
    agent = ObservedResearch(config, library=library)
    budget, actions = SearchBudget(config.compute_budget), []
    claims = agent._research_view(spec, rows, checked, budget, actions, [], ResearchGraph(),
                                 transitions=transitions)
    return {'agent': agent, 'actions': actions, 'claims': claims,
            'best': agent._best(claims), 'budget': budget, 'validation': checked,
            'laboratory': laboratory, 'training': rows, 'spec': spec}


def _choices(result):
    return [event for event in result['actions'] if event['action'] == 'choose_research_action']


def _feedback(result):
    return [event for event in result['actions'] if event['action'] == 'research_action_feedback']


@pytest.fixture(scope='module')
def learned_directions():
    laboratory = StructureLab(readout='total', seed=71)
    source = laboratory.acquire(laboratory.random_probe(89, length=24))
    candidates = propose_structures(laboratory.spec, source.observations, SearchBudget(50000),
                                    transitions=source.transitions)
    library = KnowledgeLibrary()
    library.learn(candidates[0]['candidate'], laboratory.spec, source.observations)
    # This acquired local response contradicts the remembered total response.
    result = _research(library)
    return result, json.loads(json.dumps(library.to_dict()))


def test_executed_directions_compare_to_current_best_and_pay_their_actual_search(learned_directions):
    result, saved = learned_directions
    events = _feedback(result)
    by_action = {event['research_action']: event for event in events}
    assert set(by_action) == {'callable_transfer', 'mechanism_revision', 'program_revision', 'structural_research'}
    assert len({event['before_loss'] for event in events}) == 3
    # Each direction must add to the explanation already available. Reducing
    # its remaining error keeps an effective later revision valuable.
    program, structure = by_action['program_revision'], by_action['structural_research']
    assert structure['after_loss'] < program['after_loss'] < program['before_loss']
    assert structure['feedback']['last_gain'] > program['feedback']['last_gain']
    assert structure['before_loss'] == program['after_loss']
    assert by_action['callable_transfer']['feedback']['last_gain'] < 0
    groups = result['agent'].assessments
    baseline = result['agent']._best([claim for group in groups for claim in group[0]
                                    if claim['kind'] == 'current_affine'])
    assert events[0]['before_loss'] == baseline['validation_loss']
    for event in events:
        if event['research_action'] == 'mechanism_revision':
            # The remembered total response has no additive mechanisms to
            # remove. Searching it still costs compute and earns no gain.
            proposal = next(action for action in result['actions']
                            if action['action'] == 'propose_mechanism_revisions')
            assert proposal['alternatives'] == []
            assert event['before_loss'] == event['after_loss']
            assert event['feedback']['last_gain'] == 0
            index, group = next((index, group) for index, group in enumerate(groups) if not group[0])
        else:
            index, group = next((index, group) for index, group in enumerate(groups)
                if any(claim.get('research_action_origin', {}).get('action') == event['research_action']
                       for claim in group[0]))
        candidates, validation, _, fallback, spent = group
        best = result['agent']._best(candidates) or baseline
        measured = loss(best['candidate'].program, validation, fallback, result['spec'].risk_quantile)
        assert event['after_loss'] == pytest.approx(measured)
        assert event['compute_cost'] == spent - groups[index - 1][4]
        assert event['observation_cost'] == 0  # This direction reuses the paid evidence.
        assert event['evidence_rows'] == len(validation) == 24
        assert event['evidence_basis'] == 'common_unfitted_validation'
        assert not set(map(id, validation)) & set(map(id, result['training']))
        assert saved['action_experience'][event['context']][event['research_action']]['trials'] == 1
    assert result['laboratory'].costs == {'observation_cost': 48, 'measurement_cost': 576, 'queries': 2}
    # The final structural interval is refitted to both trajectories. Its tiny
    # fitting error cannot replace the earlier direction's measured return.
    combined = next(event for event in result['actions']
                    if event['action'] == 'revise_structure_from_combined_evidence')
    assert combined['answered']
    assert result['best']['validation_loss'] < events[-1]['after_loss']
    assert result['actions'].index(combined) > result['actions'].index(events[-1])


@pytest.mark.parametrize('ablation', ['forget_policy', 'disable_policy'])
def test_json_restored_experience_avoids_worse_reuse_and_preserves_successor_answer(learned_directions, ablation):
    _, saved = learned_directions
    retained = KnowledgeLibrary.from_dict(deepcopy(saved))
    absent = KnowledgeLibrary.from_dict(deepcopy(saved))
    config = CONFIG
    if ablation == 'forget_policy':
        absent.action_experience.clear()
    else:
        config = replace(CONFIG, learn_research_policy=False)
    absent_before = deepcopy(absent.action_experience)
    improved = _research(retained, training_seed=43, validation_seed=67, renamed=True)
    control = _research(absent, training_seed=43, validation_seed=67, renamed=True, config=config)
    assert _choices(improved)[0]['selected_action'] == 'program_revision'
    assert _choices(improved)[0]['reason'] == 'measured_improvement_per_resource'
    assert all(choice['selected_action'] != 'callable_transfer' for choice in _choices(improved))
    assert any(event['action'] == 'compare_structural_explanations' for event in improved['actions'])
    assert any(event['action'] == 'execute_research_programs' for event in control['actions'])
    assert improved['budget'].spent <= control['budget'].spent
    if ablation == 'disable_policy':
        assert improved['budget'].spent < control['budget'].spent
    assert improved['laboratory'].costs == control['laboratory'].costs
    assert improved['best']['validation_loss'] < 1e-7
    assert control['best']['validation_loss'] < 1e-7
    # Score new response rows only after both selections have finished.
    fresh = improved['laboratory'].evaluation_rows(seed=733, length=48)
    _, fresh, _ = _rename(improved['laboratory'].spec, fresh, [], 'next_')
    assert [improved['best']['candidate'].program.predict(row) for row in fresh] == pytest.approx(
        [control['best']['candidate'].program.predict(row) for row in fresh])
    if ablation == 'disable_policy':
        assert absent.action_experience == absent_before
        assert not _feedback(control)
    assert sum(record['trials'] for bucket in saved['action_experience'].values()
               for record in bucket.values()) == 4


def test_already_fitted_validation_does_not_teach_the_action_policy(learned_directions):
    _, saved = learned_directions
    library = KnowledgeLibrary.from_dict(deepcopy(saved))
    before = deepcopy(library.action_experience)
    result = _research(library, reuse_validation=True)
    assert result['claims']
    assert not _feedback(result)
    assert library.action_experience == before


def test_old_negative_preferences_cannot_veto_current_counterexample_research(learned_directions):
    _, saved = learned_directions
    library = KnowledgeLibrary.from_dict(deepcopy(saved))
    # Inject pessimistic historical opinions to exercise the stopping boundary;
    # the current observations, searches, and successful answer remain real.
    for bucket in library.action_experience.values():
        for record in bucket.values():
            record.update(trials=2, expected_gain=-1., mean_gain=-1.)
    result = _research(library, training_seed=43, validation_seed=67)
    choices = _choices(result)
    assert {item['selected_action'] for item in choices} == {
        'callable_transfer', 'mechanism_revision', 'program_revision', 'structural_research'}
    assert all(item['reason'] == 'retry_remaining_direction_on_current_evidence' for item in choices)
    assert result['best']['validation_loss'] < 1e-7
    assert {item['research_action'] for item in _feedback(result)} == {
        'callable_transfer', 'mechanism_revision', 'program_revision', 'structural_research'}
    assert all(item['feedback']['trials'] == 3 for item in _feedback(result))
    assert result['budget'].spent <= CONFIG.compute_budget


def test_memory_disabled_runs_current_research_without_writing_action_experience(learned_directions):
    _, saved = learned_directions
    library = KnowledgeLibrary.from_dict(deepcopy(saved))
    before = deepcopy(library.to_dict())
    result = _research(library, training_seed=43, validation_seed=67,
                       config=replace(CONFIG, use_memory=False))
    assert result['best']['validation_loss'] < 1e-7
    assert any(event['action'] == 'compare_structural_explanations' for event in result['actions'])
    assert not any(event['action'] == 'execute_research_programs' for event in result['actions'])
    assert not _feedback(result)
    assert not _choices(result)
    assert library.to_dict() == before
