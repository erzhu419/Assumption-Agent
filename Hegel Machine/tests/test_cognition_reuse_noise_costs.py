"""A paid noise explanation saves search while genuine changes remain learnable."""

from copy import deepcopy
from dataclasses import replace
import gzip
import json

import pytest

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.unified_revisits import offer_successor_knowledge
import hegel_machine.cognition.research_programs as procedures
import hegel_machine.cognition.unified_research as controller
from test_cognition_cold_temporal_research import CONFIG, UnspecifiedResponseLab, _paid_confirmation


def _rows(batch):
    return [Observation(item['features'], item['target'], tuple(item['history']))
            for item in batch['observations']]


def _research(lab, library, seed):
    config = replace(CONFIG, seed=seed)
    report = UnifiedResearchAgent(config, library=library, measurement_budget=0).run(lab)
    bought = [event for event in report['actions'] if event['action'] == 'acquire']
    assert report['costs']['queries'] == len(bought) == len(lab.acquisitions)
    assert report['costs']['observation_cost'] == sum(
        len(item['observations']) for item in lab.acquisitions) <= config.observation_budget
    assert report['costs']['measurement_cost'] == 0
    assert report['costs']['compute_cost'] <= config.compute_budget
    return report


@pytest.fixture(scope='module')
def acquired_response():
    source_library = KnowledgeLibrary()
    source = _research(UnspecifiedResponseLab('acquired_response', 3), source_library, 9951301)
    assert source['status'] == 'completed' and source['concepts_added']
    saved = json.loads(json.dumps(source_library.to_dict()))
    return source, saved


@pytest.fixture(scope='module')
def response_changes(acquired_response, tmp_path_factory):
    source, saved = acquired_response
    learned = []
    for position, (width, gain) in enumerate(((3, -.65), (5, -.65), (3, .35))):
        trace = []
        original_composition = procedures.propose_research_programs
        original_compilation = controller.compile_claims

        def composition(*args, **kwargs):
            trace.append('compose')
            return original_composition(*args, **kwargs)

        def compilation(*args, **kwargs):
            trace.append('compile')
            return original_compilation(*args, **kwargs)

        lab = UnspecifiedResponseLab(f'new_measurement_{position}', width, gain=gain, noise=.01)
        library = KnowledgeLibrary.from_dict(deepcopy(saved))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(procedures, 'propose_research_programs', composition)
            patch.setattr(controller, 'compile_claims', compilation)
            report = _research(lab, library, 9952301 + position * 1009)
        learned.append({'report': report, 'width': width, 'gain': gain,
                        'trace': trace, 'knowledge': library.to_dict()})

    # All three researchers finish before a terminal mean response is acquired.
    for position, item in enumerate(learned):
        terminal = UnspecifiedResponseLab(f'terminal_{position}', item['width'], gain=item['gain'])
        rows = [row for repeat, length in enumerate((1, 3, 9, 64, 64))
                for row in terminal.acquire(terminal.random_probe(9959301 + position * 5003 + repeat,
                    length=length, name=f'terminal_{repeat}')).observations]
        item['mean_error'] = loss(Program.from_dict(item['report']['program']),
                                 rows, item['report']['fallback'], .5)
    path = tmp_path_factory.mktemp('reuse_noise_costs') / 'changes.json.gz'
    path.write_bytes(gzip.compress(json.dumps({'source': source, 'initial_knowledge': saved,
                                              'changes': learned}).encode()))
    return source, saved, learned, path


def test_noisy_reuse_buys_a_repeat_before_expanding_the_response_language(response_changes, record_property):
    source, _, learned, path = response_changes
    item, = learned[:1]
    report, actions = item['report'], item['report']['actions']
    record_property('research_reports', str(path))
    record_property('research_costs', json.dumps(report['costs']))
    record_property('mean_response_mse', item['mean_error'])
    assert report['status'] == 'completed' and item['mean_error'] < 3e-4
    assert set(source['concepts_added']) & set(report['used_sources'])
    tested = next(event for event in actions
                  if event['action'] == 'test_response_noise_before_expansion' and event['supported'])
    paused = next(event for event in actions
                  if event['action'] == 'defer_research_expansion_after_repeat')
    assert actions.index(tested) < actions.index(paused)
    assert tested['noise_supported'] and not tested['repeatable_error']
    assert tested['repeat_noise_variance'] > CONFIG.minimum_improvement
    assert item['trace'] == []
    batches = {batch['probe']['probe_id']: batch for batch in report['batches']}
    original = batches[tested['probe_id']]
    repeated = next(batch for batch in report['batches']
                    if batch['probe']['probe_id'] != tested['probe_id']
                    and batch['probe']['sequence'] == original['probe']['sequence'])
    program = Program.from_dict(tested['program'])
    assert tested['original_model_loss'] == pytest.approx(loss(program, _rows(original), 0., .5))
    assert tested['repeated_model_loss'] == pytest.approx(loss(program, _rows(repeated), 0., .5))
    assert tested['compute_cost'] == 4 * len(original['observations'])
    _paid_confirmation(report)
    confirmed = next(event for event in actions
                     if event['action'] == 'validate_frozen_program' and event['accepted'])
    assert actions.index(paused) < actions.index(confirmed)
    assert confirmed['program'] == report['program']
    assert any(event['action'] == 'compare_noise_replication' and event['noise_supported']
               for event in actions[actions.index(paused) + 1:])


@pytest.mark.parametrize('position', (1, 2))
def test_noisy_temporal_and_coefficient_changes_still_buy_research_and_fresh_support(
        response_changes, position, record_property):
    source, saved, learned, path = response_changes
    item, report = learned[position], learned[position]['report']
    record_property('research_reports', str(path))
    record_property('mean_response_mse', item['mean_error'])
    first = _rows(report['batches'][0])
    assert loss(Program.from_dict(source['program']), first, source['fallback'], .5) > .05
    assert report['status'] == 'completed' and item['mean_error'] < 3e-4
    assert report['concepts_added']
    assert report['program'] != source['program']
    assert any(event['action'] in {'propose_mechanism_revisions', 'replay_research_programs',
                                  'propose_coordinate_reconstruction'} for event in report['actions'])
    _paid_confirmation(report)
    assert all(definition in item['knowledge']['definitions'] for definition in saved['definitions'])
    if position == 1:
        assert Program.from_dict(report['program']).history_depth >= item['width'] - 1


def test_saved_response_reopens_renamed_paid_evidence_without_rediscovering_parameters(
        acquired_response, monkeypatch, record_property):
    source, saved = acquired_response
    lab = UnspecifiedResponseLab('unfinished_renamed_apparatus', 3, names=('rate', 'stress'))
    limited = replace(CONFIG, compute_budget=512, seed=9961301)
    previous = UnifiedResearchAgent(limited, library=KnowledgeLibrary(), measurement_budget=0).run(lab)
    assert previous['status'] == 'unresolved'
    old_acquisitions = len(lab.acquisitions)
    library = KnowledgeLibrary.from_dict(deepcopy(saved))
    generic_fits = []
    original = library.propose

    def recorded_generic_fit(*args, **kwargs):
        generic_fits.append(kwargs.get('source_concept_ids'))
        return original(*args, **kwargs)

    monkeypatch.setattr(library, 'propose', recorded_generic_fit)
    budget = SearchBudget(2000)
    agent = UnifiedResearchAgent(CONFIG, library=library, measurement_budget=0)
    offer = offer_successor_knowledge(agent, lab, previous, source['concepts_added'], budget)
    assert offer is not None and offer['old_evidence_gain'] > CONFIG.minimum_improvement
    assert generic_fits == [] and len(lab.acquisitions) == old_acquisitions
    assert 0 < budget.spent <= budget.limit
    candidate = offer['candidate']
    paid = [row for batch in previous['batches'] for row in _rows(batch)]
    measured = loss(Program.from_dict(candidate['program']), paid, candidate['fallback'], .5)
    assert candidate['selection_loss'] == pytest.approx(measured)
    assert measured < CONFIG.minimum_improvement
    assert set(candidate['source_concept_ids']) & set(source['concepts_added'])
    assert Program.from_dict(candidate['program']).required_features() == {'rate', 'stress'}
    # Screening is a quote. It has not replaced or confirmed the old answer.
    assert previous['status'] == 'unresolved' and candidate['program'] != previous['program']
    assert offer['minimum_resources']['observation_cost'] >= 48
    record_property('successor_replay_compute_cost', budget.spent)
