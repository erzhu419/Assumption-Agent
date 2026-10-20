"""Actual archived revisions, complete old search and a held-future label."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.evaluation import predict
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget
from hegel_machine.cognition.selection import rank_candidates


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def replay_module():
    specification = importlib.util.spec_from_file_location('cross_skeleton_replay',
        ROOT / 'scripts/replay_cross_skeleton_relations.py')
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / 'scripts'))
        specification.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def history(replay_module):
    module = replay_module
    if not module.POINTER.is_file():
        pytest.skip('The historical source pointer is not retained in this checkout')
    if not Path(json.loads(module.POINTER.read_text())['archive']).is_file():
        pytest.skip('The retained historical archive is not mounted')
    # The actual Common replay is retained separately; do not repeat its full
    # source search for each evidence check.
    receipt = json.loads(module.HISTORICAL_OUTPUT.read_text())
    return module, receipt


def test_a_better_training_rank_still_defers_parent_adoption_without_fresh_rows(replay_module):
    incumbent = Program('lag', name='x', lag=1)
    winner = Program('const', value=0.)
    observed = [Observation({'x': 2.}, 0., ({'x': .1},))]
    config = {'selection_policy': 'prospective', 'complexity_weight': .001,
              'minimum_improvement': .002}
    ranked = rank_candidates([Candidate('parent', incumbent, 'reuse'),
                              Candidate('screen', winner, 'composition')], observed, [],
                             fallback=.25, q=.5, budget=SearchBudget(2), complexity_weight=.001)
    current, screening, adoption = replay_module.retained_parent_selection(ranked, incumbent, config)
    assert screening.candidate.program == winner
    assert adoption['ranking_gain_exceeds_minimum']
    assert current.candidate.program == incumbent
    assert adoption['status'] == 'deferred_missing_fresh_comparison'
    assert adoption['action'] == 'keep_current_parent'
    assert adoption['fresh_comparison_rows'] == 0
    assert not adoption['controller_adoption_rules_executed']
    startup = Observation({'x': 5.}, 999., ())
    assert predict(current.candidate.program, startup, .25) == .25
    assert predict(screening.candidate.program, startup, .25) == 0.


def test_original_config_and_actual_prefix_scores_do_not_turn_ranking_into_adoption(history):
    module, _ = history
    payload, _ = module.load_source(module.POINTER)
    config = payload['config']
    assert config['selection_policy'] == 'prospective'
    assert config['minimum_improvement'] == .002
    assert config['complexity_weight'] == .001
    episode = payload['episodes'][6]
    incumbent = Program.from_dict(episode['actions'][44]['details']['after_program'])
    winner = Program.from_dict(payload['episodes'][1]['actions'][36]['details']['after_program'])
    prefix = [*module.recovered_prefix(episode, 2, 14), *module.recovered_prefix(episode, 3, 40)]
    ranked = rank_candidates([Candidate('archive:4', winner, 'retained_archive'),
                              Candidate('archive:17', incumbent, 'retained_archive')], prefix, [],
                             fallback=episode['fallback'], q=.5, budget=SearchBudget(16),
                             complexity_weight=config['complexity_weight'])
    current, screening, adoption = module.retained_parent_selection(ranked, incumbent, config)
    assert current.evidence_loss == pytest.approx(.0007329565380917861)
    assert screening.evidence_loss == pytest.approx(.0026079565282924847)
    assert current.candidate.program == incumbent and screening.candidate.program == winner
    assert adoption['ranking_gain'] == pytest.approx(.000125000009799302)
    assert not adoption['ranking_gain_exceeds_minimum']
    assert adoption['status'] == 'deferred_missing_fresh_comparison'
    assert module.OUTPUT != module.HISTORICAL_OUTPUT


def test_actual_different_source_bodies_yield_an_executable_context_and_constraints(history):
    _, receipt = history
    relation = receipt['frozen_relation']
    definition = Definition.from_dict(relation['definition'])
    assert definition.body.op != 'arg' and len(definition.arguments) >= 2
    assert len(set(relation['source_ids'])) == 2
    assert len(relation['source_reconstruction']) == 2
    assert all(item['exact'] and item['target'] == 'after_program'
               for item in relation['source_reconstruction'])
    assert relation['learned_constraints']
    assert relation['construction_only'] and not relation['active_library']
    assert not relation['global_support'] and not relation['independent_confirmation']
    assert receipt['mainline_application']['generated_count'] > 0
    assert receipt['mainline_application']['concrete_programs_checked_through_existing_transfer']


def test_full_old_capabilities_and_original_cutoffs_are_shared(history):
    _, receipt = history
    protocol = receipt['protocol']
    assert protocol['target_action_cutoff'] == 44
    assert protocol['prefix_row_count'] == 8
    assert protocol['continuation_action_index'] == 65
    assert not protocol['future_trajectory_labels_used_for_generation_or_selection']
    assert not protocol['final_target_program_used_for_generation']
    capabilities = receipt['old_baseline']['capabilities']
    for key in ('archive_programs', 'extract_template_count', 'old_propose_candidates',
                'old_composition_candidates', 'old_sparse_candidates', 'old_warm_start_candidates'):
        assert capabilities[key] > 0
    for arm in receipt['arms'].values():
        assert arm['all_old_candidates_retained']
        assert arm['old_candidate_count'] == capabilities['unique_candidates']
        assert not any(arm['no_acquisition'].values())
        assert arm['compute']['total'] <= protocol['same_total_compute_budget']
        assert not arm['plan']['selected_action_executed']
    for reference in receipt['old_baseline']['archive_program_refs']:
        field = reference['field']
        position = int(field.split('episodes[')[1].split(']')[0])
        assert position <= 6
        if position == 6:
            assert int(field.split('actions[')[1].split(']')[0]) <= 44


def test_real_prefix_counterexample_survives_object_removal_and_ordinary_alias(history):
    _, receipt = history
    check = receipt['counterexample']
    assert 'memory_events' in check['source_ref']['field']
    for name, arm in receipt['arms'].items():
        failure = arm['counterexample']
        assert failure['actual_transfer_status'] == 'refuted'
        assert failure['ordinary_alias_compatible'] is False
        assert failure['failed_transfer_preserved']
        assert failure['source_reconstruction_preserved']
        assert failure['object_present'] == (name == 'keep')
        assert any(item['status'] == 'failed' for item in failure['checks'])


def test_comparison_reports_a_changed_prediction_instead_of_forcing_a_null(history):
    module, receipt = history
    keep, remove = deepcopy(receipt['arms']['keep']), receipt['arms']['remove']
    keep['parent']['continuation_prediction'] = remove['parent']['continuation_prediction'] + 1.
    changed = module.compare_arms(keep, remove)
    assert changed['incremental_behavior_effect_observed']
    assert not changed['same_continuation_prediction']


def test_changing_future_labels_leaves_the_real_inputs_to_induction_fitting_and_selection_unchanged(history):
    module, _ = history
    payload, source = module.load_source(module.POINTER)
    baseline = module.old_baseline(payload, source)
    sources = module.source_declarations(payload, source)
    plan = module.plan_inputs(payload['episodes'][6], ['same_finite_pool'])
    changed = deepcopy(payload)
    future = changed['episodes'][6]['memory_events'][0]['observation']
    for point in future['history']:
        point['__target__'] = 1e6
    future['target'] = 1e6
    checked = module.old_baseline(changed, source)
    assert checked['prefix'] == baseline['prefix']
    assert checked['candidates'] == baseline['candidates']
    assert checked['costs'] == baseline['costs']
    assert module.source_declarations(changed, source) == sources
    assert module.plan_inputs(changed['episodes'][6], ['same_finite_pool']) == plan
