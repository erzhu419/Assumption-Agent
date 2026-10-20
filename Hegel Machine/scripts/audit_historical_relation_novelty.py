#!/usr/bin/env python3
"""Audit whether retained revisions actually exceed earlier archive capabilities.

Target constants are used only for an expressivity/reconstruction audit. They
are never supplied to a predictor proposer or called prospective evidence.
Earlier rejected Programs remain available sources for old template extraction.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from hegel_machine.cognition.abstractions import extract_templates
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_episodes import extract_episodes, load_episode_sources
from replay_theory_consolidation import POINTER, ROOT, load_source


OUTPUT = ROOT / 'docs/evidence/historical_relation_novelty_20261020.json'


def reference(position, *path):
    return {'source': 'retained_cognitive_sequence', 'json_path': ['episodes', position, *path]}


def constants(program):
    return ([program.value] if program.op == 'const' else
            [value for child in program.children for value in constants(child)])


def literal(value):
    return Program('const', value=value)


def times(value, program):
    return Program('mul', children=(literal(value), program))


def plus(left, right):
    return Program('add', children=(left, right))


def find_path(program, wanted, path=()):
    if program == wanted:
        return path
    for index, child in enumerate(program.children):
        found = find_path(child, wanted, (*path, 'children', index))
        if found is not None:
            return found
    return None


def _actual_revision(episodes, position, action_index):
    episode = episodes[position]
    action = episode['actions'][action_index]
    if action['action'] != 'revise':
        raise AssertionError('An adoption witness must reference an actual revision')
    matches = Program.from_dict(action['details']['after_program']) == Program.from_dict(episode['program'])
    if not matches:
        raise AssertionError('The cited revision is not the Episode final Program')
    return {'episode_index': position, 'task_id': episode['task_id'],
        'revision_ref': reference(position, 'actions', action_index),
        'operation': action['details']['operation'],
        'after_program_matches_episode_final': matches,
        'development_loss': action['details'].get('development_loss'),
        'terminal_evaluation_loss': episode['evaluation']['loss'],
        'terminal_evaluation_rows': episode['evaluation']['rows'],
        'independent_confirmation': False}


def _witness(target, reconstructed, *, target_position, old_refs, method, capabilities):
    exact = reconstructed == target
    if not exact:
        raise AssertionError(f'Incorrect old-capability reconstruction for episodes[{target_position}]')
    return {'target_ref': reference(target_position, 'program'),
        'source_cutoff': {'strictly_before_episode_index': target_position},
        'earlier_program_refs': old_refs, 'method': method, 'old_capability_sources': capabilities,
        'target_constants_used_only_for_representation_audit': True,
        'exact_ordered_Program_reconstruction': exact, 'prospective_prediction': False,
        'scientific_support_added': False}


def _external_leads():
    """Read five already-located small sources, without expanding their traces."""
    base = 'artifacts/response_family_recovery_20261009'
    cubic_path = f'{base}/changed_mechanism/brief.json'
    recovery = json.loads((ROOT / cubic_path).read_text())
    before = next(item for item in recovery['declarations_before'] if item['id'] == 'grammar_signal_1')
    after = next(item for item in recovery['declarations_after'] if item['id'] == 'signal_cubic_5')
    rebuilt = Program('mul', children=tuple(Program.from_dict(item) for item in before['basis']))
    cubic = Program.from_dict(after['basis'][0])
    if rebuilt != cubic:
        raise AssertionError('The cited earlier square and signal do not reconstruct the cubic Program')
    transfer_path = 'artifacts/response_structure_transfer_20261009/summary.json'
    transfer = json.loads((ROOT / transfer_path).read_text())['groups']['cubic_transfer']
    failure_path = 'docs/evidence/conditional_online_failure_20261013/representation_repair_20261014.json'
    diagnosis = json.loads((ROOT / failure_path).read_text())['diagnosis']
    family_path = 'artifacts/response_family_revision_20261009/result.json'
    family_cases = json.loads((ROOT / family_path).read_text())['cases']
    missing = json.loads((ROOT / base / 'uninformative_sensor/brief.json').read_text())
    return [
        {'id': 'signal_cubic_recovery', 'source_refs': [
            {'file': cubic_path, 'field': 'declarations_before[id=grammar_signal_1].basis'},
            {'file': cubic_path, 'field': 'declarations_after[id=signal_cubic_5].basis[0]'},
            {'file': transfer_path, 'json_path': ['groups', 'cubic_transfer']}],
         'representation_status': 'reconstructed_by_old',
         'representation_witness': {'operation': 'mul(old.square, old.signal)', 'exact_ordered_Program': True},
         'historical_outcome': {'without_retained_cubic': transfer['without_cubic']['successes'],
             'with_retained_cubic': transfer['retain_cubic']['successes'],
             'cases_per_arm': transfer['retain_cubic']['cases']},
         'shared_relation_status': 'UNKNOWN_no_new_multi_episode_relation_object',
         'interpretation': 'The archived cubic recovery and its 8/8 transfer support remain valid. Multiplication expressivity alone does not show the earlier fixed grammar already knew the cubic relation, or replace a full-old-search control.'},
        {'id': 'conditional_three_step_window', 'source_refs': [
            {'file': 'docs/evidence/conditional_online_failure_20261013/luna_window_confirmed_candidate_20261017.py', 'function': '_design'},
            {'file': 'docs/evidence/conditional_online_failure_20261013/structural_method_revision_20261017.json', 'json_path': ['same_interface_history_check']},
            {'file': 'src/hegel_machine/cognition/composed_conditions.py', 'function': 'compose_condition_response'}],
         'representation_status': 'old_window_and_condition_constructor_available',
         'representation_witness': 'window_sum(if(condition, response_coordinate, 0)) with width three sums only existing history and current coordinates',
         'full_python_model_reconstruction': 'UNKNOWN_not_compiled_or_refitted',
         'shared_relation_status': 'UNKNOWN_no_multiple_executable_revision_sources',
         'interpretation': 'Existing window/condition primitives cover the stated coordinate operation. This is not an exact proof for the complete Python polynomial fit, nor proof that an old library knew a new shared constraint.'},
        {'id': 'missing_history_zero_collision', 'source_refs': [
            {'file': failure_path, 'json_path': ['diagnosis', 'witness']},
            {'file': 'src/hegel_machine/cognition/language.py', 'operations': ['available', 'window_sum']}],
         'retained_counterexample': diagnosis['witness'],
         'retained_scope': {'rows': diagnosis['rows'], 'conflicting_groups': diagnosis['conflicting_groups'],
             'empirical_mse_floor': diagnosis['empirical_mse_floor']},
         'representation_status': 'old_availability_and_partial_window_primitives_available',
         'shared_relation_status': 'UNKNOWN_no_multiple_successful_executable_revisions',
         'interpretation': 'The collision is real and remains unresolved material. Availability primitives do not themselves supply a learned cross-Episode repair or its future value.'},
        {'id': 'shared_two_feature_response', 'source_refs': [
            {'file': family_path, 'json_path': ['cases', 1, 'initial', 'families']},
            {'file': family_path, 'json_path': ['cases', 1, 'confirmations']}],
         'initial_declared_families': family_cases[1]['initial']['families'],
         'retained_confirmation_count': len(family_cases[1]['confirmations']),
         'representation_status': 'already_declared_shared_x_z_family',
         'shared_relation_status': 'already_declared_before_this_research',
         'interpretation': 'The existing two-coordinate shared family was already declared and produced two confirmations; another name would not establish a new shared relation.'},
        {'id': 'unidentified_measurement_state', 'source_refs': [
            {'file': family_path, 'json_path': ['cases', 2, 'scientific_diagnosis', 'same_measurement_conflict']},
            {'file': f'{base}/uninformative_sensor/brief.json'}],
         'retained_counterexamples': family_cases[2]['scientific_diagnosis']['same_measurement_conflict'],
         'refuted_family_ids': family_cases[2]['refuted_family_ids'],
         'retained_confirmation_count': len(family_cases[2]['confirmations']),
         'recovery_status': missing['status'], 'recovery_new_actions': missing['new_actions'],
         'representation_status': 'UNKNOWN_missing_distinguishing_observable',
         'shared_relation_status': 'UNKNOWN_no_executable_successful_revision',
         'interpretation': 'The same measured setting has contradictory outcomes and the recovery added no action. No new measurement or successful executable relation is supplied by these files.'},
    ]


def _mainline_identity_check(source, count):
    refs = [{key: source[key] for key in ('archive', 'unit', 'member')} |
            {'json_path': ['episodes', i]} for i in range(count)]
    retained = load_episode_sources(refs, ROOT)
    agent = SimpleNamespace(workspace=ROOT, questions=[], theory_index={}, observations=[],
                            revisions=[], events=[], relation_research={})
    extracted = extract_episodes(agent, retained_episodes=retained)['episodes']
    repeated = [episode for episode in extracted
                if episode.get('source_ref', {}).get('json_path') in (['episodes', 1], ['episodes', 16])]
    if len(extracted) != count or len(repeated) != 2:
        raise AssertionError('Mainline Episode extraction merged distinct retained source positions')
    return {'retained_episode_count': count, 'extracted_episode_count': len(extracted),
        'same_task_id_distinct_sources_preserved': True,
        'repeated_task_episodes': [{key: episode[key] for key in ('episode_id', 'source_ref', 'costs')}
                                  for episode in repeated]}


def audit(pointer=POINTER, output=OUTPUT):
    payload, source = load_source(pointer)
    episodes = payload['episodes']
    old_start = Program.from_dict(episodes[1]['actions'][18]['details']['after_program'])
    old_guard = old_start.children[0]
    old_lag_template = next(template for template in extract_templates(old_start, 'old_t02_startup')
        if template.kind == 'subexpression' and template.pattern.op == 'lag'
        and template.source_features == ('x',))
    lag_one = old_lag_template.instantiate({'role_0': 'x'})
    target_seven = Program.from_dict(episodes[6]['program'])
    mature, startup = target_seven.children[1:]
    rebuilt_seven = Program('if', children=(old_guard,
        plus(plus(literal(mature.children[0].children[0].value),
                  times(mature.children[0].children[1].children[0].value, lag_one)),
             times(mature.children[1].children[0].value, Program('feature', name='x'))),
        plus(literal(startup.children[0].value),
             times(startup.children[1].children[0].value, Program('feature', name='x')))))
    target_fourteen = Program.from_dict(episodes[13]['program'])
    mature, startup = target_fourteen.children[1:]
    rebuilt_fourteen = Program('if', children=(Program('available', lag=2),
        plus(times(mature.children[0].children[0].value, Program('feature', name='x')),
             times(mature.children[1].children[0].value, Program('lag', name='x', lag=2))),
        times(startup.children[0].value, Program('feature', name='x'))))
    startup_ref = reference(1, 'actions', 18, 'details', 'after_program')
    startup_witnesses = [
        _witness(target_seven, rebuilt_seven, target_position=6, old_refs=[startup_ref],
            method='old extracted lag fragment plus existing startup/mature constructor',
            capabilities=['src/hegel_machine/cognition/selection.py:warm_start_candidates',
                          'src/hegel_machine/cognition/abstractions.py:extract_templates']),
        _witness(target_fourteen, rebuilt_fourteen, target_position=13, old_refs=[startup_ref,
            reference(6, 'program')], method='existing lag-depth enumeration and startup constructor',
            capabilities=['src/hegel_machine/cognition/synthesis.py:_terms',
                          'src/hegel_machine/cognition/selection.py:warm_start_candidates'])]
    source_three = Program.from_dict(episodes[2]['program'])
    threshold_template = next(template for template in extract_templates(source_three, 'old_t03')
        if template.kind == 'subexpression' and template.pattern.op == 'ge'
        and template.pattern.children[0].op == 'feature')
    target_eight = Program.from_dict(episodes[7]['program'])
    nested, lag_threshold = target_eight.children[1].children
    rebuilt_eight = Program('if', children=(old_guard, Program('or', children=(
        Program('ge', children=(threshold_template.instantiate({'role_0': 'x'},
            [nested.children[0].children[1].value]), literal(nested.children[1].value))),
        Program('ge', children=(lag_one, literal(lag_threshold.children[1].value))))),
        threshold_template.instantiate({'role_0': 'x'}, [target_eight.children[2].children[1].value])))
    boolean_witness = _witness(target_eight, rebuilt_eight, target_position=7,
        old_refs=[startup_ref, reference(2, 'program')],
        method='old threshold/lag fragments and existing Boolean OR composition',
        capabilities=['src/hegel_machine/cognition/synthesis.py:synthesize.compositions'])
    failed_nine = Program.from_dict(episodes[8]['actions'][20]['details']['after_program'])
    failed_nine_ref = reference(8, 'actions', 20, 'details', 'after_program')
    templates = extract_templates(failed_nine, 'retained_replaced_t09_program')
    square = next(template for template in templates if template.kind == 'subexpression'
        and template.pattern.op == 'mul' and len(template.roles) == 1)
    interaction = next(template for template in templates if template.kind == 'subexpression'
        and template.pattern.op == 'mul' and len(template.roles) == 2)
    square_program = square.instantiate({'role_0': 'x'})
    interaction_program = interaction.instantiate({'role_0': 'x', 'role_1': 'z'})
    target_thirteen = Program.from_dict(episodes[12]['program'])
    rebuilt_thirteen = plus(times(target_thirteen.children[0].children[0].value,
                                 Program('feature', name='x')),
                           times(target_thirteen.children[1].children[0].value, square_program))
    target_sixteen = Program.from_dict(episodes[15]['program'])
    rebuilt_sixteen = plus(literal(target_sixteen.children[0].value),
                          times(target_sixteen.children[1].children[0].value, interaction_program))
    square_witness = _witness(target_thirteen, rebuilt_thirteen, target_position=12,
        old_refs=[failed_nine_ref], method='old extracted square fragment plus existing current-coordinate linear head',
        capabilities=['src/hegel_machine/cognition/abstractions.py:extract_templates',
                      'src/hegel_machine/cognition/abstractions.py:fit_template',
                      'src/hegel_machine/cognition/synthesis.py:_fit_linear'])
    interaction_witness = _witness(target_sixteen, rebuilt_sixteen, target_position=15,
        old_refs=[failed_nine_ref], method='old extracted interaction fragment plus existing affine calibration',
        capabilities=['src/hegel_machine/cognition/abstractions.py:extract_templates',
                      'src/hegel_machine/cognition/abstractions.py:fit_template'])
    old_tail = extract_templates(Program.from_dict(episodes[3]['program']), 'old_t04')[0]
    target_nine = Program.from_dict(episodes[8]['program'])
    rebuilt_nine = old_tail.instantiate({'role_0': 'x'}, constants(target_nine))
    tail_witness = _witness(target_nine, rebuilt_nine, target_position=8,
        old_refs=[reference(3, 'program')], method='one original AbstractTemplate with role/constant rebinding',
        capabilities=['src/hegel_machine/cognition/abstractions.py:AbstractTemplate.instantiate',
                      'src/hegel_machine/cognition/synthesis.py:synthesize.robustifications'])
    cases = [
        {'id': 'startup_history_availability', 'theme': 'Separate unavailable history from mature history',
         'proposed_source_episode_indices': [1, 6, 13], 'status': 'reconstructed_by_old',
         'witnesses': startup_witnesses,
         'actual_old_generation': [_actual_revision(episodes, 6, 69), _actual_revision(episodes, 13, 36)],
         'retained_failure_refs': [reference(6, 'actions', 68), reference(13, 'actions', 18)],
         'interpretation': 'Lag depth two is absent from earlier single templates, but the old bounded lag/startup constructor and the actual composition already handle it.'},
        {'id': 'boolean_history_relation', 'theme': 'Compose current and historical threshold relations',
         'proposed_source_episode_indices': [2, 7], 'status': 'reconstructed_by_old',
         'witnesses': [boolean_witness], 'actual_old_generation': [_actual_revision(episodes, 7, 42)],
         'retained_failure_refs': [reference(7, 'actions', 16)],
         'no_counterexample_gain_at_final_revision': {
             'source_ref': reference(7, 'actions', 41), 'fresh_gain': episodes[7]['actions'][41]['details']['fresh_gain'],
             'saved_terminal_loss': episodes[7]['evaluation']['loss']},
         'interpretation': 'The OR revision tied its predecessor on adaptive fresh data; nonzero retained terminal error remains. It does not establish a new supported logic.'},
        {'id': 'nonlinear_square', 'theme': 'Use a squared observable coordinate',
         'proposed_source_episode_indices': [8, 12], 'status': 'reconstructed_by_old',
         'witnesses': [square_witness], 'actual_old_generation': [_actual_revision(episodes, 12, 44)],
         'retained_failure_refs': [reference(12, 'actions', 43)],
         'old_fragment_ref': reference(8, 'actions', 20, 'details', 'after_program',
                                       *find_path(failed_nine, square_program)),
         'interpretation': 'A replaced t09 archive Program already contains the square. Ignoring rejected Programs would manufacture novelty.'},
        {'id': 'cross_variable_interaction', 'theme': 'Use a product of two distinct public coordinates',
         'proposed_source_episode_indices': [8, 15], 'status': 'reconstructed_by_old',
         'witnesses': [interaction_witness], 'actual_old_generation': [_actual_revision(episodes, 15, 42)],
         'retained_failure_refs': [reference(15, 'actions', 16), reference(15, 'actions', 41)],
         'old_fragment_ref': reference(8, 'actions', 20, 'details', 'after_program',
                                       *find_path(failed_nine, interaction_program)),
         'interpretation': 'An injective name change of a one-role square does not make a two-role product, but the earlier replaced archive and existing composition already supply that product.'},
        {'id': 'tail_calibration', 'theme': 'Calibrate a retained response by an offset and slope',
         'proposed_source_episode_indices': [3, 8], 'status': 'reconstructed_by_old',
         'witnesses': [tail_witness], 'actual_old_generation': [_actual_revision(episodes, 8, 39)],
         'retained_failure_refs': [reference(8, 'actions', 40)],
         'interpretation': 'The complete ordered t09 structure is an instance of the existing t04 template; changed numerical calibration is not a new shared relation.'},
    ]
    for case in cases:
        case['representation_status'] = case['status']
        case['status_scope'] = 'existing_raw_Program_and_constructor_coverage'
        case['historical_generation_status'] = 'already_generated_by_old'
        case['shared_relation_status'] = 'no_multi_episode_object'
    external = _external_leads()
    identities = _mainline_identity_check(source, len(episodes))
    qualified = sum(item['shared_relation_status'] == 'executable_new_shared_relation_candidate'
                    for item in [*cases, *external])
    receipt = {'status': 'passed', 'scope': 'bounded_retained_history_relation_novelty_audit',
        'sources': {'retained_cognitive_sequence': source},
        'protocol': {'episode_identity': 'archive member plus episode list position; task_id alone is insufficient',
            'duplicate_task_id': {'task_id': 't02', 'distinct_episode_indices': [1, 16]},
            'source_cutoff': 'Each witness cites raw Programs from strictly earlier Episode positions; actual target revision is reported separately',
            'rejected_programs_retained': True,
            'target_parameter_use': 'Only exact representation reconstruction; no proposer receives target constants',
            'comparison': 'Ordered Program equality; absence of one template match does not exclude old composition',
            'new_paid_data_or_sampling': False},
        'source_episode_count': len(episodes),
        'mainline_episode_identity_check': identities,
        'repeated_task_identity': [{'episode_index': i, 'task_id': episodes[i]['task_id'],
            'phase': episodes[i]['evaluation']['task_metadata']['phase'], 'source_ref': reference(i)}
            for i in (1, 16)],
        'interpretation_boundary': 'Raw Program reconstructibility, actual old generation, and evidence for a new cross-Episode relation are separate claims. A general constructor does not establish that the old library already knew a shared relation.',
        'retained_replaced_program': {'source_ref': failed_nine_ref,
            'replacement_ref': reference(8, 'actions', 39),
            'square_and_interaction_fragments_extracted': True, 'treated_as_global_scientific_law': False},
        'cases': cases, 'external_leads': external, 'qualified_new_shared_relation_count': qualified,
        'ablation_status': 'not_started_no_qualified_relation',
        'unresolved': [
            {'status': 'UNKNOWN_missing_paid_labels',
             'question': 'Whether a different compiled prior improves search cost on genuinely withheld history-dependent tasks',
             'reason': 'The retained evaluation examples omit complete measured histories; no new labelled target trajectory is created here.'},
            {'status': 'no_multi_episode_object',
             'question': 'A new cross-Episode constraint coupling guard depth or shared coefficients',
             'reason': 'This audit found old constructors and actual revisions, not an independently authored new executable relation with a separate target.'}],
        'no_acquisition': {'provider_calls': 0, 'lab_execute_calls': 0, 'new_paid_observations': 0,
                           'new_experiment_fees': 0, 'source_payload_bytes_exported': 0},
        'limitations': ['The audit covers one retained 17-Episode development sequence and five located small external leads, not every historical file or every possible relation.',
            'Reconstruction proves old representability only; target constants were read for this audit and give no prospective prediction or cost advantage.',
            'Current old-capability code locations explain the constructor paths; the archived revision actions independently establish that the historical execution already generated these structures.',
            'The two occurrences of t02 are distinct Episodes; later meanings and target outcomes never enter earlier source archives.']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pointer', type=Path, default=POINTER)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.source_pointer, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output),
        'qualified_new_shared_relation_count': result['qualified_new_shared_relation_count'],
        'ablation_status': result['ablation_status'],
        'source_episode_count': result['source_episode_count']}, ensure_ascii=False))
