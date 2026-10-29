"""Restore the paid local-response scene without another fit or prediction."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition.knowledge_evolution import bind_answers
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate
from hegel_machine.cognition.object_continuation import decode_entry, observations


def load_paid_scene(root, seed_acceptance, previous_repair):
    root, seed_acceptance, previous_repair = map(Path, (root, seed_acceptance, previous_repair))
    report_path = root / 'runs/106029101/pr3/response_report.json'
    new_path = seed_acceptance / 'new_claims.json'
    selection_path = previous_repair / 'working_selection.json'
    frozen_path = previous_repair / 'frozen_predictor.json'
    confirmation_path = previous_repair / 'confirmation_observations.json'
    receipt_path = previous_repair / 'receipt.json'
    diagnostic_path = previous_repair / 'paid_residual_readout.json'

    def read(path):
        return json.loads(path.read_text())

    report, registration = read(report_path), read(root / 'registration.json')
    selection, frozen = read(selection_path), read(frozen_path)
    receipt, diagnostic = read(receipt_path), read(diagnostic_path)
    if (Path(selection['retained_pool_ref']['report']).resolve() != report_path.resolve()
            or Path(selection['seed_claims_ref']).resolve() != new_path.resolve()):
        raise ValueError('Paid scores belong to another retained pool')
    if frozen['fallback'] != report['fallback']:
        raise ValueError('The paid scores and frozen response use different fallbacks')

    state, pool, refs = report['object_continuation_state'], [], []

    def retain(payload, file, path):
        pool.append(decode_entry(payload))
        refs.append({'file': str(file), 'json_path': path})

    for i, payload in enumerate(state['finalists']):
        retain(payload, report_path, ['object_continuation_state', 'finalists', i])
    for i, frontier in enumerate(state['candidate_frontiers']):
        for j, payload in enumerate(frontier['entries']):
            retain(payload, report_path, ['object_continuation_state', 'candidate_frontiers', i, 'entries', j])
    for i, payload in enumerate(read(new_path)):
        retain(payload, new_path, [i])
    retain(state['selected'], report_path, ['object_continuation_state', 'selected'])

    # Restore the exact previous Graph substitution, not a new residual fit.
    support_index = selection['source_support_action']
    support = report['actions'][support_index]['noise_assessment']
    question = selection['action']['question_id']
    expression = Program('add', children=(Program.from_dict(support['counterexample_baseline_program']),
                                         Program('arg', name=question)))
    child = bind_answers(expression,
        {question: Program.from_dict(support['structural_counterexample']['program'])})
    pool.append({'candidate': Candidate(f'answer:{question}:combined', child, 'research_answer'),
                 'kind': 'noise_residual_answer', 'family_id': 'T09',
                 'selection_basis': 'all_acquired_evidence', 'fallback': report['fallback']})
    refs.append({'file': str(report_path), 'json_path': ['actions', support_index, 'noise_assessment'],
                 'construction': 'ResearchGraph.add_residual_question plus bind_answers'})

    prior_scores = {}
    scores = selection['scores']
    if len(pool) != len(scores):
        raise ValueError('The paid comparison does not cover the restored pool')
    for item, score in zip(pool, scores):
        candidate = item['candidate']
        if candidate.candidate_id != score['candidate_id']:
            raise ValueError('Paid score order differs from the retained candidate versions')
        if candidate.program in prior_scores and prior_scores[candidate.program] != score['loss']:
            raise ValueError('One paid program has inconsistent scores on the same rows')
        prior_scores[candidate.program] = score['loss']
    pool[-1].update(training_loss=scores[-1]['loss'], validation_loss=scores[-1]['loss'])

    program = Program.from_dict(frozen['program'])
    match = next((i for i, item in enumerate(pool)
                  if item['candidate'].program == program and item.get('structure')), None)
    if match is None:
        raise ValueError('The selected program has no retained local-structure metadata')
    parent = deepcopy(pool[match])
    parent.update(fallback=frozen['fallback'], training_loss=prior_scores[program],
                  validation_loss=prior_scores[program], selection_basis='all_acquired_evidence')

    channels = tuple(frozen['selected_channels'])
    prior_rows, transitions, row_refs, control_refs = [], [], [], []
    prior_batch_offsets = {}
    for i, batch in enumerate(report['batches']):
        ref = {'file': str(report_path), 'json_path': ['batches', i],
               'rows': len(batch['observations']), 'channels': batch['channels']}
        if tuple(batch['channels']) == channels:
            prior_batch_offsets[i] = (len(prior_rows), len(batch['observations']))
            prior_rows.extend(observations(batch['observations']))
            transitions.extend(deepcopy(batch.get('transitions', [])))
            row_refs.append(ref)
        else:
            control_refs.append(ref)
    if len(prior_rows) != selection['paid_rows'] or selection['action']['uncompared_candidate_ids']:
        raise ValueError('Retained scores do not cover this entire paid instrument view')
    counterexample_indices = []
    for group in diagnostic['groups']:
        for counterexample in group['counterexamples']:
            offset, count = prior_batch_offsets[group['batch']]
            if not 0 <= counterexample['row'] < count:
                raise ValueError('The retained counterexample lies outside its original trajectory')
            counterexample_indices.append(offset + counterexample['row'])
    confirmation = read(confirmation_path)
    if tuple(confirmation['batch']['channels']) != channels:
        raise ValueError('The exposed confirmation uses another instrument view')
    exposed_rows = observations(confirmation['batch']['observations'])
    if len(exposed_rows) != receipt['confirmation']['rows']:
        raise ValueError('Exposed confirmation score and trajectory length differ')
    transitions.extend(deepcopy(confirmation['batch'].get('transitions', [])))
    exposed_scores = {
        Program.from_dict(report['program']): receipt['confirmation']['incumbent_mse'],
        program: receipt['confirmation']['selected_mse']}
    return {'report': report, 'registration': registration, 'parent': parent, 'pool': pool,
        'paid_rows': [*prior_rows, *exposed_rows], 'transitions': transitions,
        'prior_rows': prior_rows, 'exposed_rows': exposed_rows,
        'counterexample_indices': counterexample_indices,
        'prior_scores': prior_scores, 'exposed_scores': exposed_scores,
        'previous_compute': diagnostic['cumulative_development_compute'],
        'source_refs': {'report': str(report_path), 'registration': str(root / 'registration.json'),
            'parent': refs[match], 'pool': refs,
            'paid_trajectories': [*row_refs, {'file': str(confirmation_path), 'json_path': ['batch'],
                'rows': len(exposed_rows), 'channels': list(channels), 'role': 'already_exposed_evidence'}],
            'other_instrument_trajectories': control_refs,
            'prior_scores': {'file': str(selection_path), 'json_path': ['scores'],
                'binding': 'original pool occurrence order and restored Program'},
            'exposed_scores': {'file': str(receipt_path), 'json_path': ['confirmation']},
            'consumed_answer': refs[-1], 'previous_diagnostic': str(diagnostic_path),
            'historical_export_note': 'The old frozen export names a consumed support source; '
                'the selected structural entry is recovered by exact Program from its own source.'}}
