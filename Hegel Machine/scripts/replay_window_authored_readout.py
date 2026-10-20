#!/usr/bin/env python3
"""Execute an already-authored frozen window readout on its retained evidence."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.inquiry_revision import analyze_revision
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.question_agenda import generate_questions


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/erzhu419/hegel-active-experiments/luna_window_parameter_transfer_20261017/frozen/target_000')
OUTPUT = ROOT / 'docs/evidence/window_authored_readout_20261021.json'


class RetainedLab:
    def public_spec(self):
        return {'controls': ['u']}

    def execute(self, action):
        raise AssertionError('This readout replay cannot acquire an observation')


def reference(path, *parts):
    return {'file': str(path), 'json_path': list(parts)}


def replay_authored_readout(source=SOURCE, output=OUTPUT):
    source = Path(source)
    state_file, records_file = source / 'research_state.json', source / 'laboratory_observations.json'
    state, observations = json.loads(state_file.read_text()), json.loads(records_file.read_text())
    original = state['revisions'][0]
    method = state['method_results'][0]['proposal']['research_question']
    readout_path = state['method_results'][1]['proposal']['arguments']['readout']
    frozen_source = state['experiment_comparisons'][0]['forecasts'][1]['source']['files'][readout_path]
    record = deepcopy(original)
    # Sources stay at their original addresses; the analyzer needs predictions,
    # observations and the single original readout, not copies of predictor code.
    for entry in [*record['forecasts'], *record['prior_fit']]:
        entry.pop('source', None)
    record['proposal']['readout'] = readout_path
    record['readout_source'] = frozen_source
    with tempfile.TemporaryDirectory(prefix='hegel-window-authored-readout-') as temporary:
        agent = CommonArchiveInquiryAgent(None, temporary, arm='hegel', source_task=False,
            config=InquiryLLMConfig(max_model_calls=0, compute_budget=state['limits']['compute_budget'],
                observation_budget=state['limits']['observation_budget'],
                execution_seconds=state['limits']['execution_seconds']))
        agent.observations = observations
        agent.rows = [Observation(row['features'], row['target'], tuple(row.get('history', [])))
                      for batch in observations for row in batch['rows']]
        agent.costs.update({key: value for key, value in state['costs'].items() if key in agent.costs})
        question = agent._open_question(deepcopy(method))
        record.update(question_id=question['id'], question_meaning_version=question.get('meaning_version', 1))
        agent.revisions.append(record)
        readout_file = agent.workspace / readout_path
        readout_file.parent.mkdir(parents=True, exist_ok=True)
        readout_file.write_text(frozen_source)
        before = deepcopy(agent.costs)
        result = analyze_revision(agent, record, RetainedLab(), question)
        if result.get('readout_error') or not result.get('analysis_is_agent_authored'):
            raise AssertionError(result.get('readout_error', 'The original readout did not execute'))
        answered = result['returned_answer']
        generated = generate_questions(agent.questions, agent.research_candidates, [],
                                       mission={'purpose': 'task_prediction'})
        compute = agent.costs['compute_cost'] - before['compute_cost']
        seconds = agent.costs['python_execution_seconds'] - before['python_execution_seconds']
    fresh = state['method_results'][2]['feedback']['fresh_confirmation']
    receipt = {'status': 'completed', 'scope': 'historical_authored_readout_connection_only',
        'source_refs': {'state': reference(state_file), 'paid_records': reference(records_file),
            'question': reference(state_file, 'method_results', 0, 'proposal', 'research_question'),
            'readout_argument': reference(state_file, 'method_results', 1, 'proposal', 'arguments', 'readout'),
            'frozen_readout_source': reference(state_file, 'experiment_comparisons', 0, 'forecasts', 1,
                                               'source', 'files', readout_path),
            'original_revision': reference(state_file, 'revisions', 0)},
        'original_disconnect': {'retained_question_ids': [q['id'] for q in state['questions']],
            'revision_question_id': original.get('question_id'),
            'revision_readout': original['proposal'].get('readout'),
            'readout_executed': bool(original.get('analysis_is_agent_authored'))},
        'evidence_coverage': {'retained_laboratory_records': len(observations),
            'retained_laboratory_response_rows': sum(len(batch['rows']) for batch in observations),
            'state_current_paid_rows': state['online_research']['current_paid_rows'],
            'readout_scope': 'the original 120-row paid prefix, including its 48-row prospective acquisition',
            'later_confirmation': 'existing saved 96-row outcome; its raw rows are not reconstructed here'},
        'replay_question': {'id': question['id'], 'question': question['question'],
            'meaning_version': question.get('meaning_version', 1), 'typed_contract_invented': False},
        'readout': {'execution': result['readout_execution'], 'status': answered['status'],
            'evidence_status': answered['evidence_status'], 'check_results': answered['check_results'],
            'supported_models': answered['answer']['supported_models'],
            'refuted_models': answered['answer']['refuted_models'],
            'prospective_mse': answered['answer']['prospective_mse'],
            'evidence_rows': {'first': min(answered['evidence']), 'last': max(answered['evidence']),
                              'count': len(answered['evidence'])},
            'analysis_scope': result['analysis_scope'], 'independent_confirmation': False},
        'current_parent': {'model': state['method_results'][2]['proposal']['model'],
            'source_ref': reference(state_file, 'method_results', 2, 'feedback'),
            'paid_fit_mse': state['method_results'][2]['feedback']['mse'], 'paid_fit_rows': 120,
            'fresh_confirmation': {key: fresh[key] for key in ('passed', 'mse', 'rows', 'mse_threshold')},
            'current_failure_established': False},
        'question_boundary': {'generated_question_count_after_readout': len(generated),
            'use_sites': question.get('use_sites', []), 'premise_refs': question.get('premise_refs', []),
            'managed_input_count': len(state['research_input_bindings']),
            'status': 'author_definition_required_for_new_typed_query_or_parent_binding'},
        'costs': {'original_account': state['costs'], 'new_readout_compute': compute,
            'new_readout_python_seconds': seconds, 'new_model_executions': 0,
            'new_acquisitions': 0, 'new_model_calls': 0},
        'limitations': ['Replays the original question/readout on already paid results; does not rerun the method or fit predictors.',
            'The original readout has no numerical relation checks; its answer remains authored, not checked typed support.',
            'Both old alternatives failed; the later parent passed fresh confirmation. No current parent failure or scientific gain is established.']}
    Path(output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = replay_authored_readout(args.source, args.output)
    print(json.dumps({'status': result['status'], 'readout_status': result['readout']['status'],
        'refuted_model_count': len(result['readout']['refuted_models']), 'costs': result['costs']}))
