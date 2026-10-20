#!/usr/bin/env python3
"""Read the real frozen publication pair through Common, without execution."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.method_effect_query import PUBLISHED_EFFECT_KIND
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/evidence/conditional_online_failure_20261013/window_parameter_transfer_20261017.json'
OUTPUT = ROOT / 'docs/evidence/published_method_effect_20261021.json'


class ReadOnlyLab:
    def public_spec(self):
        return {'controls': []}

    def execute(self, action):
        raise AssertionError('Published MethodEffect replay must not execute methods or acquire data')


def replay(source=SOURCE, output=OUTPUT):
    query = {'op': 'MethodEffect', 'effect_kind': PUBLISHED_EFFECT_KIND,
        'comparison_ref': {'file': str(Path(source).resolve()), 'json_path': ['execution', 'final_outcome']},
        'left_arm': 'baseline_13of13', 'right_arm': 'published_14of14'}
    question = {'id': 'retained_published_method_effect',
        'question': 'What outcomes and costs did the registered frozen publication pair obtain?',
        'parent_id': 'task', 'depends_on': [], 'kind': 'method_effect', 'answer_type': 'relation',
        'answer_resolution_ref': {'kind': 'complete_registered_target_coverage_and_cost'},
        'query_semantics': {'mode': 'model_query', 'query_expression': query}, 'use_sites': []}
    with tempfile.TemporaryDirectory(prefix='hegel-published-method-effect-') as temporary:
        agent = CommonArchiveInquiryAgent(None, temporary, arm='hegel', source_task=False,
            config=InquiryLLMConfig(max_model_calls=0))
        agent.research_mission = {'purpose': 'method_learning', 'mission_id': 'retained_published_method_effect'}
        lab, before = ReadOnlyLab(), deepcopy(agent.costs)
        result = agent._execute('run', {'operator': 'research_step', 'research_question': question}, lab)
        if result['status'] != 'answer_ready_no_parent_use':
            raise AssertionError(result)
        repeated = agent._execute('run', {'operator': 'research_step',
            'research_question': {'id': question['id']}}, lab)
        if agent.costs != before or agent.observations or agent.confirmations:
            raise AssertionError('The read-only MethodEffect query executed new research')
        answer = result['answer']
        value, scope = answer['value'], answer['scope']
        arms = {side: {key: deepcopy(value[side][key]) for key in ('arm', 'method_source_ref',
            'publication_ref', 'protocol_ref', 'summary_ref', 'case_count', 'submitted',
            'confirmation_passed', 'terminal_passed', 'costs', 'recorded_author_costs',
            'recorded_author_costs_ref', 'author_cost_scope', 'prior_author_phases_ref',
            'prior_transfer_phases_ref')} for side in ('left', 'right')}
        lineage = value['source_cost_lineage']
        receipt = {'status': 'completed', 'scope': 'frozen_published_method_package_transfer_effect',
            'query': query, 'source_ref': answer['source_ref'],
            'Common': {'operator': 'research_step', 'status': result['status'],
                'mission': deepcopy(agent.research_mission), 'parent_revision_completed': result['parent_revision_completed'],
                'same_repeated_answer': repeated.get('answer') == answer, 'new_costs': {
                    key: agent.costs[key] - before[key] for key in agent.costs},
                'new_paid_records': 0, 'new_confirmations': 0, 'new_method_executions': 0},
            **arms, 'right_minus_left': value['right_minus_left'],
            'registered_case_ids': scope['registered_case_ids'],
            'contract': {key: deepcopy(scope[key]) for key in ('registration_ref', 'independent_parameter_groups',
                'conditions_per_group', 'same_original_prediction_archive', 'runner_ref', 'prefix_runner_ref',
                'target_runner', 'prefix_rows', 'methods_include_publication_support_provenance',
                'literal_source_bytes_only', 'general_method_superiority', 'prediction_knowledge_gain',
                'new_independent_confirmation')},
            'source_cost_lineage': {'shared_prior_author_phase_count': len(lineage['shared_prior_author_phase_refs']),
                'nonidentical_prior_author_phase_counts': {key: len(refs) for key, refs in
                    lineage['nonidentical_prior_author_phase_refs'].items()},
                'phase_accounts_summed': False, 'nonidentical_phases_are_independent': 'not_established'},
            'failure_scope': {'baseline_no_submission_count': value['left']['case_count'] - value['left']['submitted'],
                'baseline_terminal_mse': None, 'terminal_mse_difference': value['right_minus_left']['terminal_mse'],
                'case_records': 'All eight outcomes and costs remain at each arm summary_ref.new_parameters.'}}
    Path(output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = replay(args.source, args.output)
    print(json.dumps({'status': result['status'], 'Common': result['Common']['status'],
        'terminal_passed': [result['left']['terminal_passed'], result['right']['terminal_passed']],
        'cost_difference': result['right_minus_left']['costs'], 'output': str(args.output)}, ensure_ascii=False))
