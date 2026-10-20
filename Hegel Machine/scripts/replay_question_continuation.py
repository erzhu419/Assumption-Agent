"""Replay the paid source006 cutoff through PR1 without a scientific run.

The small bridge is an operator-supplied integration fixture. It selects an
already authored experiment after a measured contrast, not a new world model.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig


class ReplayLab:
    def public_spec(self):
        return {'controls': ['x', 'z', 'c'], 'action_kinds': ['trajectory']}

    def execute(self, action):
        raise AssertionError('Historical replay cannot acquire observations')


def replay(observations_file, authored_action_file):
    records = json.loads(Path(observations_file).read_text())[:127]
    if len(records) != 127:
        raise ValueError('This replay requires the source006 127-record cutoff')
    draft = json.loads(Path(authored_action_file).read_text())['source_code']
    with TemporaryDirectory(prefix='hegel_question_replay_') as directory:
        agent = CommonArchiveInquiryAgent(None, directory, arm='hegel', source_task=False,
            config=InquiryLLMConfig(observation_budget=0, compute_budget=100, execution_seconds=30.))
        agent.observations = records
        agent.rows = [Observation(row['features'], row['target'], tuple(row.get('history', [])))
                      for record in records for row in record['rows']]
        agent._open_question({'id': 'source_method_repair', 'question': 'Revise the current predictor'})
        agent._save('shared/paid_next.py', draft)
        agent._save('shared/bridge.py', '''import json, runpy
def select_actions(spec, records, memory):
    state=json.load(open('research_state.json'))
    child=next(q for q in state['questions'] if q['id']=='paid_order_contrast')
    if child['semantic_answer']['value'] == 0:
        return {'stop':'This finite contrast supplies no difference for this branch.'}
    return runpy.run_path('shared/paid_next.py')['select_actions'](spec, records, memory)
''')

        def readout(index):
            record = records[index]
            row = record['rows'][12]
            return {'op': 'Readout', 'record_index': index, 'step': 12,
                    'binding': {'action': record['action'], 'features': row['features'],
                                'history': row.get('history', [])}}

        question = {'id': 'paid_order_contrast', 'question': 'What is the measured first washout contrast?',
            'parent_id': 'source_method_repair', 'depends_on': [], 'kind': 'value',
            'context_ref': {'target': 'response', 'protocol': 'retained_reset_trajectory'},
            'answer_type': 'number', 'answer_resolution_ref': {'scope': 'these two paid rows'},
            'query_semantics': {'mode': 'model_query', 'query_expression': {
                'op': 'Difference', 'left': readout(125), 'right': readout(126)}},
            'use_sites': [{'kind': 'choose_parent_experiment', 'parent_id': 'source_method_repair',
                           'path': 'shared/bridge.py'}]}
        result = agent._execute('run', {'operator': 'research_step', 'research_question': question}, ReplayLab())
        if not result.get('parent_revision_completed'):
            raise RuntimeError(json.dumps(result))
        parent = next(q for q in agent.questions if q['id'] == 'source_method_repair')
        actions = parent['pending_query']['actions']
        assert result['parent_revision_completed'] and len(actions) == 2
        assert [len(action['sequence']) for action in actions] == [32, 32]
        before = deepcopy(agent.costs)
        reused = agent._execute('run', {'operator': 'research_step',
                                       'research_question': {'id': question['id']}}, ReplayLab())
        assert reused['status'] == 'reused_answer' and agent.costs == before
        assert agent.costs['observation_cost'] == agent.costs['model_calls'] == 0
        return {'scope': 'operator_supplied_bridge_on_historical_cutoff_not_autonomous_discovery',
            'observations_ref': str(Path(observations_file).resolve()), 'record_cutoff': 127,
            'paid_row_cutoff': len(agent.rows), 'authored_next_ref': str(Path(authored_action_file).resolve()),
            'readout': {'records': [125, 126], 'step': 12, 'operation': 'Difference'},
            'answer': result['answer'], 'parent_revision_completed': result['parent_revision_completed'],
            'prepared_revision': parent['pending_query']['revision']['id'],
            'prepared_action_rows': [len(action['sequence']) for action in actions],
            'purchased_action_rows': 0, 'repeat_status': reused['status'],
            'costs': asdict(agent.config) | {'actual': agent.costs},
            'limitations': 'No new predictor, five-fold comparison, confirmation or autonomous research claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--authored-action', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(replay(args.observations, args.authored_action), indent=2) + '\n')
