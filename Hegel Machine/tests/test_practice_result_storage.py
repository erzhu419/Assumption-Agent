"""Practice traces stay local; online source results remain portable archives."""

import json
from pathlib import Path

import pytest

from hegel_machine.cognition.anonymous_method_learning import (
    AnonymousPreparedMethodAgent, SingleSubmissionMethodAgent,
)
from hegel_machine.cognition.anonymous_research_archive import load_research_archive
from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from test_common_archive_online_research import ConfirmationLab
from test_inquiry_revision import action


MODEL = '''import json
import basis_config
from pathlib import Path
parameters = json.loads(Path('parameters.json').read_text())
def predict(row):
    return basis_config.GAIN * row['features']['u'] + parameters['offset']
'''

METHOD = '''def select_actions(spec, records, memory):
    if not records:
        return {'actions': [{'kind': 'trajectory', 'sequence': [{'u': 1.}]}]}
    return {'model': 'model.py', 'lesson': 'The current paid data support this predictor.'}
'''


def knowledge():
    return {'files': {'model.py': MODEL, 'basis_config.py': 'GAIN = 1.\n',
        'parameters.json': '{"offset": 0}', 'notes.md': 'inherited notes'},
        'manifest': {'hypotheses': [{'id': 'predictor', 'path': 'model.py'}],
                     'research_programs': [{'id': 'method', 'path': METHOD_PATH}]},
        'lessons': ['Check the new paid data before adoption.']}


@pytest.mark.parametrize('agent_class', [AnonymousPreparedMethodAgent, SingleSubmissionMethodAgent])
def test_practice_result_keeps_executable_disk_context_without_reading_snapshot_tree(
        tmp_path, monkeypatch, agent_class):
    agent = agent_class(None, tmp_path, arm='hegel', source_task=False, knowledge=knowledge(),
        config=InquiryLLMConfig(observation_budget=10, confirmation_rows=3,
                               compute_budget=100, execution_seconds=30.))
    agent.bind_method(METHOD)
    (tmp_path / 'notes.md').write_text('updated by this practice')
    history = tmp_path / '_experiment_models/comparison_0000/candidate_0000/_evaluated_model/context.json'
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps({'retained_evidence': 'x' * 100_000}))
    read_text = Path.read_text

    def read_without_reexport(path, *args, **kwargs):
        assert path != history, 'practice result read a complete old candidate context for reexport'
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_without_reexport)
    result = agent.execute_method(ConfirmationLab(), max_steps=2)
    assert result['status'] == 'completed' and result['fresh_confirmation']['passed']
    assert result['costs']['observation_cost'] == 4
    assert len(result['method_results']) == 2
    assert result['inherited_files_changed'] == ['notes.md']
    outgoing = result['outgoing_knowledge']
    assert 'files' not in outgoing and outgoing['workspace'] == str(tmp_path)
    assert outgoing['manifest']['hypotheses']
    assert 'The current paid data support this predictor.' in outgoing['lessons']
    assert result['knowledge_protocol'] == 'method_practice_workspace'
    assert 'retained_evidence' not in json.dumps(result)
    assert history.exists()
    evaluation = evaluate_python_model(tmp_path, result['model'], [Observation({'u': 7.}, 7.)])
    assert evaluation['returncode'] == 0 and evaluation['mse'] == 0, evaluation


def test_online_source_export_still_supports_training_only_archive_fallback(tmp_path):
    root = tmp_path / 'source'
    workspace = root / 'case_000' / 'code_react'
    agent = CommonArchiveInquiryAgent(None, workspace, arm='code_react', source_task=False,
        knowledge=knowledge(), config=InquiryLLMConfig(observation_budget=10,
            confirmation_rows=3, compute_budget=100, execution_seconds=30.))
    lab = ConfirmationLab()
    agent._execute('query', {'actions': [action(1.)]}, lab)
    finished = agent._execute('finish', {'model': 'model.py'}, lab)
    assert finished['fresh_confirmation']['passed']
    result = agent.result('completed')
    assert result['outgoing_knowledge']['files']['model.py'] == MODEL
    assert result['outgoing_knowledge']['files']['parameters.json'] == '{"offset": 0}'
    assert result['knowledge_protocol'] == 'common_executable_archive'
    (workspace / 'training.json').write_text(json.dumps(result))
    (root / 'protocol.json').write_text(json.dumps({
        'cases': [{'id': 'case_000'}], 'arms': ['code_react']}))
    assert not (workspace / 'archive.json').exists()

    exported, sources = load_research_archive(root)
    target = tmp_path / 'target'
    for name, content in exported['files'].items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    evaluation = evaluate_python_model(target, sources[0]['model'], [Observation({'u': 9.}, 9.)])
    assert evaluation['returncode'] == 0 and evaluation['mse'] == 0, evaluation
