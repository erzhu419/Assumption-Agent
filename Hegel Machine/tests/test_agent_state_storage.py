"""Scientific state writers keep prior paid evidence through failed replacement."""

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from hegel_machine.cognition import checkpoint_storage as storage
from hegel_machine.cognition.code_agent import CodeAgent
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMAgent


class TransitionLab:
    def __init__(self):
        self.calls = 0

    def acquire(self, probe, channels):
        self.calls += 1
        return SimpleNamespace(observations=[Observation({'u': self.calls}, self.calls * 2)],
            measurement_cost=0, transitions=[{'before': {'u': self.calls - 1},
                                            'after': {'u': self.calls}}])


def test_code_query_and_state_restore_keep_rows_transitions_and_fitting_views(tmp_path):
    agent = CodeAgent(None, tmp_path / 'source')
    lab = TransitionLab()
    for _ in range(2):
        agent._execute('query', {'sequence': [{'u': 1}], 'channels': []}, lab)
    state = agent.to_dict()
    restored = CodeAgent(None, tmp_path / 'restored', state=state)
    assert restored.to_dict() == state
    assert json.loads((restored.workspace / 'observations.json').read_text()) == json.loads(json.dumps(state['observations']))
    assert json.loads((restored.workspace / 'transitions.json').read_text()) == state['transition_batches']
    restored._write_model_observations(lab, [])
    observations = restored.workspace / 'observations.json'
    fitting = restored.workspace / 'model_observations.json'
    assert fitting.read_bytes() == observations.read_bytes()
    assert fitting.stat().st_ino != observations.stat().st_ino
    fitting.write_text('[]')
    assert json.loads(observations.read_text()) == json.loads(json.dumps(state['observations']))


def test_code_query_failed_transition_replacement_preserves_existing_evidence(tmp_path, monkeypatch):
    agent, lab = CodeAgent(None, tmp_path), TransitionLab()
    agent._execute('query', {'sequence': [{'u': 1}]}, lab)
    original = {name: (tmp_path / name).read_bytes() for name in
                ('observations.json', 'transitions.json', 'model_observations.json')}

    def failed(_fd):
        raise OSError('storage full')

    monkeypatch.setattr(storage.os, 'fsync', failed)
    with pytest.raises(OSError, match='storage full'):
        agent._execute('query', {'sequence': [{'u': 2}]}, lab)
    for name, content in original.items():
        assert (tmp_path / name).read_bytes() == content
    assert not list(tmp_path.glob('*.writing'))


@pytest.mark.parametrize('kind', ['numerical', 'world'])
def test_failed_evidence_replacement_keeps_all_previous_records(tmp_path, monkeypatch, kind):
    if kind == 'numerical':
        agent = InquiryLLMAgent(None, tmp_path, arm='code_react')
        agent.rows = [Observation({'u': 1}, 2)]
        agent.observations = [{'index': 0, 'rows': [asdict(agent.rows[0])]}]
    else:
        agent = DiscoveryWorldAgent(None, tmp_path, arm='hegel',
            limits={'python_seconds_per_task': 10})
        agent.observations = [{'index': 0, 'action': {'action': 'READ'},
            'feedback': {'reading': 2}, 'observation': {'lastActionMessage': '2'}}]
    agent._write_evidence()
    original = {path.name: path.read_bytes() for path in tmp_path.glob('*.json')}
    agent.observations.append(deepcopy(agent.observations[0]))
    if kind == 'numerical':
        agent.rows.append(Observation({'u': 2}, 4))

    def failed(_fd):
        raise OSError('storage full')

    monkeypatch.setattr(storage.os, 'fsync', failed)
    with pytest.raises(OSError, match='storage full'):
        agent._write_evidence()
    assert {path.name: path.read_bytes() for path in tmp_path.glob('*.json')} == original
    assert not list(tmp_path.glob('*.writing'))


def test_world_research_state_failure_retains_previous_question_and_costs(tmp_path, monkeypatch):
    agent = DiscoveryWorldAgent(None, tmp_path, arm='hegel',
        limits={'python_seconds_per_task': 10})
    agent.lab = SimpleNamespace(public_observation=lambda: {'lastActionMessage': 'ready'})
    agent._save_state()
    state_path = tmp_path / 'research_state.json'
    original = state_path.read_bytes()
    clone = storage._clone_file

    def failed(source, destination):
        if str(source) == str(state_path):
            raise OSError('storage full')
        return clone(source, destination)

    monkeypatch.setattr(storage, '_clone_file', failed)
    agent.costs['compute_cost'] = 23
    with pytest.raises(OSError, match='storage full'):
        agent._save_state()
    assert state_path.read_bytes() == original
    assert json.loads(original)['costs']['compute_cost'] == 0
    assert not list(tmp_path.glob('*.writing'))
