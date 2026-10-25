"""Incremental filesystem writes preserve the complete public research JSON."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition import online_inquiry_llm
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from test_common_archive_resume import SEED, agent_at, checkpoint


@pytest.mark.parametrize('result_name', ['result.json', 'training.json'])
def test_restoration_seeds_state_but_does_not_publish_a_stale_result(
        tmp_path, monkeypatch, result_name):
    old, lab, previous = checkpoint(tmp_path)
    original_state = (old.workspace / 'research_state.json').read_bytes()
    (old.workspace / result_name).write_text(json.dumps(previous))
    original_result = (old.workspace / result_name).read_bytes()
    continued = agent_at(tmp_path / 'continued')
    writes = []
    write = online_inquiry_llm.write_json_checkpoint

    def record_previous_state(path, text, **kwargs):
        if path == continued.workspace / 'research_state.json':
            writes.append(path.read_bytes())
        return write(path, text, **kwargs)

    monkeypatch.setattr(online_inquiry_llm, 'write_json_checkpoint', record_previous_state)
    target = TemporalInquiryLab(task_index=1)
    monkeypatch.setattr(target, 'execute', lambda *_args, **_kwargs:
                        pytest.fail('restoring paid evidence must not purchase it again'))
    restored = continued.restore_research(previous, old.workspace, target,
                                         confirmation_seed=SEED)

    assert writes == [original_state]
    assert json.loads((continued.workspace / 'research_state.json').read_text()) == (
        json.loads(json.dumps(continued._program_state())))
    assert not (continued.workspace / 'result.json').exists()
    assert continued.events == old.events
    assert continued.observations == old.observations
    assert continued.method_results == old.method_results
    assert continued.costs == old.costs == restored['costs_before']
    assert target.costs == lab.costs
    assert (old.workspace / 'research_state.json').read_bytes() == original_state
    assert (old.workspace / result_name).read_bytes() == original_result


def test_checkpoint_updates_keep_complete_readable_state_and_original_scene(tmp_path):
    old, lab, previous = checkpoint(tmp_path)
    original = {name: (old.workspace / name).read_bytes() for name in
                ('research_state.json', 'observations.json', 'model_observations.json',
                 'laboratory_observations.json')}
    continued = agent_at(tmp_path / 'continued')
    continued.restore_research(previous, old.workspace, TemporalInquiryLab(task_index=1),
                               confirmation_seed=SEED)
    old_costs = deepcopy(continued.costs)
    continued.questions[-1]['status'] = 'additional_evidence_needed'
    continued.revisions.append({'index': len(continued.revisions),
                                'status': 'authored', 'source': '# revised source\n'})
    continued.method_results.append({'step': len(continued.method_results),
                                     'path': 'inherited.py', 'feedback': {'error': 'fit incomplete'}})
    continued.costs['compute_cost'] += 7
    continued._write_evidence()
    continued._prepare_program_state()
    saved = json.loads((continued.workspace / 'research_state.json').read_text())
    assert saved == json.loads(json.dumps(continued._program_state()))
    assert saved['revisions'][-1]['source'] == '# revised source\n'
    assert saved['method_results'][-1]['feedback']['error'] == 'fit incomplete'
    assert saved['costs']['compute_cost'] == old_costs['compute_cost'] + 7
    assert json.loads((continued.workspace / 'laboratory_observations.json').read_text()) == (
        json.loads(original['laboratory_observations.json']))
    for name, content in original.items():
        assert (old.workspace / name).read_bytes() == content
    assert old.costs == old_costs
