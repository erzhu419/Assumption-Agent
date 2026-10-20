"""New execution failures survive an otherwise unchanged consolidation cache."""

from copy import deepcopy
import json

from test_evidence_procedure_execution import prepared
from test_theory_consolidation import consolidate


def test_new_real_execution_error_is_intaken_without_repeating_completed_induction(tmp_path):
    agent, lab = prepared(tmp_path)
    original = consolidate(agent, lab)
    stored = deepcopy(agent.theory_consolidation['candidates'])
    cursor = agent.theory_consolidation['material_cursors'][str(tmp_path.resolve())]
    assert cursor['observations'] == len(agent.observations)
    measured = [item for item in agent.theory_consolidation['materials'] if item['kind'] == 'measured']
    assert len(measured) == len(agent.observations)
    assert all(set(item['source_ref']) == {'file', 'json_path'} for item in measured)
    failed = agent._execute('evaluate', {'model': 'failed.py',
        'code': "def predict(row): raise RuntimeError('actual retained failure')\n"}, lab)
    assert failed['returncode'] != 0 and 'actual retained failure' in failed['stderr']
    # The outer controller records this exact execution feedback in live runs.
    agent.events.append({'action': 'evaluate', 'arguments': {'model': 'failed.py'}, 'feedback': failed})
    costs = deepcopy(agent.costs)
    result = consolidate(agent, lab)
    assert result['cache_hit'] and result['new_compute_cost'] == 0
    assert result['frozen_candidates'] == original['frozen_candidates']
    assert agent.theory_consolidation['candidates'] == stored and agent.costs == costs
    error = next(item for item in agent.theory_consolidation['materials'] if item['kind'] == 'error')
    assert error['source_ref'] == {'file': 'research_context.json', 'json_path': ['events', 0]}
    assert error['adds_independent_support'] is False
    material_count = len(agent.theory_consolidation['materials'])
    assert consolidate(agent, lab)['cache_hit']
    assert len(agent.theory_consolidation['materials']) == material_count
    saved = json.loads((tmp_path / result['library_path']).read_text())
    assert saved['materials'] == agent.theory_consolidation['materials']
    assert saved['material_cursors'][str(tmp_path.resolve())]['events'] == 1
    assert agent.costs['model_calls'] == 0 and not agent.confirmations
