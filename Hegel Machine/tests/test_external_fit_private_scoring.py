"""External historical fitting uses private files and retains the scored state."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_python_paid_score_reuse import choose, no_execution, scene, scoring_costs


def test_external_mutating_fit_keeps_source_and_adopts_actual_scored_parameters(tmp_path):
    agent, lab = scene(tmp_path / 'current')
    source = tmp_path / 'external' / '_evaluated_model'
    source.mkdir(parents=True)
    model = source / 'model.py'
    model.write_text('from pathlib import Path\ndef fit(rows):\n'
        ' p=Path("counter.txt")\n p.write_text(str(int(p.read_text())+1))\n'
        'def predict(row): return float(Path("counter.txt").read_text())\n')
    (source / 'counter.txt').write_text('0')
    (source / 'model_observations.json').write_text(json.dumps([asdict(row) for row in agent.rows]))
    before = agent.costs['compute_cost']
    for iteration in (1, 2):
        result = choose(agent, lab, [str(model)])
        evaluation = result['evaluations'][0]
        assert evaluation['returncode'] == 0 and evaluation['mse'] == .5
        assert not evaluation['paid_score_reuse']['reusable']
        assert evaluation['paid_score_reuse']['unavailable_reason'] == 'execution_context_changed_during_scoring'
        private = Path(evaluation['execution_context']['workspace'])
        assert (source / 'counter.txt').read_text() == '0'
        assert (private / 'counter.txt').read_text() == '1'
        assert evaluation['fitting_view']['file'] == str(private / 'model_observations.json')
        assert agent.evaluated_model['source_model'] == str(model)
        retained = agent.workspace / '_evaluated_model'
        assert (retained / 'counter.txt').read_text() == '1'
        previous_directory = Path.cwd()
        try:
            os.chdir(retained)
            predictor = runpy.run_path(str(retained / 'model.py'))['predict']
            assert predictor({'features': {'u': 1.}, 'history': []}) == 1.
        finally:
            os.chdir(previous_directory)
        assert agent.costs['compute_cost'] - before == iteration * len(agent.rows)


def test_external_literal_cache_hit_retains_saved_actual_context_without_execution(
        tmp_path, monkeypatch):
    agent, lab = scene(tmp_path / 'current')
    source = tmp_path / 'common_models'
    source.mkdir()
    model = source / 'program_000.py'
    model.write_text(export_temporal_program(Program('feature', name='u')))
    first = choose(agent, lab, [str(model)])
    actual = deepcopy(first['evaluations'][0]['execution_context'])
    assert first['new_scored_count'] == 1
    before = scoring_costs(agent)
    agent.evaluated_model = None
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    repeated = choose(agent, lab, [str(model)])
    assert repeated['reused_score_count'] == 1 and repeated['new_scored_count'] == 0
    assert repeated['evaluations'][0]['execution_context'] == actual
    assert agent.evaluated_model['source_model'] == str(model)
    assert scoring_costs(agent) == before
