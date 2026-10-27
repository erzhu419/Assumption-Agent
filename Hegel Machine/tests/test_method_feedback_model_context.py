"""Host method feedback is not a predictor input or another paid snapshot."""

from pathlib import Path

import pytest

from hegel_machine.cognition.code_agent import python_model_context_matches
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.research_episodes import load_source_values
from hegel_machine.cognition.temporal_operator_agent import export_temporal_program
from test_python_paid_score_reuse import scene, choose, no_execution


def assessment(agent, result):
    reference = result['evaluations'][0]['paid_score_reuse']['execution_basis_ref']
    return reference, load_source_values([reference], agent.workspace)[0]['value']


@pytest.mark.parametrize('literal', [False, True])
def test_host_feedback_updates_reuse_the_same_paid_score_without_copying_records(tmp_path, monkeypatch, literal):
    agent, lab = scene(tmp_path)
    model = 'old.py'
    agent._save(model, export_temporal_program(Program('const', value=0.)) if literal
                else 'def predict(row): return 0.\n')
    agent._save('method_feedback/use_000000.json', '{"status":"pending_comparison"}')
    first = choose(agent, lab, [model])
    reference, paid = assessment(agent, first)
    retained = Path(paid['execution_context']['workspace'])
    assert not (retained / 'method_feedback').exists()
    if literal:
        assert paid['closed_prediction_state']
    before = agent.costs['compute_cost']
    agent._save('method_feedback/use_000000.json', '{"status":"completed","paired_future":{"rows":8}}')
    agent._save('method_feedback/paid_000001.json', '[{"features":{"u":1},"target":1}]')
    monkeypatch.setattr('hegel_machine.cognition.online_inquiry_llm.evaluate_python_model', no_execution)
    repeated = choose(agent, lab, [model])
    reused_ref, reused = assessment(agent, repeated)
    assert repeated['new_scored_count'] == 0 and repeated['reused_score_count'] == 1
    assert reused_ref == reference and reused == paid
    assert agent.costs['compute_cost'] == before
    assert python_model_context_matches(agent.workspace, model, retained, paid['execution_context']['model'])


@pytest.mark.parametrize('change', ['program', 'parameters'])
def test_actual_program_or_parameters_still_invalidate_the_score(tmp_path, change):
    agent, lab = scene(tmp_path)
    model = 'old.py'
    agent._save(model, 'import json\nfrom pathlib import Path\n'
        'def predict(row): return json.loads(Path("params.json").read_text())["value"]\n')
    agent._save('params.json', '{"value":0}')
    agent._save('method_feedback/use_000000.json', '{"status":"pending"}')
    choose(agent, lab, [model])
    if change == 'program':
        agent._save(model, export_temporal_program(Program('const', value=3.)))
    else:
        agent._save('params.json', '{"value":3}')
    before = agent.costs['compute_cost']
    result = choose(agent, lab, [model])
    assert result['new_scored_count'] == 1 and result['reused_score_count'] == 0
    assert agent.costs['compute_cost'] == before + len(agent.rows)


def test_same_name_in_an_ordinary_data_directory_remains_a_real_input(tmp_path):
    agent, lab = scene(tmp_path)
    model = 'old.py'
    agent._save(model, 'import json\nfrom pathlib import Path\n'
        'def predict(row): return json.loads(Path("data/method_feedback/params.json").read_text())["value"]\n')
    agent._save('data/method_feedback/params.json', '{"value":0}')
    first = choose(agent, lab, [model])
    _, original = assessment(agent, first)
    assert (Path(original['execution_context']['workspace']) / 'data/method_feedback/params.json').is_file()
    agent._save('data/method_feedback/params.json', '{"value":2}')
    result = choose(agent, lab, [model])
    assert result['new_scored_count'] == 1 and result['reused_score_count'] == 0


def test_explicitly_bound_child_under_host_directory_keeps_its_execution_context(tmp_path):
    agent, lab = scene(tmp_path)
    model = 'parent.py'
    agent._save(model, 'def bind_models(models):\n global child\n child=models["child"]\n'
        'def predict(row): return child(row)\n')
    agent._save('model_bindings.json', '{"parent.py":{"child":"method_feedback/_evaluated_model/model.py"}}')
    agent._save('method_feedback/use_000000.json', '{"status":"completed"}')
    agent._save('method_feedback/_evaluated_model/model.py', 'import json\nfrom pathlib import Path\n'
        'def predict(row): return json.loads(Path("params.json").read_text())["value"]\n')
    agent._save('method_feedback/_evaluated_model/params.json', '{"value":0}')
    first = choose(agent, lab, [model])
    _, paid = assessment(agent, first)
    retained = Path(paid['execution_context']['workspace'])
    assert not (retained / 'method_feedback/use_000000.json').exists()
    assert (retained / 'method_feedback/_evaluated_model/model.py').is_file()
    assert python_model_context_matches(agent.workspace, model, retained, model)
    agent._save('method_feedback/_evaluated_model/params.json', '{"value":2}')
    assert not python_model_context_matches(agent.workspace, model, retained, model)
    result = choose(agent, lab, [model])
    assert result['new_scored_count'] == 1 and result['reused_score_count'] == 0
