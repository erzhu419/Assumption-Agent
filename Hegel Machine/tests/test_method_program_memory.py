"""Actual method phases survive restoration without replaying completed fits.

These tiny models and programs are test fixtures, not live research inputs.
"""

from copy import deepcopy

from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.experimental_design_control import ExperimentalDesignMethodAgent
from test_inquiry_revision import action, write
from test_temporal_python_refit import LawLab


MODEL = '''from pathlib import Path
def fit(rows):
    count = Path('actual_fit_count.txt')
    count.write_text(str((int(count.read_text()) if count.exists() else 0) + 1))
    return {'rows':len(rows)}
def predict(row): return row['features']['u']
'''

METHOD = '''def select_actions(spec, records, memory):
    assert memory['program_path'] == 'shared/consolidated_method.py'
    assert memory['program_arguments']['path'] == memory['program_path']
    complete = [step for step in memory['program_results']
        if (step.get('proposal') or {}).get('action') == 'evaluate'
        and step['feedback'].get('returncode') == 0
        and step['evidence_after'] == memory['program_evidence']]
    if complete:
        return {'stop':'Current paid evidence has already been fitted and evaluated.'}
    return {'action':'evaluate','arguments':{'model':'candidate.py'}}
'''


def make_scene(path, source=METHOD):
    agent = ExperimentalDesignMethodAgent(None, path, arm='hegel', source_task=False,
        config=InquiryLLMConfig(observation_budget=100, confirmation_rows=3,
            compute_budget=2000, execution_seconds=30., max_model_calls=0, token_budget=0))
    lab = LawLab()
    agent._execute('query', {'actions': [action(0., channels=[])]}, lab)
    write(agent, 'candidate.py', MODEL, lab)
    agent.bind_method(source)
    return agent, lab


def evaluations(result):
    return [row for row in result['method_results']
            if (row.get('proposal') or {}).get('action') == 'evaluate']


def test_direct_program_feedback_advances_after_one_actual_fit(tmp_path):
    agent, lab = make_scene(tmp_path)
    result = agent.execute_method(lab)
    assert result['status'] == 'method_stopped'
    assert len(evaluations(result)) == 1
    assert evaluations(result)[0]['feedback']['mse'] == 0.
    assert (tmp_path / 'actual_fit_count.txt').read_text() == '1'
    assert result['program_contexts'][METHOD_PATH] == {'source': METHOD, 'start': 0}
    assert result['costs']['model_calls'] == 0 and lab.costs['observation_cost'] == 1
    assert result['fresh_confirmation'] is None


def restore(agent, result, path):
    resumed = ExperimentalDesignMethodAgent(None, path, arm='hegel', source_task=False,
        config=agent.config, knowledge=deepcopy(agent.knowledge))
    lab = LawLab()
    resumed.restore_practice_research(result, agent.workspace, lab, confirmation_seed=1)
    resumed.bind_method(METHOD)
    return resumed, lab


def test_restored_current_source_keeps_completed_phase_without_another_fit(tmp_path):
    agent, lab = make_scene(tmp_path / 'first')
    result = agent.execute_method(lab)
    resumed, new_lab = restore(agent, result, tmp_path / 'resumed')
    final = resumed.execute_method(new_lab)
    assert final['status'] == 'method_stopped'
    assert len(evaluations(final)) == 1
    assert final['costs']['compute_cost'] == result['costs']['compute_cost']
    assert (resumed.workspace / 'actual_fit_count.txt').read_text() == '1'
    assert final['program_contexts'] == result['program_contexts']
    assert final['paid_observations'] == result['paid_observations']


def test_revised_program_gets_new_scope_but_retains_old_completed_records(tmp_path):
    agent, lab = make_scene(tmp_path)
    initial = agent.execute_method(lab)
    revised = METHOD.replace('Current paid evidence', 'Revised program: current paid evidence')
    agent.bind_method(revised)
    final = agent.execute_method(lab)
    assert final['status'] == 'method_stopped'
    assert len(evaluations(final)) == 2
    assert (tmp_path / 'actual_fit_count.txt').read_text() == '2'
    assert final['method_results'][:2] == initial['method_results']
    assert final['program_contexts'][METHOD_PATH] == {'source': revised, 'start': 2}


def test_new_observations_preserve_acquisition_stage_and_allow_current_fit(tmp_path):
    source = METHOD.replace(
        "    if complete:\n",
        "    acquired = any((step.get('proposal') or {}).get('actions')\n"
        "        and step['feedback'].get('observation_cost', 0) > 0\n"
        "        for step in memory['program_results'])\n"
        f"    if complete and not acquired: return {{'actions':[{action(.5)!r}]}}\n"
        "    if complete:\n")
    agent, lab = make_scene(tmp_path, source)
    result = agent.execute_method(lab)
    assert result['status'] == 'method_stopped'
    rows = result['method_results']
    assert len(rows) == 4 and len(evaluations(result)) == 2
    assert rows[1]['evidence_before'] == {'records': 1, 'rows': 1}
    assert rows[1]['evidence_after'] == {'records': 2, 'rows': 2}
    assert rows[2]['evidence_before'] == rows[1]['evidence_after']
    assert result['program_contexts'][METHOD_PATH]['start'] == 0
    assert lab.costs['observation_cost'] == 2


def test_unscoped_old_checkpoint_is_history_not_current_draft_completion(tmp_path):
    agent, lab = make_scene(tmp_path / 'first')
    result = agent.execute_method(lab)
    old = deepcopy(result)
    old.pop('program_contexts')
    resumed, new_lab = restore(agent, old, tmp_path / 'resumed')
    final = resumed.execute_method(new_lab)
    assert len(evaluations(final)) == 2
    assert final['program_contexts'][METHOD_PATH]['start'] == 2
    assert final['method_results'][:2] == result['method_results']


def test_other_programs_steps_are_not_current_program_completed_stages(tmp_path):
    agent, lab = make_scene(tmp_path)
    write(agent, 'other.py', "def select_actions(spec, records, memory):\n"
        "    return {'action':'evaluate','arguments':{'model':'candidate.py'}}\n", lab)
    agent._execute('query', {'path': 'other.py'}, lab)
    final = agent.execute_method(lab)
    assert len(evaluations(final)) == 2
    assert [row['path'] for row in evaluations(final)] == ['other.py', METHOD_PATH]
    assert final['program_contexts'][METHOD_PATH]['start'] == 1
