"""Missing public instruments leave failed Python models available to research."""

from copy import deepcopy

from hegel_machine.cognition.common_archive_inquiry import CommonArchiveInquiryAgent
from hegel_machine.cognition.finite_native_author import export_finite_native_author
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_research_routing import instrument_action
from hegel_machine.cognition.temporal_paired_method import execute_procedure
from test_finite_native_author import ChoiceLab, POLICY, declaration
from test_inquiry_revision import action


class NoMeterLab(ChoiceLab):
    def public_spec(self):
        return {**super().public_spec(), 'channels': []}


def test_unavailable_meter_does_not_block_failed_python_choice_and_native_research(tmp_path):
    contract = declaration()
    contract['predictor_paths'] = ['archive/failed.py']
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        knowledge={'files': {POLICY: export_finite_native_author(),
            'archive/failed.py': 'def predict(row): return 0.\n'}, 'finite_native_author': contract},
        config=InquiryLLMConfig(observation_budget=20, compute_budget=3000,
            confirmation_rows=2, execution_seconds=30., max_model_calls=0))
    lab = NoMeterLab()
    agent._execute('query', {'actions': [action(1., 2.), action(3., 4.)]}, lab)
    before = deepcopy(agent.costs)
    result = execute_procedure(agent, lab, POLICY, max_steps=4, allow_confirmation=False,
                               record_events=True)
    assert [step['proposal']['arguments']['operator'] for step in result['steps'][:3]] == [
        'research_choice', 'refit', 'native_condition_research'], result
    # A failed refit changes neither the model roster nor paid rows. Reuse the
    # completed choice, then return the native search's actual unresolved work.
    stopped = result['steps'][3]['proposal']['stop']
    assert stopped['status'] == 'author_work_request'
    assert stopped['request']['reason'] == 'executable_revision_or_scope_required'
    assert stopped['request']['native_status'] == 'needs_matched_evidence'
    choices = [item for item in agent.operator_results if item.get('operator') == 'research_choice']
    assert len(choices) == 1
    assert choices[0]['next_action']['arguments']['operator'] == 'refit'
    assert choices[-1]['evaluated_model']['mse'] > 0
    assert (tmp_path / 'archive/failed.py').read_text() == 'def predict(row): return 0.\n'
    assert any(item.get('status') == 'refit_unavailable' for item in agent.operator_results)
    native = agent.operator_results[-1]
    assert native['operator'] == 'native_condition_research' and 'error' not in native
    assert 0 < native['native_condition_research']['compute_cost'] <= contract['max_compute']
    # Exercise the original missing-meter caller explicitly without requiring
    # the finite policy to repeat an unchanged complete comparison.
    before_recheck = deepcopy(agent.costs)
    recheck = agent._execute('run', {'operator': 'research_choice', 'models': ['archive/failed.py']}, lab)
    assert recheck['diagnosis'] == 'instrument_channel_unavailable'
    assert recheck['recommendation'] is None and not recheck.get('next_action')
    assert recheck['new_scored_count'] == 0 and recheck['reused_score_count'] == 1
    assert agent.costs['compute_cost'] == before_recheck['compute_cost']
    assert agent.costs['compute_cost'] > before['compute_cost']
    assert agent.costs['observation_cost'] == lab.paid == 4 and len(lab.actions) == 2
    assert agent.costs['measurement_cost'] == agent.costs['model_calls'] == 0
    assert not agent.confirmations


def test_available_meter_keeps_the_existing_paired_calibration_proposal(tmp_path):
    agent = CommonArchiveInquiryAgent(None, tmp_path, arm='code_react', source_task=False,
        config=InquiryLLMConfig(observation_budget=40, compute_budget=3000,
            confirmation_rows=2, execution_seconds=30., max_model_calls=0))
    lab = NoMeterLab()
    agent._execute('query', {'actions': [action(1., 2., 3., 4., 5., 6.),
                                       action(6., 5., 4., 3., 2., 1.)]}, lab)
    before = deepcopy(agent.costs)
    spec = {**lab.public_spec(), 'channels': [{'name': 'meter', 'cost_per_row': 3}]}
    offered = instrument_action(agent, spec)
    proposal = offered['next_action']
    assert proposal['action'] == 'query'
    assert [item['channels'] for item in proposal['arguments']['actions']] == [['meter'], ['meter']]
    assert [item['operator'] for item in proposal['arguments']['after_observation']] == [
        'instrument', 'assess_instrument']
    assert offered['measurement_cost'] == 36
    assert agent.costs == before and lab.paid == 12
