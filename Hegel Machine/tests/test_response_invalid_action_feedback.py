"""A recorded model action-name error remains paid, recoverable feedback."""

from scripts.consolidate_response_measurement import TransferSignalLab
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS


def test_wrong_dialog_action_name_is_recorded_then_explicit_correction_can_continue(tmp_path):
    lab = TransferSignalLab('changed_mechanism')
    agent = DiscoveryWorldAgent(None, tmp_path, arm='code_archive', limits=LIMITS)
    agent.lab = lab
    wrong = {'action': 'chosen_dialog_option_int', 'arg1': 6}
    failed = agent._execute('query', {'actions': [
        {'action': 'TALK', 'arg1': 105}, wrong, {'chosen_dialog_option_int': 6}]}, lab)
    assert failed['error']
    assert failed['observation_indices'] == [0, 1]
    assert agent.observations[-1]['action'] == wrong
    assert agent.observations[-1]['feedback']['success'] is False
    assert agent.observations[-1]['feedback']['errors'] == [
        'Unsupported laboratory action: chosen_dialog_option_int']
    assert lab.environment_steps == agent.costs['environment_steps'] == len(agent.observations) == 2
    assert lab.in_dialog and lab.frequencies[5] == 500.
    # The caller has to submit the valid interface shape; the fixture does not
    # silently reinterpret arg1 or execute the remainder after a failed action.
    resumed = agent._execute('query', {'actions': [
        {'chosen_dialog_option_int': 6}, {'chosen_dialog_option_int': 1}]}, lab)
    assert 'error' not in resumed
    assert resumed['observation_indices'] == [2, 3]
    assert lab.frequencies[5] == 0. and not lab.in_dialog
    assert lab.environment_steps == agent.costs['environment_steps'] == len(agent.observations) == 4
    assert lab.actions == [row['action'] for row in agent.observations]
