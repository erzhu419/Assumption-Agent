from scripts.analyze_conditional_event_prefixes import duplicate_groups, records_at_prefix


def record(index, level, target):
    return {'row_indices': [index], 'rows': [{'target': target}],
            'action': {'kind': 'trajectory', 'channels': [], 'sequence': [{'x': level}]}}


def test_prefix_excludes_later_evidence_but_includes_nested_confirmation():
    initial, confirmed, future = record(0, 0, 1), record(1, 1, 2), record(2, 2, 3)
    training = {'paid_observations': [initial, confirmed, future], 'events': [
        {'feedback': {'observations': [initial]}},
        {'feedback': {'fresh_confirmation': {'observations': [confirmed]}}},
        {'feedback': {'observations': [future]}}]}
    assert records_at_prefix(training, 1) == [initial]
    assert records_at_prefix(training, 2) == [initial, confirmed]


def test_duplicate_actions_stay_together_without_grouping_by_labels():
    first = record(0, 0, 1)
    repeat_with_noise = record(1, 0, 1.1)
    other_action_same_target = record(2, 1, 1)
    different_readout = record(3, 0, 1)
    different_readout['action']['channels'] = ['meter']
    assert duplicate_groups([first, repeat_with_noise, other_action_same_target,
                             different_readout]) == [0, 0, 1, 2]
