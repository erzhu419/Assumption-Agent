def _point_needs_response(state, number):
    """Use acquired target responses after a new target's point failure."""
    other_targets = [test for test in state.get('tests', [])
                     if test['number'] != number
                     and test['number'] in state.get('target_numbers', [])
                     and not test.get('authored_experiment')]
    supported = {test['family'] for test in other_targets if test['status'] == 'supported'}
    # Tuning can solve a target without confirming the requested point or
    # naming a winning formula. completed is refreshed from paid responses
    # and current installed/activated observations before this decision.
    completed_target = any(target != number and target in state.get('target_numbers', [])
                           for target in state.get('completed', []))
    response_progress = any(test.get('response_question')
        and 'restore_frequency' not in test
        and (test['status'] == 'supported' or test['number'] in state.get('completed', []))
        for test in other_targets)
    return any(test['number'] == number and test['status'] == 'refuted'
               and not test.get('response_question')
               and not test.get('authored_experiment')
               and (test['family'] in supported or response_progress or completed_target)
               for test in state.get('tests', []))
