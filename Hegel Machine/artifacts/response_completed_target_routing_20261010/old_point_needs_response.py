def _point_needs_response(state, number):
    """Continue useful response research after a new target's point failure."""
    other_targets = [test for test in state.get('tests', [])
                     if test['number'] != number
                     and test['number'] in state.get('target_numbers', [])]
    supported = {test['family'] for test in other_targets if test['status'] == 'supported'}
    # A response search can solve a target without confirming the requested
    # point or naming a winning formula. completed is refreshed from actual
    # installed/activated observations before this routing decision.
    response_progress = any(test.get('response_question')
        and not test.get('authored_experiment') and 'restore_frequency' not in test
        and (test['status'] == 'supported' or test['number'] in state.get('completed', []))
        for test in other_targets)
    return any(test['number'] == number and test['status'] == 'refuted'
               and not test.get('response_question')
               and (test['family'] in supported
                    or response_progress and not test.get('authored_experiment'))
               for test in state.get('tests', []))
