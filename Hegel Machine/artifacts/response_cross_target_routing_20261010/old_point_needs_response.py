def _point_needs_response(state, number):
    """A locally successful basis can outlive one failed extrapolated point."""
    supported = {test['family'] for test in state.get('tests', [])
                 if test['status'] == 'supported' and test['number'] != number
                 and test['number'] in state.get('target_numbers', [])}
    return any(test['number'] == number and test['status'] == 'refuted'
               and not test.get('response_question') and test['family'] in supported
               for test in state.get('tests', []))
