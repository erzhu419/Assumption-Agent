"""Same-source prior confirmation policy; retain complete-visit cost learning."""
def prior_confirmation(planner, spec, rows, selected, budget, **kwargs):
    kwargs.pop('confirmation_baseline', None)
    state = selected.get('condition_uncertainty') or {}
    tolerance = kwargs.get('tolerance', 1e-7)
    if (state.get('status') != 'unresolved'
            or state.get('original_target_loss') is None
            or state['original_target_loss'] > tolerance):
        return None
    return planner(spec, rows, selected, budget, **kwargs)


def install_prior_confirmation():
    from hegel_machine.cognition import observed_condition_experiments as experiments
    original = experiments.plan_observed_condition_probe
    def planner(spec, rows, selected, budget, **kwargs):
        # Preserve ordinary discovery calls that were never confirmation calls.
        if 'confirmation_baseline' not in kwargs:
            return original(spec, rows, selected, budget, **kwargs)
        if kwargs['confirmation_baseline'] is not None:
            from hegel_machine.cognition.observed_condition_uncertainty import assess_observed_condition_uncertainty
            state = assess_observed_condition_uncertainty(spec, rows, selected, budget,
                previous=selected.get('condition_uncertainty'),
                tolerance=kwargs.get('tolerance', 1e-7))
            if (state is None or state.get('status') != 'unresolved'
                    or state.get('original_target_loss') is None
                    or state['original_target_loss'] > kwargs.get('tolerance',1e-7)):
                return None
            selected['condition_uncertainty'] = state
        return prior_confirmation(original, spec, rows, selected, budget, **kwargs)
    experiments.plan_observed_condition_probe = planner
