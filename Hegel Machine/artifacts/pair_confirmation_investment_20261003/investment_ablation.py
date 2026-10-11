"""Study-only 0.172 single-batch procurement; keep the coherent pair planner."""
def acquire_fixed_pair_confirmation(spec, selected, plan, budget, actions, *, acquire, **kwargs):
    return acquire(kwargs['channels'], 'confirm_continuation_program', probe=plan['probe'])


def install_prior_investment():
    from hegel_machine.cognition import condition_confirmation
    condition_confirmation._acquire_pair_confirmation = acquire_fixed_pair_confirmation
