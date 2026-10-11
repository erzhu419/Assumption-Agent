"""Same-source first scored phase cost; preserve all ordinary measured feedback."""
def install_first_phase_cost():
    from hegel_machine.cognition import condition_marginal_value as value
    original_phase = value.record_condition_followup
    original_whole = value.record_instrument_revisit_cost
    first_phases = {}
    def phase(policy, **kwargs):
        if kwargs.get('before_loss') is not None and kwargs.get('after_loss') is not None:
            first_phases.setdefault(id(policy), (policy, {key: kwargs[key] for key in
                ('observation_cost', 'measurement_cost', 'compute_cost')}))
        return original_phase(policy, **kwargs)
    def whole(policy, *, observation_cost, measurement_cost, compute_cost,
              sampling_context, screening_compute_cost=0):
        _, costs = first_phases.pop(id(policy))
        return original_whole(policy, **costs, sampling_context=sampling_context,
            screening_compute_cost=screening_compute_cost)
    value.record_condition_followup = phase
    value.record_instrument_revisit_cost = whole
