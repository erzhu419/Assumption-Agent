"""Substrate and temperature feedback for the supplied fed-batch model."""

from math import exp


def compute_control_step(obs: dict, state_memory: dict) -> tuple[dict, dict]:
    dt = obs['dt_h']
    volume = obs['V_L']
    biomass = obs['X_meas_g_L']
    substrate = obs['S_meas_g_L']
    temperature = obs['T_meas_C']

    # The previous command is still applied over the upcoming interval.
    pending_feed = state_memory.get('pending_feed', 0.2)
    next_volume = volume + dt * pending_feed
    mu = 0.42 * exp(0.065 * (temperature - 30.0)) * substrate / (
        0.8 + substrate + substrate * substrate / 15.0
    )
    uptake = 2.0 * mu * biomass
    next_substrate = max(0.0, substrate + dt * (
        pending_feed / volume * (45.0 - substrate) - uptake
    ))

    # Track the rising side of the Haldane curve. The feed inventory cap
    # reserves cooling capacity and leaves time to consume residual substrate.
    if next_substrate >= 2.5 and substrate >= 45.0:
        feed = 0.0
    else:
        requested = next_volume * (uptake + 1.6 * (2.5 - next_substrate)) / max(1.0, 45.0 - next_substrate)
        feed = max(0.0, min(2.5, requested, (8.0 - next_volume) / dt))

    # Reach full cooling well below the safety limit even with the delay.
    coolant = max(0.0, min(1.0, (temperature - 30.0) / 2.0))
    return {'feed_rate_L_per_h': float(feed), 'coolant_flow_frac': float(coolant)}, {'pending_feed': float(feed)}
