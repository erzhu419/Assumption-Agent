"""Feedback control for the supplied delayed fed-batch simulator."""

from math import exp


def compute_control_step(obs: dict, state_memory: dict) -> tuple[dict, dict]:
    """Return the next feed and coolant commands and one-step memory."""
    dt = obs['dt_h']
    volume = obs['V_L']
    biomass = obs['X_meas_g_L']
    substrate = obs['S_meas_g_L']
    temp = obs['T_meas_C']
    pending_feed = state_memory.get('feed', 0.2)
    actuation_volume = volume + dt * pending_feed

    # Use the growth model to replace substrate as it is consumed, with
    # feedback on S kept below the peak over the stated inhibition range.
    mu = 0.42 * exp(0.065 * (temp - 30.0)) * substrate / (
        0.8 + substrate + substrate * substrate / 15.0
    )
    growth = mu * biomass
    substrate_slope = pending_feed / volume * (45.0 - substrate) - 2.0 * growth
    actuation_substrate = substrate + dt * substrate_slope
    feed = actuation_volume * (2.0 * growth + 1.8 * (2.35 - actuation_substrate)) / (45.0 - 2.35)

    # Limit carbon inventory as well as volume: after feeding 3.5 L,
    # complete conversion would give about 89 g of biomass in this model.
    feed_capacity = (7.5 - actuation_volume) / dt
    feed = max(0.0, min(2.5, feed, feed_capacity))

    # Cancel estimated heat production, then use temperature feedback
    # to handle plant mismatch and the unmeasured heat pulse.
    cooling_gain = 3.6 * 0.85 * (temp - 18.0) / (volume / 4.0)
    heat_load = 2.8 * growth + 0.15 * (29.0 - temp) + feed / volume * (28.0 - temp)
    coolant = max(0.0, min(1.0, (heat_load + 13.0 * (temp - 33.05)) / cooling_gain))
    if temp >= 33.85:
        coolant = 1.0

    return {'feed_rate_L_per_h': feed, 'coolant_flow_frac': coolant}, {'feed': feed}
