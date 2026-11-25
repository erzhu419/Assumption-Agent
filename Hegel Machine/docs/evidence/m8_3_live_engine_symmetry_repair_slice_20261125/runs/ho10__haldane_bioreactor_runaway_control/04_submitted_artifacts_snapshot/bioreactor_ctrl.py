"""Delay-aware feedback for the supplied fed-batch simulator."""

from math import exp


def compute_control_step(obs: dict, state_memory: dict) -> tuple[dict, dict]:
    dt = obs['dt_h']
    v = obs['V_L']
    x = obs['X_meas_g_L']
    s = obs['S_meas_g_L']
    temp = obs['T_meas_C']
    pending = state_memory.get('feed', 0.2)
    next_v = v + dt * pending

    # Predict substrate when the new command first takes effect.
    mu = 0.42 * exp(0.065 * (temp - 30.0)) * s / (0.8 + s + s*s/15.0)
    uptake = 2.0 * mu * x
    next_s = s + dt * (pending / v * (45.0 - s) - uptake)
    feed = next_v * (uptake + 1.8 * (2.9 - next_s)) / (45.0 - 2.9)
    # A finite feed budget leaves room under the volume limit and avoids
    # adding substrate that cannot be consumed before the end of the batch.
    feed = max(0.0, min(2.5, feed, (6.8 - next_v) / dt))

    heat = 2.8 * mu * x
    if 8.95 <= obs['t_h'] <= 12.5:
        # Reserve cooling for the scheduled heat-disturbance window.
        heat += 6.0
    cooling_gain = 14.4 * 0.75 * (temp - 18.0) / v
    cooling = max(0.0, min(1.0, (heat + 5.0 * (temp - 32.7)) / max(1.0, cooling_gain)))
    return {'feed_rate_L_per_h': feed, 'coolant_flow_frac': cooling}, {'feed': feed}
