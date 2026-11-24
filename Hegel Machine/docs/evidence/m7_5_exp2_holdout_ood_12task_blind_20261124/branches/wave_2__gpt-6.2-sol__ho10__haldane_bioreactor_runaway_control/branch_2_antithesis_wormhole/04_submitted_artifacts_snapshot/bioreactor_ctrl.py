"""Delayed feed and temperature feedback for the supplied fed-batch model."""

from math import exp


def _clip(value, low, high):
    return min(high, max(low, value))


def compute_control_step(obs: dict, state_memory: dict) -> tuple[dict, dict]:
    dt = float(obs['dt_h'])
    volume = float(obs['V_L'])
    biomass = float(obs['X_meas_g_L'])
    substrate = float(obs['S_meas_g_L'])
    temp = float(obs['T_meas_C'])
    old_feed = state_memory.get('feed', 0.2)
    old_cool = state_memory.get('cool', 0.3)
    integral = state_memory.get('integral', 0.0)

    # The old command acts over the next interval, before the new one takes effect.
    mu = 0.42 * exp(0.065 * (temp - 30.0)) * substrate / (0.8 + substrate + substrate**2 / 15.0)
    uptake = 2.0 * mu * biomass
    heat = 2.8 * mu * biomass
    next_volume = volume + dt * old_feed
    substrate += dt * (old_feed / volume * (45.0 - substrate) - uptake)
    f = next_volume * (uptake + 1.8 * (2.5 - substrate)) / (45.0 - substrate)
    # Bound the total substrate inventory through the volume budget, which also
    # leaves cooling headroom in the high-biomass phase.
    feed = _clip(f, 0.0, min(2.5, max(0.0, (8.7 - next_volume) / dt)))

    cooling_gain = 3.6 * 0.85 * (temp - 18.0) / (volume / 4.0)
    drift = heat + 0.15 * (29.0 - temp) + old_feed / volume * (28.0 - temp)
    target = 30.5
    predicted_temp = temp + dt * (drift + integral - cooling_gain * old_cool)
    error = predicted_temp - target
    demand = (drift + integral + 4.0 * error) / cooling_gain
    if (demand < 1.0 or error < 0.0) and (demand > 0.0 or error > 0.0):
        integral = _clip(integral + 4.0 * dt * error, -6.0, 6.0)
        demand = (drift + integral + 4.0 * error) / cooling_gain
    coolant = _clip(demand, 0.0, 1.0)
    return {'feed_rate_L_per_h': float(feed), 'coolant_flow_frac': float(coolant)}, {
        'feed': feed, 'cool': coolant, 'integral': integral,
    }
