def compute_control_step(obs: dict, state_memory: dict):
    t = float(obs.get('t_h', 0.0))
    S = float(obs['S_meas_g_L'])
    T = float(obs['T_meas_C'])
    X = float(obs['X_meas_g_L'])
    V = float(obs['V_L'])
    dt = float(obs.get('dt_h', 0.05))
    # Keep substrate below the Haldane optimum; use modest feed so inhibited
    # substrate cannot accumulate during the heat pulse.
    target_s = 2.4
    feed = max(0.0, min(1.15, 0.55 * (target_s - S)))
    if S > 2.15:
        feed = 0.0
    # Productive growth is favored by moderate temperature, but thermal safety
    # dominates; full cooling is anticipatory because the actuator has one-step lag.
    coolant = max(0.0, min(1.0, 0.70 + 0.55 * (T - 29.7)))
    if T >= 31.0:
        coolant = 1.0
    # Volume limit with actuator delay and conservative headroom.
    if V >= 11.35 or t >= 19.0:
        feed = 0.0
    elif V >= 10.8:
        feed = min(feed, 0.15)
    return {'feed_rate_L_per_h': float(feed), 'coolant_flow_frac': float(coolant)}, state_memory
