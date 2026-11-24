import math

def compute_control_step(obs: dict, state_memory: dict) -> tuple[dict, dict]:
    t = float(obs.get('t_h', 0.0))
    dt = float(obs.get('dt_h', 0.05))
    v = float(obs.get('V_L', 4.0))
    s = max(0.0, float(obs.get('S_meas_g_L', 2.2)))
    temp = float(obs.get('T_meas_C', 30.0))
    x = max(0.05, float(obs.get('X_meas_g_L', 1.5)))
    a = 0.4
    sf = (1-a)*float(state_memory.get('s_f', s)) + a*s
    tf = (1-a)*float(state_memory.get('t_f', temp)) + a*temp
    xf = (1-a)*float(state_memory.get('x_f', x)) + a*x
    mu = 0.42 * math.exp(0.065*(tf-30.0)) * sf / (0.8 + sf + sf*sf/15.0)
    feed = max(0.0, 0.60 * mu * xf / 0.50 + 0.10 * (2.0-sf))
    if sf > 2.4:
        feed *= max(0.0, (3.5-sf)/1.1)
    if sf > 3.2:
        feed = 0.0
    if tf > 32.0:
        feed *= max(0.0, min(1.0, (34.2-tf)/2.2))
    remaining = max(dt, 20.0-t)
    volume_limit = max(0.0, (11.85-v)/remaining)
    feed = min(2.5, feed, volume_limit)
    mem = dict(state_memory)
    mem.update(s_f=sf, t_f=tf, x_f=xf)
    return {'feed_rate_L_per_h': float(feed), 'coolant_flow_frac': 1.0}, mem
