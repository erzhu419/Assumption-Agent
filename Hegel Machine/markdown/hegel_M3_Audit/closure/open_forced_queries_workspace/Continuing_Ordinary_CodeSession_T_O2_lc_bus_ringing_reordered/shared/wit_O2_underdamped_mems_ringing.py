import math

_SCHEMA_FAMILY_TAG = 'underdamped_ringing_2d'
_CORE_LAW_EXPR = "(_PARAMS['p0'] * f['step_force'] * (1.0 - math.exp(-f['delay_s'] / _PARAMS['p1']) * math.cos(_PARAMS['p2'] * f['delay_s'])))"
_PARAMS = {'p0': 1.0, 'p1': 1.0, 'p2': 0.0, 'p3': 0.0}

def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    us = [float(r['features']['step_force']) for r in observations]
    ts = [float(r['features']['delay_s']) for r in observations]
    ys = [(float(r['target']) - (0)) / ((1) * (1)) for r in observations]
    best_sol = None
    for w_init in (0.9, 1.4, 2.0, 2.6):
        sol = least_squares(
            lambda p: [(p[0] * u * (1.0 - math.exp(-t / p[1]) * math.cos(p[2] * t))) - y for u, t, y in zip(us, ts, ys)],
            x0=[1.5, 1.8, w_init],
            bounds=([1e-4, 0.1, 0.0], [200.0, 50.0, 10.0]),
        )
        if best_sol is None or sol.cost < best_sol.cost:
            best_sol = sol
    _PARAMS['p0'], _PARAMS['p1'], _PARAMS['p2'] = [float(v) for v in best_sol.x]

def predict(row):
    f = row['features']
    latent = (_PARAMS['p0'] * f['step_force'] * (1.0 - math.exp(-f['delay_s'] / _PARAMS['p1']) * math.cos(_PARAMS['p2'] * f['delay_s'])))
    return 0 + (1) * (1) * latent
