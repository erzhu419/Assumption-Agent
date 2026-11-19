import math

_SCHEMA_FAMILY_TAG = 'overdamped_step_2d'
_CORE_LAW_EXPR = "(_PARAMS['p0'] * f['photon_flux'] * (1.0 - math.exp(-f['bleed_valve'] / _PARAMS['p1'])))"
_PARAMS = {'p0': 1.0, 'p1': 1.0, 'p2': 0.0, 'p3': 0.0}

def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    us = [float(r['features']['photon_flux']) for r in observations]
    ts = [float(r['features']['bleed_valve']) for r in observations]
    ys = [(float(r['target']) - (4)) / ((-1) * (1.4)) for r in observations]
    def res(p):
        return [(p[0] * u * (1.0 - math.exp(-t / p[1]))) - y for u, t, y in zip(us, ts, ys)]
    sol = least_squares(res, x0=[1.5, 1.2], bounds=([1e-4, 0.05], [200.0, 50.0]))
    _PARAMS['p0'], _PARAMS['p1'] = float(sol.x[0]), float(sol.x[1])

def predict(row):
    f = row['features']
    latent = (_PARAMS['p0'] * f['photon_flux'] * (1.0 - math.exp(-f['bleed_valve'] / _PARAMS['p1'])))
    return 4 + (-1) * (1.4) * latent
