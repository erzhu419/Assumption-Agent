import math

_SCHEMA_FAMILY_TAG = 'monod_1d'
_CORE_LAW_EXPR = "(_PARAMS['p0'] * f['v_step'] / (_PARAMS['p1'] + f['v_step']))"
_PARAMS = {'p0': 1.0, 'p1': 1.0, 'p2': 0.0, 'p3': 0.0}

def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    xs = [float(r['features']['v_step']) for r in observations]
    ys = [(float(r['target']) - (2.5)) / ((-1) * (1.2)) for r in observations]
    def res(p):
        return [(p[0] * x / (p[1] + x)) - y for x, y in zip(xs, ys)]
    sol = least_squares(res, x0=[max(ys) * 1.3 + 0.5, 2.0], bounds=([1e-4, 1e-4], [500.0, 500.0]))
    _PARAMS['p0'], _PARAMS['p1'] = float(sol.x[0]), float(sol.x[1])

def predict(row):
    f = row['features']
    latent = (_PARAMS['p0'] * f['v_step'] / (_PARAMS['p1'] + f['v_step']))
    return 2.5 + (-1) * (1.2) * latent
