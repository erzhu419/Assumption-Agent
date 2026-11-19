import math

_SCHEMA_FAMILY_TAG = 'monod_1d'
_CORE_LAW_EXPR = "(_PARAMS['p0'] * f['substrate_mM'] / (_PARAMS['p1'] + f['substrate_mM']))"
_PARAMS = {'p0': 1.0, 'p1': 1.0, 'p2': 0.0, 'p3': 0.0}

def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    xs = [float(r['features']['substrate_mM']) for r in observations]
    ys = [(float(r['target']) - (0)) / ((1) * (1)) for r in observations]
    def res(p):
        return [(p[0] * x / (p[1] + x)) - y for x, y in zip(xs, ys)]
    sol = least_squares(res, x0=[max(ys) * 1.3 + 0.5, 2.0], bounds=([1e-4, 1e-4], [500.0, 500.0]))
    _PARAMS['p0'], _PARAMS['p1'] = float(sol.x[0]), float(sol.x[1])

def predict(row):
    f = row['features']
    latent = (_PARAMS['p0'] * f['substrate_mM'] / (_PARAMS['p1'] + f['substrate_mM']))
    return 0 + (1) * (1) * latent
