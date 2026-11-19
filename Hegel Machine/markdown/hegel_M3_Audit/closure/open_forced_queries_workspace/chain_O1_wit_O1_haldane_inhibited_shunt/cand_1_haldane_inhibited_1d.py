import math

_SCHEMA_FAMILY_TAG = 'haldane_inhibited_1d'
_CORE_LAW_EXPR = "(_PARAMS['p0'] * f['substrate_mM'] / (_PARAMS['p1'] + f['substrate_mM'] + _PARAMS['p2'] * (f['substrate_mM'] ** 2)))"
_PARAMS = {'p0': 1.0, 'p1': 1.0, 'p2': 0.0, 'p3': 0.0}

def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    xs = [float(r['features']['substrate_mM']) for r in observations]
    ys = [(float(r['target']) - (0)) / ((1) * (1)) for r in observations]
    def res(p):
        return [(p[0] * x / (p[1] + x + p[2] * (x ** 2))) - y for x, y in zip(xs, ys)]
    sol = least_squares(res, x0=[max(ys) * 2.0 + 1.0, 2.5, 0.05], bounds=([1e-4, 1e-4, 0.0], [500.0, 500.0, 50.0]))
    _PARAMS['p0'], _PARAMS['p1'], _PARAMS['p2'] = float(sol.x[0]), float(sol.x[1]), float(sol.x[2])

def predict(row):
    f = row['features']
    latent = (_PARAMS['p0'] * f['substrate_mM'] / (_PARAMS['p1'] + f['substrate_mM'] + _PARAMS['p2'] * (f['substrate_mM'] ** 2)))
    return 0 + (1) * (1) * latent
