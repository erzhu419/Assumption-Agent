import math

_SCHEMA_FAMILY_TAG = 'haldane_inhibited_plus_bypass_2d'
_CORE_LAW_EXPR = "((_PARAMS['p0'] * f['substrate_mM'] / (_PARAMS['p1'] + f['substrate_mM'] + _PARAMS['p2'] * (f['substrate_mM'] ** 2))) + _PARAMS['p3'] * f['shunt_flow'])"
_PARAMS = {'p0': 1.0, 'p1': 1.0, 'p2': 0.0, 'p3': 0.0}

def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    x0 = [float(r['features']['substrate_mM']) for r in observations]
    x1 = [float(r['features']['shunt_flow']) for r in observations]
    ys = [(float(r['target']) - (0)) / ((1) * (1)) for r in observations]
    def res(p):
        return [((p[0] * s / (p[1] + s + p[2] * (s ** 2))) + p[3] * b) - y for s, b, y in zip(x0, x1, ys)]
    sol = least_squares(res, x0=[8.0, 2.5, 0.06, 0.5], bounds=([1e-4, 1e-4, 0.0, -50.0], [500.0, 500.0, 50.0, 50.0]))
    _PARAMS['p0'], _PARAMS['p1'], _PARAMS['p2'], _PARAMS['p3'] = [float(v) for v in sol.x]

def predict(row):
    f = row['features']
    latent = ((_PARAMS['p0'] * f['substrate_mM'] / (_PARAMS['p1'] + f['substrate_mM'] + _PARAMS['p2'] * (f['substrate_mM'] ** 2))) + _PARAMS['p3'] * f['shunt_flow'])
    return 0 + (1) * (1) * latent
