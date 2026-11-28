import numpy as np
from scipy.optimize import least_squares


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-9)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _cross(x):
    return (x[:, [0, 0, 1, 2]] * x[:, [1, 3, 2, 3]])


def _profile(fun, x, y, start, bounds):
    def residual(p):
        d = fun(x, p)
        return (d @ _ls(d, y) - y).ravel()
    p = least_squares(residual, start, bounds=bounds, max_nfev=100).x
    return p, _ls(fun(x, p), y)


def _calibrate(study):
    a, b, c = (study['panel_' + k] for k in 'abc')
    ad = _aff(a['u'])
    aw = np.vstack((np.zeros(4), np.eye(4)))
    for _ in range(10):
        r = ad @ aw - a['v']
        w = 1 / np.sqrt(1 + np.sum(r*r, axis=1) / .02**2)
        aw = _ls(ad*w[:, None], a['v']*w[:, None])
    def odd(z, p):
        return z*(1-p[1]*np.exp(-p[0]*z*z))
    def bd(z, p):
        return _aff(np.c_[odd(z, p), z*z])
    def cd(z, p):
        return _aff(np.c_[z, np.sqrt(1+p[0]*z*z)])
    bp, bw = _profile(bd, b['u'], b['v'], [1., 1.1], ([.1, .2], [5., 2.]))
    cp, cw = _profile(cd, c['u'], c['v2'], [.9], ([.01], [4.]))
    pw = _ls(_aff(_cross(c['u'])), c['v1'])
    return aw, lambda z: bd(z, bp) @ bw, lambda z: _aff(_cross(z)) @ pw, lambda z: cd(z, cp) @ cw, lambda z: odd(z, bp)


def _readout_gram(ref, query, config):
    mode, length, interaction, fitted = config
    z, g = ref
    v, h = query
    nz = len(z)
    a = np.r_[z, v]
    b = np.r_[g, h]
    # Fit normalization on the training features unless using the batch-design variant.
    za = a[:nz] if fitted else a
    zb = b[:nz] if fitted else b
    a = a / np.maximum(za.std(0), 1e-8)
    b = (b-zb.mean(0)) / np.maximum(zb.std(0), 1e-8)
    lin = a @ a[:nz].T / 4
    joint = b @ b[:nz].T / b.shape[1]
    parts = []
    for j in range(4):
        w = b[:, j::4]
        dot = w @ w[:nz].T / w.shape[1]
        if mode == 'poly':
            parts.append((1+dot/length)**3-1)
        else:
            norm = np.mean(w*w, axis=1)
            r2 = np.maximum(norm[:, None]+norm[None, :nz]-2*dot, 0) / length**2
            if mode == 'rbf':
                parts.append(np.exp(-r2/2))
            else:
                r = np.sqrt(r2)
                parts.append((1+r)*np.exp(-r))
    kernel = 1 + lin + sum(parts)/4
    if interaction == 'mixed':
        kernel += (.1*lin + .02*lin**2)*joint
    else:
        from itertools import combinations
        kernel += sum(parts[i]*parts[j] for i,j in combinations(range(4),2)) * (.05/6)
        kernel += sum(parts[i]*parts[j]*parts[k] for i,j,k in combinations(range(4),3)) * (.02/4)
    return kernel[:nz], kernel[nz:]


def _readout_features(x, group, aw, odd):
    coord, lift = group
    if coord == 1:
        x = np.linalg.solve(aw[1:].T, (x-aw[0]).T).T
    elif coord == 2:
        x = x @ aw[1:] + aw[0]
    f = x**3 if lift == 'cube' else odd(x)
    return x, np.c_[x, f]


def _predict_readout(ref, query, y, cfg):
    _, mode, length, interaction, fitted, reg = cfg
    k, q = _readout_gram(ref, query, (mode, length, interaction, fitted))
    return q @ np.linalg.solve(k + reg*np.eye(len(k)), y)


def _predict(study, specs):
    aw, _, _, _, odd = _calibrate(study)
    tr = study['train']
    x, q = tr['x'], study['test']['x']
    results = []
    for choices, target in zip(specs, (tr['y'], tr['traj'])):
        pred = []
        for cfg in choices:
            ref = _readout_features(x, cfg[0], aw, odd)
            query = _readout_features(q, cfg[0], aw, odd)
            pred.append(_predict_readout(ref, query, target, cfg))
        results.append(np.mean(pred, axis=0))
    return results


_READOUTS = [[((1, 'cube'), 'matern', 8.0, 'mixed', False, 1e-05), ((1, 'cube'), 'matern', 12.0, 'mixed', False, 0.0003)], [((0, 'odd'), 'matern', 6.0, 'mixed', False, 0.0001), ((1, 'odd'), 'poly', 12.0, 'limited', True, 0.0001)]]


def run_node(inputs: dict) -> dict:
    study = inputs.get('__input__', inputs)
    y, state = _predict(study, _READOUTS)
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
