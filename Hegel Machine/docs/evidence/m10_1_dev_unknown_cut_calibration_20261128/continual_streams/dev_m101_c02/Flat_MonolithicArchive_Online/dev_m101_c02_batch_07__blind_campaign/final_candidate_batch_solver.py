"""Calibrated panel-B transform and two independent response readouts."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _fit(study):
    weight = .5
    a, b = study['panel_a'], study['panel_b']
    ok = np.max(abs(a['u']-a['v']), axis=1) < 1
    u, v = a['u'][ok], a['v'][ok]
    du = u-u.mean(axis=0)
    gain = np.sum(du*(v-v.mean(axis=0)), axis=0)/np.sum(du**2, axis=0)
    offset = (v-u*gain).mean(axis=0)
    t = study['eval_times']
    data = study['train']
    x, y, traj = (data[k] for k in ('x', 'y', 'traj'))
    w = (x-offset)/gain

    def transform(x, p):
        k, power, linear = p[:3]
        z = x**3/(1+k*np.abs(x)**power)+linear*x
        for amplitude, frequency in p[3:].reshape(-1, 2):
            z += amplitude*np.sin(frequency*x)
        return _affine(z)

    def scalar(z):
        return np.column_stack((_affine(z), z[:, 0]*z[:, 2]+z[:, 1]*z[:, 3]))

    def link(d, p):
        h = d@p
        return np.column_stack((np.ones(len(h)), h + np.arctan(h)))

    def time_basis(p):
        return np.column_stack((np.log1p(p[0]*t), -np.expm1(-p[1]*t)))

    def coordinates(p, query):
        panel = transform(b['u'], p)
        operator = _ls(panel, b['v'])
        return transform(query, p)@operator, panel@operator-b['v']

    def profile(p, output=False):
        z, rp = coordinates(p[:-8], w)
        a = _affine(z)
        d = link(scalar(z), p[-8:-2])
        tb = time_basis(p[-2:])
        cy = _ls(d, y)
        ct = _ls(a, _ls(tb, traj.T).T)
        if output:
            return cy, ct, tb
        return np.r_[rp.ravel(), weight*(d@cy-y), weight*(a@ct@tb.T-traj).ravel()]

    # Initialize in the four-harmonic basin identified from panel B alone.
    start = np.array([7.59, 2.355, -.09492, .00312, 9.631, .000627,
                      13.545, -.00005, 21.5, .00966, 5.90])
    z, _ = coordinates(start, w)
    ini = _ls(scalar(z), y)
    n = (len(start)-3)//2
    lo = np.r_[[.01, .7, -2], [-.25, 2]*n, [-100]*6, [.01, .01]]
    hi = np.r_[[30, 4, 2], [.25, 34]*n, [100]*6, [10, 10]]
    fits = [least_squares(profile, np.r_[start, scale*ini, 1.4, .7],
                          bounds=(lo, hi), max_nfev=220)
            for scale in (.7, 1.5)]
    opt = min(fits, key=lambda fit: fit.cost)
    params = opt.x
    cy, ct, tb = profile(params, True)

    def predict(query):
        z, _ = coordinates(params[:-8], (query-offset)/gain)
        return link(scalar(z), params[-8:-2])@cy, _affine(z)@ct@tb.T

    return predict, params, cy


def solve_batch(study: dict) -> dict:
    predict, _, _ = _fit(study)
    y, state = predict(study['test']['x'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
