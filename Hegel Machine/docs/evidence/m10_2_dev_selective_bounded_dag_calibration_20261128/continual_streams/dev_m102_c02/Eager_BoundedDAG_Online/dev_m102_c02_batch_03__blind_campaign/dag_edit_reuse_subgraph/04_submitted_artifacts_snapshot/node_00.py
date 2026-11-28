"""Calibrate raw observations by inverting the cyclic reference-panel map."""
import numpy as np


def _fit_panel(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    n, p = u.shape
    neighbor = np.roll(u, -1, axis=1)
    # Per-channel bias and gain, with a common cubic and cyclic coupling.
    d = np.zeros((n, p, 2*p+2))
    j = np.arange(p)
    d[:, j, j] = 1.
    d[:, j, p+j] = u
    d[:, :, -2] = u**3
    d[:, :, -1] = (u-1.)*neighbor
    center = np.median(v, axis=0)
    scale = np.maximum(1.4826*np.median(np.abs(v-center), axis=0), 0.25)
    keep = np.max(np.abs(v-center)/scale, axis=1) <= 6.
    if keep.sum() < n//2:
        keep = np.ones(n, dtype=bool)
    for _ in range(5):
        beta = np.linalg.lstsq(d[keep].reshape(-1, 2*p+2), v[keep].ravel(), rcond=None)[0]
        err = np.max(np.abs(v-d@beta), axis=1)
        updated = err <= max(0.002, 4.*np.median(err))
        if updated.sum() < n//2 or np.array_equal(updated, keep):
            break
        keep = updated
    beta = np.linalg.lstsq(d[keep].reshape(-1, 2*p+2), v[keep].ravel(), rcond=None)[0]
    return beta[:p], beta[p:2*p], beta[-2], beta[-1]


def _inverse(x, params):
    bias, gain, cubic, coupling = params
    p = len(gain)
    j = np.arange(p)
    y = x-bias
    linear = np.diag(gain)
    linear[j, (j+1)%p] -= coupling
    z = np.linalg.solve(linear, y.T).T

    def residual(a):
        return gain*a + cubic*a**3 + coupling*(a-1.)*np.roll(a, -1, axis=1)-y

    r = residual(z)
    for _ in range(32):
        if np.max(np.abs(r), initial=0.) < 1e-11:
            break
        jac = np.zeros((len(z), p, p))
        jac[:, j, j] = gain + 3.*cubic*z*z + coupling*np.roll(z, -1, axis=1)
        jac[:, j, (j+1)%p] += coupling*(z-1.)
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        old = np.sum(r*r, axis=1)
        trial = z-step
        rt = residual(trial)
        for _ in range(16):
            bad = np.sum(rt*rt, axis=1) > old
            if not np.any(bad):
                break
            step[bad] *= 0.5
            trial = z-step
            rt = residual(trial)
        ok = np.sum(rt*rt, axis=1) <= old
        z, r = np.where(ok[:, None], trial, z), np.where(ok[:, None], rt, r)
    return z


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    params = _fit_panel(study['panel_a'])
    panel = study['panel_b']
    mix = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    c = _affine(_inverse(np.asarray(study['train']['x'], dtype=float), params)) @ mix
    query = _affine(_inverse(np.asarray(study['test']['x'], dtype=float), params)) @ mix
    return {'c': c, 'query': query}
