import numpy as np


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _terms(u, j):
    a, b = u[:, j], u[:, (j + 1) % u.shape[1]]
    return np.column_stack((np.ones(len(a)), a, b, a*b, a**3))


def _fit_panel_a(u, v):
    # Whole-row weights prevent a corrupted panel observation from leaking
    # into the fit of any channel.
    designs = [_terms(u, j) for j in range(u.shape[1])]
    center = np.median(v, axis=0)
    spread = np.maximum(np.median(np.abs(v - center), axis=0), 1e-8)
    score = np.max(np.abs(v - center) / spread, axis=1)
    keep = score <= max(6., 2.5*np.median(score))
    if keep.sum() < len(u)//2:
        keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        coef = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]
                         for j, d in enumerate(designs)])
        err = np.linalg.norm(np.column_stack([d @ c for d, c in zip(designs, coef)]) - v, axis=1)
        new = err <= max(0.001, 4*np.median(err))
        if np.array_equal(keep, new) or new.sum() < len(u)//2:
            break
        keep = new
    return coef


def _inverse(x, c):
    p = x.shape[1]
    j = np.arange(p)
    k = (j + 1) % p
    a = np.diag(c[:, 1]); a[j, k] += c[:, 2]
    u = np.linalg.solve(a, (x - c[:, 0]).T).T

    def residual(z):
        return c[:, 0] + c[:, 1]*z + c[:, 2]*z[:, k] + c[:, 3]*z*z[:, k] + c[:, 4]*z**3 - x

    for _ in range(40):
        r = residual(u)
        if np.max(np.abs(r), initial=0.0) < 1e-11:
            break
        jac = np.zeros((len(u), p, p))
        jac[:, j, j] = c[:, 1] + c[:, 3]*u[:, k] + 3*c[:, 4]*u**2
        jac[:, j, k] += c[:, 2] + c[:, 3]*u
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        baseline = np.sum(r*r, axis=1)
        trial = u - step
        for _ in range(15):
            bad = np.sum(residual(trial)**2, axis=1) > baseline
            if not np.any(bad):
                break
            step[bad] *= 0.5
            trial[bad] = u[bad] - step[bad]
        bad = np.sum(residual(trial)**2, axis=1) > baseline
        trial[bad] = u[bad]
        u = trial
    return u


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    a, b = s['panel_a'], s['panel_b']
    c = _fit_panel_a(a['u'], a['v'])
    x = np.concatenate((s['train']['x'], s['test']['x']), axis=0)
    bmap = np.linalg.lstsq(_aff(b['u']), b['v'], rcond=None)[0]
    return {'w': _aff(_inverse(x, c)) @ bmap}
