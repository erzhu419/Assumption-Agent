import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit_panel(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    n, d = u.shape
    adj = np.roll(u, -1, axis=1)
    eye = np.broadcast_to(np.eye(d), (n, d, d))
    design = np.concatenate((eye, eye * u[..., None], adj[..., None],
                             (u * adj)[..., None], (u ** 3)[..., None]), axis=2)

    def fit(mask):
        return np.linalg.lstsq(design[mask].reshape(-1, 2*d+3), v[mask].ravel(), rcond=None)[0]

    # Gross contamination affects entire rows. Start with a trimmed fit,
    # then recover all rows consistent with the fitted calibration.
    norm = np.linalg.norm(v, axis=1)
    keep = norm <= max(4 * np.median(norm), 1e-3)
    q = fit(keep)
    for _ in range(4):
        err = np.max(abs(design @ q - v), axis=1)
        keep = err <= max(8 * np.median(err), 0.002)
        q = fit(keep)
    return q[:d], q[d:2*d], *q[2*d:]


def _invert(x, params):
    offset, scale, neighbor, pair, cubic = params
    d = x.shape[1]
    j = np.arange(d)
    nxt = (j + 1) % d
    rhs = x - offset
    lin = np.diag(scale)
    lin[j, nxt] += neighbor
    u = np.linalg.solve(lin, rhs.T).T

    def residual(z):
        return z * scale + (neighbor + pair * z) * z[:, nxt] + cubic * z**3 - rhs

    for _ in range(32):
        r = residual(u)
        err = np.max(abs(r), axis=1)
        if np.max(err, initial=0.0) < 1e-12:
            break
        jac = np.zeros((len(u), d, d))
        jac[:, j, j] = scale + pair * u[:, nxt] + 3*cubic*u*u
        jac[:, j, nxt] += neighbor + pair * u
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        rates = np.ones(len(u))
        for _ in range(18):
            trial = u - rates[:, None] * step
            bad = (np.max(abs(residual(trial)), axis=1) >= err) & (err > 1e-12)
            if not bad.any():
                break
            rates[bad] *= 0.5
        u = np.where((~bad & (err > 1e-12))[:, None], trial, u)
    return u


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    u = _invert(x, _fit_panel(study['panel_a']))
    panel = study['panel_b']
    mapping = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    return {'w': _affine(u) @ mapping}
