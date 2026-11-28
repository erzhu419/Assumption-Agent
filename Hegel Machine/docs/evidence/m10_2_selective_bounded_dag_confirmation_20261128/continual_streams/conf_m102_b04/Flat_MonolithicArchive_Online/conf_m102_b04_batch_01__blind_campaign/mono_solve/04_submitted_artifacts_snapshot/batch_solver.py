"""Panel-calibrated kernel regression for the scalar and trajectory readouts."""
import numpy as np


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _least(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _coordinates(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    # Initialize a diagonal calibration with high-breakdown pair slopes.
    i, j = np.triu_indices(len(u), 1)
    du, dv = u[i] - u[j], v[i] - v[j]
    gain = np.array([np.median(dv[abs(du[:, k]) > .1, k] / du[abs(du[:, k]) > .1, k]) for k in range(u.shape[1])])
    offset = np.median(v - u * gain, axis=0)
    residual = np.linalg.norm(v - offset - u * gain, axis=1)
    median = np.median(residual)
    keep = residual <= max(8 * median, 1e-3)
    if keep.sum() >= 2 * u.shape[1]:
        coef = [_least(_affine(u[keep, k:k+1]), v[keep, k]) for k in range(u.shape[1])]
        offset, gain = np.array(coef).T
    cal = (x - offset) / gain
    b = study['panel_b']
    op = _least(_affine(b['u']), b['v'])
    return x, cal, _affine(x) @ op, _affine(cal) @ op


def _panel_features(study, coords):
    c = study['panel_c']
    def design(w):
        return np.column_stack((_affine(w), np.expm1(-.29 * w * w)))
    coef = _least(design(c['u']), c['v2'])
    return tuple(np.column_stack((p, p*p)) for p in (design(w) @ coef for w in coords))


def _sqdist(z):
    return np.sum((z[:, None] - z[None, :])**2, axis=2)


def _predict(kernel, target, n, penalty):
    avg = target.mean(axis=0)
    k = kernel[:n, :n].copy()
    k.flat[::n+1] += penalty
    return avg + kernel[n:, :n] @ np.linalg.solve(k, target - avg)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    coords = _coordinates(study, x)
    c = _panel_features(study, coords)
    r = [np.exp(-_sqdist(z) / 4) for z in coords[2:]]
    g = [f @ f.T for f in c]
    y = .30 * _predict(r[0] + .25*g[0], train['y'], n, .0003)
    y += .35 * _predict(r[0] + .25*g[2], train['y'], n, .0003)
    y += .35 * _predict(r[1] + .25*g[3], train['y'], n, .0003)
    traj = np.asarray(train['traj'])
    st = .30 * _predict(r[0] + 2*g[2], traj, n, .0003)
    st += .70 * _predict(r[1] + 2*g[3], traj, n, .0003)
    # Blend smooth readouts in raw and panel-B coordinates for the scalar.
    dx, db = _sqdist(coords[0]), _sqdist(coords[2])
    gp = .3*g[0] + .7*g[2]
    broad = _predict(np.exp(-.2*db) + .15*gp, train['y'], n, .001)
    mixed = _predict(np.exp(-.06*(dx + db)) + .15*gp, train['y'], n, .0003)
    y = .55*y + .25*broad + .20*mixed

    # Learn the common time profile within the training fold; shrink the
    # smaller time-dependent deviations toward a calibrated ridge fit.
    mean = traj.mean(axis=0)
    centered = traj - mean
    q = np.linalg.svd(centered, full_matrices=False)[2][0]
    amplitude = _predict(np.exp(-.2*(.2*dx + .8*db)) + 1.2*gp,
                         centered @ q, n, .00003)
    residual = centered - np.outer(centered @ q, q)
    u = np.column_stack((_affine(coords[1]), c[3][:, :x.shape[1]]))
    ui = u[:n]
    coef = np.linalg.solve(ui.T @ ui + .4*np.eye(ui.shape[1]), ui.T @ residual)
    st = .55*st + .45*(mean + np.outer(amplitude, q) + u[n:] @ coef)
    return {'y_hat': y, 'state_pred': st, 'diagnostics_vec': np.array([y.mean(), y.std(), st[:, -1].mean(), st[:, -1].std()])}
