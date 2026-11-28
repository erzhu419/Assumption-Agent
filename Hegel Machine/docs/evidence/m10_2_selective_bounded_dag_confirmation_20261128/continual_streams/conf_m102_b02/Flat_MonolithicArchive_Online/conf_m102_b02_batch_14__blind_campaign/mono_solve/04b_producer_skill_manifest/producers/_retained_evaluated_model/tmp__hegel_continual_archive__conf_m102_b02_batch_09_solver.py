"""Invert the reference calibration and predict with the measured B→C1 path."""
import numpy as np
from scipy.optimize import root


def _intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _latent(panel, x):
    u, v = panel['u'], panel['v']

    def correction(w):
        return .18 * np.tanh(1.8 * w) - .32 * np.sin(2.4 * np.roll(w, 2, axis=-1))

    target = v - correction(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        offset, scale = np.array([_ls(_intercept(u[keep, j]), target[keep, j])
                                  for j in range(u.shape[1])]).T
        err = np.linalg.norm(u * scale + offset - target, axis=1)
        new = err < max(.001, 4 * np.median(err))
        if np.array_equal(new, keep) or new.sum() < len(u) // 2:
            break
        keep = new
    return np.array([root(lambda w: scale * w + offset + correction(w) - r,
                          (r - offset) / scale).x for r in x])


def _features(study, x):
    z = _latent(study['panel_a'], x)
    pb, pc = study['panel_b'], study['panel_c']
    w = _intercept(z) @ _ls(_intercept(pb['u']), pb['v'])

    def design(a):
        return _intercept(np.column_stack((a, np.arcsinh(2.65 * a),
                                          a / (1 + (5.47 * a)**2))))

    h = design(w) @ _ls(design(pc['u']), pc['v1'])
    return z, h, (z[:, :, None] * h[:, None, :]).reshape(len(z), -1)


def _ridge(f, response, n, alpha):
    mean = f[:n].mean(axis=0)
    scale = np.maximum(f[:n].std(axis=0), 1e-10)
    q = (f - mean) / scale
    base = response.mean(axis=0)
    a = q[:n]
    coef = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (response - base))
    return q[n:] @ coef + base


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    test = study['test']['x']
    z, h, cross = _features(study, np.vstack((train['x'], test)))
    trace = cross[:, [0, 5, 10, 15]].sum(axis=1)
    y = _ridge(np.column_stack((z, h, trace)), train['y'], n, .003)
    # The trajectory shares the trace term and adds the z[1] * h[0] coupling.
    state_design = _intercept(np.column_stack((z, h, trace, cross[:, 4])))
    state = state_design[n:] @ _ls(state_design[:n], train['traj'])
    end = state[:, -1]
    return dict(y_hat=y, state_pred=state, diagnostics_vec=np.array([y.mean(), y.std(), end.mean(), end.std()]))
