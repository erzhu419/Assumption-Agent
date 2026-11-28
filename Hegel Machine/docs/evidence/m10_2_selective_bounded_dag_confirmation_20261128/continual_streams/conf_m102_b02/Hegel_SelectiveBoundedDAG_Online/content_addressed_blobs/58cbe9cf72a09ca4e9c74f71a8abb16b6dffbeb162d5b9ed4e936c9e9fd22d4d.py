"""Fit panel calibrations and emit coordinates for the frozen bilinear basis."""
import numpy as np
from scipy.optimize import least_squares


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _invert(panel, measured):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    keep = np.max(np.abs(v-u), axis=1) < 3
    u, v = u[keep], v[keep]
    n, width = u.shape
    nxt = np.roll(u, -1, axis=1)
    design = np.zeros((n, width, 2*width+2))
    for j in range(width):
        design[:, j, j] = 1
        design[:, j, width+j] = u[:, j]
    design[:, :, -2] = (u-1)*nxt
    design[:, :, -1] = u**3
    p = _fit(design.reshape(n*width, -1), v.ravel())
    offset, gain, cross, cubic = p[:width], p[width:2*width], p[-2], p[-1]

    def forward(x):
        return offset + gain*x + cross*(x-1)*np.roll(x, -1, axis=-1) + cubic*x**3

    x = (measured-offset)/gain
    j = np.arange(width)
    for _ in range(25):
        res = forward(x)-measured
        if np.max(np.abs(res), initial=0) < 1e-12:
            break
        jac = np.zeros((len(x), width, width))
        jac[:, j, j] = gain + cross*np.roll(x, -1, axis=1) + 3*cubic*x*x
        jac[:, j, (j+1)%width] = cross*(x-1)
        step = np.linalg.solve(jac, res[..., None])[..., 0]
        old = np.sum(res**2, axis=1)
        trial = x-step
        for _ in range(20):
            bad = np.sum((forward(trial)-measured)**2, axis=1) > old
            if not np.any(bad):
                break
            trial[bad] = (trial[bad]+x[bad])/2
        x = trial
    return x


def _panel_b(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def design(q, p):
        k, a, b, c = p
        z = np.tanh(np.exp(k)*q+a*q**3)+b*q**3+c*q
        return np.column_stack((np.ones(len(q)), z))
    def residual(p):
        d = design(u, p)
        return (d@_fit(d, v)-v).ravel()
    p = least_squares(residual, [.71, 6.28, -.61, -2.11], max_nfev=180).x
    return design(x, p)@_fit(design(u, p), v)


def _panel_c(panel, x):
    u = np.asarray(panel['u'], dtype=float)
    def design(q):
        return np.column_stack((np.ones(len(q)), q*np.roll(q, -1, axis=1)))
    return design(x)@_fit(design(u), panel['v1'])


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train']['x'], study['test']['x']
    x = _invert(study['panel_a'], np.vstack((train, test)))
    c = _panel_c(study['panel_c'], _panel_b(study['panel_b'], x))
    n = len(train)
    return dict(x_train=x[:n], x_test=x[n:], c_train=c[:n], c_test=c[n:])
