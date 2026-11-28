"""Invert the coupled calibration and apply the calibrated panel-B map."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _ls(design, target):
    return np.linalg.lstsq(design, target, rcond=1e-10)[0]


def _frequency(loss, lower, upper):
    grid = np.linspace(lower, upper, 51)
    i = int(np.argmin([loss(k) for k in grid]))
    return minimize_scalar(loss, method='bounded', bounds=(
        grid[max(0, i-1)], grid[min(len(grid)-1, i+1)])).x


def _a_design(u, k, j):
    p, q = u[:, j], u[:, (j+2) % 4]
    return np.column_stack((np.ones(len(u)), p, q, np.sin(k*p), np.sin(k*q)))


def _inverse(panel, points):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    pilot = np.column_stack((np.ones(len(u)), u, u**3, u*np.roll(u, -1, axis=1)))
    keep = np.arange(len(u))
    for _ in range(4):
        err = np.linalg.norm(v-pilot@_ls(pilot[keep], v[keep]), axis=1)
        med = np.median(err)
        mad = np.median(abs(err-med))
        selected = np.flatnonzero(err <= med+max(12*mad, 0.15))
        if len(selected) < pilot.shape[1]+2 or np.array_equal(selected, keep):
            break
        keep = selected
    u, v = u[keep], v[keep]
    def fit(k):
        designs = [_a_design(u, k, j) for j in range(4)]
        coeff = np.array([_ls(d, v[:, j]) for j,d in enumerate(designs)])
        error = np.column_stack([d@coeff[j] for j,d in enumerate(designs)])-v
        return np.mean(error**2), coeff
    k = _frequency(lambda k: fit(k)[0], 1.7, 3.5)
    coeff = fit(k)[1]
    def forward(z):
        q = np.roll(z, -2)
        return (coeff[:, 0] + coeff[:, 1]*z + coeff[:, 2]*q
                + coeff[:, 3]*np.sin(k*z) + coeff[:, 4]*np.sin(k*q))
    coords = []
    for point in points:
        start = u[np.argmin(np.sum((v-point)**2, axis=1))]
        fun = lambda z: forward(z)-point
        trial = root(fun, start)
        if np.linalg.norm(trial.fun) > 1e-7 or np.max(abs(trial.x)) > 1.4:
            trial = least_squares(fun, np.clip(start, -1.39, 1.39), bounds=(-1.4, 1.4),
                                  ftol=1e-11, xtol=1e-11, gtol=1e-11)
        coords.append(trial.x)
    return np.asarray(coords).reshape(-1, 4)


def _panel_b(panel, z):
    u, v = (np.asarray(panel[key], dtype=float) for key in ('u', 'v'))
    def design(x, k):
        return np.column_stack((np.ones(len(x)), x, np.sin(k*x)))
    def loo(k):
        d = design(u, k)
        q, _ = np.linalg.qr(d, mode='reduced')
        return np.mean(abs((v-d@_ls(d, v))/np.maximum(1e-8, 1-np.sum(q*q, axis=1))[:, None]))
    k = _frequency(loo, 1.9, 3.8)
    return design(z, k)@_ls(design(u, k), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.asarray(study['train']['x'], dtype=float)
    query = np.asarray(study['test']['x'], dtype=float)
    z = _inverse(study['panel_a'], np.concatenate((x, query)))
    c = _panel_b(study['panel_b'], z)
    return {'c': c[:len(x)], 'query': c[len(x):]}
