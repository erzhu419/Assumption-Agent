"""Invert the calibrated sensor map and evaluate the cosine panel operator."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _profile(loss, lo, hi):
    grid = np.linspace(lo, hi, 81)
    i = int(np.argmin([loss(a) for a in grid]))
    return minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),
                           method='bounded', options={'xatol': 1e-9}).x


def _sensor_model(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    d = _affine(u)
    err = np.linalg.norm(v - d @ _fit(d, v), axis=1)
    med = np.median(err)
    keep = err <= med + 6 * max(np.median(np.abs(err-med)), med/4, 1e-9)
    u, v = u[keep], v[keep]
    w = np.roll(u, 2, axis=1)

    def regress(freq):
        coeff, loss = [], 0.
        for j in range(u.shape[1]):
            x, z = u[:, j], w[:, j]
            d = _affine(np.column_stack((x, x**3, np.sin(freq*x), np.sin(freq*z))))
            b = _fit(d, v[:, j])
            q = np.linalg.qr(d, mode='reduced')[0]
            h = np.sum(q*q, axis=1)
            r = (v[:, j] - d @ b) / np.maximum(1-h, 1e-8)
            loss += r @ r
            coeff.append(b)
        return loss, np.asarray(coeff).T

    freq = _profile(lambda a: regress(a)[0], 0.7, 5.)
    return freq, regress(freq)[1]


def _invert(points, freq, coeff):
    b, gain, cubic, own, cross = coeff
    diag = np.arange(len(gain))
    opposite = np.roll(diag, 2)

    def forward(z):
        s = np.sin(freq*z)
        return b + gain*z + cubic*z**3 + own*s + cross*s[opposite]

    def jac(z):
        c = freq*np.cos(freq*z)
        j = np.diag(gain + 3*cubic*z*z + own*c)
        j[diag, opposite] = cross*c[opposite]
        return j

    result = []
    for x in points:
        guess = (x-b)/gain
        def residual(z):
            return forward(z)-x
        sol = root(residual, guess, jac=jac, options={'xtol': 1e-10})
        if np.max(np.abs(residual(sol.x))) > 1e-8:
            sol = least_squares(residual, guess, jac=jac, gtol=1e-12, max_nfev=100)
        result.append(sol.x)
    return np.asarray(result).reshape(points.shape)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x, q = (np.asarray(study[k]['x'], dtype=float) for k in ('train', 'test'))
    latent = _invert(np.concatenate((x, q)), *_sensor_model(study['panel_a']))
    panel = study['panel_b']
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(z, freq):
        return _affine(np.cos(freq*z))

    def loss(freq):
        d = design(u, freq)
        r = v - d @ _fit(d, v)
        h = np.sum(np.linalg.qr(d, mode='reduced')[0]**2, axis=1)
        return np.mean((r / np.maximum(1-h[:, None], 1e-8))**2)

    freq = _profile(loss, 0.7, 6.)
    coords = design(latent, freq) @ _fit(design(u, freq), v)
    return {'c': coords[:len(x)], 'query': coords[len(x):]}
