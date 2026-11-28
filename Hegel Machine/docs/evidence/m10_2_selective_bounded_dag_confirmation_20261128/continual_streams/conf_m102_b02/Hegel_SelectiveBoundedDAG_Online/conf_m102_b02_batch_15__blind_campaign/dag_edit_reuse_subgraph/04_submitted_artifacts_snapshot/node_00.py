"""Recover latent study coordinates from the three reference panels."""
import numpy as np
from scipy.optimize import minimize_scalar


def _calibrate(panel, measured):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    affine = np.column_stack((np.ones(len(u)), u))
    residual = np.linalg.norm(affine @ np.linalg.lstsq(affine, v, rcond=None)[0] - v, axis=1)
    keep = residual < max(0.05, 4 * np.median(residual))
    if keep.sum() >= 8:
        u, v = u[keep], v[keep]
    d = u.shape[1]
    opposite = (np.arange(d) + d // 2) % d

    def profile(freq):
        s = np.sin(freq*u)
        coefs = np.array([np.linalg.lstsq(np.column_stack((np.ones(len(u)), u[:, j], s[:, j], s[:, opposite[j]])), v[:, j], rcond=None)[0] for j in range(d)]).T
        err = coefs[0] + u*coefs[1] + s*coefs[2] + s[:, opposite]*coefs[3] - v
        return np.sum(err**2), coefs

    freq = minimize_scalar(lambda f: profile(f)[0], bounds=(1.6, 3.2), method='bounded').x
    offset, gain, self_amp, cross_amp = profile(freq)[1]
    x = (measured - offset)/gain
    eye = np.eye(d)
    for _ in range(35):
        sine = np.sin(freq*x)
        cos = freq*np.cos(freq*x)
        err = offset + gain*x + self_amp*sine + cross_amp*sine[:, opposite] - measured
        if np.max(np.abs(err)) < 1e-10:
            break
        jac = (gain + self_amp*cos)[:, :, None]*eye + (cross_amp*cos[:, opposite])[:, :, None]*eye[opposite]
        delta = np.linalg.solve(jac, err[..., None])[..., 0]
        x -= np.clip(delta, -.25, .25)
    return x


def _cosine_map(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(z, freq):
        return np.column_stack((np.ones(len(z)), np.cos(freq*z)))

    def profile(freq):
        h = design(u, freq)
        coef = np.linalg.lstsq(h, v, rcond=None)[0]
        return np.sum((h @ coef - v)**2), coef

    freq = minimize_scalar(lambda f: profile(f)[0], bounds=(1.8, 3.4), method='bounded').x
    return design(x, freq) @ profile(freq)[1]


def _quadratic_map(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v1'))

    def design(z):
        # The fitted response uses only products between even and odd channels.
        return np.column_stack((np.ones(len(z)), z[:, 0]*z[:, 1],
                                z[:, 0]*z[:, 3], z[:, 2]*z[:, 1],
                                z[:, 2]*z[:, 3]))

    coef = np.linalg.lstsq(design(u), v, rcond=None)[0]
    return design(x) @ coef


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']['x']
    n = len(train)
    x = _calibrate(study['panel_a'], np.vstack((train, study['test']['x'])))
    b = _cosine_map(study['panel_b'], x)
    c = _quadratic_map(study['panel_c'], b)
    return dict(x_train=x[:n], x_test=x[n:], c_train=c[:n], c_test=c[n:])
