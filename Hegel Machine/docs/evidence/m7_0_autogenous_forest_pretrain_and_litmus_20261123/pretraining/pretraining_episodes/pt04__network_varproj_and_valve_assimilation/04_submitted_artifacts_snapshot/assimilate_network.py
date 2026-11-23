"""Recover the valve and Fourier forcing from affine-calibrated sensor traces."""
import numpy as np
from scipy.optimize import least_squares


def _response(problem):
    T = int(problem['T'])
    dt = float(problem['dt'])
    x0 = np.asarray(problem['x0'], dtype=float)
    n = x0.size
    A = np.eye(n) - dt * (np.asarray(problem['L']) + np.diag(problem['damp']))
    phase = 2 * np.pi * np.arange(T)[:, None] * np.arange(1, 6) / T
    F = np.concatenate((np.sin(phase), np.cos(phase)), axis=1)
    F -= F.mean(axis=0)
    drives = np.column_stack((np.ones(T), F))
    inlet = dt * np.asarray(problem['b_in'])[:, None]
    states = np.zeros((T, n, drives.shape[1] + 1))
    states[0, :, 0] = x0
    for t in range(T - 1):
        states[t + 1] = A @ states[t]
        states[t + 1, :, 1:] += inlet * drives[t]
    return F, states


def _mad(a):
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=0)), axis=0)


def _project(base, design, y, weights):
    """Eliminate the gain and offset of each sensor in weighted least squares."""
    w = weights / weights.sum(axis=0)
    yc = y - np.sum(w * y, axis=0)
    bc = base - np.sum(w * base, axis=0)
    dc = design - np.sum(w[..., None] * design, axis=0)
    sw = np.sqrt(weights)

    def evaluate(c):
        z = bc + np.einsum('tsk,k->ts', dc, c)
        den = np.maximum(np.sum(w * z * z, axis=0), 1e-30)
        slope = np.sum(w * z * yc, axis=0) / den
        gain = np.maximum(0.2, slope)
        r = gain * z - yc
        return r, z, den, slope, gain

    def jac(c):
        _, z, den, slope, gain = evaluate(c)
        da = np.einsum('ts,tsk->sk', w * (yc - 2.0 * gain * z), dc) / den[:, None]
        da[slope <= 0.2] = 0.0
        return (sw[..., None] * (gain[None, :, None] * dc + z[..., None] * da)).reshape(-1, design.shape[-1])

    return evaluate, lambda c: (sw * evaluate(c)[0]).ravel(), jac


def _fit(base, design, y, initial_weights, noise):
    c = np.zeros(design.shape[-1])
    w = initial_weights.copy()
    penalty = 0.06 * np.tile(np.arange(1., 6.) ** 0.65, 2)
    penalty_jac = np.diag(penalty)
    for scale in (3.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5):
        evaluate, fun, jac = _project(base, design, y, w)
        c = least_squares(lambda q: np.r_[fun(q), penalty*q], c, jac=lambda q: np.vstack((jac(q), penalty_jac)), max_nfev=100, ftol=1e-9).x
        r = evaluate(c)[0]
        w = 1.0 / (1.0 + (r / (scale * noise)) ** 2)
    # Screen large spikes once, then compare nested Fourier fits on the same samples.
    clean = np.abs(r - np.median(r, axis=0)) <= 8.0 * noise
    rms_noise = max(np.median(noise), 1e-8)
    weights_clean = clean * (rms_noise / noise)**2
    fits = []
    for order in range(1, 6):
        indices = np.r_[0:order, 5:5+order]
        ev, fun, jac = _project(base, design[..., indices], y, weights_clean)
        # Higher orders need more shrinkage on the partially observed interval.
        p = np.full(len(indices), (0.05 + 0.03 * (order - 1)) * rms_noise / 0.012)
        q = least_squares(lambda q: np.r_[fun(q), p*q], c[indices],
                          jac=lambda q: np.vstack((jac(q), np.diag(p))),
                          max_nfev=100, ftol=1e-9).x
        res = ev(q)[0]
        coeff = np.zeros_like(c)
        coeff[indices] = q
        criterion = np.sum(clean * (res / noise)**2) + 0.65 * np.log(y.size) * order
        fits.append((criterion, coeff))
    criteria = np.array([f[0] for f in fits])
    weights = np.exp(-(criteria-criteria.min()) / 4.0)
    c = np.average([f[1] for f in fits], weights=weights, axis=0)
    r = _project(base, design, y, weights_clean)[0](c)[0]
    score = np.log1p((r / (1.5 * noise)) ** 2).sum()
    return c, r, score


def _estimate(problem):
    y = np.asarray(problem['Y_obs'], dtype=float)
    F, states = _response(problem)
    measured = states[:len(y), np.asarray(problem['obs_nodes'], dtype=int), :]
    local_error = y[1:-1] - (y[:-2] + y[2:]) / 2
    noise = np.maximum(_mad(local_error) / np.sqrt(1.5), 1e-8)
    # A short median window supplies robust initial weights, not replacement data.
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(y, ((2, 2), (0, 0)), mode='edge'), 5, axis=0)
    med = np.median(windows, axis=-1)
    init = 1.0 / (1.0 + ((y - med) / (5.0 * noise)) ** 2)
    fits = []
    for valve in (0.35, 1.0):
        base = measured[..., 0] + valve * measured[..., 1]
        c, residual, score = _fit(base, valve * measured[..., 2:], y, init, noise)
        fits.append((score, valve, c, residual))
    return F, states, fits


def recover_network_state(problem: dict) -> dict:
    """Return the inferred valve, full trajectory, drive, and spike mask."""
    F, states, fits = _estimate(problem)
    _, valve, c, residual = min(fits, key=lambda fit: fit[0])
    centered = residual - np.median(residual, axis=0)
    threshold = np.maximum(5.0 * _mad(centered), 1e-8)
    return {'valve_open': bool(valve == 1.0),
            'x_full': states[..., 0] + valve * (states[..., 1] + states[..., 2:] @ c),
            'u_drive': 1.0 + F @ c,
            'outlier_mask': np.abs(centered) > threshold}
