"""Elastic stiffness and persistent loading slips from force–displacement curves."""

import numpy as np
from scipy.optimize import least_squares


def _mad(y):
    return 1.4826 * np.median(np.abs(y - np.median(y)))


def _robust_fit(x, y, scale=None):
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    if scale is None:
        scale = max(_mad(y - x @ beta), 1e-8)
    for _ in range(12):
        r = (y - x @ beta) / scale
        w = (1 + r * r) ** -0.5
        beta = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)[0]
    return beta


def _force_stages(force):
    peak = np.max(force)
    at_peak = np.flatnonzero(np.abs(force - peak) <= 1e-8 * np.ptp(force))
    return int(at_peak[0]), int(at_peak[-1])


def _first_slip(force, depth, load_end):
    # Subtract the local smooth loading trend before checking the step height.
    # A candidate needs a sustained change in level on both sides, not just a
    # large first difference. Score adjacent splits using the same observations.
    f = force[:load_end + 1]
    h = depth[:load_end + 1]
    for i in np.flatnonzero(np.diff(h) > 0.25) + 1:
        start, end = max(0, i - 9), min(len(h), i + 9)
        x, y = f[start:end] - f[i], h[start:end]
        if end == len(h):
            # The force is still at its peak in these hold samples.
            tail = depth[load_end + 1:load_end + 4]
            x, y = np.r_[x, np.repeat(x[-1], len(tail))], np.r_[y, tail]
        split = i - start
        fits = []
        for j in range(max(1, split - 2), min(end - start, split + 3)):
            after = np.arange(len(y)) >= j
            # Center each side separately so the jump does not tilt the local
            # loading slope. Robust weights downweight isolated sensor spikes.
            A = np.column_stack((np.ones(len(y)), x, after))
            slope = np.median(np.diff(y[:end - start]) / np.diff(x[:end - start]))
            r = y - slope * x
            height = np.median(r[after]) - np.median(r[~after])
            baseline = np.where(after, np.median(r[after]), np.median(r[~after]))
            fit = _robust_fit(A, y, max(_mad(r - baseline), 1e-5))
            loss = np.minimum(np.abs(y - A @ fit), 0.12).sum()
            fits.append((loss, abs(split - j), j, fit[2]))
        if fits:
            _, _, j, height = min(fits)
            if height > 0.35:
                return float(f[start + j])
    return None


def _hold_rate(t, h, start, end):
    span = t[end] - t[start]
    idx = np.arange(start, end + 1)
    idx = idx[(t[idx] >= t[start] + 0.05 * span)
              & (t[idx] <= t[end] - min(0.18, 0.12 * span))]
    if len(idx) < 5:
        idx = np.arange(start, end + 1)
    z = t[idx] - t[end]
    y = h[idx] - np.median(h[idx])
    x = np.column_stack((np.ones_like(z), z))
    beta = _robust_fit(x, y)
    noise = max(_mad(y - x @ beta), 1e-8)
    return max(0.0, beta[1]), noise


def _unloading_stiffness(t, f, h, start, end):
    creep, noise = _hold_rate(t, h, start, end)
    z = t[end + 1:] - t[end]
    x = f[end + 1:] - f[end]
    y = h[end + 1:] - np.median(h[end + 1:])
    # Skip samples in the 15 ms onset-glitch window. For very short traces
    # retain enough points to fit the elastic slope.
    keep = z > 0.015
    if np.count_nonzero(keep) >= 5:
        z, x, y = z[keep], x[keep], y[keep]
    design = np.column_stack((np.ones_like(x), x))
    rate = -_robust_fit(np.column_stack((np.ones_like(z), z)), x)[1]
    apparent = _robust_fit(design, y, noise)
    # A linear creep term is indistinguishable from elastic response during
    # a linear ramp. If subtracting the measured hold rate leaves no resolved
    # curvature, estimate compliance with that rate fixed, not a fitted tau.
    corrected = y - creep * z
    linear = _robust_fit(design, corrected, noise)
    residual = corrected - design @ linear
    bins = np.array_split(residual, min(8, max(1, len(residual) // 5)))
    coherent = np.sqrt(np.mean([np.median(part)**2 for part in bins]))
    if linear[1] > 0 and coherent <= 0.8 * noise:
        return 1.0 / linear[1]


    def fit_residual(p):
        off, compliance, log_tau = p
        tau = np.exp(log_tau)
        creep_displacement = -creep * tau * np.expm1(-z / tau)
        return off + compliance * x + creep_displacement - y

    # The creep displacement has derivative equal to the measured hold rate
    # at onset. Thus the fitted elastic compliance includes the specified
    # creep/ramp-rate correction without applying it again later in unloading.
    dt = np.median(np.diff(t))
    upper = 30 * z[-1]
    p0 = [apparent[0], max(1e-9, apparent[1] + creep / rate), np.log(0.1 * z[-1])]
    fit = least_squares(fit_residual, p0,
                        bounds=([-np.inf, 1e-10, np.log(0.1 * dt)],
                                [np.inf, np.inf, np.log(upper)]),
                        loss='soft_l1', f_scale=noise)
    return 1.0 / fit.x[1]


def analyze_specimen_curves(curves: list[dict], frame_compliance_um_per_N: float) -> dict:
    """Return the median corrected stiffness and the first slip for each curve."""
    stiffness, slips = [], []
    for curve in curves:
        t, f, raw = (np.asarray(curve[key], dtype=float)
                     for key in ('time_s', 'force_N', 'raw_disp_um'))
        h = raw - frame_compliance_um_per_N * f
        start, end = _force_stages(f)
        stiffness.append(_unloading_stiffness(t, f, h, start, end))
        slips.append(_first_slip(f, h, start))
    return {'true_initial_stiffness_N_per_um': float(np.median(stiffness)),
            'first_slip_force_N_per_curve': slips}
