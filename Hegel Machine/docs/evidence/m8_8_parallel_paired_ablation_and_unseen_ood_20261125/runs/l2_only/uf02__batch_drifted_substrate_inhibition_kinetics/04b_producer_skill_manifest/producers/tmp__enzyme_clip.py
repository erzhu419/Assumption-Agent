"""Recover Haldane kinetics from blank- and reference-calibrated plates."""

import numpy as np
from scipy.optimize import least_squares


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _basis(row, col):
    r, c = (row - 3.5) / 3.5, (col - 5.5) / 5.5
    return (1., r, c, r*r, r*c, c*c)


def _rate(s, log_params):
    vmax, km, ki = np.exp(log_params)
    return vmax * s / (km + s + s*s/ki)


def _raw(w):
    return w['raw_rate'] if 'raw_rate' in w else w['rate_raw']


def _calibrate(wells):
    blanks = np.array([_raw(w) for w in wells if w['well_type'] == 'BLANK'])
    refs = [w for w in wells if w['well_type'] == 'REF_WT']
    x = np.array([_basis(w['row'], w['col']) for w in refs])
    offset = np.median(blanks)
    y = np.array([_raw(w) for w in refs]) - offset
    initial = np.linalg.lstsq(x, y, rcond=None)[0]
    floor = max(1e-8, 1e-6 * np.median(np.abs(y)))
    scale = max(_mad(blanks), floor)
    gain = least_squares(lambda p: x @ p - y, initial, jac=lambda p: x.copy(),
                         loss='soft_l1', f_scale=1.5*scale).x
    residual = x @ gain - y
    scale = max(_mad(np.r_[blanks-offset, residual]), floor)
    keep = np.abs(residual - np.median(residual)) <= 4.5 * scale
    if not keep.all() and np.linalg.matrix_rank(x[keep]) == x.shape[1]:
        gain = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
        scale = max(_mad(np.r_[blanks-offset, x @ gain-y]), floor)
    return offset, gain, scale


def estimate_variant_haldane_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate positive V_max, K_m, and K_i for each variant across plates."""
    groups = {}
    for plate in plates:
        wells = plate['wells']
        offset, gain, scale = _calibrate(wells)
        for w in wells:
            if w['well_type'] == 'VARIANT':
                g = np.dot(_basis(w['row'], w['col']), gain)
                groups.setdefault(w['variant_id'], []).append(
                    (w['substrate_mM'], _raw(w)-offset, g, scale))
    results = {}
    for variant, rows in groups.items():
        s, y, g, scale = np.asarray(rows).T
        lo, hi = min(s[s > 0]), max(s)
        v = max(np.percentile(y / g, 85), 1e-6)
        initial = np.log([2*v, np.sqrt(lo*hi)/2, np.sqrt(lo*hi)*2])
        lower = np.log([v*1e-4, lo*1e-5, lo*1e-5])
        upper = np.log([v*1e5, hi*1e5, hi*1e5])
        fit = least_squares(lambda p: (g*_rate(s, p)-y)/scale, initial,
                            bounds=(lower, upper), loss='soft_l1', f_scale=1.5)
        residual = (g*_rate(s, fit.x)-y)/scale
        keep = np.abs(residual - np.median(residual)) <= 4.5 * max(1., _mad(residual))
        if not keep.all() and np.unique(s[keep & (s > 0)]).size >= 3:
            fit = least_squares(lambda p: ((g*_rate(s, p)-y)/scale)[keep], fit.x,
                                bounds=(lower, upper), loss='soft_l1', f_scale=2.5)
        results[variant] = dict(zip(('V_max', 'K_m', 'K_i'), map(float, np.exp(fit.x))))
    return results
