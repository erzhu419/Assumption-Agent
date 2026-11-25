"""Control-referenced Hill fitting for a 384-well fluorescence assay."""

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit


def _spatial_basis(row, col, length):
    depth = np.minimum.reduce((row, 15 - row, col, 23 - col))
    return np.column_stack((np.ones_like(row), np.exp(-depth / length),
                            (row - 7.5) / 7.5, (col - 11.5) / 11.5))


def _profile_gain(row, col, signal):
    scale = np.median(signal)
    target = signal / scale
    # Fit in fractional signal units so the outlier rule follows plate brightness.
    def profile(length):
        design = _spatial_basis(row, col, length)
        initial = np.linalg.lstsq(design, target, rcond=None)[0]
        return least_squares(lambda b: design @ b - target, initial,
                             jac=lambda b: design, loss='soft_l1', f_scale=0.009)

    length = minimize_scalar(lambda t: profile(t).cost, bounds=(0.55, 6.0),
                             method='bounded', options={'xatol': 0.01}).x
    return length, profile(length).x * scale


def _fit_gain(row, col, signal):
    length, coeff = _profile_gain(row, col, signal)
    residual = signal - _spatial_basis(row, col, length) @ coeff
    # Clogged DMSO wells lose far more signal than normal readout noise.
    # Exclude them before refitting the perimeter decay, not just the gain.
    keep = residual > -0.07 * np.median(signal)
    if not keep.all():
        return _profile_gain(row[keep], col[keep], signal[keep])
    return length, coeff


def _fit_hill(dose, scaled_signal, gain):
    # Residuals in the readout domain, scaled by a typical full-viability signal.
    scale = np.median(gain)
    gain = gain / scale
    observed = scaled_signal / scale
    log10 = np.log(10.0)

    def residual(b):
        return gain * expit(log10 * b[1] * (b[0] - dose)) - observed

    def jacobian(b):
        z = b[0] - dose
        v = expit(log10 * b[1] * z)
        d = gain * log10 * v * (1.0 - v)
        return np.column_stack((d * b[1], d * z))

    bounds = ([dose.min() - 3, 0.05], [dose.max() + 3, 8.0])
    # The convex loss in residual space gives a stable starting point before
    # the redescending loss suppresses badly clogged or dead wells.
    init = least_squares(residual, [np.median(dose), 1.0], jac=jacobian,
                         bounds=bounds, loss='soft_l1', f_scale=0.035).x
    return least_squares(residual, init, jac=jacobian, bounds=bounds,
                         loss='cauchy', f_scale=0.02).x


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate the midpoint (log10 µM) and positive Hill slope of each compound."""
    observations = defaultdict(list)
    for plate in plates:
        wells = plate['wells']
        if not wells:
            continue
        row, col, readout = np.array([(w['row'], w['col'], w['fluorescence'])
                                      for w in wells], dtype=float).T
        kind = np.array([w['well_type'] for w in wells])
        background = np.median(readout[kind == 'POS_KILL'])
        signal = readout - background
        dmso = kind == 'DMSO'
        length, coeff = _fit_gain(row[dmso], col[dmso], signal[dmso])
        gain = _spatial_basis(row, col, length) @ coeff
        for i in np.flatnonzero(kind == 'COMPOUND'):
            w = wells[i]
            observations[w['compound_id']].append((w['log10_conc_uM'], signal[i], gain[i]))

    result = {}
    for cid, items in observations.items():
        ic50, slope = _fit_hill(*np.asarray(items, dtype=float).T)
        result[cid] = {'log10_ic50_uM': float(ic50), 'hill_slope': float(slope)}
    return result
