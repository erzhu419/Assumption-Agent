"""Estimate Hill curves after control-based spatial plate calibration."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def _coordinates(wells):
    r, c = np.array([(w['row'], w['col']) for w in wells], dtype=float).T
    distance = np.minimum.reduce([r, 15 - r, c, 23 - c])
    return distance, r / 15 - 0.5, c / 23 - 0.5


def _gain(p, coords):
    d, r, c = coords
    base, edge, row, col, log_decay = p
    return base + edge * np.exp(-d / np.exp(log_decay)) + row * r + col * c


def _calibrate_plate(wells):
    kind = np.array([w['well_type'] for w in wells])
    y = np.array([w['fluorescence'] for w in wells], dtype=float)
    pos = kind == 'POS_KILL'
    neg = kind == 'DMSO'
    background = np.median(y[pos])
    scale = np.median(y[neg]) - background
    coords = _coordinates(wells)
    controls = tuple(x[neg] for x in coords)
    target = (y[neg] - background) / scale
    residual = lambda p: _gain(p, controls) - target
    bounds = ([0.1, 0., -1., -1., np.log(0.25)],
              [3., 2., 1., 1., np.log(12.)])
    fit = least_squares(residual, [0.9, 0.4, 0., 0., np.log(2.)],
                        bounds=bounds, loss='soft_l1', f_scale=0.008)
    # Refine from the convex-loss basin so grossly corrupted controls have
    # vanishing influence instead of continuing to pull the gain surface.
    fit = least_squares(residual, fit.x, bounds=bounds, loss='cauchy', f_scale=0.015)
    gain = scale * _gain(fit.x, coords)
    return (y - background) / gain


def _fit_hill(conc, viability):
    doses, ix = np.unique(conc, return_inverse=True)
    medians = np.array([np.median(viability[ix == i]) for i in range(len(doses))])
    start = doses[np.argmin(np.abs(medians - 0.5))]
    bounds = ([doses[0] - 5, -3.], [doses[-1] + 5, 3.])

    def residual(p):
        return expit(-np.log(10.) * np.exp(p[1]) * (conc - p[0])) - viability

    fit = least_squares(residual, [start, 0.], bounds=bounds,
                        loss='soft_l1', f_scale=0.02)
    fit = least_squares(residual, fit.x, bounds=bounds,
                        loss='cauchy', f_scale=0.02)
    return {'log10_ic50_uM': float(fit.x[0]), 'hill_slope': float(np.exp(fit.x[1]))}


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Return one robust concentration–response fit per compound across plates."""
    observations = {}
    for plate in plates:
        wells = plate['wells']
        viability = _calibrate_plate(wells)
        for well, value in zip(wells, viability):
            if well['well_type'] == 'COMPOUND':
                observations.setdefault(well['compound_id'], []).append((well['log10_conc_uM'], value))
    return {name: _fit_hill(*np.asarray(pairs, dtype=float).T)
            for name, pairs in observations.items()}
