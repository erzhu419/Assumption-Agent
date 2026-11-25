"""Estimate Hill curves using spatially calibrated 384-well fluorescence."""

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


_LN10 = np.log(10.0)


def _coordinates(wells):
    row, col, signal = np.array([
        (w['row'], w['col'], w['fluorescence']) for w in wells
    ], dtype=float).T
    distance = np.minimum.reduce([row, 15 - row, col, 23 - col])
    return distance, (row - 7.5) / 7.5, (col - 11.5) / 11.5, signal


def _design(distance, row, col, decay, family):
    u = distance / decay
    edge = np.exp(-u) if family == 'exponential' else 1 / (1 + u*u)
    return np.column_stack((np.ones_like(row), edge, row, col))


def _calibrate(plates):
    prepared = []
    controls = []
    for plate in plates:
        wells = plate['wells']
        d, r, c, signal = _coordinates(wells)
        kinds = np.array([w['well_type'] for w in wells])
        bg = np.median(signal[kinds == 'POS_KILL'])
        dmso = kinds == 'DMSO'
        scale = np.median(signal[dmso]) - bg
        y = (signal - bg) / scale
        prepared.append((wells, d, r, c, y))
        controls.append((d[dmso], r[dmso], c[dmso], y[dmso]))

    def residual(params, family):
        decay = params[0]
        return np.concatenate([
            _design(d, r, c, decay, family) @ beta - y
            for (d, r, c, y), beta in zip(controls, params[1:].reshape(-1, 4))
        ])

    # Share the decay length, not the baseline, gain, or illumination coefficients.
    start = [2.0] + [v for _ in controls for v in (0.9, 0.3, 0.0, 0.0)]
    lo = [0.35] + [v for _ in controls for v in (0.05, 0.0, -np.inf, -np.inf)]
    hi = [8.0] + [np.inf] * (4 * len(controls))
    candidates = []
    for family in ('exponential', 'rational'):
        f = lambda p: residual(p, family)
        fit = least_squares(f, start, bounds=(lo, hi), loss='soft_l1', f_scale=0.012)
        fit = least_squares(f, fit.x, bounds=(lo, hi), loss='cauchy', f_scale=0.018)
        candidates.append((fit.cost, family, fit.x))
    _, family, params = min(candidates, key=lambda item: item[0])
    return prepared, params, family


def _fit_hill(log_dose, viability):
    def residual(params):
        mid, slope = params
        return expit(_LN10 * slope * (mid - log_dose)) - viability

    doses = np.unique(log_dose)
    medians = [np.median(viability[log_dose == x]) for x in doses]
    start = [doses[np.argmin(np.abs(np.array(medians) - 0.5))], 1.0]
    bounds = ([doses[0] - 4.0, 0.03], [doses[-1] + 4.0, 8.0])
    params = least_squares(residual, start, bounds=bounds, loss='soft_l1', f_scale=0.035).x
    params = least_squares(residual, params, bounds=bounds, loss='cauchy', f_scale=0.022).x
    err = residual(params)
    sigma = 1.4826 * np.median(np.abs(err - np.median(err)))
    keep = np.abs(err) < max(0.04, 4.0 * sigma)
    if np.unique(log_dose[keep]).size >= 3:
        params = least_squares(lambda p: residual(p)[keep], params, bounds=bounds,
                               loss='soft_l1', f_scale=0.022).x
    return {'log10_ic50_uM': float(params[0]), 'hill_slope': float(params[1])}


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate each compound's log10 IC50 in µM and positive Hill slope."""
    if not plates:
        return {}
    prepared, params, family = _calibrate(plates)
    points = defaultdict(list)
    for (wells, d, r, c, y), beta in zip(prepared, params[1:].reshape(-1, 4)):
        gain = _design(d, r, c, params[0], family) @ beta
        for well, viability in zip(wells, y / gain):
            if well['well_type'] == 'COMPOUND':
                points[well['compound_id']].append((well['log10_conc_uM'], viability))
    return {compound: _fit_hill(*np.array(values).T) for compound, values in points.items()}
