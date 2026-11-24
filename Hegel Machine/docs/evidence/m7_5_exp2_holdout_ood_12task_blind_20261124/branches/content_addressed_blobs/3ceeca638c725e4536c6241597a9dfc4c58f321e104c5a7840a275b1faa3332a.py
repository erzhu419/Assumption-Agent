"""Robust, spatially corrected Hill fits for a 384-well viability assay."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


_LN10 = np.log(10.0)


def _coordinates(wells):
    row = np.array([w['row'] for w in wells], dtype=float)
    col = np.array([w['col'] for w in wells], dtype=float)
    distance = np.minimum.reduce([row, 15 - row, col, 23 - col])
    return distance, (row - 7.5) / 7.5, (col - 11.5) / 11.5


def _gain(coordinates, params):
    distance, row, col = coordinates
    interior, edge, decay, row_tilt, col_tilt = params
    return interior + edge * np.exp(-distance / decay) + row_tilt * row + col_tilt * col


def _mad(values):
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def _calibrate(wells):
    kill = np.array([w['fluorescence'] for w in wells if w['well_type'] == 'POS_KILL'])
    background = np.median(kill)
    dmso = [w for w in wells if w['well_type'] == 'DMSO']
    signal = np.array([w['fluorescence'] for w in dmso]) - background
    coords = _coordinates(dmso)
    scale = np.median(signal)
    noise = max(_mad(kill) / scale, 0.005)
    signal = signal / scale
    fit = least_squares(
        lambda p: _gain(coords, p) - signal,
        [0.85, 0.30, 2.4, 0, 0],
        bounds=([0.3, 0.0, 0.25, -0.4, -0.4], [1.8, 1.5, 15, 0.4, 0.4]),
        loss='cauchy', f_scale=max(3 * noise, 0.03),
    )
    return background, scale, fit.x, noise


def _fit_hill(dose, response, noise):
    levels = np.unique(dose)
    medians = np.array([np.median(response[dose == level]) for level in levels])
    start = levels[np.argmin(np.abs(medians - 0.5))]

    def residual(p):
        midpoint, slope = p
        return (expit(_LN10 * slope * (midpoint - dose)) - response) / noise

    bounds = ([levels[0] - 5, 0.05], [levels[-1] + 5, 8.0])
    fit = least_squares(residual, [start, 1.0], bounds=bounds, loss='soft_l1', f_scale=2.0)
    fit = least_squares(residual, fit.x, bounds=bounds, loss='cauchy', f_scale=3.0)
    return {'log10_ic50_uM': float(fit.x[0]), 'hill_slope': float(fit.x[1])}


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate a shared dose-response curve per compound across all plates."""
    observations = {}
    for plate in plates:
        wells = plate['wells']
        background, scale, params, noise = _calibrate(wells)
        treated = [w for w in wells if w['well_type'] == 'COMPOUND']
        gains = _gain(_coordinates(treated), params)
        for w, gain in zip(treated, gains):
            response = (w['fluorescence'] - background) / (scale * gain)
            observations.setdefault(w['compound_id'], []).append(
                (w['log10_conc_uM'], response, noise / gain)
            )
    return {key: _fit_hill(*np.array(values).T) for key, values in observations.items()}
