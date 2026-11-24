"""Control-calibrated Hill fits for 384-well fluorescence viability screens."""

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit


def _spatial_basis(row, col, decay):
    depth = np.minimum.reduce((row, 15 - row, col, 23 - col))
    return np.column_stack((np.ones_like(row), np.exp(-depth / decay),
                            row / 15 - 0.5, col / 23 - 0.5))


def _fit_control_gain(row, col, signal, decay):
    design = _spatial_basis(row, col, decay)
    initial = np.linalg.lstsq(design, signal, rcond=None)[0]
    return least_squares(lambda coef: design @ coef - signal, initial,
                         jac=lambda coef: design, loss='cauchy', f_scale=0.025)


def _fit_hill(dose, viability, relative_gain):
    weight = relative_gain / np.median(relative_gain)
    scale = 0.025
    log10 = np.log(10.)

    def residual(par):
        potency, slope = par
        return weight * (expit(log10 * slope * (potency - dose)) - viability)

    # A coarse robust search avoids choosing a failed low-dose well as the
    # midpoint when the whole upper plateau is sparsely sampled.
    lo, hi = dose.min() - 4., dose.max() + 4.
    midpoint = np.linspace(dose.min() - 1., dose.max() + 1., 61)
    start = np.array(np.meshgrid(midpoint, [0.45, 0.8, 1.2, 1.8, 2.8])).reshape(2, -1).T
    prediction = expit(log10 * start[:, 1, None] * (start[:, 0, None] - dose))
    score = np.log1p(((prediction - viability) * weight / scale) ** 2).sum(axis=1)
    return least_squares(residual, start[np.argmin(score)],
                         bounds=([lo, 0.08], [hi, 8.]),
                         loss='cauchy', f_scale=scale).x


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate each compound's log10 IC50 in µM and positive Hill slope."""
    prepared = []
    for plate in plates:
        wells = plate['wells']
        row, col = np.array([(w['row'], w['col']) for w in wells], dtype=float).T
        kind = np.array([w['well_type'] for w in wells])
        fluorescence = np.array([w['fluorescence'] for w in wells], dtype=float)
        background = np.median(fluorescence[kind == 'POS_KILL'])
        positive = kind == 'DMSO'
        signal = fluorescence - background
        # A failed DMSO well loses signal on a scale much larger than the
        # spatial gain variation; exclude it before fitting the gain field.
        positive &= signal > 0.5 * np.median(signal[positive])
        reference = np.median(signal[positive])
        prepared.append((wells, row, col, kind, signal / reference, positive, reference))

    samples = defaultdict(list)
    for wells, row, col, kind, signal, dmso, reference in prepared:
        r_ctrl, c_ctrl, y_ctrl = row[dmso], col[dmso], signal[dmso]

        def profile(decay):
            return _fit_control_gain(r_ctrl, c_ctrl, y_ctrl, decay).cost

        decay = minimize_scalar(profile, bounds=(0.5, 8.), method='bounded',
                                options={'xatol': 0.005}).x
        coef = _fit_control_gain(r_ctrl, c_ctrl, y_ctrl, decay).x
        gain = _spatial_basis(row, col, decay) @ coef
        for i in np.flatnonzero(kind == 'COMPOUND'):
            well = wells[i]
            samples[well['compound_id']].append(
                (well['log10_conc_uM'], signal[i] / gain[i], gain[i] * reference))

    estimates = {}
    for compound, observations in samples.items():
        potency, slope = _fit_hill(*np.array(observations).T)
        estimates[compound] = {'log10_ic50_uM': float(potency), 'hill_slope': float(slope)}
    return estimates
