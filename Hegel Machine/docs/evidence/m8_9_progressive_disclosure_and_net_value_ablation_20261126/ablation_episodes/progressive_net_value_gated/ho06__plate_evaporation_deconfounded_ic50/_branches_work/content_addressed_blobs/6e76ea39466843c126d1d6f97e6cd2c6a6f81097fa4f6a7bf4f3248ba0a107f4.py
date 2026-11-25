"""Control-calibrated Hill fits for a 16 by 24 viability assay."""
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit


def _positions(wells):
    row = np.array([w['row'] for w in wells], dtype=float)
    col = np.array([w['col'] for w in wells], dtype=float)
    edge = np.minimum(np.minimum(row, 15 - row), np.minimum(col, 23 - col))
    return np.column_stack((np.ones_like(row), row / 7.5 - 1, col / 11.5 - 1, edge))


def _basis(pos, decay):
    x = pos.copy()
    x[:, 3] = np.exp(-x[:, 3] / decay)
    return x


def _control_fit(pos, signal, decay, noise):
    x = _basis(pos, decay)
    initial = np.linalg.lstsq(x, signal, rcond=None)[0]
    return least_squares(lambda a: x @ a - signal, initial, jac=lambda a: x,
                         loss='soft_l1', f_scale=2 * noise)


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate IC50 in log10 micromolar units and the positive Hill slope."""
    controls, samples = [], []
    for plate in plates:
        groups = defaultdict(list)
        for well in plate['wells']:
            groups[well['well_type']].append(well)
        kills = np.array([w['fluorescence'] for w in groups['POS_KILL']])
        background = np.median(kills)
        dmso = np.array([w['fluorescence'] for w in groups['DMSO']]) - background
        scale = np.median(dmso)
        noise = max(0.005, 1.4826 * np.median(np.abs(kills - background)) / scale)
        controls.append((_positions(groups['DMSO']), dmso / scale, noise))
        wells = groups['COMPOUND']
        readings = np.array([w['fluorescence'] for w in wells])
        samples.append((wells, _positions(wells), (readings - background) / scale))

    # Pool information about the range of evaporation, but fit its strength
    # and the illumination plane separately for each plate.
    def pooled_cost(log_decay):
        decay = np.exp(log_decay)
        return sum(_control_fit(p, y, decay, n).cost / n**2 for p, y, n in controls)

    bounds = (np.log(0.4), np.log(7.0))
    shared = minimize_scalar(pooled_cost, bounds=bounds, method='bounded').x
    by_compound = defaultdict(list)
    for (pos, signal, noise), (wells, sample_pos, readings) in zip(controls, samples):
        def local_cost(log_decay):
            cost = _control_fit(pos, signal, np.exp(log_decay), noise).cost / noise**2
            return cost + 0.5 * ((log_decay - shared) / 0.30)**2

        decay = np.exp(minimize_scalar(local_cost, bounds=bounds, method='bounded').x)
        coeff = _control_fit(pos, signal, decay, noise).x
        gain = _basis(sample_pos, decay) @ coeff
        for well, y, g in zip(wells, readings, gain):
            by_compound[well['compound_id']].append((well['log10_conc_uM'], y, g, noise))

    result = {}
    log10 = np.log(10)
    for compound, records in by_compound.items():
        conc, y, gain, noise = np.array(records).T
        # Keep the residual in fluorescence units; division by spatial gain
        # would otherwise overweight the low-gain wells.
        sigma = np.maximum(noise, 0.008)
        levels = np.unique(conc)
        medians = np.array([np.median((y / gain)[conc == d]) for d in levels])
        mid = np.interp(0.5, np.minimum.accumulate(medians)[::-1], levels[::-1])
        def residual(theta):
            viability = expit(log10 * theta[1] * (theta[0] - conc))
            return (gain * viability - y) / sigma

        fit = least_squares(residual, [mid, 1.0], loss='cauchy', f_scale=3.0,
                            bounds=([levels[0] - 5, 0.05], [levels[-1] + 5, 8.0])).x
        result[compound] = {'log10_ic50_uM': float(fit[0]), 'hill_slope': float(fit[1])}
    return result
