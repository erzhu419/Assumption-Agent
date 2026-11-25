"""Control-calibrated Hill fits for a 384-well viability assay."""

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


_LOG10 = np.log(10.0)


def _geometry(rows, cols):
    distance = np.minimum.reduce((rows, 15 - rows, cols, 23 - cols))
    return (rows - 7.5) / 7.5, (cols - 11.5) / 11.5, distance


def _gain_basis(geometry, decay):
    row, col, distance = geometry
    return np.column_stack((np.ones_like(row), row, col, np.exp(-distance / decay)))


def _calibrate(plates):
    """Use POS_KILL to anchor background and DMSO to map viable signal."""
    controls = []
    prepared = []
    for plate in plates:
        wells = plate['wells']
        kind = np.asarray([well['well_type'] for well in wells])
        readout = np.asarray([well['fluorescence'] for well in wells], dtype=float)
        geometry = _geometry(np.asarray([well['row'] for well in wells]),
                             np.asarray([well['col'] for well in wells]))
        background = np.median(readout[kind == 'POS_KILL'])
        dmso = kind == 'DMSO'
        unit = np.median(readout[dmso] - background)
        signal = (readout - background) / unit
        controls.append((tuple(axis[dmso] for axis in geometry), signal[dmso]))
        prepared.append((wells, kind == 'COMPOUND', geometry, signal))

    # Pool the decay length, not the plate-specific signal or illumination.
    def residual(params):
        decay = params[0]
        return np.concatenate([
            _gain_basis(geometry, decay) @ beta - signal
            for (geometry, signal), beta in zip(controls, params[1:].reshape(-1, 4))
        ])

    initial_decay = 1.5
    initial = [np.linalg.lstsq(_gain_basis(geometry, initial_decay), signal, rcond=None)[0]
               for geometry, signal in controls]
    lower = np.r_[0.45, np.tile([0.05, -0.5, -0.5, 0.0], len(controls))]
    upper = np.r_[6.0, np.tile([2.0, 0.5, 0.5, 2.0], len(controls))]
    start = np.clip(np.r_[initial_decay, np.concatenate(initial)], lower, upper)
    fit = least_squares(residual, start, bounds=(lower, upper), loss='cauchy', f_scale=0.015)
    return prepared, fit.x[0], fit.x[1:].reshape(-1, 4)


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate each compound's log10 IC50 (µM) and positive Hill slope."""
    prepared, decay, coefficients = _calibrate(plates)
    observations = defaultdict(list)
    for (wells, compound, geometry, signal), beta in zip(prepared, coefficients):
        gain = _gain_basis(tuple(axis[compound] for axis in geometry), decay) @ beta
        for i, viability in zip(np.flatnonzero(compound), signal[compound] / gain):
            well = wells[i]
            observations[well['compound_id']].append((well['log10_conc_uM'], viability))

    estimates = {}
    for compound_id, pairs in observations.items():
        dose, viability = np.asarray(pairs).T
        levels = np.unique(dose)
        medians = np.array([np.median(viability[dose == level]) for level in levels])
        start = levels[np.argmin(abs(medians - 0.5))]

        def residual(params):
            ic50, slope = params
            return expit(_LOG10 * slope * (ic50 - dose)) - viability

        fit = least_squares(residual, [start, 1.0],
                            bounds=([levels[0] - 3, 0.08], [levels[-1] + 3, 6.0]),
                            loss='cauchy', f_scale=0.023)
        estimates[compound_id] = {'log10_ic50_uM': float(fit.x[0]),
                                  'hill_slope': float(fit.x[1])}
    return estimates
