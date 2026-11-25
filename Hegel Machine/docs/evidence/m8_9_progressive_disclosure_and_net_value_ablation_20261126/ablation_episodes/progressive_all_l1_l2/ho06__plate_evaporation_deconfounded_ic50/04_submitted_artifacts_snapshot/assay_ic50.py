"""Robust Hill-curve fitting with per-plate spatial calibration."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


_LN10 = np.log(10.0)


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Fit each compound across plates, using kill and DMSO wells as anchors."""
    compounds = sorted({w['compound_id'] for plate in plates for w in plate['wells']
                        if w['well_type'] == 'COMPOUND'})
    if not compounds:
        return {}
    compound_index = {name: i for i, name in enumerate(compounds)}
    n, m = len(plates), len(compounds)
    records = []
    for i, plate in enumerate(plates):
        for well in plate['wells']:
            kind = well['well_type']
            if kind == 'COMPOUND':
                k = compound_index[well['compound_id']]
                dose = well['log10_conc_uM']
            elif kind in ('DMSO', 'POS_KILL'):
                k, dose = (-1 if kind == 'DMSO' else -2), 0.0
            else:
                continue
            records.append((i, k, dose, well['row'], well['col'], well['fluorescence']))
    p, k, dose, row, col, signal = np.asarray(records, dtype=float).T
    p, k = p.astype(int), k.astype(int)
    dmso = k == -1
    kill = k == -2
    treated = k >= 0
    background = np.array([np.median(signal[(p == i) & kill]) for i in range(n)])
    scale = np.array([np.median(signal[(p == i) & dmso]) - background[i]
                      for i in range(n)])
    observed = (signal - background[p]) / scale[p]
    x, z = (col - 11.5) / 11.5, (row - 7.5) / 7.5
    edge = np.minimum.reduce((row, 15 - row, col, 23 - col))

    midpoint = []
    for j in range(m):
        mask = k == j
        levels = np.unique(dose[mask])
        medians = [np.median(observed[mask & (dose == level)]) for level in levels]
        midpoint.append(levels[np.argmin(np.abs(np.asarray(medians) - 0.5))])

    # Parameters: plate offsets and scales, compound midpoints and slopes,
    # then each plate's perimeter amplitude, decay length and tilt coefficients.
    spatial = 2 * n + 2 * m
    start = np.r_[np.zeros(n), np.full(n, 0.88), midpoint, np.ones(m),
                  np.tile([0.4, 2.0, -0.02, 0.03], n)]
    lower = np.r_[np.full(n, -0.2), np.full(n, 0.3), np.full(m, -7.0),
                  np.full(m, 0.15), np.tile([0.0, 0.15, -0.5, -0.5], n)]
    upper = np.r_[np.full(n, 0.2), np.full(n, 2.0), np.full(m, 7.0),
                  np.full(m, 5.0), np.tile([1.5, 12.0, 1.0, 1.0], n)]
    ii = np.arange(p.size)
    tt = ii[treated]
    mu_col = 2 * n + k[treated]
    slope_col = mu_col + m
    field_cols = spatial + 4 * p[:, None] + np.arange(4)

    def evaluate(theta, derivative=False):
        amplitude, decay, tilt_x, tilt_z = theta[field_cols].T
        attenuation = np.exp(-edge / decay)
        field = 1 + amplitude * attenuation + tilt_x * x + tilt_z * z
        delta = theta[mu_col] - dose[treated]
        slope = theta[slope_col]
        viability = dmso.astype(float)
        v = expit(_LN10 * slope * delta)
        viability[treated] = v
        plate_scale = theta[n + p]
        if not derivative:
            return theta[p] + plate_scale * field * viability - observed
        jac = np.zeros((p.size, theta.size))
        jac[ii, p] = 1.0
        jac[ii, n + p] = field * viability
        hill_factor = _LN10 * plate_scale[tt] * field[tt] * v * (1 - v)
        jac[tt, mu_col] = hill_factor * slope
        jac[tt, slope_col] = hill_factor * delta
        field_derivative = np.column_stack((attenuation,
            amplitude * attenuation * edge / decay**2, x, z))
        jac[ii[:, None], field_cols] = (plate_scale * viability)[:, None] * field_derivative
        return jac

    def jacobian(theta):
        return evaluate(theta, derivative=True)

    # Start with a gentler loss; then downweight dead or clogged wells sharply.
    fit = least_squares(evaluate, start, jac=jacobian, bounds=(lower, upper),
                        loss='soft_l1', f_scale=0.025)
    fit = least_squares(evaluate, fit.x, jac=jacobian, bounds=(lower, upper),
                        loss='cauchy', f_scale=0.012)
    return {name: {'log10_ic50_uM': float(fit.x[2 * n + j]),
                   'hill_slope': float(fit.x[2 * n + m + j])}
            for j, name in enumerate(compounds)}
