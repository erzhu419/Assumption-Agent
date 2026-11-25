"""Estimate Hill curves after control-based spatial calibration of each plate."""

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def _surface(row, col, decay):
    distance = np.minimum.reduce((row, 15 - row, col, 23 - col))
    return np.column_stack((np.ones_like(row), np.exp(-distance / decay),
                            (row - 7.5) / 7.5, (col - 11.5) / 11.5))


def _calibrate(row, col, signal, kill, dmso):
    background = np.median(signal[kill])
    control = signal[dmso] - background
    scale = np.median(control)
    r, c = row[dmso], col[dmso]

    def residual(p):
        return _surface(r, c, p[4]) @ p[:4] - control / scale

    fit = least_squares(residual, [0.85, 0.30, 0, 0, 1.8],
                        bounds=([0.35, 0, -0.25, -0.25, 0.5],
                                [1.5, 1.0, 0.25, 0.25, 5.0]),
                        loss='cauchy', f_scale=0.018)
    gain = scale * np.maximum(_surface(row, col, fit.x[4]) @ fit.x[:4], 0.25)
    return background, gain, fit.x


def _hill(dose, viability, weight):
    levels = np.unique(dose)
    medians = np.array([np.median(viability[dose == x]) for x in levels])
    midpoint = levels[np.argmin(np.abs(medians - 0.5))]

    def residual(p):
        prediction = expit(np.log(10.0) * p[1] * (p[0] - dose))
        return weight * (prediction - viability)

    bounds = ([dose.min() - 3, 0.15], [dose.max() + 3, 6.0])
    start = least_squares(residual, [midpoint, 1.0], bounds=bounds,
                          loss='soft_l1', f_scale=0.055)
    fit = least_squares(residual, start.x, bounds=bounds,
                        loss='cauchy', f_scale=0.027)
    return {'log10_ic50_uM': float(fit.x[0]), 'hill_slope': float(fit.x[1])}


def _initial_curves(plates, calibrations):
    """Fit shared compound curves to independently calibrated replicate plates."""
    observations = defaultdict(list)
    for plate, (background, gain, _) in zip(plates, calibrations):
        wells = plate['wells']
        signal = np.array([w['fluorescence'] for w in wells], dtype=float)
        kind = np.array([w['well_type'] for w in wells])
        for i in np.flatnonzero(kind == 'COMPOUND'):
            well = wells[i]
            observations[well['compound_id']].append((well['log10_conc_uM'],
                                                       (signal[i] - background) / gain[i],
                                                       gain[i] / np.median(gain)))
    return {compound: _hill(*np.asarray(values).T)
            for compound, values in observations.items()}


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate compound Hill parameters with plate-specific spatial calibration."""
    plate_arrays, calibrations = [], []
    for plate in plates:
        wells = plate['wells']
        r, c, signal = np.asarray([(w['row'], w['col'], w['fluorescence']) for w in wells], dtype=float).T
        kind = np.array([w['well_type'] for w in wells])
        kill, ctrl = kind == 'POS_KILL', kind == 'DMSO'
        plate_arrays.append((wells, r, c, signal, kill, ctrl))
        calibrations.append(_calibrate(r, c, signal, kill, ctrl))
    seed = _initial_curves(plates, calibrations)
    if not seed:
        return {}
    ids = list(seed)
    lookup = {name: i for i, name in enumerate(ids)}
    nplate = len(plates)
    offset = 5*nplate
    npar = offset + 3 + 2*len(ids)
    data, init = [], []
    for (wells, r, c, signal, kill, ctrl), (bg, _, spatial) in zip(plate_arrays, calibrations):
        scale = np.median(signal[ctrl] - bg)
        init.extend(spatial)
        use = np.flatnonzero(~kill)
        cid = np.array([lookup.get(wells[i]['compound_id'], 0) for i in use])
        dose = np.array([wells[i]['log10_conc_uM'] if not ctrl[i] else 0. for i in use])
        data.append((np.minimum(r[use], 15-r[use]), np.minimum(c[use], 23-c[use]),
                     (r[use]-7.5)/7.5, (c[use]-11.5)/11.5, (signal[use]-bg)/scale,
                     ctrl[use], cid, dose))
    # Share the shape, but fit each plate’s amplitude, decay and illumination.
    init += [1., .04, 1.]
    for name in ids:
        init.extend([seed[name]['log10_ic50_uM'], seed[name]['hill_slope']])
    sizes = np.cumsum([0]+[len(d[0]) for d in data])
    ln10 = np.log(10.)

    def evaluate(x, jac=False):
        power, mix, aspect = x[offset:offset+3]
        hill = x[offset+3:].reshape(-1, 2)
        result = np.zeros((sizes[-1]+3, npar)) if jac else np.empty(sizes[-1]+3)
        for j, (dr, dc, r, c, y, ctrl, cid, dose) in enumerate(data):
            base, amp, tr, tc, decay = x[5*j:5*j+5]
            cc = dc/aspect
            lo, hi = np.minimum(dr, cc)/decay, np.maximum(dr, cc)/decay
            zl, zh = lo**power, hi**power
            el, eh = np.exp(-zl), np.exp(-zh)
            per = el + mix*eh
            gain = base + amp*per + tr*r + tc*c
            mid, slope = hill[cid].T
            v = expit(ln10*slope*(mid-dose))
            # DMSO anchors the full-viability gain independently of the Hill curve.
            v[ctrl] = 1.
            start, end = sizes[j:j+2]
            if not jac:
                result[start:end] = gain*v - y
                continue
            ql, qh = el*zl, eh*zh
            result[start:end, 5*j:5*j+5] = v[:,None]*np.column_stack((np.ones_like(r), per, r, c, amp*power/decay*(ql+mix*qh)))
            result[start:end, offset:offset+3] = (v*amp)[:,None]*np.column_stack((
                -ql*np.log(np.maximum(lo, 1e-10))-mix*qh*np.log(np.maximum(hi, 1e-10)),
                eh, power/aspect*np.where(cc <= dr, ql, mix*qh)))
            d = gain*v*(1-v)*ln10
            ii = np.arange(start, end)
            jj = offset+3+2*cid
            result[ii, jj] = d*slope
            result[ii, jj+1] = d*(mid-dose)
        reg = np.array([.012, .012, .012])
        if jac:
            result[-3:, offset:offset+3] = np.diag(reg)
        else:
            result[-3:] = reg*(x[offset:offset+3]-[1, 0, 1])
        return result

    lower = [.30, 0, -.25, -.25, .45]*nplate + [.65, 0., .70] + [-8., .15]*len(ids)
    upper = [1.6, 1., .25, .25, 5.5]*nplate + [2.6, 1.15, 1.65] + [8., 6.]*len(ids)
    fit = least_squares(evaluate, init, jac=lambda x: evaluate(x, True), bounds=(lower, upper),
                        loss='cauchy', f_scale=.022, x_scale='jac', max_nfev=100)
    params = fit.x[offset+3:].reshape(-1, 2)
    return {name:{'log10_ic50_uM': float(t), 'hill_slope': float(s)} for name,(t,s) in zip(ids,params)}
