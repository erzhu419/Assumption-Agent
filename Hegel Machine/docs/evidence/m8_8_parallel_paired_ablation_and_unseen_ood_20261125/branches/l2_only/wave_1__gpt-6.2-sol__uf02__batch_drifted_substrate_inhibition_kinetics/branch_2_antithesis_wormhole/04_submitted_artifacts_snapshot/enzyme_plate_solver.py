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
    scale = max(_mad(np.r_[blanks-offset, x @ gain-y]), floor)
    return offset, gain, scale


def _initial_kinetics(plates):
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
        results[variant] = dict(zip(('V_max', 'K_m', 'K_i'), map(float, np.exp(fit.x))))
    return results


KEYS = ('V_max', 'K_m', 'K_i')

def estimate_variant_haldane_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Fit shared positive Haldane parameters and per-plate optical terms."""
    start = _initial_kinetics(plates)
    ids = {v:i for i,v in enumerate(start)}
    if not ids:
        return {}
    params = []
    plate_index, variant_index, x, s, rate, y, sigma = [], [], [], [], [], [], []
    for i, plate in enumerate(plates):
        wells = plate['wells']
        offset, gain, scale = _calibrate(wells)
        params.extend([offset, *gain])
        for w in wells:
            kind = w['well_type']
            plate_index.append(i)
            variant_index.append(ids[w['variant_id']] if kind == 'VARIANT' else -1)
            x.append(_basis(w['row'], w['col']))
            s.append(w['substrate_mM'])
            rate.append(float(kind == 'REF_WT'))
            y.append(_raw(w))
            sigma.append(scale)
    nplate = len(params)
    params.extend(np.log([start[v][k] for v in ids for k in KEYS]))
    plate_index, variant_index, x, s, rate, y, sigma = map(np.asarray, (plate_index, variant_index, x, s, rate, y, sigma))
    optical = 7*plate_index
    mask = variant_index >= 0
    ii = np.arange(y.size)
    jj = ii[mask]
    kinetic = nplate + 3*variant_index[mask]
    sv = s[mask]

    def model(p):
        vmax, km, ki = np.exp(p[kinetic[:, None] + np.arange(3)]).T
        inhib = sv**2/ki
        denom = km + sv + inhib
        v = rate.copy()
        v[mask] = vmax*sv/denom
        g = np.sum(x*p[optical[:, None] + np.arange(1,7)], axis=1)
        return v, g, -km/denom, inhib/denom

    v, g, _, _ = model(np.asarray(params))
    # Freeze noise weights at the control-calibrated starting prediction.
    weight = 1/np.hypot(sigma, 0.012*g*v)

    def fun(p):
        v, g, _, _ = model(p)
        return (p[optical]+g*v-y)*weight

    def jac(p):
        v, g, km, ki = model(p)
        j = np.zeros((len(y),len(params)))
        j[ii,optical] = weight
        j[ii[:,None], optical[:,None] + np.arange(1,7)] = x * (weight*v)[:,None]
        j[jj[:,None], kinetic[:,None]+np.arange(3)] = (weight*g*v)[mask,None]*np.column_stack((np.ones_like(km),km,ki))
        return j

    initial = np.asarray(params)
    lower, upper = np.full_like(initial,-np.inf), np.full_like(initial,np.inf)
    lower[nplate:], upper[nplate:] = initial[nplate:]-5, initial[nplate:]+5
    fit = least_squares(fun, initial, jac=jac, bounds=(lower,upper), x_scale='jac', loss='cauchy', f_scale=3.0, max_nfev=80)
    return {v:dict(zip(KEYS,map(float,np.exp(fit.x[nplate+3*i:nplate+3*i+3])))) for v,i in ids.items()}
