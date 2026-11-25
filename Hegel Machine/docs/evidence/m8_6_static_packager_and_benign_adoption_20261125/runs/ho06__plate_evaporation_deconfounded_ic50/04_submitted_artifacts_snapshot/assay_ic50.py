"""Control-referenced Hill fitting for the 384-well viability assay."""
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def _spatial_gain(plate, row, col, signal, dmso, replicate_pairs):
    """Fit a bounded perimeter response and illumination tilt on each plate."""
    r = (row - 7.5) / 7.5
    c = (col - 11.5) / 11.5
    u, v = np.minimum(row, 15 - row), np.minimum(col, 23 - col)
    d = np.minimum(u, v)
    left, right = np.asarray(replicate_pairs, dtype=int).reshape(-1, 2).T
    strength = np.minimum(signal[left], signal[right])
    use = (strength > 0.26) & (np.abs(np.log(np.maximum(signal[left], .01) / np.maximum(signal[right], .01))) < .38)
    left, right, strength = left[use], right[use], strength[use]
    ratio = np.log(signal[left] / signal[right])
    gain = np.empty_like(signal)
    for i in np.unique(plate):
        at = np.flatnonzero(plate == i)
        controls = np.flatnonzero(dmso[at])
        pair = plate[left] == i
        a, b = np.searchsorted(at, left[pair]), np.searchsorted(at, right[pair])
        rr, cc, uu, vv, dd = r[at], c[at], u[at], v[at], d[at]
        z = signal[at]
        def predict(q):
            base, tr, tc, amp, ir, ic, width, mix = q
            nearest = np.exp(-dd / width)
            sides = np.exp(-uu / width) + np.exp(-vv / width) - np.exp(-(uu+vv) / width)
            edge = nearest + mix * (sides - nearest)
            return base + tr*rr + tc*cc + (amp + ir*rr + ic*cc)*edge
        bounds = ([.65,-.14,-.14,0.,-.10,-.10,.5,0.], [1.20,.14,.14,.65,.10,.10,4.5,1.0])
        center = np.clip(np.median(z[controls]) - .08,.66,1.19)
        def residual(q):
            g = predict(q)
            control_err = (g[controls]-z[controls])/.010
            contrast = (np.log(g[a]/g[b])-ratio[pair])*strength[pair]/.014
            prior = np.r_[q[4:6]/.055, np.log(q[6]/1.9)/.65, q[7]/1.25]
            return np.r_[control_err,contrast,prior]
        warm = least_squares(residual,[center,0,0,.30,0,0,1.9,.2],bounds=bounds,loss='soft_l1',f_scale=1.5).x
        final = least_squares(residual,warm,bounds=bounds,loss='cauchy',f_scale=1.5).x
        gain[at] = predict(final)
    return gain


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Return each assayed compound's log10 IC50 (µM) and positive Hill slope."""
    records = [
        (i, w['row'], w['col'], w['well_type'], w['fluorescence'])
        for i, p in enumerate(plates) for w in p['wells']
    ]
    plate, row, col, well_type, readout = map(np.asarray, zip(*records))
    dmso = well_type == 'DMSO'
    kill = well_type == 'POS_KILL'
    signal = np.empty(len(plate), dtype=float)
    for i in range(len(plates)):
        mask = plate == i
        background = np.median(readout[mask & kill])
        scale = np.median(readout[mask & dmso]) - background
        signal[mask] = (readout[mask] - background) / scale

    compounds = defaultdict(list)
    cells = defaultdict(list)
    j = 0
    for i, p in enumerate(plates):
        for w in p['wells']:
            if w['well_type'] == 'COMPOUND':
                cid = w['compound_id']
                dose = float(w['log10_conc_uM'])
                compounds[cid].append((j, dose))
                cells[i, cid, dose].append(j)
            j += 1
    pairs = [indices for indices in cells.values() if len(indices) == 2]
    gain = _spatial_gain(plate, row, col, signal, dmso, pairs)

    result = {}
    for cid, observations in compounds.items():
        idx, dose = map(np.asarray, zip(*observations))
        y, g = signal[idx], gain[idx]
        levels, group = np.unique(dose, return_inverse=True)
        typical = np.array([np.median(y[group == i] / g[group == i]) for i in range(len(levels))])
        middle = levels[np.argmin(np.abs(typical - 0.5))]

        def residual(params):
            ic50, hill = params
            return g * expit(np.log(10.0) * hill * (ic50 - dose)) - y

        # Fit in fluorescence coordinates so edge normalization does not inflate noise.
        bounds = ([levels[0] - 4.0, 0.05], [levels[-1] + 4.0, 8.0])
        warm = least_squares(lambda v: residual(v) / 0.025, [middle, 1.0],
                             bounds=bounds, loss='soft_l1').x
        ic50, hill = least_squares(lambda v: residual(v) / 0.015, warm,
                                  bounds=bounds, loss='cauchy').x
        result[cid] = {'log10_ic50_uM': float(ic50), 'hill_slope': float(hill)}
    return result
