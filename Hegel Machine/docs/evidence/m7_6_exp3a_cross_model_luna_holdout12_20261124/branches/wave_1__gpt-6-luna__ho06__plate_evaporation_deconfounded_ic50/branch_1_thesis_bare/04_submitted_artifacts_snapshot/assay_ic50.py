import numpy as np
from scipy.optimize import least_squares


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    ids = sorted({w['compound_id'] for p in plates for w in p['wells'] if w.get('well_type') == 'COMPOUND'})
    ci = {k: i for i, k in enumerate(ids)}
    rows, values, types, cidx, doses, pidx = [], [], [], [], [], []
    for pi, plate in enumerate(plates):
        for w in plate['wells']:
            r, c = w['row'] / 15.0, w['col'] / 23.0
            rows.append([1.0, 4*r*(1-r), 4*c*(1-c), (r-.5)*(c-.5), (r-.5)**2, (c-.5)**2, (4*r*(1-r))**2, (4*c*(1-c))**2])
            values.append(float(w['fluorescence']))
            types.append(w.get('well_type'))
            pidx.append(pi)
            if w.get('well_type') == 'COMPOUND':
                cidx.append(ci[w['compound_id']])
                doses.append(float(w['log10_conc_uM']))
            else:
                cidx.append(-1)
                doses.append(0.0)
    R, Y = np.asarray(rows), np.asarray(values)
    typ, pidx, cidx, dose = np.asarray(types), np.asarray(pidx), np.asarray(cidx), np.asarray(doses)
    npl, stride = len(plates), 9
    q0 = np.zeros(npl * stride + 2 * len(ids))
    for j, plate in enumerate(plates):
        kill = [w['fluorescence'] for w in plate['wells'] if w.get('well_type') == 'POS_KILL']
        dmso = [w['fluorescence'] for w in plate['wells'] if w.get('well_type') == 'DMSO']
        b = float(np.median(kill))
        q0[stride*j] = b
        q0[stride*j+1] = np.log(max(float(np.median(dmso)) - b, 1.0))
    q0[npl*stride::2] = np.linspace(-2.0, 1.7, len(ids))
    q0[npl*stride+1::2] = 1.0
    ctrl = np.where(typ == 'DMSO', 1.0, np.where(typ == 'POS_KILL', 0.0, np.nan))

    def pred(q):
        out = np.empty(len(Y))
        for j in range(npl):
            ix = pidx == j
            b, amp = q[stride*j], np.exp(q[stride*j+1])
            g = np.exp(np.clip(R[ix, 1:] @ q[stride*j+2:stride*j+stride], -0.5, 0.5))
            v = ctrl[ix].copy()
            cc = cidx[ix] >= 0
            if np.any(cc):
                h = q[npl*stride + 2*cidx[ix][cc]]
                slope = q[npl*stride + 2*cidx[ix][cc] + 1]
                z = np.clip(slope * (dose[ix][cc] - h) * np.log(10.0), -35, 35)
                v[cc] = 1.0 / (1.0 + np.exp(z))
            out[ix] = b + amp * g * v
        return out

    scale = np.maximum(np.abs(Y) * 0.015, 8.0)
    lb = np.r_[np.tile([0, np.log(100), -1, -1, -1, -1, -1, -1, -1], npl), np.tile([-4, .3], len(ids))]
    ub = np.r_[np.tile([1000, np.log(10000), 1, 1, 1, 1, 1, 1, 1], npl), np.tile([3, 3], len(ids))]
    fit = least_squares(lambda q: (pred(q) - Y) / scale, q0, loss='soft_l1', f_scale=1.0, max_nfev=3000, bounds=(lb, ub))
    return {k: {'log10_ic50_uM': float(fit.x[npl*stride+2*i]), 'hill_slope': float(fit.x[npl*stride+2*i+1])} for k, i in ci.items()}
