"""Experimental joint fit of plate gains and compound Hill curves."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
import assay_ic50 as baseline


def joint(plates):
    seed = baseline.estimate_compound_hill_parameters(plates)
    ids = list(seed)
    lookup = {name: i for i, name in enumerate(ids)}
    nplate = len(plates)
    offset = 5*nplate
    npar = offset + 3 + 2*len(ids)
    data, init = [], []
    for plate in plates:
        wells = plate['wells']
        r, c, signal = np.asarray([(w['row'], w['col'], w['fluorescence']) for w in wells]).T
        kind = np.array([w['well_type'] for w in wells])
        kill, ctrl = kind == 'POS_KILL', kind == 'DMSO'
        bg, g = baseline._calibrate(r, c, signal, kill, ctrl)
        scale = np.median(signal[ctrl] - bg)
        def surface_res(x):
            return baseline._surface(r, c, x[4]) @ x[:4] - g / scale
        init.extend(least_squares(surface_res, [0.85, .3, 0, 0, 1.8],
                                 bounds=([.35, 0, -.25, -.25, .5], [1.5, 1, .25, .25, 5])).x)
        use = np.flatnonzero(~kill)
        cid = np.array([lookup.get(wells[i]['compound_id'], 0) for i in use])
        dose = np.array([wells[i]['log10_conc_uM'] if not ctrl[i] else 0. for i in use])
        data.append((np.minimum(r[use], 15-r[use]), np.minimum(c[use], 23-c[use]),
                     (r[use]-7.5)/7.5, (c[use]-11.5)/11.5, (signal[use]-bg)/scale,
                     ctrl[use], cid, dose))
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
