"""Estimate Hill curves using spatially calibrated 384-well fluorescence."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


_LN10 = np.log(10.0)


def _coordinates(wells):
    row, col, signal = np.array([
        (w['row'], w['col'], w['fluorescence']) for w in wells
    ], dtype=float).T
    distance = np.minimum.reduce([row, 15 - row, col, 23 - col])
    return distance, (row - 7.5) / 7.5, (col - 11.5) / 11.5, signal


def _design(distance, row, col, decay):
    return np.column_stack((np.ones_like(row), np.exp(-distance / decay), row, col))


def _calibrate(plates):
    prepared = []
    controls = []
    for plate in plates:
        wells = plate['wells']
        d, r, c, signal = _coordinates(wells)
        kinds = np.array([w['well_type'] for w in wells])
        bg = np.median(signal[kinds == 'POS_KILL'])
        dmso = kinds == 'DMSO'
        scale = np.median(signal[dmso]) - bg
        y = (signal - bg) / scale
        prepared.append((wells, d, r, c, y))
        controls.append((d[dmso], r[dmso], c[dmso], y[dmso]))

    def residual(params):
        decay = params[0]
        return np.concatenate([
            _design(d, r, c, decay) @ beta - y
            for (d, r, c, y), beta in zip(controls, params[1:].reshape(-1, 4))
        ])

    # Share the decay length, not the baseline, gain, or illumination coefficients.
    start = [2.0] + [v for _ in controls for v in (0.9, 0.3, 0.0, 0.0)]
    lo = [0.35] + [v for _ in controls for v in (0.05, 0.0, -np.inf, -np.inf)]
    hi = [8.0] + [np.inf] * (4 * len(controls))
    fit = least_squares(residual, start, bounds=(lo, hi), loss='soft_l1', f_scale=0.012)
    fit = least_squares(residual, fit.x, bounds=(lo, hi), loss='cauchy', f_scale=0.018)
    return prepared, fit.x


def _fit_hill(log_dose, viability):
    def residual(params):
        mid, slope = params
        return expit(_LN10 * slope * (mid - log_dose)) - viability

    doses = np.unique(log_dose)
    medians = [np.median(viability[log_dose == x]) for x in doses]
    start = [doses[np.argmin(np.abs(np.array(medians) - 0.5))], 1.0]
    bounds = ([doses[0] - 4.0, 0.03], [doses[-1] + 4.0, 8.0])
    params = least_squares(residual, start, bounds=bounds, loss='soft_l1', f_scale=0.035).x
    params = least_squares(residual, params, bounds=bounds, loss='cauchy', f_scale=0.022).x
    err = residual(params)
    sigma = 1.4826 * np.median(np.abs(err - np.median(err)))
    keep = np.abs(err) < max(0.04, 4.0 * sigma)
    if np.unique(log_dose[keep]).size >= 3:
        params = least_squares(lambda p: residual(p)[keep], params, bounds=bounds,
                               loss='soft_l1', f_scale=0.022).x
    return {'log10_ic50_uM': float(params[0]), 'hill_slope': float(params[1])}


def estimate_compound_hill_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate each compound's log10 IC50 in µM and positive Hill slope."""
    if not plates:
        return {}
    # DMSO anchors spatial gain; POS_KILL independently anchors background.
    prepared, cal = _calibrate(plates)
    ids = sorted({w['compound_id'] for p in plates for w in p['wells'] if w['well_type']=='COMPOUND'})
    index = {k:i for i,k in enumerate(ids)}
    ds, rs, cs, ys, pages, codes, doses = [], [], [], [], [], [], []
    for j,(wells,d,r,c,y) in enumerate(prepared):
        use = np.array([w['well_type']!='POS_KILL' for w in wells])
        ds.extend(d[use]); rs.extend(r[use]); cs.extend(c[use]); ys.extend(y[use]); pages.extend([j]*sum(use))
        for w in np.asarray(wells, dtype=object)[use]:
            codes.append(index.get(w['compound_id'],-1)); doses.append(w['log10_conc_uM'] if codes[-1]>=0 else 0.)
    d,r,c,y,x = map(np.array,(ds,rs,cs,ys,doses))
    p,k = np.array(pages), np.array(codes)
    treated = k>=0
    kt,xt = k[treated],x[treated]
    base = 2+4*len(plates)
    ar = np.arange(len(y)); at = ar[treated]
    pp = 2+4*p
    ones = np.ones_like(d)
    vi = y/(cal[1:].reshape(-1,4)[p]*np.column_stack((ones,np.exp(-d/cal[0]),r,c))).sum(1)
    par = [_fit_hill(x[k==j],vi[k==j]) for j in range(len(ids))]
    # Refine the gain and curves together in fluorescence space. Replicated
    # compound doses constrain the profile where DMSO coverage is sparse.
    start = np.r_[cal[0],.05,cal[1:],[a[field] for a in par for field in ('log10_ic50_uM', 'hill_slope')]]
    lo = [.35,0]+[.05,0,-np.inf,-np.inf]*len(plates)+[xt.min()-4,.03]*len(ids)
    hi = [8,1]+[np.inf]*(4*len(plates))+[xt.max()+4,8]*len(ids)
    def components(q):
        t,mix=q[:2]; u=d/t; ex=np.exp(-u); ra=1/(1+u*u)
        edge=ex+mix*(ra-ex)
        beta=q[1+pp]
        gain=q[pp]+beta*edge+q[2+pp]*r+q[3+pp]*c
        mid,slope=q[base+2*kt],q[base+2*kt+1]
        v=ones.copy(); v[treated]=expit(_LN10*slope*(mid-xt))
        return gain,v,edge,ex,ra,u,beta,mid,slope
    def fun(q):
        gain,v,*_=components(q)
        return gain*v-y
    def jac(q):
        gain,v,edge,ex,ra,u,beta,mid,slope=components(q)
        J=np.zeros((len(y),len(q)))
        J[:,0]=beta*v/q[0]*((1-q[1])*u*ex+q[1]*2*u*u*ra*ra)
        J[:,1]=beta*v*(ra-ex)
        J[ar,pp]=v; J[ar,1+pp]=v*edge; J[ar,2+pp]=v*r; J[ar,3+pp]=v*c
        common=(gain*v*(1-v))[treated]*_LN10
        J[at,base+2*kt]=common*slope; J[at,base+2*kt+1]=common*(mid-xt)
        return J
    fit=least_squares(fun,start,jac=jac,bounds=(lo,hi),loss='cauchy',f_scale=.024,max_nfev=150)
    q=fit.x
    return {name:{'log10_ic50_uM':float(q[base+2*j]),'hill_slope':float(q[base+2*j+1])} for j,name in enumerate(ids)}
