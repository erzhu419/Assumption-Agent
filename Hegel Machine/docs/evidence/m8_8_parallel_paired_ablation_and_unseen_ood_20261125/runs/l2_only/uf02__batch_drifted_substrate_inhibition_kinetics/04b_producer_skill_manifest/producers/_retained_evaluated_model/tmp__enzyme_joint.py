import numpy as np
from scipy.optimize import least_squares
from enzyme_plate_solver import _basis, _calibrate, _raw, estimate_variant_haldane_parameters as baseline

KEYS = ('V_max', 'K_m', 'K_i')

def estimate(plates, proportional=0., loss="soft_l1", f_scale=1.5, pool=0.):
    start = baseline(plates)
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
    sigma = np.exp((1-pool)*np.log(sigma) + pool*np.log(np.median(sigma)))
    weight = 1/np.hypot(sigma, proportional*g*v)

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
    fit = least_squares(fun, initial, jac=jac, bounds=(lower,upper), x_scale='jac', loss=loss, f_scale=f_scale, max_nfev=80)
    return {v:dict(zip(KEYS,map(float,np.exp(fit.x[nplate+3*i:nplate+3*i+3])))) for v,i in ids.items()}
