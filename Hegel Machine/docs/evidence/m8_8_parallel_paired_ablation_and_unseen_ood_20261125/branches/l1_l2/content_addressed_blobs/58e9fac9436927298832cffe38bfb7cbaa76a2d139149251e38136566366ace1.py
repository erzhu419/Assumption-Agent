"""Control-anchored inference for substrate-inhibition assay plates."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix


def _mad(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _surface(wells):
    r = (np.array([w['row'] for w in wells]) - 3.5) / 3.5
    c = (np.array([w['col'] for w in wells]) - 5.5) / 5.5
    return np.column_stack((np.ones(len(r)), r, c, r*r, r*c, c*c))


def _calibrate(f, y, blank, ref):
    b = np.median(y[blank])
    fr, yr = f[ref], y[ref]
    q = np.linalg.lstsq(fr, yr, rcond=None)[0]
    scale = max(_mad(y[blank]), _mad(fr @ q - yr), abs(q[0] - b) * 1e-4, 1e-9)
    res = lambda a: fr @ a - yr
    q = least_squares(res, q, jac=lambda a: fr.copy(), loss='soft_l1', f_scale=scale).x
    scale = max(_mad(res(q)), 0.5 * _mad(y[blank]), abs(q[0] - b) * 1e-4, 1e-9)
    q = least_squares(res, q, jac=lambda a: fr.copy(), loss='cauchy', f_scale=2.5 * scale).x
    return b, q, scale


def _kinetics(s, logp):
    v, km, ki = np.exp(logp.T)
    inhibition = s*s / ki
    den = km + s + inhibition
    rate = v*s / den
    return rate, np.stack((rate, -rate*km/den, rate*inhibition/den), axis=-1)


def _fit(plates, joint=True):
    names = sorted({w['variant_id'] for p in plates for w in p['wells']
                    if w['well_type'] == 'VARIANT'})
    if not names:
        return {}
    index = {v: j for j, v in enumerate(names)}
    plate_coef, fs, ys, ss, ps, ids, scales = [], [], [], [], [], [], []
    for i, plate in enumerate(plates):
        wells = plate['wells']
        f = _surface(wells)
        y = np.array([w['raw_rate'] for w in wells], dtype=float)
        s = np.array([w['substrate_mM'] for w in wells], dtype=float)
        variant = np.array([index[w['variant_id']] if w['well_type'] == 'VARIANT'
                            else -1 if w['well_type'] == 'REF_WT' else -2 for w in wells])
        b, q, scale = _calibrate(f, y, (variant == -2) & (s == 0), variant == -1)
        plate_coef.append(np.r_[b, q])
        fs.append(f); ys.append(y); ss.append(s); ids.append(variant)
        ps.extend([i] * len(wells)); scales.append(scale)
    f, y, s, ids = map(np.concatenate, (fs, ys, ss, ids))
    ps = np.asarray(ps)
    plate_coef = np.asarray(plate_coef)
    # Stabilize noisy scale estimates from the small per-plate reference sets.
    scales = np.asarray(scales)
    sigma = np.sqrt(scales * np.median(scales))[ps]
    nplate = len(plates)
    mask = ids >= 0
    rate = np.where(ids == -1, 1.0, 0.0)
    bg = plate_coef[ps, 0]
    gain = np.einsum('ij,ij->i', f, plate_coef[ps, 1:]) - bg
    kinetic, lower, upper = [], [], []
    for k in range(len(names)):
        m = ids == k
        sk, gk, bk, yk, ek = s[m], gain[m], bg[m], y[m], sigma[m]
        lo, hi = min(sk[sk > 0]), max(sk)
        typical = np.quantile((yk-bk)/gk, 0.8)
        typical = max(typical, 0.01)
        bounds = (np.log([typical * 1e-4, lo * 1e-4, lo * 1e-4]),
                  np.log([typical * 1e5, hi * 1e4, hi * 1e4]))
        lower.extend(bounds[0]); upper.extend(bounds[1])
        start = np.log([2*typical, np.sqrt(lo*hi)/3, np.sqrt(lo*hi)*6])
        def residual(a):
            return (bk + gk*_kinetics(sk, a)[0] - yk) / ek
        def jac(a):
            return (gk/ek)[:, None]*_kinetics(sk, a)[1]
        a = least_squares(residual, start, jac=jac, bounds=bounds,
                          loss='soft_l1', f_scale=2., max_nfev=100).x
        a = least_squares(residual, a, jac=jac, bounds=bounds,
                          loss='cauchy', f_scale=2.5, max_nfev=100).x
        kinetic.extend(a)

    if joint:
        # Parameterize the plate intercept as the raw reference intercept.
        # Then each measurement is (1-rate)*background + rate*raw_reference.
        start = np.r_[plate_coef.ravel(), kinetic]
        nv = 7*nplate
        bounds = (np.r_[np.full(nv, -np.inf), lower], np.r_[np.full(nv, np.inf), upper])
        rows = np.r_[np.repeat(np.arange(len(y)), 7), np.repeat(np.flatnonzero(mask), 3)]
        cols = np.r_[(7*ps[:, None] + np.arange(7)).ravel(),
                     (nv + 3*ids[mask, None] + np.arange(3)).ravel()]
        def evaluate(a):
            p = a[:nv].reshape(-1, 7)[ps]
            b = p[:, 0]
            g = np.einsum('ij,ij->i', f, p[:, 1:]) - b
            v = rate.copy()
            v[mask], dv = _kinetics(s[mask], a[nv:].reshape(-1, 3)[ids[mask]])
            return b + g*v, g, v, dv
        def rfun(a):
            return (evaluate(a)[0]-y)/sigma
        def jfun(a):
            _, g, v, dv = evaluate(a)
            data = np.r_[(np.column_stack((1-v, v[:, None]*f))/sigma[:, None]).ravel(),
                         ((g[mask]/sigma[mask])[:, None]*dv).ravel()]
            j = csr_matrix((data, (rows, cols)), shape=(len(y), len(a)))
            return j.toarray() if len(a) <= 80 else j
        opts = dict(bounds=bounds, x_scale='jac', max_nfev=100)
        a = least_squares(rfun, start, jac=jfun, loss='cauchy', f_scale=2.5, **opts).x
        # Remove gross errors rather than sacrificing the precision of the inliers.
        r = rfun(a)
        threshold = 4.5 * max(0.8, _mad(r))
        keep = np.abs(r) <= threshold
        a = least_squares(lambda x: rfun(x)[keep], a, jac=lambda x: jfun(x)[keep],
                          loss='soft_l1', f_scale=3., **opts).x
        kinetic = a[nv:]
    return {name: dict(zip(('V_max', 'K_m', 'K_i'), map(float, val)))
            for name, val in zip(names, np.exp(np.asarray(kinetic).reshape(-1, 3)))}


def estimate_variant_haldane_parameters(plates: list[dict]) -> dict[str, dict[str, float]]:
    """Estimate shared variant kinetics after calibrating each plate's optics."""
    return _fit(plates)
