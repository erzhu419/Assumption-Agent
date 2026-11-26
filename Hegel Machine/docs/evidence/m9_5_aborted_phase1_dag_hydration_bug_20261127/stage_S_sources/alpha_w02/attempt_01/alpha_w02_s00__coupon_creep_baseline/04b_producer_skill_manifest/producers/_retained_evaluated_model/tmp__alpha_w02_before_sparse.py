"""Robust calibration and two-branch sub-yield creep identification."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x, y, prior):
    """Fit an affine correction using consensus over pairs of spot checks."""
    if not len(x):
        return prior
    # Fit corrections rather than the dominant identity component.
    dy = y - x
    a, b = prior[0] - 1, prior[1]
    if len(x) == 1:
        return a + 1, dy[0] - a*x[0]
    i, j = np.triu_indices(len(x), 1)
    keep = np.abs(x[i] - x[j]) >= max(0.04, 0.12*np.ptp(x))
    i, j = i[keep], j[keep]
    if not len(i):
        return a + 1, np.median(dy - a*x)
    slopes = np.r_[a, (dy[j] - dy[i])/(x[j] - x[i])]
    offsets = np.r_[np.median(dy - a*x), dy[i] - slopes[1:]*x[i]]
    err = abs(dy - slopes[:, None]*x - offsets[:, None])
    consensus = err < 0.0007
    score = consensus.sum(axis=1)
    loss = np.minimum(err/0.0007, 1).sum(axis=1)
    win = np.lexsort((np.abs(slopes-a), loss, -score))[0]
    m = consensus[win]
    if m.sum() < 2 or np.ptp(x[m]) < 0.04:
        return a + 1, np.median(dy - a*x)
    if len(x) <= 3 and m.sum() < len(x):
        # Too few checks to identify a new slope and reject a glitch independently.
        return a + 1, np.median(dy - a*x)
    u = x[m] - x[m].mean()
    slope = np.dot(u, dy[m])/np.dot(u, u)
    return 1 + slope, dy[m].mean() - slope*x[m].mean()


def _calibration(d):
    c = d['calibration_load_cells']
    lab, x, y = map(np.asarray, (c['lab_id'], c['cmd_stress'], c['true_stress']))
    finite = np.isfinite(x) & np.isfinite(y)
    out = np.empty((4, 2))
    for l in range(4):
        m = finite & (lab == l)
        out[l] = _affine(x[m], y[m], (1., 0.))
    return out


def _align(d):
    tr, te = d['train'], d['test']
    features, ref, lab, stratum = [np.concatenate((tr[k], te[k])) for k in
        ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')]
    x = features.astype(float, copy=True)
    observed = np.isfinite(x)
    for l in np.unique(lab):
        m = lab == l
        r = ref[m]
        center = np.nanmedian(np.nanmean(r, axis=0))
        scale = np.nanmedian(np.nanstd(r, axis=0, ddof=0))
        x[m] = (x[m] - center)/scale
    _, cells = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    counts = np.bincount(cells)
    # Average available values within each cell, then weight cells equally.
    cell_means = np.full((len(counts), x.shape[1]), np.nan)
    for cell in range(len(counts)):
        v = x[cells == cell]
        n = np.isfinite(v).sum(axis=0)
        np.divide(np.nansum(v, axis=0), n, out=cell_means[cell], where=n>0)
    x = np.where(observed, x, np.nanmean(cell_means, axis=0))
    n = len(tr['lab_id'])
    return x[:n], x[n:], observed[:n].all(axis=1)


def _state_inputs(split, calibration):
    a, b = calibration[np.asarray(split['lab_id'])].T
    hist = a[:, None]*split['past_stress_cmd'] + b[:, None]
    hold = a*split['stress_hold_cmd'] + b
    return hist, hold


def _basis(tau, hist, hold, dt, times, x):
    dt = np.asarray(dt)
    after = np.cumsum(dt[::-1])[::-1] - dt
    coeff = -np.expm1(-dt[:, None]/tau)*np.exp(-after[:, None]/tau)
    diff = hold[:, None] - hist @ coeff
    t = np.asarray(times)
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((diff[:, None, :] * (-np.expm1(-t[:, None]/tau))[None, :, :],
                           drift[:, None, :]*t[None, :, None]), axis=2)
    rate = np.column_stack((diff/tau, drift))
    return curve, rate


def _noise(r):
    return max(1e-9, 1.4826*np.median(np.abs(r - np.median(r))))


def _fit(hist, hold, dt, t, x, curve, rate):
    k = len(t)
    yc = curve.ravel()
    mc, mr = np.isfinite(yc), np.isfinite(rate)
    yc, yr = yc[mc], rate[mr]
    y = np.r_[yc, yr]
    n = len(yc)

    def design(tau):
        bc, br = _basis(tau, hist, hold, dt, t, x)
        return np.vstack((bc.reshape(-1, 7)[mc], br[mr]))

    def search(scale, starts):
        def residual(p):
            return (design(p[:2]) @ p[2:] - y)/scale
        best = None
        lo = np.r_[0.2, 0.2, np.full(7, -np.inf)]
        hi = np.r_[8., 8., np.full(7, np.inf)]
        for p in starts:
            fit = least_squares(residual, p, bounds=(lo, hi), loss='cauchy',
                                x_scale='jac', max_nfev=110, ftol=2e-9)
            if best is None or fit.cost < best.cost:
                best = fit
        return best.x

    starts = []
    for tau in ([0.35, 2.4], [0.7, 4.0], [1.3, 6.5]):
        b = design(np.asarray(tau))
        w = np.linalg.lstsq(b, y, rcond=None)[0]
        starts.append(np.r_[tau, w])
    r0 = design(starts[1][:2]) @ starts[1][2:] - y
    def scales(r, multiple):
        sc = _noise(r[:n])
        sr = _noise(r[n:]) if len(yr) else sc
        return multiple*np.r_[np.full(n, sc), np.full(len(yr), sr)]
    p = search(scales(r0, 1.7), starts)
    r = design(p[:2]) @ p[2:] - y
    p = search(scales(r, 2.0), [p])
    if p[0] > p[1]:
        p[:4] = p[[1, 0, 3, 2]]
    return p[:2], p[2:]


def solve_alpha_w02_s00(dataset: dict) -> dict:
    tr, te = dataset['train'], dataset['test']
    dt, t = dataset['past_dt'], dataset['eval_times']
    cal = _calibration(dataset)
    x, xt, complete = _align(dataset)
    hist, hold = _state_inputs(tr, cal)
    hist_t, hold_t = _state_inputs(te, cal)
    # Do not fit material parameters to proxies for missing coupon covariates.
    taus, weights = _fit(hist[complete], hold[complete], dt, t, x[complete],
                        (tr['strain_curves_obs'] - tr['strain_0'][:, None])[complete],
                        tr['instant_rate_obs'][complete])
    b, v = _basis(taus, hist_t, hold_t, dt, t, xt)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': te['strain_0'][:, None] + b @ weights,
            'initial_rate_pred': v @ weights}
