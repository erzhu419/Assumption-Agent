"""Two-branch hereditary strain with yield-gated creep and history work."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, hstack


def _align(train, test):
    """Estimate lab location and gain from reference coupons, not study coupons."""
    n = len(train['lab_id'])
    labs = np.r_[train['lab_id'], test['lab_id']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = refs[mask]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        x[mask] = (x[mask] - offset) / gain
    cells = np.column_stack((train['lab_id'], train['stratum']))
    _, cell, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weights = 1.0 / count[cell]
    observed = np.isfinite(x[:n])
    weight_sum = weights @ observed
    fill = np.divide(weights @ np.where(observed, x[:n], 0.0), weight_sum,
                     out=np.zeros(x.shape[1]), where=weight_sum > 0)
    return tuple(np.where(np.isfinite(x), x, fill)[sl] for sl in (slice(None, n), slice(n, None)))


class _Design:
    def __init__(self, split, x, dt, time, yield_stress):
        self.time = time
        self.stress = split['stress_hold_0']
        self.history = split['past_stress']
        self.dt = dt
        self.remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
        s = self.stress
        self.work = (np.maximum(np.abs(self.history) - yield_stress, 0.0)**2 @ dt) * s
        overstress = np.sign(s) * np.maximum(np.abs(s) - yield_stress, 0.0)**2
        self.drift = np.column_stack((np.ones(len(s)), x, overstress))

    def matrices(self, taus):
        # Exact zero-initialized Prony state under piecewise constant history.
        kernels = np.exp(-self.remaining[:, None] / taus) * -np.expm1(-self.dt[:, None] / taus)
        gaps = self.stress[:, None] - self.history @ kernels
        fast = np.column_stack((gaps[:, 0], self.work))
        slow = gaps[:, 1:2]
        rate = np.column_stack((self.drift, fast / taus[0], slow / taus[1]))
        f = -np.expm1(-self.time[:, None] / taus)
        curves = np.concatenate((self.drift[:, None, :] * self.time[None, :, None],
                                 fast[:, None, :] * f[None, :, :1],
                                 slow[:, None, :] * f[None, :, 1:]), axis=2)
        return curves, rate


def _fit(design, train):
    target = train['strain_curves_obs'] - train['strain_0'][:, None]
    onset = train['instant_rate_obs']
    time = design.time
    # Differences against the onset extrapolation isolate curvature and cancel
    # the coupon drift exactly, even when a coupon entry had to be imputed.
    scale = 0.035 + 0.016 * time
    def system(taus):
        curves, rate = design.matrices(taus)
        a = ((curves - time[None, :, None] * rate[:, None, :]) / scale[None, :, None]).reshape(-1, 9)
        return np.vstack((a, rate / 0.04))

    b = np.r_[((target - onset[:, None] * time) / scale).ravel(), onset / 0.04]
    candidates = []
    for a in (0.28, 0.58, 1.1, 2.1):
        for c in (0.95, 2.0, 4.0, 7.0):
            if c <= a:
                continue
            m = system(np.array([a, c]))
            coeff = np.linalg.lstsq(m, b, rcond=None)[0]
            r = m @ coeff - b
            candidates.append((np.log1p(r*r).sum(), np.r_[a, (c-a)/(8-a), coeff]))

    def unpack(p):
        return np.array((p[0], p[0] + (8.0 - p[0]) * p[1]))

    def residual(p):
        return system(unpack(p)) @ p[2:] - b

    bounds = (np.r_[0.20, 1e-6, np.full(9, -np.inf)],
              np.r_[8.0-1e-6, 1.0, np.full(9, np.inf)])
    results = []
    for _, start in sorted(candidates, key=lambda p: p[0])[:2]:
        warm = least_squares(residual, start, bounds=bounds, loss='soft_l1', f_scale=2.0, x_scale='jac', max_nfev=80)
        fit = least_squares(residual, warm.x, bounds=bounds, loss='cauchy', f_scale=2.0, x_scale='jac', max_nfev=100)
        results.append(fit)
    fit = min(results, key=lambda r: r.cost)
    return unpack(fit.x), fit.x[2:]



def _fit_joint(design, train, prior_sigma=0.018, curve_sigma=0.025, onset_sigma=0.016):
    """Fit onset and curves jointly, allowing a small coupon-specific drift."""
    taus, coef = _fit(design, train)
    t = design.time
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    onset = train['instant_rate_obs']
    n, k = y.shape
    # Standardized independent curve and onset observations; the last block
    # estimates coupon drift deviations not explained by the panel model.
    obs = np.r_[(y/curve_sigma).ravel(), onset/onset_sigma, np.zeros(n)]
    a = csr_matrix((np.r_[np.tile(t/curve_sigma,n), np.full(n,1/onset_sigma), np.full(n,1/prior_sigma)], (np.arange(n*(k+2)), np.r_[np.repeat(np.arange(n),k), np.arange(n), np.arange(n)])),shape=(n*(k+2),n))
    def un(q): return np.array([q[0], q[0]+(8-q[0])*q[1]])
    def regular(q):
        c,r = design.matrices(un(q))
        return np.vstack(((c/curve_sigma).reshape(-1,9), r/onset_sigma, np.zeros((n,9))))
    q = np.r_[taus[0], (taus[1]-taus[0])/(8-taus[0]), coef]
    def fun(p): return regular(p)@p[2:11] + a@p[11:] - obs
    def jac(p):
        b = np.zeros((len(obs),11)); b[:,2:] = regular(p)
        for j in range(2):
            delta = np.zeros(11); delta[j] = 1e-5
            b[:,j] = ((regular(p[:11]+delta)-regular(p[:11]-delta))@p[2:11])/(2e-5)
        return hstack((csr_matrix(b),a),format='csr')
    def loss(z):
        result = np.stack((z,np.ones_like(z),np.zeros_like(z)))
        for sl,c in [(slice(None,n*k),1.0),(slice(n*(k+1),None),4.0)]:
            u = z[sl]/c
            result[:,sl] = np.stack((c*np.log1p(u),1/(1+u),-1/(c*(1+u)**2)))
        return result
    res=least_squares(fun,np.r_[q,onset-design.matrices(taus)[1]@coef],jac=jac,loss=loss,bounds=(np.r_[.2,1e-6,np.full(9+n,-np.inf)],np.r_[8-1e-6,1,np.full(9+n,np.inf)]),x_scale='jac',max_nfev=60,tr_options={'atol':1e-9,'btol':1e-9})
    return un(res.x), res.x[2:11]

def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, x_test = _align(train, test)
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    y = float(dataset['yield_stress'])
    taus, coef = _fit_joint(_Design(train, x_train, dt, time, y), train)
    design = _Design(test, x_test, dt, time, y)
    curves, rate = design.matrices(taus)
    # Shrink small onset discrepancies toward the calibrated coupon drift.
    # Keep the measured onset when the coupon is incomplete or disagrees strongly.
    correction = test['instant_rate_obs'] - rate @ coef
    complete = np.isfinite(test['coupon_features_raw']).all(axis=1)
    shrinkage = (0.016**2 / (0.018**2 + 0.016**2)) / (1 + (correction / 0.036)**2)
    correction = correction * (1 - np.where(complete, shrinkage, 0.0))
    prediction = test['strain_0'][:, None] + curves @ coef + correction[:, None] * time
    return {'estimated_taus': taus, 'viscoplastic_weights': coef[[5, 7]],
            'predicted_strain': prediction}
