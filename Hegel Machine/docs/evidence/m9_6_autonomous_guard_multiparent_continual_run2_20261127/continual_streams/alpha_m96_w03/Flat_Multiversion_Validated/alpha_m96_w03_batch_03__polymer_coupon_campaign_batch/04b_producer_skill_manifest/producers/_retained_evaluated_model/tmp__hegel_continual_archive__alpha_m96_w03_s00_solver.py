"""Calibrated two-branch hereditary creep for sub-yield coupon holds."""
import numpy as np
from scipy.optimize import least_squares


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _calibrate(checks):
    gain = np.ones(4)
    bias = np.zeros(4)
    for i in range(4):
        m = np.asarray(checks['lab_id']) == i
        x = np.asarray(checks['cmd_stress'])[m]
        y = np.asarray(checks['true_stress'])[m]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        a, b = np.triu_indices(len(x), 1)
        sep = np.abs(x[a] - x[b]) > 1e-10
        a, b = a[sep], b[sep]
        if not len(a):
            if len(x):
                bias[i] = np.median(y - x)
            continue
        g = np.median((y[a] - y[b]) / (x[a] - x[b]))
        r = y - g*x
        keep = np.abs(r - np.median(r)) <= max(4 * _mad(r), 1e-8)
        if np.ptp(x[keep]) > 1e-10:
            gain[i], bias[i] = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
        else:
            gain[i], bias[i] = g, np.median(r)
    return gain, bias


def _align(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    out = np.empty_like(raw, dtype=float)
    for i in np.unique(lab):
        m = lab == i
        ref_i = ref[m]
        ref_i = ref_i[:, np.isfinite(ref_i).sum(axis=0) >= 2]
        loc = np.median(np.nanmean(ref_i, axis=0))
        scale = np.median(np.nanstd(ref_i, axis=0, ddof=0))
        out[m] = (raw[m] - loc) / scale
    # Each observed (lab, stratum) cell has one vote in the pooled fallback.
    cells = [(lab == i) & (stratum == s) for i, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    finite = np.isfinite(out)
    means = []
    for m in cells:
        count = finite[m].sum(axis=0)
        means.append(np.divide(np.nansum(out[m], axis=0), count, out=np.full(out.shape[1], np.nan), where=count > 0))
    means = np.array(means)
    fallback = np.nanmean(means, axis=0)
    for m, mean in zip(cells, means):
        out[m] = np.where(finite[m], out[m], np.where(np.isfinite(mean), mean, fallback))
    n = len(train['lab_id'])
    return out[:n], out[n:]


class _Hold:
    def __init__(self, interval, times, block, x, calibration):
        self.times = times
        self.dt = interval
        self.end = np.r_[np.cumsum(interval[:0:-1])[::-1], 0.]
        lab = block['lab_id']
        g, b = (c[lab] for c in calibration)
        self.hist = np.asarray(block['past_stress_cmd']) * g[:, None] + b[:, None]
        self.stress = np.asarray(block['stress_hold_cmd']) * g + b
        self.features = np.column_stack((np.ones(len(x)), x))
        self.base = np.asarray(block['strain_0'])
        self.t = np.r_[times, 1.]
        self.linear = self.features[:, None, :] * self.t[None, :, None]

    def design(self, taus, gradient=False):
        matrices, derivs = [], []
        for tau in taus:
            e1 = np.exp(-self.end / tau)
            e2 = np.exp(-(self.end + self.dt) / tau)
            w = e1 * -np.expm1(-self.dt / tau)
            delta = self.stress - np.einsum('ij,j->i', self.hist, w)
            mode = np.r_[-np.expm1(-self.times / tau), 1. / tau]
            matrices.append(delta[:, None] * mode)
            if gradient:
                dw = (self.end * e1 - (self.end + self.dt) * e2) / tau**2
                dd = -np.einsum('ij,j->i', self.hist, dw)
                dm = -np.r_[self.times*np.exp(-self.times/tau), 1.] / tau**2
                derivs.append(dd[:, None] * mode + delta[:, None] * dm)
        design = np.concatenate((np.stack(matrices, axis=-1), self.linear), axis=-1)
        return design, derivs


def _fit(hold, block):
    curves = np.asarray(block['strain_curves_obs']) - hold.base[:, None]
    rate = np.asarray(block['instant_rate_obs'])
    obs = np.column_stack((curves, rate))
    good = np.isfinite(obs)
    y = obs[good]
    lo = np.r_[[.2, .2], np.full(7, -np.inf)]
    hi = np.r_[[8., 8.], np.full(7, np.inf)]

    def optimize(p, sigma, loss):
        w = np.broadcast_to(1. / sigma, obs.shape)[good]
        def residual(v):
            a, _ = hold.design(v[:2])
            return (np.einsum('ijk,k->ij', a, v[2:])[good] - y) * w
        def jacobian(v):
            a, ds = hold.design(v[:2], gradient=True)
            return np.column_stack((ds[0][good]*v[2], ds[1][good]*v[3], a[good])) * w[:, None]
        return least_squares(residual, p, jac=jacobian, bounds=(lo, hi), loss=loss, x_scale='jac', max_nfev=130, ftol=1e-9)

    p0 = np.array([.5, 3.])
    a, _ = hold.design(p0)
    theta = np.linalg.lstsq(a[good], y, rcond=None)[0]
    err = obs - np.einsum('ijk,k->ij', a, theta)
    k = curves.shape[1]
    pilot_scales = np.r_[np.full(k, max(_mad(err[:, :-1][good[:, :-1]]), 1e-7)), max(_mad(err[:, -1][good[:, -1]]), 1e-7)]
    p = optimize(np.r_[p0, theta], pilot_scales, 'soft_l1').x
    for _ in range(2):
        a, _ = hold.design(p[:2])
        err = obs - np.einsum('ijk,k->ij', a, p[2:])
        scales = np.r_[np.full(k, max(_mad(err[:, :-1][good[:, :-1]]), 1e-7)), max(_mad(err[:, -1][good[:, -1]]), 1e-7)]
        p = optimize(p, 2.3849 * scales, 'cauchy').x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p


def _calibration_options(x, y, limits, yield_stress, baseline):
    # The check noise on the public panel is about 1e-4 stress units.
    tol = 6e-4
    if len(x) < 4:
        return [baseline]
    a, b = np.triu_indices(len(x), 1)
    distinct = np.abs(x[a] - x[b]) > 1e-10
    a, b = a[distinct], b[distinct]
    g = (y[a] - y[b]) / (x[a] - x[b])
    r = y[None, :] - y[a, None] - g[:, None] * (x[None, :] - x[a, None])
    masks = np.unique(np.abs(r) <= tol, axis=0)
    design = np.column_stack((x, np.ones(len(x))))
    minimum = len(x) - (len(x) - 1) // 3
    options = []
    for m in masks:
        n = m.sum()
        if n < minimum or np.ptp(x[m]) <= 1e-10:
            continue
        fit = np.linalg.lstsq(design[m], y[m], rcond=None)[0]
        if np.max(np.abs(fit[0] * limits + fit[1])) > yield_stress + tol:
            continue
        err = y - design @ fit
        options.append((n, np.ptp(x[m]), err[m] @ err[m], fit))
    if not options:
        return [baseline]
    n = max(v[0] for v in options)
    if n == len(x):
        return [baseline]
    ranked = sorted((v for v in options if v[0] == n), key=lambda v: (-v[1], v[2]))
    return [v[-1] for v in ranked]


def _select_calibration(options, fit, block, x, dt, times, hold_cls):
    obs = block['instant_rate_obs']
    good = np.isfinite(obs)
    curves = block['strain_curves_obs']
    scores = []
    for gb in options:
        # Only the affine calibration varies; coefficients come from other labs.
        arrays = np.broadcast_to(np.asarray(gb)[:, None], (2, 4))
        hold = hold_cls(dt, times, block, x, arrays)
        pred = hold.design(fit[:2])[0] @ fit[2:]
        de = np.abs(pred[:, :-1] + hold.base[:, None] - curves)
        score = np.nanmedian(de)
        if good.any():
            score += np.median(np.abs(pred[good, -1] - obs[good]))
        scores.append(score)
    return options[np.argmin(scores)]


def _resolve_calibration(dataset, train, test, x, dt, times):
    checks = dataset['calibration_load_cells']
    calibration = np.array(_calibrate(checks))
    ambiguous = []
    donor_calibration = []
    for lab in range(4):
        cmd = np.concatenate([
            block[key][block['lab_id'] == lab].ravel()
            for block in (train, test) for key in ('past_stress_cmd', 'stress_hold_cmd')
        ])
        if not cmd.size:
            continue
        limits = np.array([cmd.min(), cmd.max()])
        xc, yc = (np.asarray(checks[k]) for k in ('cmd_stress', 'true_stress'))
        use = (np.asarray(checks['lab_id']) == lab) & np.isfinite(xc) & np.isfinite(yc)
        options = _calibration_options(xc[use], yc[use], limits, dataset['yield_stress'], calibration[:, lab].copy())
        calibration[:, lab] = options[0]
        differences = (np.asarray(options) - options[0]) @ np.vstack((limits, np.ones(2)))
        if np.max(np.abs(differences)) > .01:
            ambiguous.append((lab, options))
        elif use.sum() >= 4 and np.ptp(xc[use]) > 1e-10:
            donor_calibration.append(lab)
    if ambiguous:
        labels = train['lab_id']
        donors = np.isin(labels, donor_calibration)
        if donors.any():
            block = {k: train[k][donors] for k in ('lab_id', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'strain_curves_obs', 'instant_rate_obs')}
            fit = _fit(_Hold(dt, times, block, x[donors], calibration), block)
            for lab, options in ambiguous:
                target = labels == lab
                if target.any():
                    subset = {k: train[k][target] for k in block}
                    calibration[:, lab] = _select_calibration(options, fit, subset, x[target], dt, times, _Hold)
    return calibration


def solve_alpha_m96_w03_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    xt, xe = _align(train, test)
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    cal = _resolve_calibration(dataset, train, test, xt, dt, time)
    p = _fit(_Hold(dt, time, train, xt, cal), train)
    target = _Hold(dt, time, test, xe, cal)
    a, _ = target.design(p[:2])
    pred = np.einsum('ijk,k->ij', a, p[2:])
    return {'estimated_taus': p[:2], 'hereditary_weights': p[2:],
            'viscoplastic_weights': np.zeros(2),
            'predicted_strain': target.base[:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
