"""Calibrated two-branch creep with hold and history overstress."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


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


def _sweep(checks, lab):
    labels, x, y = (np.asarray(checks[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    use = (labels == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _line_options(x, y):
    """Find positive affine lines with the largest spot-check consensus."""
    a = np.column_stack((x, np.ones(len(x))))
    candidates = []
    for i in range(len(x)):
        for j in range(i):
            if abs(x[i] - x[j]) < 1e-6:
                continue
            g = (y[i] - y[j]) / (x[i] - x[j])
            if g <= 0:
                continue
            keep = np.abs(a @ [g, y[i] - g*x[i]] - y) < .002
            p = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
            e = a @ p - y
            keep = abs(e) < .002
            candidates.append((int(keep.sum()), np.ptp(x[keep]), float(e[keep] @ e[keep]), p))
    if not candidates:
        return []
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
    best = candidates[0]
    options = []
    for n, span, _, p in candidates:
        if n != best[0]:
            break
        if span > 1e-5 and not any(np.max(abs(p - q)) < .003 for q in options):
            options.append(p)
    return options


def _calibration_options(data):
    checks = data['calibration_load_cells']
    history_path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    history = None
    all_options = []
    for lab in range(4):
        x, y = _sweep(checks, lab)
        options = _line_options(x, y)
        if len(x) < 4 or np.ptp(x) < .2 or len(options) != 1:
            if history is None:
                history = json.loads(history_path.read_text()) if history_path.is_file() else []
            priors = [p[0] for s in history if (p := _line_options(*_sweep(s, lab)))]
            if not priors:
                priors = [np.array([1., 0.])]
            for p in priors:
                trials = [p]
                if len(x) and (len(x) < 4 or np.ptp(x) < .2):
                    trials.append(np.array([p[0], np.median(y - p[0]*x)]))
                for q in trials:
                    if not any(np.max(abs(q - r)) < .003 for r in options):
                        options.append(q)
            if len(x):
                counts = [np.count_nonzero(abs(g*x + b - y) < .004) for g, b in options]
                options = [q for q, n in zip(options, counts) if n == max(counts)]
        all_options.append(options)
    return all_options


class _Hold:
    def __init__(self, dt, times, block, x, cal, yield_stress):
        self.times = times
        self.dt = dt
        self.end = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
        lab = block['lab_id']
        g, b = cal[:, lab]
        self.hist = block['past_stress_cmd'] * g[:, None] + b[:, None]
        self.stress = block['stress_hold_cmd'] * g + b
        self.base = block['strain_0']
        self.ov = np.maximum(0., abs(self.stress) - yield_stress)**2 * np.sign(self.stress)
        work = np.maximum(0., abs(self.hist) - yield_stress)**2 @ dt
        self.memory = work * self.stress
        drift = np.column_stack((np.ones(len(x)), x, self.ov))
        self.linear = drift[:, None, :] * np.r_[times, 1.][None, :, None]

    def design(self, taus, gradient=False):
        modes, derivs = [], []
        for tau in taus:
            e = np.exp(-self.end / tau)
            ee = np.exp(-(self.end + self.dt) / tau)
            delta = self.stress - np.einsum('ij,j->i', self.hist, e * -np.expm1(-self.dt/tau))
            mode = np.r_[-np.expm1(-self.times/tau), 1./tau]
            modes.append(delta[:, None] * mode)
            if gradient:
                dw = (self.end*e - (self.end + self.dt)*ee) / tau**2
                dm = -np.r_[self.times*np.exp(-self.times/tau), 1.] / tau**2
                ddelta = -np.einsum('ij,j->i', self.hist, dw)
                derivs.append(ddelta[:, None]*mode + delta[:, None]*dm)
            if len(modes) == 1:
                plastic = self.memory[:, None] * mode
                dplastic = self.memory[:, None] * dm if gradient else None
        return np.concatenate((np.stack(modes, axis=-1), self.linear, plastic[..., None]), axis=-1), derivs, dplastic


def _predict(hold, p):
    return np.einsum('ijk,k->ij', hold.design(p[:2])[0], p[2:])


def _fit(hold, block):
    obs = np.column_stack((block['strain_curves_obs'] - hold.base[:, None], block['instant_rate_obs']))
    good = np.isfinite(obs)
    y = obs[good]
    low = np.r_[[.05001, .05002], np.full(9, -np.inf)]
    high = np.r_[[24.999, 24.999], np.full(9, np.inf)]

    def scales(err):
        def mad(v):
            return max(1e-7, 1.4826*np.median(abs(v - np.median(v)))) if len(v) else 1.
        return np.where(np.arange(obs.shape[1]) == obs.shape[1] - 1,
                        mad(err[:, -1][good[:, -1]]), mad(err[:, :-1][good[:, :-1]]))

    def optimize(p, sigma, loss):
        w = np.broadcast_to(1./sigma, obs.shape)[good]
        def residual(v):
            return (_predict(hold, v)[good] - y) * w
        def jacobian(v):
            a, ds, dp = hold.design(v[:2], gradient=True)
            return np.column_stack((ds[0][good]*v[2] + dp[good]*v[-1], ds[1][good]*v[3], a[good])) * w[:, None]
        return least_squares(residual, p, jac=jacobian, bounds=(low, high), loss=loss,
                             x_scale='jac', max_nfev=150, ftol=1e-9).x

    p0 = np.array([.55, 3.1])
    a = hold.design(p0)[0]
    theta = np.linalg.lstsq(a[good], y, rcond=None)[0]
    p = np.r_[p0, theta]
    p = optimize(p, scales(obs - _predict(hold, p)), 'soft_l1')
    for _ in range(2):
        p = optimize(p, 2.3849*scales(obs - _predict(hold, p)), 'cauchy')
    return p


def _estimate(dataset, x):
    train = dataset['train']
    args = dataset['past_dt'], dataset['eval_times']
    ys = dataset['yield_stress']
    options = _calibration_options(dataset)
    cal = np.array([q[0] for q in options]).T.copy()
    labels = train['lab_id']
    keys = ('lab_id', 'past_stress_cmd', 'stress_hold_cmd', 'strain_0', 'strain_curves_obs', 'instant_rate_obs')

    def subset(m):
        return {k: train[k][m] for k in keys}

    trust = np.array([len(q) == 1 for q in options])
    for lab, q in enumerate(options):
        target = labels == lab
        donor = trust[labels]
        if not donor.any():
            donor = ~target
        if len(q) == 1 or not target.any() or not donor.any():
            continue
        block = subset(donor)
        p = _fit(_Hold(*args, block, x[donor], cal, ys), block)
        block = subset(target)
        scores = []
        for line in q:
            cal[:, lab] = line
            hold = _Hold(*args, block, x[target], cal, ys)
            pred = _predict(hold, p)
            scores.append(np.nanmedian(abs(pred[:, :-1] + hold.base[:, None] - block['strain_curves_obs'])) +
                          np.nanmedian(abs(pred[:, -1] - block['instant_rate_obs'])))
        cal[:, lab] = q[np.argmin(scores)]
    return cal, _fit(_Hold(*args, train, x, cal, ys), train)


def solve_alpha_m96_w03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    xt, xe = _align(train, test)
    cal, p = _estimate(dataset, xt)
    target = _Hold(dataset['past_dt'], dataset['eval_times'], test, xe, cal, dataset['yield_stress'])
    pred = _predict(target, p)
    # The missing coupon covariates enter only the linear drift. Use the
    # supplied initial-rate measurement to recover that specimen's drift,
    # without changing its Prony or plastic-work response.
    if 'instant_rate_obs' in test:
        rate = np.asarray(test['instant_rate_obs'])
        missing = ~np.isfinite(test['coupon_features_raw'])
        use = np.any(missing & (p[5:9] != 0.), axis=1) & np.isfinite(rate)
        correction = rate[use] - pred[use, -1]
        pred[use, :-1] += correction[:, None] * target.times
        pred[use, -1] += correction
    return {'estimated_taus': p[:2], 'hereditary_weights': p[2:9],
            'viscoplastic_weights': p[9:], 'predicted_strain': target.base[:, None] + pred[:, :-1],
            'initial_rate_pred': pred[:, -1]}
