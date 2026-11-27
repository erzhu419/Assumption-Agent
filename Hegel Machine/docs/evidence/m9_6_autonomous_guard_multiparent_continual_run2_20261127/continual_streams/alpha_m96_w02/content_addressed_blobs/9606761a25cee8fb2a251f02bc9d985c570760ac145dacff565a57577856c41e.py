"""Calibrate the four load cells and fit a two-branch creep law."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _sweep(checks, lab):
    x = np.asarray(checks['cmd_stress'], float)
    y = np.asarray(checks['true_stress'], float)
    m = (np.asarray(checks['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _line(x, y, prior=None):
    # Find consensus at the measured sweep precision, not at the EMI scale.
    i, j = np.triu_indices(len(x), 1)
    distinct = np.abs(x[j] - x[i]) >= .18
    i, j = i[distinct], j[distinct]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    candidates = list(zip(slopes, y[i] - slopes * x[i]))
    if prior is not None:
        candidates.append(prior)
        candidates.extend((prior[0], v - prior[0] * u) for u, v in zip(x, y))
    best = None
    for g, b in candidates:
        if g <= 0:
            continue
        error = np.abs(y - (g * x + b))
        use = error <= 8e-4
        key = (int(use.sum()), -float(np.minimum(error, 8e-4).sum()))
        if best is None or key > best[0]:
            best = key, use
    if best is None:
        return (np.array([1., np.median(y - x)]) if len(x) else np.array([1., 0.])) if prior is None else prior
    use = best[1]
    u, v = x[use], y[use]
    if len(u) >= 3 and np.ptp(u) >= .35:
        return np.linalg.lstsq(np.column_stack((u, np.ones(len(u)))), v, rcond=None)[0]
    if prior is not None:
        # With fewer than three well-spaced inliers, a pair can fit exactly
        # even if one check is an EMI spike. Keep the archived gain and
        # update the intercept only from checks consistent with that archive.
        offset = y - prior[0] * x
        supported = np.abs(offset - prior[1]) <= .003
        if np.any(supported):
            return np.array([prior[0], np.median(offset[supported])])
        return prior
    if len(u) >= 2 and np.ptp(u) >= .18:
        return np.linalg.lstsq(np.column_stack((u, np.ones(len(u)))), v, rcond=None)[0]
    return np.array([1., np.median(y - x) if len(x) else 0.])


def _calibrate(checks):
    history = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.is_file() else []
    out = []
    for lab in range(4):
        prior = None
        for episode in history:
            x, y = _sweep(episode, lab)
            if len(x):
                prior = _line(x, y, prior)
        out.append(_line(*_sweep(checks, lab), prior))
    return np.array(out)


def _align(train, test):
    lab, stratum, raw, refs = (np.concatenate((train[k], test[k])) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    x = raw.astype(float)
    for c in np.unique(lab):
        use = lab == c
        panel = refs[use]
        offset = np.nanmedian(np.nanmean(panel, axis=0))
        spread = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (x[use] - offset) / spread
    _, cells, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                 return_inverse=True, return_counts=True)
    w = (1. / counts[cells])[:, None]
    finite = np.isfinite(x)
    mass = (w * finite).sum(axis=0)
    fill = np.divide((w * np.where(finite, x, 0.)).sum(axis=0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0)
    return np.where(finite, x, fill)


class _Design:
    def __init__(self, dataset, split, x, cal):
        gain, bias = cal[np.asarray(split['lab_id'], int)].T
        self.past = split['past_stress_cmd'] * gain[:, None] + bias[:, None]
        self.hold = split['stress_hold_cmd'] * gain + bias
        self.dt = np.asarray(dataset['past_dt'])
        self.t = np.asarray(dataset['eval_times'])
        sy = dataset['yield_stress']
        ov = np.maximum(0., np.abs(self.hold) - sy)**2 * np.sign(self.hold)
        self.hist = (np.maximum(0., np.abs(self.past) - sy)**2 * self.dt).sum(axis=1) * self.hold
        self.active = np.array([np.any(ov), np.any(self.hist)])
        drift = np.column_stack((np.ones(len(x)), x, ov))
        self.base = np.zeros((len(x), len(self.t)+1, 9))
        self.base[:, :-1, 2:8] = drift[:, None, :] * self.t[None, :, None]
        self.base[:, -1, 2:8] = drift

    def make(self, taus, derivative=False):
        a = self.base.copy()
        deriv = np.zeros(a.shape[:2] + (2, 2)) if derivative else None
        for branch, tau in enumerate(taus):
            e = np.exp(-self.dt / tau)
            f = -np.expm1(-self.dt / tau)
            z = np.zeros(len(self.hold))
            dz = np.zeros_like(z)
            for j in range(len(e)):
                if derivative:
                    dz = e[j] * dz + e[j] * self.dt[j] / tau**2 * (z - self.past[:, j])
                z = e[j] * z + f[j] * self.past[:, j]
            gap = self.hold - z
            h = -np.expm1(-self.t / tau)
            a[:, :-1, branch] = gap[:, None] * h
            a[:, -1, branch] = gap / tau
            if derivative:
                dh = -np.exp(-self.t / tau) * self.t / tau**2
                deriv[:, :-1, branch, 0] = gap[:, None] * dh - dz[:, None] * h
                deriv[:, -1, branch, 0] = -dz / tau - gap / tau**2
            if branch == 0:
                a[:, :-1, 8] = self.hist[:, None] * h
                a[:, -1, 8] = self.hist / tau
                if derivative:
                    deriv[:, :-1, 0, 1] = self.hist[:, None] * dh
                    deriv[:, -1, 0, 1] = -self.hist / tau**2
        return (a, deriv) if derivative else a


def _mad(x):
    return max(1e-7, 1.4826 * np.median(np.abs(x - np.median(x))))


def _fit(design, split):
    y = np.column_stack((split['strain_curves_obs'] - split['strain_0'][:, None],
                         split['instant_rate_obs']))
    present = np.isfinite(y)
    rate = present.copy()
    rate[:, :-1] = False
    is_rate = rate[present]
    y = y[present]
    cols = np.r_[np.ones(7, dtype=bool), design.active]
    n = int(cols.sum())
    has_history = cols[8]
    seeds = []
    grid = (.19, .35, .65, 1.15, 2.1, 3.8, 7., 12.)
    for i, t1 in enumerate(grid[:-1]):
        for t2 in grid[i+1:]:
            a = design.make((t1, t2))[present][:, cols]
            b = np.linalg.lstsq(a, y, rcond=None)[0]
            for _ in range(2):
                r = np.einsum('ij,j->i', a, b) - y
                s = 2.5 * np.where(is_rate, _mad(r[is_rate]) if np.any(is_rate) else 1., _mad(r[~is_rate]))
                w = 1. / np.hypot(s, r)
                b = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]
            r = np.einsum('ij,j->i', a, b) - y
            seeds.append((np.r_[t1, t2, b], r))
    _, rr = min(seeds, key=lambda z: np.median(np.abs(z[1][~is_rate])))
    scale = 2.5 * np.where(is_rate, _mad(rr[is_rate]) if np.any(is_rate) else 1., _mad(rr[~is_rate]))
    seeds.sort(key=lambda z: np.log1p((z[1] / scale)**2).sum())

    # Optimize the slow/fast separation directly; the plastic history always uses the fast branch.
    def pack(p):
        t1, t2 = p[:2]
        return np.r_[t1, (t2 - t1) / (24.9 - t1), p[2:]]

    def taus(p):
        return np.array([p[0], p[0] + (24.9 - p[0]) * p[1]])

    def residual(p):
        a = design.make(taus(p))[present][:, cols]
        return (np.einsum('ij,j->i', a, p[2:]) - y) / scale

    def jacobian(p):
        a, dp = design.make(taus(p), derivative=True)
        d = dp[present]
        d1 = d[:, 0, 0] * p[2]
        if has_history:
            d1 += d[:, 0, 1] * p[-1]
        d2 = d[:, 1, 0] * p[3]
        return np.column_stack((d1 + (1 - p[1]) * d2, (24.9 - p[0]) * d2,
                                a[present][:, cols])) / scale[:, None]

    bounds = (np.r_[.051, 1e-5, np.full(n, -np.inf)], np.r_[19., .99999, np.full(n, np.inf)])
    polished = [least_squares(residual, pack(p), jac=jacobian, bounds=bounds, loss='cauchy',
                              x_scale='jac', max_nfev=100, ftol=1e-8, xtol=1e-8)
                for p, _ in seeds[:3]]
    p = min(polished, key=lambda fit: fit.cost).x
    weights = np.zeros(9)
    weights[cols] = p[2:]
    return taus(p), weights


def solve_alpha_m96_w02_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x = _align(train, test)
    n = len(train['lab_id'])
    cal = _calibrate(dataset['calibration_load_cells'])
    # Missing covariates have an unknown coupon-specific drift. Fit the shared
    # coefficients on observed covariates when they span the full design.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    design = _Design(dataset, train, x[:n], cal)
    if not complete.all() and complete.any():
        subset = {k: v[complete] for k, v in train.items()}
        clean = _Design(dataset, subset, x[:n][complete], cal)
        cols = np.r_[np.ones(7, dtype=bool), design.active]
        observed = np.isfinite(np.column_stack((subset['strain_curves_obs'],
                                               subset['instant_rate_obs'])))
        if np.linalg.matrix_rank(clean.make((.55, 3.8))[observed][:, cols]) == cols.sum():
            train, design = subset, clean
    tau, weights = _fit(design, train)
    test_design = _Design(dataset, test, x[n:], cal)
    pred = test_design.make(tau) @ weights

    # The imputed features supply the prior drift. If covariates are missing,
    # an observed onset identifies its remaining linear-in-time contribution.
    onset = np.asarray(test.get('instant_rate_obs', np.full(len(pred), np.nan)))
    update = (~np.isfinite(test['coupon_features_raw']).all(axis=1)
              & np.isfinite(onset))
    delta = onset[update] - pred[update, -1]
    pred[update, :-1] += delta[:, None] * test_design.t
    pred[update, -1] += delta
    return dict(estimated_taus=tau, hereditary_weights=weights[:7],
                viscoplastic_weights=weights[7:],
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None],
                initial_rate_pred=pred[:, -1])
