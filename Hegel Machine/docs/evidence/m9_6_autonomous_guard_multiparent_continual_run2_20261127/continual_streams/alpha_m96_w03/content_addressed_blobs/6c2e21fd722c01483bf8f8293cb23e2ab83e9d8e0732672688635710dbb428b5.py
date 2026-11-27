"""Calibrated hereditary creep with hold and history plasticity."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _calibration_candidates(x, y):
    i, j = np.triu_indices(x.size, 1)
    use = np.abs(x[j] - x[i]) > 0.05
    i, j = i[use], j[use]
    if not i.size:
        return 0, []
    gain = (y[j] - y[i]) / (x[j] - x[i])
    bias = y[i] - gain*x[i]
    masks = np.unique(np.abs(gain[:, None]*x + bias[:, None] - y) <= 6e-4, axis=0)
    count = int(masks.sum(axis=1).max())
    a = np.c_[x, np.ones(x.size)]
    candidates = []
    for mask in masks[masks.sum(axis=1) == count]:
        if np.ptp(x[mask]) > 0.05:
            p = np.linalg.lstsq(a[mask], y[mask], rcond=None)[0]
            candidates.append((p, np.linalg.norm(a[mask] @ p - y[mask])))
    return count, candidates


def _calibrate(checks):
    lab_id, cmd, true = (np.asarray(checks[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    valid = np.isfinite(cmd) & np.isfinite(true)
    observations, sparse = [], []
    result = np.tile([1., 0.], (4, 1))
    for lab in range(4):
        m = valid & (lab_id == lab)
        x, y = cmd[m], true[m]
        observations.append((x, y))
        count, candidates = _calibration_candidates(x, y)
        if count >= 3 and candidates:
            result[lab] = min(candidates, key=lambda item: item[1])[0]
        else:
            sparse.append((lab, candidates))
    if not sparse:
        return result.T

    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    history = json.loads(path.read_text()) if path.is_file() else []
    profiles = []
    for sweep in history:
        labs, x, y = (np.asarray(sweep[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
        profile = np.full((4, 2), np.nan)
        for lab in range(4):
            m = (labs == lab) & np.isfinite(x) & np.isfinite(y)
            _, candidates = _calibration_candidates(x[m], y[m])
            if candidates:
                profile[lab] = min(candidates, key=lambda item: item[1])[0]
        profiles.append(profile)

    # Choose a calibration epoch by agreement with this batch, not by
    # how often its coefficients occurred in the archive. Cap EMI influence.
    def agreement(profile):
        return sum(np.sum(1. - np.minimum(1., np.abs(y - (p[0]*x + p[1])) / .003))
                   for p, (x, y) in zip(profile, observations) if np.isfinite(p).all())

    scores = np.array([agreement(p) for p in profiles])
    selected = profiles[np.argmax(scores)] if scores.size and scores.max() > 0 else result.copy()
    for lab, candidates in sparse:
        x, y = observations[lab]
        prior = selected[lab].copy()
        if not np.isfinite(prior).all():
            prior = result[lab].copy()
        residual = y - (prior[0]*x + prior[1])
        agree = np.abs(residual) <= .003
        if agree.any():
            # A trusted archived gain is better conditioned than a two-point slope.
            prior[1] += np.median(residual[agree])
        elif candidates and not (scores.size and scores.max() > 0):
            prior = min(candidates, key=lambda item: np.linalg.norm(item[0]-prior))[0]
        elif y.size and not (scores.size and scores.max() > 0):
            prior[1] += np.median(residual)
        result[lab] = prior
    return result.T


def _align(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    for c in np.unique(lab):
        use = lab == c
        r = refs[use]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        loc = np.median(np.nanmean(r, axis=0))
        scale = np.median(np.nanstd(r, axis=0, ddof=0))
        raw[use] = (raw[use] - loc) / scale
    cells = [(lab == c) & (stratum == s) for c, s in np.unique(np.c_[lab, stratum], axis=0)]
    means = []
    for use in cells:
        col = raw[use]
        n = np.isfinite(col).sum(axis=0)
        means.append(np.divide(np.nansum(col, axis=0), n,
                               out=np.full(raw.shape[1], np.nan), where=n > 0))
    means = np.array(means)
    n = np.isfinite(means).sum(axis=0)
    fill = np.divide(np.nansum(means, axis=0), n, out=np.zeros(raw.shape[1]), where=n > 0)
    raw = np.where(np.isfinite(raw), raw, fill)
    return np.split(raw, [len(train['lab_id'])])


class _Hold:
    def __init__(self, dataset, block, x, calibration):
        self.dt = np.asarray(dataset['past_dt'])
        self.times = np.asarray(dataset['eval_times'])
        self.end = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.]
        lab = block['lab_id']
        gain, bias = (p[lab] for p in calibration)
        self.hist = block['past_stress_cmd'] * gain[:, None] + bias[:, None]
        self.stress = block['stress_hold_cmd'] * gain + bias
        self.base = block['strain_0']
        self.features = np.c_[np.ones(len(x)), x]
        sy = dataset['yield_stress']
        self.over = np.maximum(np.abs(self.stress) - sy, 0.)**2 * np.sign(self.stress)
        self.work = (np.maximum(np.abs(self.hist) - sy, 0.)**2 @ self.dt) * self.stress
        self.time_rate = np.r_[self.times, 1.]
        self.linear = self.time_rate[None, :, None] * self.features[:, None, :]

    def design(self, taus, columns, gradient=False):
        modes, grads = [], []
        for tau in taus:
            elapsed = np.exp(-self.end / tau)
            earlier = np.exp(-(self.end + self.dt) / tau)
            hist_weight = elapsed * -np.expm1(-self.dt / tau)
            delta = self.stress - self.hist @ hist_weight
            f = np.r_[-np.expm1(-self.times / tau), 1. / tau]
            modes.append(delta[:, None] * f)
            if gradient:
                dh = (self.end*elapsed - (self.end + self.dt)*earlier) / tau**2
                dd = -(self.hist @ dh)
                df = -np.r_[self.times*np.exp(-self.times/tau), 1.] / tau**2
                grads.append(dd[:, None]*f + delta[:, None]*df)
        f = np.r_[-np.expm1(-self.times / taus[0]), 1. / taus[0]]
        a = np.concatenate((np.stack(modes, axis=-1), self.linear,
                            (self.over[:, None]*self.time_rate)[..., None],
                            (self.work[:, None]*f)[..., None]), axis=-1)
        if gradient:
            df = -np.r_[self.times*np.exp(-self.times/taus[0]), 1.] / taus[0]**2
            grads.append(self.work[:, None]*df)
        return a[..., columns], grads


def _fit(hold, block):
    y = np.c_[block['strain_curves_obs'] - hold.base[:, None], block['instant_rate_obs']]
    good = np.isfinite(y)
    target = y[good]
    rate = np.broadcast_to(np.arange(y.shape[1]) == y.shape[1]-1, y.shape)[good]
    cols = np.r_[np.arange(7), np.flatnonzero([np.any(hold.over), np.any(hold.work)])+7]
    lo = np.r_[0.05001, 0.05002, np.full(len(cols), -np.inf)]
    hi = np.r_[24.998, 24.999, np.full(len(cols), np.inf)]

    def residual(p):
        a, _ = hold.design(p[:2], cols)
        return a[good] @ p[2:] - target

    def run(p, scale, loss):
        def jac(p):
            a, da = hold.design(p[:2], cols, True)
            first = da[0][good]*p[2]
            if cols[-1] == 8:
                first += da[2][good]*p[-1]
            return np.c_[first, da[1][good]*p[3], a[good]] / scale[:, None]
        return least_squares(lambda p: residual(p)/scale, p, jac=jac, bounds=(lo, hi),
                             loss=loss, x_scale='jac', max_nfev=120, ftol=1e-9)

    scale = np.ones(target.shape)
    linear_fits = []
    for taus in ([.25, 2.], [.65, 4.], [1.5, 9.]):
        a, _ = hold.design(taus, cols)
        initial = np.r_[taus, np.linalg.lstsq(a[good], target, rcond=None)[0]]
        linear_fits.append(run(initial, scale, 'linear'))
    p = min(linear_fits, key=lambda result: result.cost).x
    for _ in range(2):
        r = residual(p)
        for group in (rate, ~rate):
            if np.any(group):
                v = r[group]
                scale[group] = max(2e-4, 1.4826*np.median(np.abs(v-np.median(v))))
        p = run(p, scale, 'soft_l1').x
    weights = np.zeros(9)
    weights[cols] = p[2:]
    order = np.argsort(p[:2])
    if cols[-1] == 8 and order[0] != 0:
        # Plastic history uses the short branch; preserve that association.
        start = np.r_[p[:2][order], p[2:4][order], p[4:]]
        p = run(start, scale, 'soft_l1').x
        weights[cols] = p[2:]
        order = np.argsort(p[:2])
    weights[:2] = weights[:2][order]
    return p[:2][order], weights


def solve_alpha_m96_w03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    xt, xe = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    taus, weights = _fit(_Hold(dataset, train, xt, calibration), train)
    hold = _Hold(dataset, test, xe, calibration)
    design, _ = hold.design(taus, np.arange(9))
    pred = design @ weights
    # A missing coupon feature leaves its specimen-specific linear drift
    # unknown. The measured onset rate identifies that drift independently
    # of the hereditary and plastic terms, whose time dependence is fixed.
    onset = np.asarray(test['instant_rate_obs'])
    missing = np.any(~np.isfinite(test['coupon_features_raw']), axis=1)
    use = missing & np.isfinite(onset)
    pred[use] += (onset[use] - pred[use, -1])[:, None] * hold.time_rate
    return dict(estimated_taus=taus, hereditary_weights=weights[:7],
                viscoplastic_weights=weights[7:], predicted_strain=hold.base[:, None]+pred[:, :-1],
                initial_rate_pred=pred[:, -1])
