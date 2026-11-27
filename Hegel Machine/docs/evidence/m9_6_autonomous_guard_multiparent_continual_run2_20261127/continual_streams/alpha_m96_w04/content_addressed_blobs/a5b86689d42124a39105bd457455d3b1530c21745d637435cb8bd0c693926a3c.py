import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
_SWEEP_TOL = 0.001


def _mad(a, axis=None):
    return 1.4826 * np.median(np.abs(a - np.median(a, axis=axis, keepdims=True)), axis=axis)


def _sweeps(checks):
    lab, x, y = (np.asarray(checks[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    return [(x[(lab == c) & np.isfinite(x) & np.isfinite(y)],
             y[(lab == c) & np.isfinite(x) & np.isfinite(y)]) for c in range(4)]


def _calibration(checks):
    sweeps = _sweeps(checks)
    fitted = np.full((4, 2), np.nan)
    for c, (x, y) in enumerate(sweeps):
        i, j = np.triu_indices(len(x), 1)
        keep = np.abs(x[i] - x[j]) > 0.1
        if not keep.any():
            continue
        i, j = i[keep], j[keep]
        a = (y[i] - y[j]) / (x[i] - x[j])
        b = y[i] - a * x[i]
        err = np.abs(y - (a[:, None] * x + b[:, None]))
        support = err <= _SWEEP_TOL
        counts = support.sum(axis=1)
        best = np.lexsort((np.where(support, err, 0).sum(axis=1), -counts))[0]
        m = support[best]
        if m.sum() >= max(3, len(x) // 2 + 1) and np.ptp(x[m]) >= 0.35:
            fitted[c] = np.polyfit(x[m], y[m], 1)
    if np.isfinite(fitted).all():
        return fitted

    templates = [np.tile([1., 0.], (4, 1))]
    if _HISTORY.is_file():
        for episode in json.loads(_HISTORY.read_text()):
            templates.append(np.array([np.polyfit(x, y, 1) for x, y in _sweeps(episode)]))
    templates = np.asarray(templates)
    for c, (x, y) in enumerate(sweeps):
        if np.isfinite(fitted[c]).all():
            continue
        # A sparse lab inherits the archived regime supported by its own and its wing's checks.
        def mismatch(lab):
            u, v = sweeps[lab]
            e = np.abs(templates[:, lab, :1] * u + templates[:, lab, 1:] - v)
            return np.median(np.minimum(e, 0.2), axis=1) if len(u) else np.zeros(len(templates))
        scores = 2 * mismatch(c) + mismatch(c ^ 1) + 0.35 * (mismatch(c ^ 2) + mismatch(c ^ 3))
        gain, bias = templates[np.argmin(scores), c]
        residual = y - (gain * x + bias)
        good = np.abs(residual) <= _SWEEP_TOL
        if good.any():
            bias += np.median(residual[good])
        fitted[c] = gain, bias
    return fitted


def _aligned(parts):
    lab, stratum, raw, ref = [np.concatenate([p[k] for p in parts]) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons')]
    x = np.empty_like(raw, dtype=float)
    for c in np.unique(lab):
        m = lab == c
        panel = ref[m]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[m] = (raw[m] - center) / scale
    observed = np.isfinite(x)
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        m = (lab == c) & (stratum == s)
        count = observed[m].sum(axis=0)
        total = np.where(observed[m], x[m], 0).sum(axis=0)
        means.append(np.divide(total, count, out=np.full(x.shape[1], np.nan), where=count > 0))
    return np.where(observed, x, np.nanmean(means, axis=0))


class _Model:
    def __init__(self, part, features, coef, dt, time, yield_stress):
        gain, bias = coef[np.asarray(part['lab_id'])].T
        self.past = np.asarray(part['past_stress_cmd']) * gain[:, None] + bias[:, None]
        self.hold = np.asarray(part['stress_hold_cmd']) * gain + bias
        self.t = np.r_[time, 0.]
        self.suffix = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
        self.work = (np.maximum(np.abs(self.past) - yield_stress, 0.) ** 2) @ dt
        self.over = np.maximum(np.abs(self.hold) - yield_stress, 0.) ** 2 * np.sign(self.hold)
        steady = np.column_stack((np.ones(len(features)), features, self.over))
        self.steady = steady[:, None, :] * np.r_[time, 1.][None, :, None]

    def _branch(self, tau):
        e = np.exp(-self.suffix / tau)
        de = e * self.suffix / tau**2
        q = self.hold - np.einsum('ij,j->i', self.past, np.diff(e))
        dq = -np.einsum('ij,j->i', self.past, np.diff(de))
        f = -np.expm1(-self.t / tau)
        df = -self.t * np.exp(-self.t / tau) / tau**2
        f[-1], df[-1] = 1. / tau, -1. / tau**2
        return q[:, None] * f, dq[:, None] * f + q[:, None] * df, f, df

    def basis(self, tau):
        a, da, f, df = self._branch(tau[0])
        b, db, _, _ = self._branch(tau[1])
        carry = (self.work * self.hold)[:, None]
        basis = np.concatenate((a[..., None], b[..., None], self.steady, (carry * f)[..., None]), axis=-1)
        return basis, da, db, carry * df


def _fit(model, y):
    n, k = y.shape
    active = np.r_[np.ones(7, dtype=bool), np.any(model.over != 0), np.any(model.work * model.hold != 0)]

    def start(tau):
        a = model.basis(tau)[0][..., active]
        return np.r_[tau[0], tau[1] - tau[0], np.linalg.lstsq(a.reshape(n*k, -1), y.ravel(), rcond=None)[0]]

    def unpack(p):
        w = np.zeros(9)
        w[active] = p[2:]
        return np.array([p[0], p[0] + p[1]]), w

    def run(p, scale):
        def fun(v, jac=False):
            tau, w = unpack(v)
            a, da, db, dc = model.basis(tau)
            if not jac:
                return ((np.einsum('ijk,k->ij', a, w) - y) / scale).ravel()
            d1 = da * w[0] + dc * w[8]
            d2 = db * w[1]
            j = np.concatenate(((d1 + d2)[..., None], d2[..., None], a[..., active]), axis=-1)
            return (j / scale[None, :, None]).reshape(n*k, -1)
        return least_squares(fun, p, jac=lambda v: fun(v, True),
                             bounds=(np.r_[.04, .025, np.full(len(p)-2, -np.inf)],
                                     np.r_[12., 20., np.full(len(p)-2, np.inf)]),
                             loss='cauchy', x_scale='jac', ftol=1e-9, max_nfev=120)

    starts = [start(t) for t in ((.35, 2.), (.65, 4.), (1.2, 6.5), (.2, 7.))]
    tau, w = unpack(starts[1])
    scale = np.maximum(2 * _mad(model.basis(tau)[0] @ w - y, axis=0), 1e-7)
    best = min((run(p, scale) for p in starts), key=lambda r: r.cost)
    for _ in range(2):
        tau, w = unpack(best.x)
        residual = model.basis(tau)[0] @ w - y
        scale = np.maximum(2 * np.r_[np.full(k-1, _mad(residual[:, :-1])), _mad(residual[:, -1])], 1e-7)
        best = run(best.x, scale)
    return unpack(best.x)


def solve_alpha_m96_w04_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x = _aligned((train, test))
    n = len(train['lab_id'])
    coef = _calibration(dataset['calibration_load_cells'])
    args = (coef, dt, time, dataset['yield_stress'])
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None], train['instant_rate_obs']))
    tau, w = _fit(_Model(train, x[:n], *args), y)
    pred = _Model(test, x[n:], *args).basis(tau)[0] @ w
    # A measured initial rate identifies the steady drift left uncertain by missing features.
    # Correct only the linear-in-time term, leaving both memory branches unchanged.
    rate = np.asarray(test['instant_rate_obs'])
    refine = np.isnan(test['coupon_features_raw']).any(axis=1) & np.isfinite(rate)
    drift = 0.97 * (rate[refine] - pred[refine, -1])
    pred[refine, :-1] += drift[:, None] * time
    pred[refine, -1] += drift
    return dict(estimated_taus=tau, hereditary_weights=w[:7], viscoplastic_weights=w[7:],
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None], initial_rate_pred=pred[:, -1])
