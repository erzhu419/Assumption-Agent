import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _mad(x, axis=None):
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)


def _checks(record):
    return tuple(np.asarray(record[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))


def _line(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx = x[i] - x[j]
    keep = np.abs(dx) > 1e-6
    if not keep.any():
        return None
    g = np.median((y[i] - y[j])[keep] / dx[keep])
    start = np.array([g, np.median(y - g*x)])
    a = np.column_stack((x, np.ones(len(x))))
    scale = max(1e-5, 1.5 * _mad(a @ start - y))
    return least_squares(lambda p: a @ p - y, start, loss='cauchy', f_scale=scale).x


def _calibration(record):
    lab, x, y = _checks(record)
    try:
        with open(_ARCHIVE) as stream:
            history = json.load(stream)
    except FileNotFoundError:
        history = []
    priors = []
    for sweep in history:
        c, u, v = _checks(sweep)
        rows = [_line(u[c == i], v[c == i]) for i in range(4)]
        if all(row is not None for row in rows):
            priors.append(rows)
    # Choose a prior regime jointly; a missing lab then inherits the regime
    # supported by the other current checks, not necessarily the latest sweep.
    prior = np.tile([1., 0.], (4, 1))
    if priors:
        def discrepancy(p):
            residual = y - p[lab, 0]*x - p[lab, 1]
            return np.mean([np.median(np.abs(residual[lab == c])) for c in np.unique(lab)]) if lab.size else 0.
        prior = min(np.asarray(priors)[::-1], key=discrepancy)
    coef = prior.copy()
    for c in range(4):
        u, v = x[lab == c], y[lab == c]
        if not len(u):
            continue
        dg = v - (prior[c, 0]*u + prior[c, 1])
        fit = _line(u, v)
        if fit is None:
            if np.median(np.abs(dg)) > .02:
                coef[c, 1] += np.median(dg)
            continue
        scale = max(.003, _mad(v - fit[0]*u - fit[1]))
        if not priors or np.median(np.abs(dg)) > min(.02, 1.5*scale):
            coef[c] = fit
    return coef


def _aligned(parts):
    lab, stratum, raw, ref = [np.concatenate([p[k] for p in parts], axis=0)
                              for k in ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons')]
    features = raw.copy()
    for c in np.unique(lab):
        m = lab == c
        panel = ref[m]
        loc = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        features[m] = (raw[m] - loc) / scale
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        rows = features[(lab == c) & (stratum == s)]
        count = np.isfinite(rows).sum(axis=0)
        means.append(np.divide(np.nansum(rows, axis=0), count,
                               out=np.full(raw.shape[1], np.nan), where=count > 0))
    return np.where(np.isfinite(features), features, np.nanmean(means, axis=0))


class _Model:
    def __init__(self, part, features, coef, dt, time, yield_stress):
        self.t = np.r_[time, 0.]
        self.suffix = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
        gain, bias = coef[np.asarray(part['lab_id'])].T
        self.past = part['past_stress_cmd']*gain[:, None] + bias[:, None]
        self.hold = part['stress_hold_cmd']*gain + bias
        ov = np.maximum(np.abs(self.hold) - yield_stress, 0.)**2 * np.sign(self.hold)
        work = np.maximum(np.abs(self.past) - yield_stress, 0.)**2 @ dt
        self.work_hold = work*self.hold
        linear = np.r_[time, 1.]
        drift = np.column_stack((np.ones(len(features)), features))
        self.drift = drift[:, None, :] * linear[None, :, None]
        self.overstress = ov[:, None] * linear

    def _relax(self, tau):
        e = np.exp(-self.t/tau)
        f, df = -np.expm1(-self.t/tau), -self.t*e/tau**2
        f[-1], df[-1] = 1./tau, -1./tau**2
        return f, df

    def _branch(self, tau, f, df):
        e = np.exp(-self.suffix/tau)
        q = self.hold - np.einsum('ij,j->i', self.past, np.diff(e))
        dq = -np.einsum('ij,j->i', self.past, np.diff(e*self.suffix/tau**2))
        return q[:, None]*f, dq[:, None]*f + q[:, None]*df

    def basis(self, taus):
        f, df = self._relax(taus[0])
        a, da = self._branch(taus[0], f, df)
        b, db = self._branch(taus[1], *self._relax(taus[1]))
        damage = self.work_hold[:, None]*f
        dd = self.work_hold[:, None]*df
        design = np.concatenate((a[..., None], b[..., None], self.drift,
                                 self.overstress[..., None], damage[..., None]), axis=-1)
        return design, da, db, dd


def _fit(model, target):
    # Keep tau_1 ordered throughout fitting: it also controls the plastic memory.
    top = 24.9
    used = np.r_[np.ones(7, dtype=bool), np.any(model.overstress), np.any(model.work_hold)]
    nw = int(used.sum())
    lower = np.r_[.051, 1e-5, np.full(nw, -np.inf)]
    upper = np.r_[24., .99999, np.full(nw, np.inf)]
    y = target.ravel()

    def unpack(p):
        taus = np.array([p[0], p[0] + (top-p[0])*p[1]])
        weights = np.zeros(9)
        weights[used] = p[2:]
        return taus, weights

    def run(start, scales):
        scale = np.broadcast_to(scales, target.shape).ravel()
        def fun(p):
            taus, w = unpack(p)
            a = model.basis(taus)[0]
            return ((a @ w).ravel() - y) / scale
        def jac(p):
            taus, w = unpack(p)
            a, da, db, dd = model.basis(taus)
            slow = db*w[1]
            fast = da*w[0] + dd*w[8]
            m = np.concatenate(((fast + (1-p[1])*slow)[..., None],
                                ((top-p[0])*slow)[..., None], a[..., used]), axis=-1)
            return m.reshape(-1, nw+2) / scale[:, None]
        return least_squares(fun, start, jac=jac, bounds=(lower, upper),
                             loss='cauchy', x_scale='jac', max_nfev=100, ftol=1e-9)

    starts, residuals = [], []
    for taus in ([.35, 2.], [.65, 4.], [1.2, 6.5], [.22, 7.]):
        a = model.basis(taus)[0][..., used].reshape(-1, nw)
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        starts.append(np.r_[taus[0], (taus[1]-taus[0])/(top-taus[0]), w])
        residuals.append((a @ w - y).reshape(target.shape))
    rough = min(residuals, key=lambda r: np.median(np.abs(r)))
    scales = np.maximum(2.*_mad(rough, axis=0), 1e-6)
    best = min((run(p, scales) for p in starts), key=lambda r: r.cost)
    for _ in range(2):
        taus, w = unpack(best.x)
        r = model.basis(taus)[0] @ w - target
        scales = np.maximum(2.*np.r_[np.full(target.shape[1]-1, _mad(r[:, :-1])), _mad(r[:, -1])], 1e-6)
        best = run(best.x, scales)
    return unpack(best.x)


def solve_alpha_m96_w04_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    time, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    coef = _calibration(dataset['calibration_load_cells'])
    x = _aligned((train, test))
    n = len(train['lab_id'])
    args = (coef, dt, time, dataset['yield_stress'])
    model = _Model(train, x[:n], *args)
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    taus, weights = _fit(model, y)
    pred = _Model(test, x[n:], *args).basis(taus)[0] @ weights
    return dict(estimated_taus=taus, hereditary_weights=weights[:7], viscoplastic_weights=weights[7:],
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None], initial_rate_pred=pred[:, -1])
