import numpy as np
from scipy.optimize import least_squares


def _mad(values, axis=None):
    center = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - center), axis=axis)


def _calibration(checks):
    lab, x, y = (np.asarray(checks[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    coef = np.empty((4, 2))
    for c in range(4):
        u, v = x[lab == c], y[lab == c]
        i, j = np.triu_indices(len(u), 1)
        keep = np.abs(u[i] - u[j]) > 1e-8
        i, j = i[keep], j[keep]
        if not len(i):
            coef[c] = (1., np.median(v - u) if len(u) else 0.)
            continue
        gain = np.median((v[i] - v[j]) / (u[i] - u[j]))
        bias = np.median(v - gain * u)
        a = np.column_stack((u, np.ones(len(u))))
        scale = max(1e-8, 2.0 * _mad(a @ [gain, bias] - v))
        coef[c] = least_squares(lambda p: a @ p - v, [gain, bias], jac=lambda p: a.copy(),
                                loss='cauchy', f_scale=scale).x
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
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    observed = np.isfinite(features)
    means = []
    for c, group in cells:
        m = (lab == c) & (stratum == group)
        count = observed[m].sum(axis=0)
        total = np.nansum(features[m], axis=0)
        means.append(np.divide(total, count, out=np.full(features.shape[1], np.nan), where=count > 0))
    fill = np.nanmean(means, axis=0)
    return np.where(observed, features, fill)


class _Model:
    def __init__(self, part, features, coef, dt, time):
        self.t = np.r_[time, 0.]
        self.suffix = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
        gain, bias = coef[np.asarray(part['lab_id'])].T
        self.past = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
        self.hold = part['stress_hold_cmd'] * gain + bias
        self.drift = np.column_stack((np.ones(len(features)), features))
        self.drift_basis = self.drift[:, None, :] * np.r_[time, 1.][None, :, None]

    def branch(self, tau):
        e = np.exp(-self.suffix / tau)
        de = e * self.suffix / tau**2
        q = self.hold - np.einsum('ij,j->i', self.past, np.diff(e))
        dq = -np.einsum('ij,j->i', self.past, np.diff(de))
        f = -np.expm1(-self.t / tau)
        df = -self.t * np.exp(-self.t / tau) / tau**2
        f[-1], df[-1] = 1. / tau, -1. / tau**2
        return q[:, None] * f, dq[:, None] * f + q[:, None] * df

    def basis(self, taus):
        a, da = self.branch(taus[0])
        b, db = self.branch(taus[1])
        return np.concatenate((a[..., None], b[..., None], self.drift_basis), axis=-1), da, db


def _fit(model, targets):
    n, k = targets.shape
    y = targets.ravel()
    bounds = (np.r_[[.2, .2], np.full(7, -np.inf)],
              np.r_[[8., 8.], np.full(7, np.inf)])

    def run(start, scales):
        scale = np.broadcast_to(scales, (n, k)).ravel()
        def fun(p):
            a, _, _ = model.basis(p[:2])
            return (np.einsum('ijk,k->ij', a, p[2:]).ravel() - y) / scale
        def jac(p):
            a, da, db = model.basis(p[:2])
            mat = np.concatenate(((da*p[2])[..., None], (db*p[3])[..., None], a), axis=-1)
            return mat.reshape(-1, 9) / scale[:, None]
        return least_squares(fun, start, jac=jac, bounds=bounds, loss='cauchy',
                             x_scale='jac', max_nfev=160, ftol=1e-9)

    initial = []
    for taus in ([.35, 2.0], [.65, 4.0], [1.2, 6.5], [.22, 7.0]):
        a = model.basis(taus)[0].reshape(-1, 7)
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        initial.append(np.r_[taus, w])
    rough = _mad((model.basis(initial[1][:2])[0] @ initial[1][2:] - targets), axis=0)
    scales = np.maximum(2.0 * rough, 1e-6)
    candidates = [run(p, scales) for p in initial]
    best = min(candidates, key=lambda r: r.cost)
    for _ in range(2):
        a = model.basis(best.x[:2])[0]
        residual = a @ best.x[2:] - targets
        curve = max(float(_mad(residual[:, :-1])), 1e-6)
        rate = max(float(_mad(residual[:, -1])), 1e-6)
        scales = np.r_[np.full(k-1, 2.*curve), 2.*rate]
        best = run(best.x, scales)
    p = best.x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p


def solve_alpha_m96_w04_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    time, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    coef = _calibration(dataset['calibration_load_cells'])
    x = _aligned((train, test))
    n = len(train['lab_id'])
    model = _Model(train, x[:n], coef, dt, time)
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    p = _fit(model, y)
    pred = _Model(test, x[n:], coef, dt, time).basis(p[:2])[0] @ p[2:]
    return dict(estimated_taus=p[:2], hereditary_weights=p[2:], viscoplastic_weights=np.zeros(2),
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None], initial_rate_pred=pred[:, -1])
