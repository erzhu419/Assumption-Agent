import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _mad(x, axis=None):
    return 1.4826 * np.median(abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)


def _checks(checks, lab):
    mask = np.asarray(checks['lab_id']) == lab
    return tuple(np.asarray(checks[k], dtype=float)[mask] for k in ('cmd_stress', 'true_stress'))


def _calibrate(checks):
    history = json.loads(_HISTORY.read_text()) if _HISTORY.exists() else []
    coef = np.empty((4, 2))
    for lab in range(4):
        prior = np.array([1., 0.])
        has_prior = False
        for sweep in history:
            x, y = _checks(sweep, lab)
            if len(x) >= 2 and np.ptp(x) > 0.1:
                prior = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
                has_prior = True
        x, y = _checks(checks, lab)
        if not len(x):
            coef[lab] = prior
            continue
        a = np.column_stack((x, np.ones(len(x))))
        if has_prior and len(x) <= 2:
            # One or two checks cannot distinguish calibration drift from EMI.
            # Update only from checks compatible with the previous sweep.
            offset = y - a @ prior
            keep = abs(offset) <= 0.003
            if keep.sum() == 2 and np.ptp(x) >= 0.1:
                p = np.linalg.solve(a, y)
                if abs(p[0] - prior[0]) <= 0.003 and abs(p[1] - prior[1]) <= 0.003:
                    coef[lab] = p
                    continue
            if np.any(keep):
                prior[1] += np.median(offset[keep])
            coef[lab] = prior
            continue
        candidates = [prior, [prior[0], np.median(y - prior[0] * x)]]
        for i in range(len(x)):
            for j in range(i):
                if abs(x[i] - x[j]) >= 0.1:
                    gain = (y[i] - y[j]) / (x[i] - x[j])
                    if gain > 0:
                        candidates.append([gain, y[i] - gain * x[i]])
        tol = 8e-4
        def score(p):
            error = (a @ p - y) / tol
            return (np.minimum(error**2, 1).sum()
                    + 0.12 * np.log1p(np.sum(((p - prior) / 0.03)**2)))
        p = np.asarray(min(candidates, key=score))
        keep = abs(a @ p - y) < tol
        if keep.sum() >= 2 and np.ptp(x[keep]) >= 0.1:
            p = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
        coef[lab] = p
    return coef


def _align(train, test):
    lab, stratum, raw, ref = (np.concatenate((train[k], test[k]), axis=0)
                              for k in ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    x = np.empty_like(raw, dtype=float)
    means = []
    for c in np.unique(lab):
        mask = lab == c
        panel = ref[mask]
        usable = np.sum(np.isfinite(panel), axis=0) >= 2
        panel = panel[:, usable]
        loc = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (raw[mask] - loc) / scale
        for group in np.unique(stratum[mask]):
            block = x[mask & (stratum == group)]
            count = np.isfinite(block).sum(axis=0)
            means.append(np.divide(np.nansum(block, axis=0), count,
                                   out=np.full(raw.shape[1], np.nan), where=count > 0))
    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0))


class _Creep:
    def __init__(self, part, features, coef, dt, time):
        self.time = np.asarray(time)
        self.suffix = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
        gain, bias = coef[np.asarray(part['lab_id'])].T
        self.past = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
        self.hold = part['stress_hold_cmd'] * gain + bias
        self.initial = part['initial_memory_states']
        self.drift = np.column_stack((np.ones(len(features)), features))[:, None, :] * np.r_[time, 1.][None, :, None]

    def branch(self, mode, tau):
        e = np.exp(-self.suffix / tau)
        de = self.suffix * e / tau**2
        q = self.hold - self.past @ np.diff(e) - self.initial[:, mode] * e[0]
        dq = -self.past @ np.diff(de) - self.initial[:, mode] * de[0]
        f = np.r_[-np.expm1(-self.time / tau), 1 / tau]
        df = np.r_[-self.time * np.exp(-self.time / tau) / tau**2, -1 / tau**2]
        return q[:, None] * f, dq[:, None] * f + q[:, None] * df

    def basis(self, taus):
        a, da = self.branch(0, taus[0])
        b, db = self.branch(1, taus[1])
        return np.concatenate((a[..., None], b[..., None], self.drift), axis=-1), da, db


def _fit(model, targets, mask=slice(None)):
    y = targets[mask]
    def design(p):
        return tuple(v[mask] for v in model.basis(p))
    def run(start, scale):
        def residual(p):
            a, _, _ = design(p[:2])
            return ((a @ p[2:] - y) / scale).ravel()
        def jacobian(p):
            a, da, db = design(p[:2])
            j = np.concatenate(((da * p[2])[..., None], (db * p[3])[..., None], a), axis=-1)
            return (j / scale[:, None]).reshape(-1, 9)
        return least_squares(residual, start, jac=jacobian, loss='soft_l1', x_scale='jac',
                             bounds=(np.r_[.051, .051, np.full(7, -np.inf)],
                                     np.r_[24.99, 24.99, np.full(7, np.inf)]),
                             max_nfev=100, ftol=1e-10)

    starts = []
    for taus in ([.35, 2.], [.65, 4.], [1.2, 6.5], [.22, 10.]):
        a, _, _ = design(taus)
        w = np.linalg.lstsq(a.reshape(-1, 7), y.ravel(), rcond=None)[0]
        starts.append(np.r_[taus, w])
    p = starts[1]
    scale = np.maximum(2 * _mad(design(p[:2])[0] @ p[2:] - y, axis=0), 1e-6)
    p = min((run(p, scale) for p in starts), key=lambda r: r.cost).x
    for _ in range(2):
        r = design(p[:2])[0] @ p[2:] - y
        scale = 2 * np.r_[np.full(y.shape[1] - 1, _mad(r[:, :-1])), _mad(r[:, -1])]
        p = run(p, np.maximum(scale, 1e-6)).x
    return p


def _box_simplex(s, y, ridge, upper, budget):
    # Feasible active-set Newton steps, restricted to the sum-zero free face.
    n = s.shape[1]
    h = s.T @ s + ridge * np.eye(n)
    w = np.full(n, budget / n)
    face = np.zeros(n, dtype=int)
    for _ in range(10 * n * n):
        g = s.T @ (s @ w - y) + ridge * w
        free = face == 0
        if not np.any(free):
            lo = np.flatnonzero(face == -1)
            hi = np.flatnonzero(face == 1)
            if not len(lo) or not len(hi):
                nu = float(np.min(g) if len(lo) else np.max(g))
                break
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            nu = float((g[i] + g[j]) / 2)
            if g[i] >= g[j] - 1e-11:
                break
            face[[i, j]] = 0
            continue
        inv_g, inv_one = np.linalg.solve(h[np.ix_(free, free)],
                                         np.column_stack((g[free], np.ones(free.sum())))).T
        nu = float(inv_g.sum() / inv_one.sum())
        step = np.zeros(n)
        step[free] = nu * inv_one - inv_g
        if np.max(abs(step)) <= 1e-12:
            violation = face * (g - nu)
            j = int(np.argmax(violation))
            if violation[j] <= 1e-10:
                break
            face[j] = 0
            continue
        limits = np.full(n, np.inf)
        neg, pos = step < -1e-14, step > 1e-14
        limits[neg] = -w[neg] / step[neg]
        limits[pos] = (upper - w[pos]) / step[pos]
        j = int(limits.argmin())
        alpha = min(1., limits[j])
        w = np.clip(w + alpha * step, 0., upper)
        if alpha < 1.:
            face[j] = 1 if step[j] > 0 else -1
            w[j] = upper if face[j] == 1 else 0.
    else:
        raise RuntimeError('Allocation active set did not converge')
    r = s @ w - y
    g = s.T @ r + ridge * w
    primal = float(0.5 * (r @ r + ridge * (w @ w)))
    shifted = g - nu
    # The box-constrained supporting hyperplane gives a global lower bound.
    dual = float(primal - g @ w + nu * budget + upper * np.minimum(0., shifted).sum())
    return dict(weights=w, nu_sum=nu, lam_lower=np.maximum(shifted, 0.),
                lam_upper=np.maximum(-shifted, 0.), primal_objective=primal,
                dual_lower_bound=min(dual, primal))


def solve_alpha_m96_w04_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x = _align(train, test)
    n = len(train['lab_id'])
    coef = _calibrate(dataset['calibration_load_cells'])
    args = coef, dataset['past_dt'], dataset['eval_times']
    model = _Creep(train, x[:n], *args)
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    p = _fit(model, y)
    response = _Creep(test, x[n:], *args).basis(p[:2])[0] @ p[2:]
    delta = response[:, :-1]
    # The rate residual estimates the linear drift left unresolved by imputation.
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    delta[missing] += (np.asarray(test['instant_rate_obs'])[missing] - response[missing, -1])[:, None] * args[2]
    pred = delta + test['strain_0'][:, None]
    s = dataset['S_alloc_modes'] * (1. + .25 * np.tanh(pred[:, -1] - test['strain_0']))[:, None]
    z = np.asarray(dataset['Z_nuisance'])
    yz = np.asarray(dataset['y_alloc_obs']).ravel()
    pz = np.linalg.solve(z.T @ z + dataset['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
    cert = _box_simplex(s - z @ (pz @ s), yz - z @ (pz @ yz), dataset['source_ridge'],
                        dataset['u_box'], dataset['total_budget'])
    return dict(estimated_taus=np.sort(p[:2]), predicted_strain=pred,
                allocation_weights=cert['weights'], kkt_certificate=cert)
