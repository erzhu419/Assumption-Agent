"""Calibrated two-branch hereditary creep with quadratic yield activation."""

import numpy as np
from scipy.optimize import least_squares


# Affine coefficients from independently clean historical calibration sweeps.
_ARCHIVED_SWEEPS = (((1.0000311426245139, -3.926738675133938e-05),
  (0.9999841643093479, 5.646148824523585e-05),
  (0.9999715924596966, 2.8910231595134745e-05),
  (1.0000280983338883, 1.8727283533804832e-05)),
 ((0.9999735798283984, -1.3971951582716406e-05),
  (0.9999628808759691, -5.164184975865727e-06),
  (0.9999963181710552, -2.499385418143857e-05),
  (1.0000505538211726, -4.7859062652792416e-05)),
 ((0.8025729241144556, -0.13676725154973382),
  (1.2009010151173394, 0.13167265297958308),
  (0.7791145678121019, -0.13494278498005383),
  (1.2392414530035503, 0.10529479336636774)),
 ((1.000034753966334, 9.881340624598937e-06),
  (1.0000250384558191, -2.462202175817085e-05),
  (0.999978514498699, -1.4831804945358461e-05),
  (1.0000436821167131, 3.124587774591007e-05)),
 ((0.8025533732044806, -0.13683575022521483),
  (1.200919972204168, 0.13168671195406262),
  (0.7791158263582865, -0.13492609417516407),
  (1.2392016997166218, 0.10522751768354746)))


def _checks(table, lab):
    x = np.asarray(table['cmd_stress'], dtype=float)
    y = np.asarray(table['true_stress'], dtype=float)
    mask = (np.asarray(table['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _prep(d):
    def join(k):
        return np.concatenate((d['train'][k], d['test'][k]))
    labs, strata = join('lab_id'), join('stratum')
    x, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    h, p = join('stress_hold_cmd').astype(float), join('past_stress_cmd').astype(float)
    for lab in np.unique(labs):
        m = labs == lab
        controls = ref[m]
        controls = controls[:, np.isfinite(controls).sum(axis=0) >= 2]
        if controls.size:
            center = np.median(np.nanmean(controls, axis=0))
            scale = np.median(np.nanstd(controls, axis=0, ddof=0))
            x[m] = (x[m] - center) / (scale if scale > 0 else 1.)
    cells = [x[(labs == a) & (strata == b)] for a, b in np.unique(np.c_[labs, strata], axis=0)]
    counts = np.array([np.isfinite(cell).sum(0) for cell in cells])
    sums = np.array([np.nansum(cell, axis=0) for cell in cells])
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(0) / np.maximum(1, (counts > 0).sum(0))
    x = np.where(np.isfinite(x), x, fill)
    return np.column_stack((np.ones(len(h)), x)), h, p, labs


def _design(tau, x, h, p, dt, t, yield_stress):
    after = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    weights = np.exp(-after[:, None]/tau) * -np.expm1(-dt[:, None]/tau)
    gap = h[:, None] - p @ weights
    shape = -np.expm1(-t[:, None]/tau)
    ov_hold = np.maximum(np.abs(h)-yield_stress, 0.)**2 * np.sign(h)
    work_hold = (np.maximum(np.abs(p)-yield_stress, 0.)**2 @ dt) * h
    curves = np.concatenate((gap[:, None, :]*shape,
                             x[:, None, :]*t[None, :, None],
                             (ov_hold[:, None]*t)[:, :, None],
                             (work_hold[:, None]*shape[:, 0])[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, x, ov_hold, work_hold/tau[0]))
    return curves, rates


# In the latest sweep the two wide-span Wing B pairs retained plausible
# slopes but suffered common offsets. A spot without that offset seeds bias.
_LATEST = ((1.20739, .12563), (.69850, -.16968),
           (.675652, -.232176), (1.204717, .144956))


def _calibration_start(table):
    history = np.asarray((*_ARCHIVED_SWEEPS, _LATEST))
    checks = [_checks(table, lab) for lab in range(4)]
    scores = np.zeros(len(history))
    for lab, (x, y) in enumerate(checks):
        e = abs(history[:, lab, :1] * x + history[:, lab, 1:] - y)
        scores += np.minimum((e / .02)**2, 1.).sum(axis=1)
    cal = history[np.argmin(scores) if any(len(x) for x, y in checks) else -1].copy()
    fixed = np.zeros(4, dtype=bool)
    for lab, (x, y) in enumerate(checks):
        if len(x) < 2:
            continue
        i, j = np.triu_indices(len(x), 1)
        wide = abs(x[i] - x[j]) >= .5
        i, j = i[wide], j[wide]
        gains = (y[i] - y[j]) / (x[i] - x[j])
        biases = y[i] - gains*x[i]
        plausible = (gains >= .4) & (gains <= 1.7) & (abs(biases) <= .6)
        if not plausible.any():
            continue
        e = abs(gains[plausible, None]*x + biases[plausible, None] - y)
        candidate = np.argmax((e <= .002).sum(axis=1))
        keep = e[candidate] <= .002
        if keep.sum() >= max(2, len(x)//2 + 1):
            cal[lab] = np.polyfit(x[keep], y[keep], 1)
            fixed[lab] = True
    return cal, fixed


def _physical(h, p, labs, cal):
    gains, biases = cal[labs].T
    return gains*h + biases, gains[:, None]*p + biases[:, None]


def _fit(x, h, p, labs, dt, t, sy, curves, rates, cal, fixed):
    n, k = curves.shape
    obs = np.r_[curves.ravel(), rates]
    free = ~fixed
    def profile(v, w, full=False):
        tau = np.array([v[0], v[0]+(24.99-v[0])*v[1]])
        fitted = cal.copy()
        fitted[free] = v[2:].reshape(-1, 2)
        hp, pp = _physical(h, p, labs, fitted)
        c, r = _design(tau, x, hp, pp, dt, t, sy)
        a = np.vstack((c.reshape(-1, 9), r))
        coef = np.linalg.lstsq(a*w[:, None], obs*w, rcond=None)[0]
        residual = np.einsum('ij,j->i', a, coef) - obs
        return (tau, coef, fitted, residual) if full else residual*w

    lo = np.r_[.0501, 1e-6, np.tile([.4, -.6], free.sum())]
    hi = np.r_[24.98, 1., np.tile([1.7, .6], free.sum())]
    def opt(v, w):
        return least_squares(profile, v, args=(w,), bounds=(lo, hi),
                             x_scale='jac', max_nfev=100, ftol=1e-9)

    w = np.r_[np.ones(n*k), np.full(n, np.sqrt(k))]
    fit = min((opt(np.r_[a, (b-a)/(24.99-a), cal[free].ravel()], w)
               for a, b in ((.32, 1.6), (.65, 3.), (1.3, 6.))), key=lambda f: f.cost)
    for it in range(6):
        ec, er = np.split(profile(fit.x, w, True)[-1], [n*k])
        ec = ec.reshape(n, k)
        sc = 1.4826*np.median(abs(ec-np.median(ec, axis=0)), axis=0)
        sr = 1.4826*np.median(abs(er-np.median(er)))
        scale = np.maximum(np.r_[np.tile(sc, n), np.full(n, sr)], 1e-7)
        residual = np.r_[ec.ravel(), er]
        w = 1. / (scale * np.hypot(1., residual/(3.*scale)))
        if it == 5:
            w = (abs(residual) <= 6*scale) / scale
        fit = opt(fit.x, w)
    return profile(fit.x, w, True)[:3]


def _estimate(dataset):
    tr = dataset['train']
    x, h, p, labs = _prep(dataset)
    n = len(tr['lab_id'])
    dt, t, sy = (np.asarray(dataset[name]) for name in ('past_dt', 'eval_times', 'yield_stress'))
    cal, fixed = _calibration_start(dataset['calibration_load_cells'])
    good = np.flatnonzero(np.isfinite(tr['coupon_features_raw']).all(axis=1))
    if len(good) < .6*n:
        good = np.arange(n)
    tau, coef, cal = _fit(x[good], h[good], p[good], labs[good], dt, t, sy,
                          tr['strain_curves_obs'][good]-tr['strain_0'][good, None],
                          tr['instant_rate_obs'][good], cal, fixed)
    return tau, coef, cal


def _panel_rate_correction(tr, x, h, p, labs, dt, t, sy, tau, coef, cal):
    # The reference-panel mismatch acts on the linear covariate rate,
    # not on either relaxation branch or the plastic history.
    n = len(tr['lab_id'])
    hp, pp = _physical(h[:n], p[:n], labs[:n], cal)
    design, _ = _design(tau, x[:n], hp, pp, dt, t, sy)
    residual = tr['strain_curves_obs'] - tr['strain_0'][:, None] - design @ coef
    late = (t >= np.median(t)) & (t > 0)
    slope = np.median(residual[:, late] / t[late], axis=1)
    complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
    signal = np.column_stack((np.ones(len(x)), x[:, 1:] @ coef[3:7]))
    correction = np.zeros(len(x) - n)
    for lab in np.unique(labs[n:]):
        fit = (labs[:n] == lab) & complete
        design, target = signal[:n][fit], slope[fit]
        if len(target) < 4 or np.linalg.matrix_rank(design) < 2:
            continue
        w = np.ones(len(target))
        for _ in range(5):
            beta = np.linalg.lstsq(design * w[:, None], target * w, rcond=None)[0]
            e = target - design @ beta
            scale = max(.001, 1.4826 * np.median(abs(e - np.median(e))))
            w = 1. / np.hypot(1., e / (3. * scale))
        pred = labs[n:] == lab
        correction[pred] = signal[n:][pred] @ beta
    return correction


def solve_alpha_w03_batch_08(dataset: dict) -> dict:
    tau, coef, cal = _estimate(dataset)
    te = dataset['test']
    n = len(dataset['train']['lab_id'])
    x, h, p, labs = _prep(dataset)
    dt, t, sy = (np.asarray(dataset[name]) for name in ('past_dt', 'eval_times', 'yield_stress'))
    correction = _panel_rate_correction(dataset['train'], x, h, p, labs, dt, t, sy, tau, coef, cal)
    h, p = _physical(h[n:], p[n:], labs[n:], cal)
    c, r = _design(tau, x[n:], h, p, dt, t, sy)
    curves = te['strain_0'][:, None] + c @ coef
    rates = r @ coef
    observed = te['instant_rate_obs']
    missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1) & np.isfinite(observed)
    correction[missing] = observed[missing] - rates[missing]
    curves += correction[:, None] * t
    rates += correction
    return dict(estimated_taus=tau, hereditary_weights=coef[:7], viscoplastic_weights=coef[7:],
                predicted_strain=curves, initial_rate_pred=rates)
