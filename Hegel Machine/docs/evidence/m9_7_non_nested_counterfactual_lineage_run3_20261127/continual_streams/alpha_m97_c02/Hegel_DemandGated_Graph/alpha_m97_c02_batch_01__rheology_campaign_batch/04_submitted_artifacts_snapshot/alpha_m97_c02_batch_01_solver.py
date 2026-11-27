"""Load-cell calibration and two-branch strain-crystallization regression."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _historical_fit(lab):
    fits = []
    for sweep in json.loads(_HISTORY.read_text()):
        take = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[take]
        y = np.asarray(sweep['true_stress'])[take]
        if len(x) >= 2 and np.ptp(x) > 0:
            fits.append(_affine(x, y))
    return np.median(fits, axis=0)


def _calibrate(cal):
    fits = np.empty((4, 2))
    for lab in range(4):
        take = (cal['lab_id'] == lab) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][take], cal['true_stress'][take]
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            use = x[i] != x[j]
            i, j = i[use], j[use]
            if len(i):
                slope = np.median((y[j] - y[i]) / (x[j] - x[i]))
                intercept = np.median(y - slope*x)
                clean = np.abs(y - (slope*x + intercept)) <= 3.
                x, y = x[clean], y[clean]
        if len(x) >= 2 and np.ptp(x) > 0:
            fits[lab] = _affine(x, y)
        else:
            fits[lab] = _historical_fit(lab)
            if len(x):
                fits[lab, 1] = np.median(y - fits[lab, 0]*x)
    return fits


def _align(train, test):
    labs, strata, raw, refs = (np.concatenate((train[k], test[k])) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        take = labs == lab
        ref = refs[take]
        center = np.nanmedian(np.nanmean(ref, axis=0))
        scale = np.nanmedian(np.nanstd(ref, axis=0, ddof=0))
        aligned[take] = (raw[take] - center) / scale
    cell_means = []
    for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0):
        values = aligned[(labs == lab) & (strata == stratum)]
        counts = np.isfinite(values).sum(axis=0)
        cell_means.append(np.divide(np.nansum(values, axis=0), counts,
                                    out=np.full(raw.shape[1], np.nan), where=counts > 0))
    return np.where(np.isfinite(aligned), aligned, np.nanmean(cell_means, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return (gain[:, None]*split['past_stress_cmd'] + bias[:, None],
            gain*split['stress_hold_cmd'] + bias)


class _Design:
    def __init__(self, past, hold, x, dt, times, crit, shear):
        self.past, self.hold, self.dt, self.times = past, hold, dt, times
        self.later = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
        damage = np.sum(np.maximum(np.abs(past)-crit, 0.)**2 * dt, axis=1)
        self.active = (damage > 0.) & (shear <= .50)
        self.act = np.where(self.active, np.tanh(damage), 0.)
        # Shear selects the campaign drift law; damage gates only its nonlinear corrections.
        phi = np.log1p(times) if shear <= .50 else times
        self.phi = np.broadcast_to(phi, (len(x), len(times)))
        self.drift = np.column_stack((np.ones(len(x)), x))

    def matrices(self, tau):
        n, k = self.phi.shape
        kernel = -np.expm1(-self.dt[:, None]/tau) * np.exp(-self.later[:, None]/tau)
        z = self.past @ kernel
        gap = self.hold[:, None] - z
        resp = -np.expm1(-self.times[:, None]/tau)
        u, v = self.act*self.hold, self.act*z[:, 0]
        a = np.empty((n, k, 9))
        a[:, :, :2] = gap[:, None, :] * resp
        a[:, :, 2:7] = self.drift[:, None, :] * self.phi[:, :, None]
        a[:, :, 7] = u[:, None] * resp[:, 0]
        a[:, :, 8] = v[:, None] * self.phi
        b = np.column_stack((gap/tau, self.drift, u/tau[0], v))
        return a, b


def _fit(design, curves, rates):
    target = np.r_[curves.ravel(), rates]
    valid = np.isfinite(target)
    is_rate = np.arange(len(target))[valid] >= curves.size
    target = target[valid]
    regime = np.any(design.active)
    p = 9 if regime else 7
    lo = np.log(min(design.dt.min(), design.times[design.times > 0].min()) / 30.)
    hi = np.log(design.times.max()*30.)

    def matrix(logtau):
        a, b = design.matrices(np.exp(logtau))
        return np.vstack((a.reshape(-1, 9), b))[valid, :p]

    def residual(params):
        return np.einsum('ij,j->i', matrix(params[:2]), params[2:]) - target

    def jac(params):
        d = matrix(params[:2])
        diff = []
        for step in ([1e-4, 0.], [0., 1e-4]):
            deriv = (matrix(params[:2]+step) - matrix(params[:2]-step)) / 2e-4
            diff.append(np.einsum('ij,j->i', deriv, params[2:]))
        return np.column_stack((*diff, d))

    def optimize(params, scale, loss):
        return least_squares(lambda q: residual(q)/scale, params,
                             jac=lambda q: jac(q)/scale[:, None], loss=loss,
                             bounds=(np.r_[[lo, lo], np.full(p, -np.inf)],
                                     np.r_[[hi, hi], np.full(p, np.inf)]),
                             x_scale='jac', max_nfev=100)

    scale = np.full(len(target), .05)
    trials = []
    for tau in ([.16, 1.1], [.38, 2.7], [.85, 5.5]):
        logtau = np.clip(np.log(tau), lo+1e-6, hi-1e-6)
        weights = np.linalg.lstsq(matrix(logtau), target, rcond=None)[0]
        trials.append(optimize(np.r_[logtau, weights], scale, 'soft_l1'))
    best = min(trials, key=lambda r: r.cost).x
    errors = residual(best)
    for mask in (is_rate, ~is_rate):
        e = errors[mask]
        if e.size:
            scale[mask] = 2.*max(1e-5, 1.4826*np.median(np.abs(e - np.median(e))))
    best = optimize(best, scale, 'cauchy').x
    taus = np.exp(best[:2])
    weights = np.zeros(9)
    weights[:p] = best[2:]
    if not regime and taus[0] > taus[1]:
        taus = taus[::-1]
        weights[:2] = weights[1::-1]
    return taus, weights


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    alignment = _align(train, test)
    calibration = _calibrate(dataset['calibration_load_cells'])
    common = (dataset['past_dt'], dataset['eval_times'], dataset['crit_stress'],
              dataset['shear_banding_index'])
    # Do not let imputed covariates bias the physical parameter estimates.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    train_past, train_hold = _physical(train, calibration)
    train_design = _Design(train_past[complete], train_hold[complete], alignment[:n][complete], *common)
    curves = train['strain_curves_obs'][complete] - train['strain_0'][complete, None]
    taus, weights = _fit(train_design, curves, train['instant_rate_obs'][complete])
    test_design = _Design(*_physical(test, calibration), alignment[n:], *common)
    curves, rates = test_design.matrices(taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
