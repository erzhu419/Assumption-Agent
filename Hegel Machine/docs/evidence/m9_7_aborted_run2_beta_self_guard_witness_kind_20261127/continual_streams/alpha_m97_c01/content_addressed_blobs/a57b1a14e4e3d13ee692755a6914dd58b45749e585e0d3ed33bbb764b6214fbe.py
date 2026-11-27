"""Fit the calibrated two-branch jammed-slip rheology model."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    lab, cmd, true = (np.asarray(rows[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    params = np.empty((4, 2))
    history = None
    for c in range(4):
        mask = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[mask], true[mask]
        clean = np.ones(len(x), dtype=bool)
        emi = False
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[i] != x[j]
            i, j = i[distinct], j[distinct]
            slopes = (y[j] - y[i]) / (x[j] - x[i])
            errors = np.abs(y[i, None] + slopes[:, None] * (x - x[i, None]) - y)
            # A three-point sweep cannot identify a bad row by pair consensus.
            # Inconsistent triples with a possible EMI excursion use the prefix
            # sweep to identify the contaminated point, not to fit clean rows.
            typical = np.median(errors, axis=1)
            tied = np.flatnonzero(typical <= typical.min() + 1e-12)
            best = tied[np.argmin(np.abs(slopes[tied] - np.median(slopes)))]
            core = errors[best] <= 8 * typical[best] + 1e-12
            residual = np.abs(np.polyval(_line(x[core], y[core]), x) - y)
            emi = np.any(residual > 3.0)
            clean = residual <= 3.0
            if len(x) == 3:
                worst_by_pair = errors.max(axis=1)
                emi = worst_by_pair.min() > 0.01 and worst_by_pair.max() > 3.0
        xc, yc = x[clean], y[clean]
        if len(xc) >= 4 or (len(x) >= 2 and not emi):
            if not emi:
                xc, yc = x, y
            params[c] = _line(xc, yc)
            continue
        if history is None:
            history = json.loads(_HISTORY.read_text())
        for sweep in reversed(history):
            m = np.asarray(sweep['lab_id']) == c
            hx, hy = (np.asarray(sweep[k])[m] for k in ('cmd_stress', 'true_stress'))
            if len(hx) >= 2:
                old = _line(hx, hy)
                break
        if len(x) < 4 or emi:
            clean = np.abs(np.polyval(old, x) - y) <= 3.0
            xc, yc = x[clean], y[clean]
        if len(xc) >= 2:
            params[c] = _line(xc, yc)
        elif len(xc) == 1:
            params[c] = old[0], yc[0] - old[0] * xc[0]
        else:
            params[c] = old
    return params


def _prepare(dataset):
    def stack(name):
        return np.concatenate([dataset[s][name] for s in ('train', 'test')])

    lab, strata = stack('lab_id'), stack('stratum')
    ref, aligned = stack('ref_coupons'), stack('coupon_features_raw')
    for c in range(4):
        m = lab == c
        if np.any(m):
            center = np.nanmedian(np.nanmean(ref[m], axis=0))
            scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
            aligned[m] = (aligned[m] - center) / scale
    cells = []
    for c, s in np.unique(np.column_stack((lab, strata)), axis=0):
        a = aligned[(lab == c) & (strata == s)]
        count = np.sum(np.isfinite(a), axis=0)
        cells.append(np.divide(np.nansum(a, axis=0), count,
                               out=np.full(a.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(cells, axis=0))
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, aligned, stack('strain_0')


class _Model:
    def __init__(self, dt, times, past, hold, features):
        self.dt = np.asarray(dt)
        self.after = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.times = np.asarray(times)
        self.past = past
        self.hold = hold
        base = np.column_stack((np.ones(len(hold)), features,
                                np.sign(hold) * np.sqrt(np.abs(hold))))
        phi = self.times / (1.0 + 0.85 * self.times)
        self.fixed = base[:, None, :] * np.r_[phi, 1.0][None, :, None]

    def design(self, taus):
        taus = np.asarray(taus)
        memory = -np.expm1(-self.dt[:, None] / taus) * np.exp(-self.after[:, None] / taus)
        gap = self.hold[:, None] - self.past @ memory
        kernel = np.vstack((-np.expm1(-self.times[:, None] / taus), 1.0 / taus))
        prony = gap[:, None, :] * kernel
        contrast = (gap[:, 0] - gap[:, 1])[:, None] * kernel[None, :, 0]
        return np.concatenate((prony, self.fixed, contrast[..., None]), axis=2)


def _initial_fit(model, y, mask=None, rate_weight=None):
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    valid = np.isfinite(y) & mask
    target = y[valid]
    n_curves = y.shape[1] - 1

    def run(weight, starts):
        scale = np.broadcast_to(np.r_[np.ones(n_curves), weight], y.shape)[valid]

        def evaluate(log_taus):
            design = model.design(np.exp(log_taus))[valid] * scale[:, None]
            coef = np.linalg.lstsq(design, target * scale, rcond=None)[0]
            resid = np.einsum('ij,j->i', design, coef) - target * scale
            return resid, coef

        opt = min((least_squares(lambda x: evaluate(x)[0], start,
                                 bounds=(np.log([0.03, 0.05]), np.log([50., 150.])),
                                 max_nfev=100, ftol=1e-10) for start in starts),
                  key=lambda o: o.cost)
        return np.exp(opt.x), evaluate(opt.x)[1]

    starts = np.log([[0.45, 4.0], [0.2, 2.0], [1.0, 10.0]])
    tau, coef = run(1.0 if rate_weight is None else rate_weight, starts)
    if rate_weight is None:
        r = np.einsum('ntp,p->nt', model.design(tau), coef) - y

        def mad(x):
            return max(np.median(np.abs(x - np.median(x))), 1e-8)

        if np.any(valid[:, -1]):
            w = np.clip(mad(r[:, :-1][valid[:, :-1]]) / mad(r[:, -1][valid[:, -1]]), 0.1, 10.0)
            tau, coef = run(w, [np.log(tau)])
    return tau, coef



def _fit(model, y, mask=None):
    valid = np.isfinite(y)
    if mask is not None:
        valid &= mask
    tau, coef = _initial_fit(model, y, mask)
    prediction = model.design(tau) @ coef
    residual = prediction - y
    sigma = np.ones(y.shape[1])
    for j in range(y.shape[1]):
        v = residual[valid[:, j], j]
        if v.size:
            sigma[j] = max(1.4826 * np.median(np.abs(v - np.median(v))), 1e-6)
    obs = y[valid]
    base = np.broadcast_to(1.0 / sigma, y.shape)[valid]
    log_tau = np.log(tau)
    for _ in range(12):
        u = (prediction[valid] - obs) * base
        scale = base / np.hypot(1.0, u / 2.5)
        target = obs * scale

        def evaluate(v):
            design = model.design(np.exp(v))[valid] * scale[:, None]
            b = np.linalg.lstsq(design, target, rcond=None)[0]
            return np.einsum('ij,j->i', design, b) - target, b

        opt = least_squares(lambda v: evaluate(v)[0], log_tau,
                            bounds=(np.log([0.03, 0.05]), np.log([50., 150.])),
                            max_nfev=40)
        log_tau = opt.x
        tau, coef = np.exp(log_tau), evaluate(log_tau)[1]
        updated = model.design(tau) @ coef
        if np.max(np.abs(updated - prediction)) < 1e-7:
            break
        prediction = updated
    return tau, coef


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    past, hold, aligned, strain0 = _prepare(dataset)
    train = dataset['train']
    n = len(train['lab_id'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    y = np.column_stack((train['strain_curves_obs'] - strain0[:n, None], train['instant_rate_obs']))
    tau, coef = _fit(_Model(dt, times, past[:n], hold[:n], aligned[:n]), y)
    design = _Model(dt, times, past[n:], hold[n:], aligned[n:]).design(tau)
    prediction = np.einsum('ntp,p->nt', design, coef)
    return dict(estimated_taus=tau, hereditary_weights=coef[:7], regime_mode_weights=coef[7:],
                predicted_strain=prediction[:, :-1] + strain0[n:, None],
                initial_rate_pred=prediction[:, -1])
