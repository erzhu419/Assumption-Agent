"""Calibrated two-branch hereditary response in the jammed-slip regime."""
import json

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = "/tmp/hegel_continual_archive/archive_calibration_history.json"


def _align(train, test):
    raw, ref, labs, strata = (np.concatenate([train[k], test[k]]) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    x = np.empty_like(raw)
    for lab in np.unique(labs):
        mask = labs == lab
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0))
        x[mask] = (raw[mask] - center) / scale
    cells = [np.flatnonzero((labs == lab) & (strata == stratum)) for lab, stratum in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = np.full((len(cells), x.shape[1]), np.nan)
    for i, rows in enumerate(cells):
        block = x[rows]
        count = np.isfinite(block).sum(axis=0)
        np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
    for i, rows in enumerate(cells):
        same_lab = [j for j, other in enumerate(cells) if labs[other[0]] == labs[rows[0]]]
        fill = means[i].copy()
        for k in np.flatnonzero(~np.isfinite(fill)):
            pool = means[same_lab, k]
            if not np.isfinite(pool).any():
                pool = means[:, k]
            fill[k] = np.nanmean(pool) if np.isfinite(pool).any() else 0.0
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], fill)
    return np.split(x, [len(train['lab_id'])])


def _calibrate(cal):
    coefficients = np.empty((4, 2))
    history = None
    # Measured clean sweeps have residuals below 2e-4 stress units.
    noise = 2e-4

    def archived_line(lab):
        nonlocal history
        if history is None:
            with open(_ARCHIVE) as f:
                history = json.load(f)
        sweep = next(row for row in reversed(history) if lab in row['lab_id'])
        keep = np.asarray(sweep['lab_id']) == lab
        return np.polyfit(np.asarray(sweep['cmd_stress'])[keep],
                          np.asarray(sweep['true_stress'])[keep], 1)

    labs = np.asarray(cal['lab_id'])
    for lab in range(4):
        x, y = (np.asarray(cal[key])[labs == lab]
                for key in ('cmd_stress', 'true_stress'))
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < 2:
            coefficients[lab] = archived_line(lab)
            continue
        design = np.column_stack((x, np.ones(len(x))))
        if len(x) == 2:
            # A two-point line has zero residual by construction. In this
            # campaign the nominal command is within half the EMI threshold
            # of physical stress on the calibration sweep, so screen against
            # the command before deciding whether the archive is needed.
            if np.any(np.abs(y - x) > 1.5):
                prior = archived_line(lab)
                if np.any(np.abs(y - design @ prior) > 3.0 - 2 * noise):
                    # At most one clean reading remains; it cannot identify
                    # both the gain and the bias.
                    coefficients[lab] = prior
                    continue
            coefficients[lab] = np.linalg.lstsq(design, y, rcond=None)[0]
            continue
        coefficients[lab] = np.linalg.lstsq(design, y, rcond=None)[0]
        candidates = []
        inconsistent = False
        possible_emi = False
        for j in range(len(x)):
            keep = np.arange(len(x)) != j
            if np.ptp(x[keep]) == 0:
                continue
            inverse = np.linalg.pinv(design[keep])
            line = inverse @ y[keep]
            predict = design[j] @ inverse
            deleted = abs(y[j] - predict @ y[keep])
            uncertainty = noise * (1 + np.abs(predict).sum())
            # Check line consistency separately from possible spike amplitude:
            # the highest-leverage reading can have the least precise prediction.
            # Three-point sweeps need a tighter consistency screen to
            # distinguish a borderline spike from a clean near-repeat.
            consistency_limit = uncertainty * (0.625 if len(x) == 3 else 1.0)
            inconsistent |= deleted > consistency_limit
            possible_emi |= deleted + uncertainty > 3.0
            error = np.max(np.abs(y[keep] - design[keep] @ line))
            if deleted + uncertainty > 3.0:
                candidates.append((error, keep, line))
        if not (inconsistent and possible_emi):
            continue
        prior = archived_line(lab)
        archive_error = np.abs(y - design @ prior)
        keep = archive_error <= 2 * noise
        if keep.sum() == len(x) - 1 and archive_error[~keep][0] > 3.0 - 2 * noise:
            # The archive identifies the spike independently of extrapolation.
            coefficients[lab] = (np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
                                 if np.ptp(x[keep]) >= 0.1 else prior)
            continue
        best_error = min(item[0] for item in candidates)
        plausible = [item for item in candidates if item[0] <= best_error + noise]
        _, keep, line = min(plausible, key=lambda item: np.max(
            np.abs(y[item[1]] - design[item[1]] @ prior)))
        # Do not estimate a new slope from two effectively repeated commands
        # when their clean readings already agree with the archived line.
        if np.ptp(x[keep]) < 0.1 and np.max(np.abs(y[keep] - design[keep] @ prior)) <= 2 * noise:
            line = prior
        coefficients[lab] = line
    return coefficients[:, 0], coefficients[:, 1]


def _design(split, aligned, gain, bias, dt, t):
    lab = split['lab_id']
    past = gain[lab, None] * split['past_stress_cmd'] + bias[lab, None]
    hold = gain[lab] * split['stress_hold_cmd'] + bias[lab]
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    flow = np.column_stack((np.ones(len(lab)), aligned, slip))
    phi = t / (1 + 0.85 * t)
    tail = flow[:, None, :] * phi[None, :, None]
    after = np.cumsum(dt[::-1])[::-1] - dt

    def evaluate(log_tau):
        tau = np.exp(log_tau)
        kernel = np.exp(-after[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
        gap = hold[:, None] - past @ kernel
        relax = -np.expm1(-t[:, None] / tau)
        cross = gap[:, 0] - gap[:, 1]
        curves = np.concatenate((gap[:, None, :] * relax, tail,
                                 (cross[:, None] * relax[:, 0])[:, :, None]), axis=2)
        rates = np.column_stack((gap / tau, flow, cross / tau[0]))
        return curves, rates

    return evaluate


def _fit(evaluate, curves, rates, dt, t):
    c_ok, r_ok = np.isfinite(curves), np.isfinite(rates)
    y = np.r_[curves[c_ok], rates[r_ok]]
    split = c_ok.sum()
    c_group = np.broadcast_to(np.arange(len(t)), curves.shape)[c_ok]
    groups = [np.flatnonzero(c_group == k) for k in range(len(t))]
    groups += [np.arange(split, len(y))]
    blocks = [slice(0, split), slice(split, len(y))]

    def matrix(p):
        a, b = evaluate(p)
        return np.concatenate((a[c_ok], b[r_ok]))

    def project(a, w):
        return np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]

    def profile(p, w):
        a = matrix(p)
        return w * (a @ project(a, w) - y)

    def scales(r):
        s = np.empty_like(r)
        for ix in groups:
            if len(ix):
                q = r[ix]
                s[ix] = max(1e-5, 1.4826 * np.median(np.abs(q - np.median(q))))
        return s

    low = min(np.min(dt), np.min(t[t > 0])) / 12
    high = max(dt.sum(), t.max()) * 12
    bounds = np.log([low, high])
    short = np.median(dt)
    long = np.median(t)
    starts = [(0.4 * short, long), (short, long), (2 * short, 2 * long), (short, 4 * long)]
    w = np.ones_like(y)
    for b in blocks:
        n = len(y[b]); w[b] = 1 / np.sqrt(max(n, 1))
    fits = [least_squares(profile, np.log(pair), args=(w,), bounds=bounds, max_nfev=75)
            for pair in starts]
    p = min(fits, key=lambda f: f.cost).x
    for _ in range(5):
        a = matrix(p)
        r = a @ project(a, w) - y
        w = 1 / np.hypot(scales(r), r / 3)
        p = least_squares(profile, p, args=(w,), bounds=bounds, max_nfev=50).x
    if p[0] > p[1]:
        p = least_squares(profile, p[::-1], args=(w,), bounds=bounds, max_nfev=75).x
    return p, project(matrix(p), w)


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    gain, bias = _calibrate(dataset['calibration_load_cells'])
    xtr, xte = _align(train, test)
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate = train['instant_rate_obs']
    p, beta = _fit(_design(train, xtr, gain, bias, dt, t),
                   np.where(complete[:, None], y, np.nan),
                   np.where(complete, rate, np.nan), dt, t)
    curve, initial = _design(test, xte, gain, bias, dt, t)(p)
    return dict(estimated_taus=np.exp(p), hereditary_weights=beta[:7],
                regime_mode_weights=beta[7:],
                predicted_strain=test['strain_0'][:, None] + curve @ beta,
                initial_rate_pred=initial @ beta)
