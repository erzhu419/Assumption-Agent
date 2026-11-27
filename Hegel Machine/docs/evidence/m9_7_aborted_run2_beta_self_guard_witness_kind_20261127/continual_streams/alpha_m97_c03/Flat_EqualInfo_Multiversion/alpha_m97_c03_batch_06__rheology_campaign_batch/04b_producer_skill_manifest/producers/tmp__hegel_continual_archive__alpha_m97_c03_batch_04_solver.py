"""Channel-calibrated two-branch strain-crystallization model."""
import json

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


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


def _design(sample, aligned, calibration, dt, times, threshold, banding):
    gain, bias = calibration[np.asarray(sample['lab_id'])].T
    history = gain[:, None] * np.asarray(sample['past_stress_cmd']) + bias[:, None]
    hold = gain * sample['stress_hold_cmd'] + bias
    damage = np.maximum(np.abs(history) - threshold, 0.0)**2 @ dt
    plateau = (banding <= 0.50) & (damage > 0)
    activation = np.where(plateau, np.tanh(damage), 0.0)
    flow = np.column_stack((np.ones(len(hold)), aligned))
    # In the low-banding phase, damage gates the two extra terms, not the base tail.
    tail = np.log1p(times) if banding <= 0.50 else times

    def evaluate(log_tau):
        tau = np.exp(log_tau)
        alpha = np.exp(-dt[:, None] / tau)
        beta = -np.expm1(-dt[:, None] / tau)
        z = np.zeros((len(hold), 2))
        for j in range(len(dt)):
            z = alpha[j] * z + beta[j] * history[:, j, None]
        gap = hold[:, None] - z
        relax = -np.expm1(-times[:, None] / tau)
        extra = activation * hold
        arrest = activation * z[:, 0]
        curves = np.concatenate((gap[:, None, :] * relax,
                                 flow[:, None, :] * tail[None, :, None],
                                 (extra[:, None] * relax[:, 0])[:, :, None],
                                 (arrest[:, None] * np.log1p(times))[:, :, None]), axis=2)
        rates = np.column_stack((gap / tau, flow, extra / tau[0], arrest))
        return curves, rates

    return evaluate


def _fit(evaluate, curves, rates, dt, times):
    curves_ok, rates_ok = np.isfinite(curves), np.isfinite(rates)
    y = np.r_[curves[curves_ok], rates[rates_ok]]
    group = np.r_[np.nonzero(curves_ok)[1], np.full(rates_ok.sum(), len(times))]
    counts = np.bincount(group, minlength=len(times) + 1)
    base_weight = np.where(group == len(times), 1.0, 1.0 / np.sqrt(len(times)))

    def matrix(p):
        a, b = evaluate(p)
        return np.concatenate((a[curves_ok], b[rates_ok]))

    def project(a, w):
        return np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]

    def residual(p, w):
        a = matrix(p)
        return (np.einsum('ij,j->i', a, project(a, w)) - y) * w

    short, long = np.median(dt), np.median(times)
    low, high = min(dt.min(), times[times > 0].min()) / 12, max(dt.sum(), times.max()) * 12
    bounds = np.log([low, high])
    starts = [(0.4*short, long), (short, long), (2*short, 2*long), (short, 4*long)]
    fits = [least_squares(residual, np.log(pair), args=(base_weight,), bounds=bounds,
                          max_nfev=70) for pair in starts]
    p = np.sort(min(fits, key=lambda fit: fit.cost).x)
    w = base_weight
    for _ in range(4):
        r = residual(p, np.ones_like(y))
        scale = np.empty_like(y)
        for j in np.flatnonzero(counts):
            ix = group == j
            q = r[ix]
            scale[ix] = max(1e-5, 1.4826 * np.median(np.abs(q - np.median(q))))
        w = base_weight / (scale * np.sqrt(1 + (r / (2.5 * scale))**2))**0.5
        # The two branches have distinct parameter roles; the fast one carries activation.
        p = least_squares(residual, p, args=(w,), bounds=bounds, max_nfev=45).x
    return np.exp(p), project(matrix(p), w)


def solve_alpha_m97_c03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    a_train, a_test = _align(train, test)
    gains, biases = _calibrate(dataset['calibration_load_cells'])
    calibration = np.column_stack((gains, biases))
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    args = calibration, dt, times, dataset['crit_stress'], dataset['shear_banding_index']
    training = _design(train, a_train, *args)
    # Imputed coupons support prediction, but their unknown covariates can bias the fit.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    def observed_design(log_tau):
        curves, rates = training(log_tau)
        return curves[complete], rates[complete]

    taus, weights = _fit(observed_design,
                         train['strain_curves_obs'][complete] - train['strain_0'][complete, None],
                         train['instant_rate_obs'][complete], dt, times)
    curves, rates = _design(test, a_test, *args)(np.log(taus))
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
