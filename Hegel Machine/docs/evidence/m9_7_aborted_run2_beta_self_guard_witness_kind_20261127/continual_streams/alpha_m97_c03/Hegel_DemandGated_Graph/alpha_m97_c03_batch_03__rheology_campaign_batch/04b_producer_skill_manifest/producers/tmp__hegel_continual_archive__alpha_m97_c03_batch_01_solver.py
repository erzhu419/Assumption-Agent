"""Calibrated two-branch hereditary model for the rheology campaign."""
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



def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _archive_line(history, lab):
    for sweep in reversed(history):
        use = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'])[use]
        y = np.asarray(sweep['true_stress'])[use]
        if len(x) >= 2 and np.ptp(x) > 0:
            return _line(x, y)
    return np.array([1., 0.])


def _calibrate(cal):
    result = np.empty((4, 2))
    history = None

    def prior(lab):
        nonlocal history
        if history is None:
            try:
                with open(_ARCHIVE) as handle:
                    history = json.load(handle)
            except FileNotFoundError:
                history = []
        return _archive_line(history, lab)

    for lab in range(4):
        use = cal['lab_id'] == lab
        x, y = cal['cmd_stress'][use], cal['true_stress'][use]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 2:
            result[lab] = prior(lab)
            continue
        # A spike at the end of a short sweep can rotate a median-slope fit.
        # Score pair lines by tight affine consensus before testing for EMI.
        pairs = []
        peak = 0.
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            for u, v in zip(i, j):
                if x[v] == x[u]:
                    continue
                gain = (y[v]-y[u])/(x[v]-x[u])
                prediction = y[u] + gain*(x-x[u])
                error = np.abs(prediction-y)
                # The load-cell sweeps are resolved below 0.001 stress units.
                inliers = error <= .003
                count = np.count_nonzero(inliers)
                if count > peak:
                    peak, pairs = count, []
                if count == peak:
                    pairs.append((inliers, np.max(error)))
        if pairs and peak < len(x) and max(error for _, error in pairs) > 3.:
            archived = prior(lab)
            archive_prediction = np.polyval(archived, x)
            candidates = [_line(x[mask], y[mask]) for mask, error in pairs if error > 3.]
            result[lab] = min(candidates, key=lambda ab: np.mean((np.polyval(ab, x)-archive_prediction)**2))
        else:
            result[lab] = _line(x, y)
    return result


def _design(split, aligned, calibration, dt, time, critical, banding):
    gain, bias = calibration[split['lab_id']].T
    stress = gain[:, None]*split['past_stress_cmd'] + bias[:, None]
    hold = gain*split['stress_hold_cmd'] + bias
    damage = np.maximum(np.abs(stress)-critical, 0.)**2 @ dt
    active = (damage > 0.) & (banding <= .50)
    act = np.where(active, np.tanh(damage), 0.)
    # Low banding selects the time basis; damage gates the two network terms.
    flow = np.log1p(time)[None, :] if banding <= .50 else time[None, :]
    covariates = np.column_stack((np.ones(len(hold)), aligned))
    elapsed_after = np.cumsum(dt[::-1])[::-1] - dt

    def build(log_tau):
        tau = np.exp(log_tau)
        memory_kernel = -np.expm1(-dt[:, None]/tau) * np.exp(-elapsed_after[:, None]/tau)
        z = stress @ memory_kernel
        gap = hold[:, None] - z
        rise = -np.expm1(-time[:, None]/tau)
        amplitude = act*hold
        log_memory = act*z[:, 0]
        n, k = len(hold), len(time)
        curve = np.empty((n, k, 9))
        curve[:, :, :2] = gap[:, None, :]*rise[None, :, :]
        curve[:, :, 2:7] = flow[:, :, None]*covariates[:, None, :]
        curve[:, :, 7] = amplitude[:, None]*rise[None, :, 0]
        curve[:, :, 8] = log_memory[:, None]*flow
        rate = np.column_stack((gap/tau, covariates, amplitude/tau[0], log_memory))
        return curve, rate
    return build


def _fit(build, strain, rate, dt, time, complete):
    # Imputed covariates add a shared drift error to every observation of a coupon.
    # Fit the shared law on measured covariates; alignment still imputes at prediction.
    use = complete if np.count_nonzero(complete) >= 18 else np.ones(len(rate), dtype=bool)
    y_curve = strain[use].ravel()
    y_rate = rate[use]
    lower = max(np.min(dt)*.025, 1e-4)
    upper = 30*max(np.sum(dt), np.max(time))
    bounds = (np.r_[np.full(2, np.log(lower)), np.full(9, -np.inf)],
              np.r_[np.full(2, np.log(upper)), np.full(9, np.inf)])

    def blocks(q):
        curve, instant = build(q)
        return curve[use].reshape(-1, 9), instant[use]

    def residual(p):
        curve, instant = blocks(p[:2])
        return curve @ p[2:] - y_curve, instant @ p[2:] - y_rate

    def run(p, scale):
        def objective(v):
            c, r = residual(v)
            return np.r_[c/scale[0], r/scale[1]]
        return least_squares(objective, p, bounds=bounds, loss='soft_l1',
                             x_scale='jac', max_nfev=160)

    scale = np.array([.02, .02])
    fits = []
    for start in ((.3, 2.), (.65, 4.5), (.15, 1.), (1.4, 8.)):
        q = np.log(np.clip(start, lower*1.01, upper*.99))
        c, r = blocks(q)
        beta = np.linalg.lstsq(np.vstack((c, r)), np.r_[y_curve, y_rate], rcond=None)[0]
        fits.append(run(np.r_[q, beta], scale))
    best = min([fit for fit in fits if fit.x[0] < fit.x[1]] or fits,
               key=lambda fit: fit.cost)
    for _ in range(2):
        scale = np.array([max(1.4826*np.median(np.abs(v-np.median(v))), 1e-6)
                          for v in residual(best.x)])
        updated = run(best.x, scale)
        if best.x[0] < best.x[1] and updated.x[0] >= updated.x[1]:
            break
        best = updated
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = dataset['past_dt'], dataset['eval_times']
    calibration = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    params = (calibration, dt, time, dataset['crit_stress'], dataset['shear_banding_index'])
    build = _design(train, x_train, *params)
    taus, beta = _fit(build, train['strain_curves_obs'] - train['strain_0'][:, None],
                      train['instant_rate_obs'], dt, time,
                      np.isfinite(train['coupon_features_raw']).all(axis=1))
    curve, rate = _design(test, x_test, *params)(np.log(taus))
    return {'estimated_taus': taus, 'hereditary_weights': beta[:7],
            'regime_mode_weights': beta[7:], 'predicted_strain': test['strain_0'][:, None] + curve @ beta,
            'initial_rate_pred': rate @ beta}
