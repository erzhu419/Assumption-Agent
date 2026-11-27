import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_CALIBRATION_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_calibration(history, channel):
    if history is None:
        return None
    if isinstance(history, list):
        for entry in reversed(history):
            estimate = _historical_calibration(entry, channel)
            if estimate is not None:
                return estimate
        return None
    if 'stress_gains' in history and 'stress_biases' in history:
        return float(history['stress_gains'][channel]), float(history['stress_biases'][channel])
    if history.get('lab_id') == channel and 'gain' in history and 'bias' in history:
        return float(history['gain']), float(history['bias'])
    return None


def _calibration(cal):
    gains = np.empty(4)
    biases = np.empty(4)
    history_loaded = False
    history = None
    for c in range(4):
        take = (cal['lab_id'] == c) & np.isfinite(cal['cmd_stress']) & np.isfinite(cal['true_stress'])
        x, y = cal['cmd_stress'][take], cal['true_stress'][take]
        clean = np.ones(len(x), dtype=bool)
        if np.unique(x).size >= 2:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[j] != x[i]
            i, j = i[distinct], j[distinct]
            slope = np.median((y[j] - y[i]) / (x[j] - x[i]))
            intercept = np.median(y - slope*x)
            clean = np.abs(y - (slope*x + intercept)) <= 3.0
        fallback = len(x) < 2 or not clean.all()
        historical = None
        if fallback:
            if not history_loaded:
                try:
                    history = json.loads(_CALIBRATION_HISTORY.read_text())
                except FileNotFoundError:
                    pass
                history_loaded = True
            historical = _historical_calibration(history, c)
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or historical is None):
            gains[c], biases[c] = np.linalg.lstsq(
                np.column_stack((x[clean], np.ones(clean.sum()))), y[clean], rcond=None
            )[0]
        elif historical is not None:
            gains[c], biases[c] = historical
        else:
            raise ValueError(f'No usable load-cell calibration for channel {c}')
    return gains, biases


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        mask = labs == lab
        controls = panel[mask]
        center = np.nanmedian(np.nanmean(controls, axis=0))
        spread = np.nanmedian(np.nanstd(controls, axis=0, ddof=0))
        aligned[mask] = (raw[mask]-center)/spread
    # Compute each cell mean from observed entries before filling any row.
    cells = [(labs == lab) & (strata == s) for lab, s in np.unique(np.column_stack((labs, strata)), axis=0)]
    means = []
    for mask in cells:
        v = aligned[mask]
        n = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), n, out=np.full(v.shape[1], np.nan), where=n > 0))
    means = np.asarray(means)
    for idx, mask in enumerate(cells):
        lab = labs[np.flatnonzero(mask)[0]]
        same_lab = np.array([labs[np.flatnonzero(m)[0]] == lab for m in cells])
        fill = means[idx].copy()
        for j in np.flatnonzero(~np.isfinite(fill)):
            v = means[same_lab, j]
            v = v[np.isfinite(v)]
            fill[j] = v.mean() if v.size else np.nanmean(means[:, j])
        aligned[mask] = np.where(np.isfinite(aligned[mask]), aligned[mask], fill)
    return aligned


def _design(logtau, stress, hold, dt, times, x):
    tau = np.exp(logtau)
    edges = np.r_[0., np.cumsum(dt)]
    remaining = edges[-1] - edges
    u = remaining[:, None]/tau
    e = np.exp(-u)
    memory = stress @ np.diff(e, axis=0)
    dmemory = stress @ np.diff(e*u, axis=0)
    gap = hold[:, None] - memory
    v = times[:, None]/tau
    ev = np.exp(-v)
    phi = -np.expm1(-v)
    n, k = len(x), len(times)
    a = np.empty((n,k+1,7))
    a[:,:k,:2] = gap[:,None,:]*phi
    a[:,k,:2] = gap/tau
    xd = np.column_stack((np.ones(n), x))
    a[:,:k,2:] = xd[:,None,:]*times[None,:,None]
    a[:,k,2:] = xd
    d = np.empty((n,k+1,2))
    d[:,:k] = -dmemory[:,None,:]*phi - gap[:,None,:]*v*ev
    d[:,k] = (-dmemory-gap)/tau
    return a, d


def _fit(stress, hold, dt, times, x, y):
    def get(p):
        return _design(p[:2], stress, hold, dt, times, x)

    def optimize(p, scale, loss):
        def fun(p):
            a, _ = get(p)
            return ((a @ p[2:] - y)/scale).ravel()

        def jac(p):
            a, d = get(p)
            j = np.concatenate((d*p[2:4], a), axis=2)
            return (j/scale[..., None]).reshape(-1, 9)

        low = np.r_[np.log(min(dt.min(), times.min())/30)*np.ones(2), np.full(7,-np.inf)]
        high = np.r_[np.log((dt.sum()+times.max())*15)*np.ones(2), np.full(7,np.inf)]
        return least_squares(fun, p, jac=jac, bounds=(low, high), loss=loss, x_scale='jac', max_nfev=150)

    best = None
    for pair in ((.35, 2.3), (.7, 3.8), (.18, 1.25)):
        logtau = np.log(pair)
        design, _ = _design(logtau, stress, hold, dt, times, x)
        w = np.linalg.lstsq(design.reshape(-1,7), y.ravel(), rcond=None)[0]
        fit = optimize(np.r_[logtau, w], np.asarray(.09), 'soft_l1')
        if best is None or fit.cost < best.cost:
            best = fit
    p = best.x
    for _ in range(2):
        a, _ = get(p)
        r = a @ p[2:] - y
        def mad(v):
            return max(1e-6, 1.4826*np.median(np.abs(v-np.median(v))))
        scale = 2.385*np.r_[np.full(len(times), mad(r[:,:-1])), mad(r[:,-1])]
        p = optimize(p, scale, 'cauchy').x
    order = np.argsort(p[:2])
    return p[:2][order], np.r_[p[2:4][order], p[4:]]


def solve_alpha_m97_c03_s00(dataset: dict) -> dict:
    tr, te = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    gains, biases = _calibration(dataset['calibration_load_cells'])
    x = _align(tr, te)
    def stresses(part):
        g, b = gains[part['lab_id']], biases[part['lab_id']]
        return part['past_stress_cmd']*g[:,None]+b[:,None], part['stress_hold_cmd']*g+b
    n = len(tr['lab_id'])
    y = np.column_stack((tr['strain_curves_obs']-tr['strain_0'][:,None], tr['instant_rate_obs']))
    logtau, w = _fit(*stresses(tr), dt, times, x[:n], y)
    a, _ = _design(logtau, *stresses(te), dt, times, x[n:])
    pred = a @ w
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': w, 'regime_mode_weights': np.zeros(2), 'predicted_strain': pred[:,:-1]+te['strain_0'][:,None], 'initial_rate_pred': pred[:,-1]}
