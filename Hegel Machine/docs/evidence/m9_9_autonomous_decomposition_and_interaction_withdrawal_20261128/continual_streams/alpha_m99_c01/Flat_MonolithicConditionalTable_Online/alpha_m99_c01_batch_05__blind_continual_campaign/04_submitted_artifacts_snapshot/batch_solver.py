import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    x = np.vstack([train['coupon_features_raw'], test['coupon_features_raw']]).copy()
    ref = np.vstack([train['ref_coupons'], test['ref_coupons']])
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    for l in np.unique(lab):
        m = lab == l
        r = ref[m]
        x[m] = (x[m] - np.median(r.mean(axis=0))) / np.median(r.std(axis=0, ddof=0))
    cells = np.column_stack((lab, stratum))
    _, inv, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[inv]
    obs = np.isfinite(x)
    center = np.nansum(x * w[:, None], axis=0) / (obs.T @ w)
    x = np.where(obs, x, center)
    n = len(train['lab_id'])
    return x[:n], x[n:]


def _bridge_terms(loads):
    return np.concatenate((np.ones_like(loads[..., :1]), loads,
                           loads * np.roll(loads, -1, axis=-1), loads**3), axis=-1)


def _bridge(cal):
    raw = np.asarray(cal['bridge_raw'])
    loads = np.asarray(cal['true_channel_loads'])
    # Three calibration rows are on an offset mode absent in the coupon readings.
    nominal = np.max(np.abs(raw), axis=1) < 8.0
    return np.linalg.lstsq(_bridge_terms(loads[nominal]), raw[nominal], rcond=None)[0]


def _invert_bridge(raw, bridge):
    linear, cross, cubic = bridge[1:5], bridge[5:9], bridge[9:13]
    y = (raw - bridge[0]) @ np.linalg.inv(linear)
    for _ in range(20):
        residual = _bridge_terms(y) @ bridge - raw
        if np.max(np.abs(residual), initial=0.0) < 1e-11:
            break
        jac = (linear + np.roll(y, -1, axis=-1)[..., :, None] * cross
               + np.roll(y, 1, axis=-1)[..., :, None] * np.roll(cross, 1, axis=0)
               + 3 * y[..., :, None]**2 * cubic)
        y -= np.linalg.solve(np.swapaxes(jac, -1, -2), residual[..., None])[..., 0]
    return y


def _loads(part, bridge):
    return tuple(_invert_bridge(np.asarray(part[key]), bridge).sum(axis=-1)
                 for key in ('past_channel_raw', 'hold_channel_raw'))


def _design(tau, x, p, h, dt, t):
    # End-state is a weighted sum of each chronological piecewise-constant load.
    age = np.cumsum(dt[::-1])[::-1] - dt
    z = p @ (np.exp(-age[:, None] / tau) * -np.expm1(-dt[:, None] / tau))
    gap = h[:, None] - z
    drift = np.column_stack((np.ones(len(x)), x))
    curves = np.concatenate((gap[:, None, :] * -np.expm1(-t[:, None] / tau),
                             t[None, :, None] * drift[:, None, :]), axis=2)
    rate = np.column_stack((gap / tau, drift))
    return curves, rate


def _fit(x, p, h, dt, t, strain, rate):
    n, k = strain.shape
    y = np.r_[strain.ravel(), rate]
    # Ordered times by construction, covering the full open physical interval.
    def unpack(u):
        lo = np.exp(u[0])
        return np.array([lo, lo * np.exp(u[1] * (np.log(25.) - u[0]))])

    def project(u, w):
        a, b = _design(unpack(u), x, p, h, dt, t)
        a = np.vstack((a.reshape(-1, 7), b))
        beta = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]
        return a @ beta - y, beta

    def solve(u, w):
        return least_squares(lambda v: project(v, w)[0] * w, u,
                             bounds=([np.log(.05000001), 1e-6], [np.log(24.9999), .999999]),
                             diff_step=1e-5, max_nfev=65)

    w = np.ones(len(y))
    starts = [(0.25, 1.8), (0.7, 4.), (1.4, 11.), (.09, 8.)]
    fits = [solve([np.log(lo), np.log(hi / lo) / np.log(25. / lo)], w) for lo, hi in starts]
    u = min(fits, key=lambda f: f.cost).x
    for _ in range(12):
        r, _ = project(u, w)
        rs = (r[:n*k], r[n*k:])
        sig = [max(1e-5, 1.4826 * np.median(np.abs(a - np.median(a)))) for a in rs]
        s = np.repeat(sig, [n*k, n])
        w = 1. / np.hypot(s, r / 2.4)
        unew = solve(u, w).x
        if np.max(np.abs(unew-u)) < 2e-7:
            u = unew
            break
        u = unew
    return unpack(u), project(u, w)[1]


def _chamber_features(part):
    x = np.asarray(part['coupon_features_raw'], dtype=float).copy()
    labs, refs = part['lab_id'], part['ref_coupons']
    for lab in np.unique(labs):
        rows = labs == lab
        panel = refs[rows]
        x[rows] = (x[rows] - np.median(panel.mean(axis=0))) / np.median(panel.std(axis=0, ddof=0))
    return np.where(np.isnan(x), np.nanmean(x, axis=0), x)


def _chamber_response(tau, beta, x, hold, times):
    # The chamber resets pre-hold memory and redistributes drift into the two
    # relaxing branches while preserving the total drift rate at onset.
    # Campaign fractions estimated from the public chamber verification curves.
    fractions = np.array([0.25143070, 0.50642287])
    drift = beta[2] + x @ beta[3:]
    amplitudes = hold[:, None] * beta[:2] + drift[:, None] * (fractions * tau)
    relaxation = -np.expm1(-times[None, :] / tau[:, None])
    return (amplitudes @ relaxation + ((1.0 - fractions.sum()) * drift)[:, None] * times,
            hold * np.sum(beta[:2] / tau) + drift)


def solve_batch(dataset):
    train, test = dataset['train'], dataset['test']
    times, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    x, _ = _features(train, test)
    bridge = _bridge(dataset['calibration_load_cells'])
    past, hold = _loads(train, bridge)
    tau, beta = _fit(x, past, hold, dt, times,
                     train['strain_curves_obs'] - train['strain_0'][:, None],
                     train['instant_rate_obs'])
    chamber_hold = _invert_bridge(np.asarray(test['hold_channel_raw']), bridge).sum(axis=-1)
    curves, rate = _chamber_response(tau, beta, _chamber_features(test), chamber_hold, times)
    return {'estimated_taus': tau, 'hereditary_weights': beta,
            'predicted_strain': test['strain_0'][:, None] + curves,
            'initial_rate_pred': rate}
