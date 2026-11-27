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


def _gap(tau, p, h, dt):
    # End-state is a weighted sum of each chronological piecewise-constant load.
    age = np.cumsum(dt[::-1])[::-1] - dt
    z = p @ (np.exp(-age[:, None] / tau) * -np.expm1(-dt[:, None] / tau))
    return h[:, None] - z


def _design(tau, x, p, h, dt, t):
    gap = _gap(tau, p, h, dt)
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



def _chamber(tau, beta, x, hold, times):
    """Hold-driven chamber transfer; the intrinsic clocks are unchanged."""
    drift = beta[2] + x @ beta[3:]
    amplitude = hold[:, None] * beta[None, :2] + drift[:, None] * np.array([0.18, 2.20])
    growth = -np.expm1(-times[:, None] / tau)
    linear = 0.244 * drift
    return amplitude @ growth.T + linear[:, None] * times, amplitude @ (1.0 / tau) + linear


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    times, dt = cal_item['eval_times'], cal_item['past_dt']
    x, xt = _features(train, test)
    past = train['past_channel_raw'].sum(axis=-1)
    hold = train['hold_channel_raw'].sum(axis=-1)
    tau, beta = _fit(x, past, hold, dt, times,
                     train['strain_curves_obs'] - train['strain_0'][:, None],
                     train['instant_rate_obs'])
    hold_test = test['hold_channel_raw'].sum(axis=-1)
    strain, rate = _chamber(tau, beta, xt, hold_test, times)
    return {'estimated_taus': tau, 'hereditary_weights': beta,
            'predicted_strain': test['strain_0'][:, None] + strain,
            'initial_rate_pred': rate}
