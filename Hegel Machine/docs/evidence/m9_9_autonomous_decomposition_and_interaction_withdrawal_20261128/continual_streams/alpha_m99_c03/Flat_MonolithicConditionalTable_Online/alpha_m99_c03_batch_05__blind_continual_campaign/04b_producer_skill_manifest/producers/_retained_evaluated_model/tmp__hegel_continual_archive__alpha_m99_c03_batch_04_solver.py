import numpy as np
from scipy.optimize import least_squares, root


def _features(train, test):
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    for group in np.unique(lab):
        mask = lab == group
        panel = ref[mask]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale
    _, cell, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = 1.0 / count[cell].reshape(-1, 1)
    observed = np.isfinite(x)
    mean = np.nansum(weight * x, axis=0) / np.sum(weight * observed, axis=0)
    return np.where(observed, x, mean)


def _calibration(cal):
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    norm = np.linalg.norm(raw, axis=1)
    median = np.median(norm)
    # The offset sweep is separated from ordinary bridge readings in raw norm.
    offset_sweep = norm > median + 8.0 * np.median(np.abs(norm - median))
    nxt = (np.arange(4) + 1) % 4
    cross = loads * loads[:, nxt]
    coeff = np.empty((7, 4))
    for j in range(4):
        basis = np.column_stack((np.ones(len(raw)), loads, loads[:, j]**3,
                                 cross[:, j], offset_sweep))
        coeff[:, j] = np.linalg.lstsq(basis, raw[:, j], rcond=None)[0][:-1]
    return coeff


def _axial(raw, bridge):
    raw = np.asarray(raw)
    base, linear, cubic, cross = bridge[0], bridge[1:5], bridge[5], bridge[6]
    nxt = (np.arange(4) + 1) % 4
    idx = np.arange(4)
    values = raw.reshape(-1, 4)
    seeds = np.linalg.solve(linear.T, (values - base).T).T

    def forward(x):
        return base + x @ linear + cubic * x**3 + cross * x * x[nxt]

    def jac(x):
        j = linear.T.copy()
        j[idx, idx] += 3 * cubic * x**2 + cross * x[nxt]
        j[idx, nxt] += cross * x
        return j

    axial = np.empty(len(values))
    for i, (value, seed) in enumerate(zip(values, seeds)):
        residual = lambda x: forward(x) - value
        sol = root(residual, seed, jac=jac, options={'xtol': 1e-10})
        if np.linalg.norm(residual(sol.x), ord=np.inf) > 1e-8:
            sol = least_squares(residual, seed, jac=jac)
        axial[i] = sol.x.sum()
    return axial.reshape(raw.shape[:-1])


def _coupon_inputs(block, x, bridge):
    past = _axial(block['past_channel_raw'], bridge)
    hold = _axial(block['hold_channel_raw'], bridge)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _design(taus, coupons, dt, times):
    past, hold, drift = coupons
    after = np.cumsum(dt[::-1])[::-1] - dt
    memory = past @ (-np.expm1(-dt[:, None] / taus) * np.exp(-after[:, None] / taus))
    gap = hold[:, None] - memory
    curve = np.concatenate((gap[:, None, :] * -np.expm1(-times[:, None] / taus),
                            drift[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap / taus, drift))
    return curve, rate


def _fit(coupons, dt, times, curve_obs, rate_obs):
    curve_ok = np.isfinite(curve_obs.ravel())
    rate_ok = np.isfinite(rate_obs)
    curve_y = curve_obs.ravel()[curve_ok]
    rate_y = rate_obs[rate_ok]
    y = np.r_[curve_y, rate_y]
    ncurve = len(curve_y)

    def build(taus):
        curve, rate = _design(taus, coupons, dt, times)
        return np.vstack((curve.reshape(-1, 7)[curve_ok], rate[rate_ok]))

    def decode(p):
        fast = np.exp(p[0])
        return np.array((fast, fast + (25.0 - fast) * p[1]))

    def encode(fast, slow):
        return np.array((np.log(fast), (slow-fast)/(25.0-fast)))

    bounds = ([np.log(0.050001), 1e-6], [np.log(24.99), 1-1e-6])
    def profile(p):
        a = build(decode(p))
        return a @ np.linalg.lstsq(a, y, rcond=None)[0] - y

    grid = [encode(f, s) for f in (.08, .18, .4, .8, 1.6, 3.2)
            for s in (.3, .65, 1.3, 2.6, 5.2, 10.4, 21.) if f < s]
    starts = sorted(grid, key=lambda p: np.linalg.norm(profile(p)))[:3]
    ordinary = min((least_squares(profile, p, bounds=bounds, max_nfev=100)
                    for p in starts), key=lambda r: r.cost)
    a = build(decode(ordinary.x))
    theta = np.linalg.lstsq(a, y, rcond=None)[0]

    def mad(r):
        return max(1e-6, 1.4826 * np.median(np.abs(r - np.median(r))))

    def scale(r):
        return np.r_[np.full(ncurve, mad(r[:ncurve])),
                     np.full(len(rate_y), mad(r[ncurve:]))]

    def robust(p, sigma, loss, f_scale):
        return least_squares(lambda v: (build(decode(v[:2])) @ v[2:] - y) / sigma,
                             p, bounds=(np.r_[bounds[0], [-np.inf]*7],
                                        np.r_[bounds[1], [np.inf]*7]),
                             loss=loss, f_scale=f_scale, x_scale='jac', max_nfev=200)

    sigma = scale(ordinary.fun)
    first = robust(np.r_[ordinary.x, theta], sigma, 'soft_l1', 1.5)
    sigma = scale(first.fun * sigma)
    result = robust(first.x, sigma, 'cauchy', 2.5)
    return decode(result.x[:2]), result.x[2:]


def _chamber(taus, theta, hold, x, times):
    # Chamber equilibration resets the branch memory. The drift rate
    # starts at its intrinsic value and decays as 1 / (1 + t).
    drift = theta[2] + x @ theta[3:]
    branch = hold[:, None] * theta[:2]
    basis = -np.expm1(-times[:, None] / taus)
    return (branch @ basis.T + drift[:, None] * np.log1p(times),
            branch @ (1.0 / taus) + drift)


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x = _features(train, test)
    n = len(train['lab_id'])
    bridge = _calibration(dataset['calibration_load_cells'])
    train_inputs = _coupon_inputs(train, x[:n], bridge)
    taus, theta = _fit(train_inputs, dt, t,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    # The latest precursor trace matches the batch 01 chamber setting.
    hold = _axial(test['hold_channel_raw'], bridge)
    curve, rate = _chamber(taus, theta, hold, x[n:], t)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curve,
            'initial_rate_pred': rate}
