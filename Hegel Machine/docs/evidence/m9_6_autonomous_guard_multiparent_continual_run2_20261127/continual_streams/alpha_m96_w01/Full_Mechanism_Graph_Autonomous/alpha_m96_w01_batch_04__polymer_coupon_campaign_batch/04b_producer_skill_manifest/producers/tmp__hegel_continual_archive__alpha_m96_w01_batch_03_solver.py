import numpy as np
from scipy.optimize import least_squares


def _load_calibration(checks, *, return_identified=False):
    gain, bias = np.ones(4), np.zeros(4)
    lab = np.asarray(checks['lab_id'])
    cmd, true = map(np.asarray, (checks['cmd_stress'], checks['true_stress']))
    pooled = []
    for j in range(4):
        m = (lab == j) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[m], true[m]
        if len(x) < 2 or np.ptp(x) < 1e-8:
            continue
        i, k = np.triu_indices(len(x), 1)
        spread = abs(x[k] - x[i]) > max(1e-8, .05 * np.ptp(x))
        i, k = i[spread], k[spread]
        slopes = (y[k] - y[i]) / (x[k] - x[i])
        a = np.median(slopes)
        b = np.median(y - a*x)
        e = y - a*x - b
        m = abs(e) <= max(.001, 5 * 1.4826 * np.median(abs(e)))
        if m.sum() >= 2 and np.ptp(x[m]) > 1e-8:
            gain[j], bias[j] = np.linalg.lstsq(np.column_stack((x[m], np.ones(m.sum()))), y[m], rcond=None)[0]
            pooled.append(j)
    for j in set(range(4)) - set(pooled):
        peer = j ^ 1
        if peer in pooled:
            gain[j], bias[j] = gain[peer], bias[peer]
        m = (lab == j) & np.isfinite(cmd) & np.isfinite(true)
        if m.any():
            bias[j] = np.median(true[m] - gain[j]*cmd[m])
    if return_identified:
        return gain, bias, np.isin(np.arange(4), pooled)
    return gain, bias


def _features(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    raw = np.vstack((train['coupon_features_raw'], test['coupon_features_raw']))
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.empty_like(raw, dtype=float)
    for j in np.unique(lab):
        m = lab == j
        r = ref[m]
        r = r[:, np.sum(np.isfinite(r), axis=0) >= 2]
        shift = np.median(np.nanmean(r, axis=0)) if r.size else 0.
        scale = np.median(np.nanstd(r, axis=0)) if r.size else 1.
        x[m] = (raw[m] - shift) / (scale if scale > 0 else 1.)
    missing = ~np.isfinite(x)
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    w = np.zeros(len(lab))
    for j, s in cells:
        m = (lab == j) & (stratum == s)
        w[m] = 1 / m.sum()
    denominator = (~missing).T @ w
    means = np.divide(np.where(missing, 0., x).T @ w, denominator, out=np.zeros(x.shape[1]), where=denominator > 0)
    return np.column_stack((np.ones(len(x)), np.where(missing, means, x))), missing


def _design(tau, past, hold, dt, t, x):
    # Unroll the two recurrences from zero at the start of the history.
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = np.exp(-trailing[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    delta = hold[:, None] - past @ kernel
    relax = -np.expm1(-t[:, None] / tau)
    a = np.concatenate((delta[:, None, :] * relax, x[:, None, :] * t[None, :, None]), axis=2)
    r = np.column_stack((delta / tau, x))
    return a, r


def _fit(past, hold, dt, t, x, y, rate):
    cm, rm = np.isfinite(y), np.isfinite(rate)
    yc, yr = y[cm], rate[rm]
    initial = []
    grid = (.24, .42, .72, 1.2, 2.15, 3.75, 6., 8.)
    for i, u in enumerate(grid[:-1]):
        for v in grid[i+1:]:
            tau = np.array([u, v])
            a, r = _design(tau, past, hold, dt, t, x)
            a, r = a[cm], r[rm]
            mat = np.vstack((a, r))
            obs = np.r_[yc, yr]
            # A clipped reweighting pass prevents a few Cauchy glitches from
            # choosing the starting time scales.
            theta = np.linalg.lstsq(mat, obs, rcond=None)[0]
            e = obs - np.einsum('ij,j->i', mat, theta)
            w = np.sqrt(np.minimum(1., max(.006, 2.5 * np.median(abs(e))) / np.maximum(abs(e), 1e-12)))
            theta = np.linalg.lstsq(mat*w[:, None], obs*w, rcond=None)[0]
            e = obs - np.einsum('ij,j->i', mat, theta)
            initial.append((np.mean(np.log1p((e/.035)**2)), np.r_[np.log(tau), theta]))
    initial.sort(key=lambda pair: pair[0])
    scales = (.04, .04)
    bounds = (np.r_[np.log([.2, .2]), np.full(7, -np.inf)], np.r_[np.log([8., 8.]), np.full(7, np.inf)])

    def residual(p):
        a, r = _design(np.exp(p[:2]), past, hold, dt, t, x)
        ce = np.einsum('ij,j->i', a[cm], p[2:]) - yc
        re = np.einsum('ij,j->i', r[rm], p[2:]) - yr
        return np.r_[ce / scales[0], re / scales[1]]

    fit = min((least_squares(residual, p, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=140) for _, p in initial[:3]), key=lambda result: result.cost)
    e = residual(fit.x)
    ec, er = e[:len(yc)] * scales[0], e[len(yc):] * scales[1]
    def robust_sigma(v):
        return 1.4826 * np.median(abs(v - np.median(v))) if v.size else .01
    scales = max(.001, 2.5*robust_sigma(ec)), max(.001, 2.5*robust_sigma(er))
    fit = least_squares(residual, fit.x, bounds=bounds, loss='cauchy', x_scale='jac', max_nfev=140)
    tau, theta = np.exp(fit.x[:2]), fit.x[2:].copy()
    if tau[0] > tau[1]:
        tau = tau[::-1].copy()
        theta[:2] = theta[:2][::-1]
    return tau, theta


def _refine_sparse_calibration(checks, gain, bias, identified, lab, cmd_past,
                               cmd_hold, dt, t, x, y, rate, complete):
    # Share the creep law across labs, not the load-cell gain. A lab with a
    # single check can recover its gain from the training response, anchored
    # to that check, without using any test measurements for calibration.
    source = complete & identified[lab]
    if source.sum() < 12:
        return
    known = lab[source]
    tau, theta = _fit(cmd_past[source]*gain[known, None] + bias[known, None],
                      cmd_hold[source]*gain[known] + bias[known], dt, t,
                      x[source], y[source], rate[source])
    check_lab, check_cmd, check_true = (np.asarray(checks[k]) for k in
                                       ('lab_id', 'cmd_stress', 'true_stress'))
    for j in np.flatnonzero(~identified):
        use = complete & (lab == j)
        if use.sum() < 4:
            continue
        m = (check_lab == j) & np.isfinite(check_cmd) & np.isfinite(check_true)
        anchor_cmd = np.median(check_cmd[m]) if m.any() else 0.
        anchor_true = np.median(check_true[m]) if m.any() else 0.
        past, hold = cmd_past[use] - anchor_cmd, cmd_hold[use] - anchor_cmd
        a, r = _design(tau, past, hold, dt, t, x[use])
        q, v = _design(tau, np.ones_like(past), np.ones_like(hold), dt, t, x[use])
        curve_base = (x[use] @ theta[2:])[:, None]*t + anchor_true*(q[:, :, :2] @ theta[:2])
        rate_base = x[use] @ theta[2:] + anchor_true*(v[:, :2] @ theta[:2])
        curve_target, rate_target = y[use] - curve_base, rate[use] - rate_base
        cm, rm = np.isfinite(curve_target), np.isfinite(rate_target)
        target = np.r_[curve_target[cm], rate_target[rm]]
        design = np.r_[(a[:, :, :2] @ theta[:2])[cm], (r[:, :2] @ theta[:2])[rm]][:, None]
        if not m.any():
            design = np.column_stack((design, np.r_[(q[:, :, :2] @ theta[:2])[cm], (v[:, :2] @ theta[:2])[rm]]))
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        start = np.array([gain[j]]) if m.any() else np.array([gain[j], bias[j]])
        fit = least_squares(lambda p: design @ p - target, start,
                            loss='cauchy', f_scale=.02)
        if fit.success and fit.x[0] > 0:
            gain[j] = fit.x[0]
            bias[j] = anchor_true - gain[j]*anchor_cmd if m.any() else fit.x[1]


def solve_alpha_m96_w01_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x, missing = _features(train, test)
    checks = dataset['calibration_load_cells']
    gain, bias, identified = _load_calibration(checks, return_identified=True)
    def stresses(split):
        lab = np.asarray(split['lab_id'])
        return np.asarray(split['past_stress_cmd'])*gain[lab, None] + bias[lab, None], np.asarray(split['stress_hold_cmd'])*gain[lab] + bias[lab]
    dt, t = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    n = len(train['lab_id'])
    complete = ~missing[:n].any(axis=1)
    keep = complete if complete.sum() >= 12 else np.ones(n, dtype=bool)
    y = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    rate_obs = np.asarray(train['instant_rate_obs'])
    if not identified.all():
        _refine_sparse_calibration(checks, gain, bias, identified, np.asarray(train['lab_id']),
                                   np.asarray(train['past_stress_cmd']), np.asarray(train['stress_hold_cmd']),
                                   dt, t, x[:n], y, rate_obs, complete)
    past, hold = stresses(train)
    tau, theta = _fit(past[keep], hold[keep], dt, t, x[:n][keep], y[keep], rate_obs[keep])
    past, hold = stresses(test)
    a, r = _design(tau, past, hold, dt, t, x[n:])
    curve = np.asarray(test['strain_0'])[:, None] + np.einsum('ntp,p->nt', a, theta)
    rate = r @ theta
    # If a feature is missing, the observed onset rate can identify the
    # otherwise uncertain linear creep slope after subtracting both branch
    # derivatives. Leave the hereditary branches and intercept unchanged.
    obs_rate = np.asarray(test['instant_rate_obs'])
    update = missing[n:].any(axis=1) & np.isfinite(obs_rate)
    curve[update] += (obs_rate[update] - rate[update])[:, None] * t
    rate[update] = obs_rate[update]
    return {'estimated_taus': tau, 'hereditary_weights': theta, 'viscoplastic_weights': np.zeros(2), 'predicted_strain': curve, 'initial_rate_pred': rate}
