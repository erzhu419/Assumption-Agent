"""Fit the damage-gated two-branch rheology model from measured data."""

import numpy as np
from scipy.optimize import least_squares


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _calibration(data, sparse=None):
    c = data['calibration_load_cells']
    result = np.empty((4, 2))
    for lab in range(4):
        take = (c['lab_id'] == lab) & np.isfinite(c['cmd_stress']) & np.isfinite(c['true_stress'])
        x, y = c['cmd_stress'][take], c['true_stress'][take]
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3:
            dx = x[:, None] - x
            dy = y[:, None] - y
            slopes = np.divide(dy, dx, out=np.full_like(dx, np.nan), where=dx != 0)
            gain = np.nanmedian(np.nanmedian(slopes, axis=1))
            clean = np.abs(y - gain*x - np.median(y - gain*x)) <= 3.0
        if clean.sum() >= 2 and (clean.all() or clean.sum() >= 4):
            result[lab] = _line(x[clean], y[clean])
        else:
            result[lab] = _archive_calibration(lab, x[clean], y[clean])
            if sparse is not None:
                sparse[lab] = (x[clean], y[clean])
    return result


def _archive_calibration(lab, x, y):
    import json
    try:
        with open('/tmp/hegel_continual_archive/archive_calibration_history.json') as file:
            records = json.load(file)
    except FileNotFoundError:
        return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)

    def rows(node):
        if isinstance(node, dict):
            if {'lab_id', 'cmd_stress', 'true_stress'} <= node.keys():
                yield from zip(*(np.atleast_1d(node[key]) for key in ('lab_id', 'cmd_stress', 'true_stress')))
            else:
                for value in node.values():
                    yield from rows(value)
        elif isinstance(node, list):
            for value in node:
                yield from rows(value)

    pairs = np.array([(u, v) for c, u, v in rows(records) if int(c) == lab and np.isfinite(u) and np.isfinite(v)]).reshape(-1, 2)
    if len(pairs) >= 2:
        gain, bias = _line(*pairs.T)
        if len(x):
            bias = np.mean(y - gain*x)
        return gain, bias
    return _line(x, y) if len(x) >= 2 else (1.0, y[0] - x[0] if len(x) else 0.0)


def _align(data):
    train, test = data['train'], data['test']
    x, ref, lab, stratum = (np.concatenate((train[k], test[k]), axis=0) for k in
                             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for c in np.unique(lab):
        sel = lab == c
        panel = ref[sel]
        shift = np.nanmedian(np.nanmean(panel, axis=0))
        gain = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[sel] = (x[sel] - shift)/gain
    means = []
    for c, s in np.unique(np.column_stack((lab, stratum)), axis=0):
        part = x[(lab == c) & (stratum == s)]
        count = np.isfinite(part).sum(axis=0)
        means.append(np.divide(np.nansum(part, axis=0), count,
                               out=np.full(x.shape[1], np.nan), where=count > 0))
    return np.where(np.isfinite(x), x, np.nanmean(means, axis=0))


def _physical(split, calibration):
    gain, bias = calibration[split['lab_id']].T
    return split['past_stress_cmd']*gain[:, None] + bias[:, None], split['stress_hold_cmd']*gain + bias


def _design(logtau, past, hold, dt, time, features, act, plateau):
    tau = np.exp(logtau)
    u = dt[:, None] / tau
    decay = np.exp(-u)
    update = -np.expm1(-u)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for j in range(len(dt)):
        stress = past[:, j, None]
        dz = dz*decay[j] + (z-stress)*(u[j]*decay[j])
        z = z*decay[j] + stress*update[j]
    gap = hold[:, None] - z
    q = time[:, None] / tau
    accum = -np.expm1(-q)
    daccum = -q*np.exp(-q)
    time_basis = np.where(plateau[:, None], np.log1p(time), time)
    ah, az = act*hold, act*z[:, 0]
    n, k = len(hold), len(time)
    curves = np.empty((n, k, 9))
    curves[:, :, :2] = gap[:, None, :]*accum
    curves[:, :, 2:7] = features[:, None, :]*time_basis[:, :, None]
    curves[:, :, 7] = ah[:, None]*accum[:, 0]
    curves[:, :, 8] = az[:, None]*time_basis
    rates = np.column_stack((gap/tau, features, ah/tau[0], az))
    dc = -dz[:, None, :]*accum + gap[:, None, :]*daccum
    dr = (-dz-gap)/tau
    mod_c = np.stack((ah[:, None]*daccum[:, 0], (act*dz[:, 0])[:, None]*time_basis), axis=2)
    mod_r = np.column_stack((-ah/tau[0], act*dz[:, 0]))
    return curves, rates, dc, dr, mod_c, mod_r


def _mad(x):
    return max(1.4826*np.median(np.abs(x-np.median(x))), 1e-7)


def _fit(args, strains, instant):
    dt, time = args[2:4]
    y = np.r_[strains.ravel(), instant]
    nc = strains.size
    low = min(dt.min(), time[time > 0].min())/15
    high = 40*max(dt.sum(), time.max())
    lb = np.r_[np.log([low, low]), np.full(9, -np.inf)]
    ub = np.r_[np.log([high, high]), np.full(9, np.inf)]
    scale = np.ones_like(y)
    cached = (None, None)

    def design(t):
        nonlocal cached
        if cached[0] is None or not np.array_equal(t, cached[0]):
            c, r, dc, dr, mc, mr = _design(t, *args)
            cached = (t.copy(), (np.vstack((c.reshape(-1, 9), r)),
                                 np.vstack((dc.reshape(-1, 2), dr)),
                                 np.vstack((mc.reshape(-1, 2), mr))))
        return cached[1]

    def residual(p):
        return (design(p[:2])[0] @ p[2:] - y)/scale

    def jac(p):
        a, d, m = design(p[:2])
        d = d*p[2:4]
        d[:, 0] += m @ p[9:11]
        return np.column_stack((d, a))/scale[:, None]

    def start(pair, w):
        t = np.log(np.clip(pair, low*1.01, high/1.01))
        a = design(t)[0]
        return np.r_[t, np.linalg.lstsq(a*w[:, None], y*w, rcond=None)[0]]

    def optimize(p, loss):
        return least_squares(residual, p, jac=jac, bounds=(lb, ub), loss=loss,
                             x_scale='jac', max_nfev=160, ftol=1e-10)

    def rescale(p):
        rr = (residual(p)*scale)[:nc].reshape(strains.shape)
        ri = (residual(p)*scale)[nc:]
        scale[:] = np.r_[np.tile([_mad(rr[:, j]) for j in range(rr.shape[1])], len(rr)),
                         np.full(len(ri), _mad(ri))]

    starts = [(0.4, 2.4), (0.8, 4.), (0.2, 1.2), (1.4, 8.), (2.4, 0.4)]
    for j in range(strains.shape[1]):
        scale[j:nc:strains.shape[1]] = _mad(strains[:, j])
    scale[nc:] = _mad(instant)
    pilot = min((optimize(start(pair, 1/scale), 'soft_l1') for pair in starts[:2]), key=lambda f:f.cost)
    rescale(pilot.x)
    best = optimize(pilot.x, 'cauchy')
    weights = 1/(scale*np.hypot(1, residual(best.x)))
    for pair in starts:
        fit = optimize(start(pair, weights), 'cauchy')
        if fit.cost < best.cost:
            best = fit
    for _ in range(2):
        rescale(best.x)
        best = optimize(best.x, 'cauchy')
    return best.x


def _inputs(dataset, split, cal, x):
    past, hold = _physical(split, cal)
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    features = np.column_stack((np.ones(len(x)), x))
    damage = np.maximum(np.abs(past)-dataset['crit_stress'], 0)**2 @ dt
    index = split.get('shear_banding_index', dataset.get('shear_banding_index', 0.0))
    # In the low-banding campaign, damage gates the two crystal terms,
    # not the logarithmic drift already present in undamaged coupons.
    plateau = np.broadcast_to(np.asarray(index) <= .50, damage.shape)
    return past, hold, dt, time, features, np.tanh(damage)*plateau, plateau


def _take_inputs(args, mask):
    return tuple(v if j in (2, 3) else v[mask] for j, v in enumerate(args))


def _refine_calibration(dataset, cal, sparse, x, strains, rates):
    train = dataset['train']
    labs = train['lab_id']
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    anchor = complete & ~np.isin(labs, list(sparse))
    if not np.any(anchor):
        return
    args = _inputs(dataset, train, cal, x)
    p = _fit(_take_inputs(args, anchor), strains[anchor], rates[anchor])
    c, r, *_ = _design(p[:2], *_take_inputs(args, anchor))
    curve_scale = np.array([_mad(v) for v in (c @ p[2:] - strains[anchor]).T])
    rate_scale = _mad(r @ p[2:] - rates[anchor])

    for lab, (cmd, true) in sparse.items():
        mask = complete & (labs == lab)
        if not np.any(mask):
            continue
        local = {k: train[k][mask] for k in ('lab_id', 'past_stress_cmd', 'stress_hold_cmd')}
        if 'shear_banding_index' in train:
            local['shear_banding_index'] = np.broadcast_to(train['shear_banding_index'], labs.shape)[mask]
        elif np.ndim(dataset.get('shear_banding_index', 0.0)):
            local['shear_banding_index'] = np.asarray(dataset['shear_banding_index'])[mask]
        pivot = cmd.mean() if len(cmd) else 0.0
        target = true.mean() if len(cmd) else 0.0

        def coefficients(q):
            return q[0], target - pivot*q[0] if len(cmd) == 1 else q[1]

        def residual(q):
            cal[lab] = coefficients(q)
            c, r, *_ = _design(p[:2], *_inputs(dataset, local, cal, x[mask]))
            return np.r_[((c @ p[2:] - strains[mask])/curve_scale).ravel(),
                         (r @ p[2:] - rates[mask])/rate_scale]

        q0 = cal[lab, :1 if len(cmd) == 1 else 2].copy()
        lower = [1e-5] if len(q0) == 1 else [1e-5, -np.inf]
        fit = least_squares(residual, q0, bounds=(lower, np.inf), loss='soft_l1', max_nfev=120)
        cal[lab] = coefficients(fit.x)


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    sparse = {}
    cal, x = _calibration(dataset, sparse), _align(dataset)
    n = len(train['lab_id'])
    strains = train['strain_curves_obs']-train['strain_0'][:, None]
    rates = train['instant_rate_obs']
    if sparse:
        _refine_calibration(dataset, cal, sparse, x[:n], strains, rates)
    p = _fit(_inputs(dataset, train, cal, x[:n]), strains, rates)
    c, r, *_ = _design(p[:2], *_inputs(dataset, test, cal, x[n:]))
    return {'estimated_taus': np.exp(p[:2]), 'hereditary_weights': p[2:9],
            'regime_mode_weights': p[9:11], 'predicted_strain': test['strain_0'][:, None]+c @ p[2:],
            'initial_rate_pred': r @ p[2:]}
