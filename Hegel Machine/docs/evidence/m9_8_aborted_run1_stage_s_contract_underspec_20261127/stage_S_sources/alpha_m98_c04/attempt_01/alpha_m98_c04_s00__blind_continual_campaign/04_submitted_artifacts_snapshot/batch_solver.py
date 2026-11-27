import numpy as np
from scipy.optimize import minimize


def _bridge(cal):
    raw = cal['bridge_raw']
    return np.linalg.lstsq(np.column_stack((np.ones(len(raw)), raw)),
                           cal['true_channel_loads'].sum(axis=1), rcond=None)[0]


def _align(groups, source):
    refs = np.concatenate([g['ref_coupons'] for g in groups])
    labs = np.concatenate([g['lab_id'] for g in groups])
    stats = {}
    for lab in np.unique(labs):
        ref = refs[labs == lab]
        stats[lab] = (np.median(np.mean(ref, axis=0)), np.median(np.std(ref, axis=0)))
    aligned = []
    for g in groups:
        x = g['coupon_features_raw'].copy()
        for lab in np.unique(g['lab_id']):
            m = g['lab_id'] == lab
            offset, gain = stats[lab]
            x[m] = (x[m] - offset) / gain
        aligned.append(x)
    g = groups[source]
    _, inverse, counts = np.unique(np.column_stack((g['lab_id'], g['stratum'])), axis=0,
                                   return_inverse=True, return_counts=True)
    w = 1 / counts[inverse, None]
    x = aligned[source]
    good = np.isfinite(x)
    fill = np.nansum(w * x, axis=0) / np.sum(w * good, axis=0)
    return [np.where(np.isfinite(x), x, fill) for x in aligned]


def _design(g, x, bridge, dt, t, taus, kind='chronological'):
    past = g['past_channel_raw'] @ bridge[1:] + bridge[0]
    hold = g['hold_channel_raw'] @ bridge[1:] + bridge[0]
    if kind == 'reversed':
        past, dt = past[:, ::-1], dt[::-1]
    age = np.r_[np.cumsum(dt[::-1])[::-1], 0.]
    mass = np.exp(-age[1:, None] / taus) * -np.expm1(-dt[:, None] / taus)
    memory = past @ mass
    if kind == 'first_extended':
        memory += past[:, :1] * np.exp(-age[0] / taus)
    elif kind == 'hold_extended':
        memory += hold[:, None] * np.exp(-age[0] / taus)
    elif kind == 'normalized':
        memory /= -np.expm1(-age[0] / taus)
    amplitude = hold[:, None] - memory
    drift = np.column_stack((np.ones(len(x)), x))
    curve = np.concatenate((amplitude[:, None, :] * -np.expm1(-t[:, None] / taus),
                           drift[:, None, :] * t[None, :, None]), axis=-1)
    rate = np.column_stack((amplitude / taus, drift))
    return np.concatenate((curve, rate[:, None, :]), axis=1)


def _obs(g):
    return np.column_stack((g['strain_curves_obs'] - g['strain_0'][:, None], g['instant_rate_obs']))


def _fit(g, x, bridge, dt, t, kind='chronological'):
    y = _obs(g)
    # Third differences largely remove the smooth response, leaving the observation noise.
    dd = np.diff(y[:, :-1], n=3, axis=1)
    sc = max(0.002, np.median(np.abs(dd - np.median(dd))) / (0.67449 * np.sqrt(20)))
    scales = np.r_[np.full(len(t), sc), max(0.012, 2 * sc)]
    target = (y / scales).ravel()

    def profile(logtau, full=False):
        taus = np.sort(np.exp(logtau))
        a = (_design(g, x, bridge, dt, t, taus, kind) / scales[None, :, None]).reshape(-1, 7)
        theta = np.linalg.lstsq(a, target, rcond=None)[0]
        for _ in range(12):
            r = a @ theta - target
            w = 1 / np.sqrt(1 + (r / 2.5)**2)
            update = np.linalg.lstsq(a * w[:, None], target * w, rcond=None)[0]
            if np.linalg.norm(update - theta) < 1e-8 * (1 + np.linalg.norm(theta)):
                theta = update
                break
            theta = update
        loss = np.sum(np.log1p(((a @ theta - target) / 2.5)**2))
        return (taus, theta, scales) if full else loss

    bounds = [(np.log(.050001), np.log(24.999))] * 2
    best = min((minimize(profile, np.log(start), bounds=bounds, method='L-BFGS-B')
                for start in ((.42, 2.8), (.85, 7.), (.18, 1.25))), key=lambda r: r.fun)
    return profile(best.x, True)


def _cell_basis(g, cells):
    return ((g['lab_id'][:, None] == cells[:, 0]) &
            (g['stratum'][:, None] == cells[:, 1])).astype(float)


def _chamber_correction(pilot, xp, test, xt, bridge, dt, t, taus, weights):
    # Fit a shared drift adjustment and shrunk lab/stratum interactions.
    cells = np.unique(np.column_stack((pilot['lab_id'], pilot['stratum'])), axis=0)
    cp, ct = _cell_basis(pilot, cells), _cell_basis(test, cells)
    hp = pilot['hold_channel_raw'] @ bridge[1:] + bridge[0]
    ht = test['hold_channel_raw'] @ bridge[1:] + bridge[0]
    center = np.median(hp)
    local_test = ct * (ht[:, None] - center)
    for j in range(len(cells)):
        seen = hp[cp[:, j] != 0]
        # Beyond a cell's observed range, retain only the shared load slope.
        local_test[:, j] = ct[:, j] * (np.clip(ht, seen.min(), seen.max()) - center)
    n = len(hp)
    basis = np.column_stack((np.full(n, .10), .08*xp, .12*cp,
                             .045*(hp-center), .035*cp*(hp[:, None]-center)))
    # A coupon-specific nuisance slope keeps individual discrepancies from
    # forcing a cell-level correction; it does not transfer to new coupons.
    design = np.column_stack((basis, .022*np.eye(n)))
    h = np.r_[t, 1.]
    r = _obs(pilot) - _design(pilot, xp, bridge, dt, t, taus) @ weights
    dd = np.diff(r[:, :-1], n=3, axis=1)
    noise = max(.009, np.median(np.abs(dd-np.median(dd))) / (.67449*np.sqrt(20)))
    scale = np.r_[np.full(len(t), noise), max(.014, 1.5*noise)]
    a = (design[:, None, :] * (h/scale)[None, :, None]).reshape(r.size, -1)
    y = (r/scale).ravel()
    reg = np.eye(a.shape[1])
    beta = np.linalg.solve(a.T @ a + reg, a.T @ y)
    for _ in range(30):
        w = 1 / (1 + ((a @ beta - y)/2.5)**2)
        new = np.linalg.solve(a.T @ (w[:, None]*a) + reg, a.T @ (w*y))
        if np.linalg.norm(new-beta) < 1e-7:
            beta = new
            break
        beta = new
    weights = weights.copy()
    weights[2:7] += np.r_[.10, np.full(4, .08)] * beta[:5]
    k = 5 + len(cells)
    remaining = .12*(ct @ beta[5:k]) + .045*(ht-center)*beta[k]
    remaining += .035*(local_test @ beta[k+1:k+1+len(cells)])
    return weights, remaining


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    pilot = dataset.get('chamber_verification_trials')
    groups = [train, test] if pilot is None else [train, test, pilot]
    x = _align(groups, 0)
    bridge = _bridge(dataset['calibration_load_cells'])
    dt, t = dataset['past_dt'], dataset['eval_times']
    taus, weights, _ = _fit(train, x[0], bridge, dt, t)
    pred = None
    if pilot is not None:
        weights, correction = _chamber_correction(pilot, x[2], test, x[1], bridge, dt, t, taus, weights)
        pred = correction[:, None] * np.r_[t, 1.]
    base = _design(test, x[1], bridge, dt, t, taus) @ weights
    pred = base if pred is None else base + pred
    return dict(estimated_taus=taus, hereditary_weights=weights,
                predicted_strain=pred[:, :-1] + test['strain_0'][:, None],
                initial_rate_pred=pred[:, -1])
