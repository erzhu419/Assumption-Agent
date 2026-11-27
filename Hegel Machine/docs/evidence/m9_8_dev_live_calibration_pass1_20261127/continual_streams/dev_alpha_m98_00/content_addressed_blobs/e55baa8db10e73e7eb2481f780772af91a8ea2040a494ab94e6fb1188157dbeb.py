import numpy as np
from scipy.optimize import least_squares


def _features(train, test=None, *, balanced=True):
    blocks = (train,) if test is None else (train, test)
    def stack(key):
        return np.concatenate([b[key] for b in blocks])
    lab, stratum = stack('lab_id'), stack('stratum')
    x = stack('coupon_features_raw').astype(float, copy=True)
    ref = stack('ref_coupons')
    for group in np.unique(lab):
        mask = lab == group
        panel = ref[mask]
        loc = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0))
        x[mask] = (x[mask] - loc) / (scale if scale > 0 else 1.)
    if balanced:
        _, ix, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                  return_inverse=True, return_counts=True)
        weight = (1. / counts[ix])[:, None]
    else:
        weight = np.ones((len(lab), 1))
    den = (weight * np.isfinite(x)).sum(axis=0)
    fill = np.divide(np.nansum(weight * x, axis=0), den,
                     out=np.zeros(x.shape[1]), where=den > 0)
    x = np.where(np.isfinite(x), x, fill)
    return (x,) if test is None else np.split(x, [len(train['lab_id'])])


def _calibration(data):
    u, raw = data['true_channel_loads'], data['bridge_raw']
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    def basis(v):
        return np.concatenate((np.ones_like(v[..., :1]), v, v**3,
                               np.stack([v[..., i]*v[..., j] for i,j in pairs], axis=-1)), axis=-1)
    a = basis(u)
    w = np.ones(len(u))
    for _ in range(14):
        c = np.linalg.lstsq(a*w[:,None], raw*w[:,None], rcond=None)[0]
        res = np.linalg.norm(a@c-raw, axis=1)
        w = np.sqrt(np.minimum(1., max(.001, 1.5*np.median(res))/np.maximum(res,1e-12)))
    keep = res < max(.005, 6*np.median(res))
    c = np.linalg.lstsq(a[keep], raw[keep], rcond=None)[0]
    return c, pairs, basis, keep


def _invert_bridge(raw, cal):
    c, pairs, basis, _ = cal
    lin_inv = np.linalg.inv(c[1:5])
    u = (raw-c[0])@lin_inv
    for _ in range(15):
        err = basis(u)@c-raw
        if np.max(np.abs(err)) < 1e-10:
            break
        jac = c[1:5] + 3*u[..., :,None]**2*c[5:9]
        for j,(p,q) in enumerate(pairs):
            jac[...,p,:] += u[...,q,None]*c[9+j]
            jac[...,q,:] += u[...,p,None]*c[9+j]
        du = np.linalg.solve(jac.swapaxes(-2,-1),err[...,None])[...,0]
        u -= du / np.maximum(1.,np.max(np.abs(du),axis=-1,keepdims=True))
    return u


def _inputs(block, x, cal):
    past = _invert_bridge(block['past_channel_raw'], cal).sum(axis=-1)
    hold = _invert_bridge(block['hold_channel_raw'], cal).sum(axis=-1)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _kernel(dt, tau):
    tail = np.cumsum(dt[::-1])[::-1] - dt
    return -np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau)


def _predict(inp, dt, t, tau, theta):
    past, hold, x = inp
    gap = hold[:, None] - past @ _kernel(dt, tau)
    amp = gap * theta[:2]
    drift = x @ theta[2:]
    return amp @ (-np.expm1(-t[:, None] / tau)).T + drift[:, None] * t, amp @ (1 / tau) + drift


def _material(block, inp, dt, t, subset=None):
    ind = slice(None) if subset is None else subset
    inputs = tuple(a[ind] for a in inp)
    y = block['strain_curves_obs'][ind] - block['strain_0'][ind, None]
    r = block['instant_rate_obs'][ind]
    def unpack(v):
        return np.array([v[0], v[0] + (25. - v[0]) * v[1]]), v[2:]
    def resid(v, scale):
        cy, cr = _predict(inputs, dt, t, *unpack(v))
        return np.r_[((cy-y)/scale[0]).ravel(), (cr-r)/scale[1]]
    bounds = ([.050001, 1e-6] + [-np.inf]*7, [24.999, .999999] + [np.inf]*7)
    best = None
    for a, b in ((.25, 2.), (.8, 5.), (2., 12.)):
        v = np.r_[a, (b-a)/(25-a), .4, .7, .1, np.zeros(4)]
        fit = least_squares(resid, v, args=((.035,.06),), bounds=bounds,
                            loss='soft_l1', x_scale='jac', max_nfev=150)
        if best is None or fit.cost < best.cost:
            best = fit
    e = resid(best.x, (1.,1.))
    def mad(q):
        return max(1e-5, 1.4826*np.median(np.abs(q-np.median(q))))
    s = (mad(e[:y.size]), mad(e[y.size:]))
    fit = least_squares(resid, best.x, args=(s,), bounds=bounds,
                        loss='cauchy', f_scale=2.385, x_scale='jac', max_nfev=130)
    return *unpack(fit.x), s


def _chamber_design(inp, dt, t, tau, theta, memory, drift):
    """Each column is a strain correction with its matching initial derivative."""
    past, hold, x = inp
    n = len(hold)
    basis = np.vstack((-np.expm1(-t[:, None] / tau), 1 / tau))
    weights = _kernel(dt, tau)
    amps = (hold[:, None] - past @ weights) * theta[:2]
    ramp = np.r_[t, 1.]
    nominal_drift = x @ theta[2:]
    columns = []
    def add(z, envelope):
        columns.append(z[:, None] * envelope)
    def branch(z):
        columns.append(z @ basis.T)
    if memory == 'offset':
        for j in range(2):
            add(np.ones(n), basis[:, j])
    elif memory == 'initial':
        for z in (np.ones(n), past[:, 0]):
            branch(z[:, None] * (theta[:2] * np.exp(-dt.sum() / tau)))
    elif memory == 'gain':
        branch((past @ weights) * theta[:2])
        branch(amps)
    elif memory == 'order':
        branch(((past[:, ::-1] - past) @ weights) * theta[:2])
    elif isinstance(memory, float):
        branch((past @ (_kernel(dt * memory, tau) - weights)) * theta[:2])
    elif memory == 'kernel':
        # Common segment gains, integrated by the two distinct physical branches.
        for j in range(len(dt)):
            branch(past[:, j, None] * weights[j] * theta[:2])
    if drift == 'linear':
        add(np.ones(n), ramp)
        add(nominal_drift, ramp)
    elif drift == 'features':
        for z in x.T:
            add(z, ramp)
    elif drift is not None:
        if drift == 'quadratic':
            curvature = np.r_[t**2 / max(t[-1], 1e-8), 0.]
        else:
            curvature = np.r_[-drift * np.expm1(-t / drift), 1.]
        add(nominal_drift, curvature - ramp)
        add(nominal_drift, ramp)
        add(np.ones(n), curvature - ramp)
        add(np.ones(n), ramp)
    if not columns:
        return np.empty((n, len(t) + 1, 0))
    return np.stack(columns, axis=-1)


def _pilot_choice(inp, pilot, dt, t, tau, theta, scale, rows=None):
    """Compare physical corrections on coupon-held-out predictions."""
    y, r = _predict(inp, dt, t, tau, theta)
    residual = np.column_stack((pilot['strain_curves_obs'] - pilot['strain_0'][:, None] - y,
                                pilot['instant_rate_obs'] - r))
    if rows is None:
        rows = np.arange(len(r))
    # Treat the curve and the initial rate as two measured outcomes.
    norm = np.r_[np.full(len(t), scale[0] * np.sqrt(len(t))), scale[1]]
    target = residual[rows] / norm
    def score(err):
        return np.sqrt(np.mean(np.sum(err**2, axis=1)))
    baseline = score(target)
    chosen = (None, None, np.empty(0), baseline)
    if baseline < 1.2 or len(rows) < 5:
        return chosen
    best = baseline
    for mem in (None, 'offset', 'initial', 'gain', 'order', 0.5, 2., 'kernel'):
        for drift in (None, 'linear', 'features', 'quadratic', 0.4, 1.5, 5.):
            if mem is None and drift is None:
                continue
            a = _chamber_design(inp, dt, t, tau, theta, mem, drift)[rows] / norm[None, :, None]
            dim = a.shape[-1]
            xx = a.transpose(0, 2, 1) @ a
            xy = np.einsum('nkp,nk->np', a, target)
            ridge = np.diag(np.maximum(xx.sum(axis=0).diagonal(), 1e-12)) * .002
            mat, rhs = xx.sum(axis=0) + ridge, xy.sum(axis=0)
            coef_loo = np.linalg.solve(mat - xx, (rhs - xy)[..., None])[..., 0]
            err = target - np.einsum('nkp,np->nk', a, coef_loo)
            cv = score(err)
            criterion = cv * (1 + .035 * dim) + .07 * dim
            if criterion < best * (0.85 if chosen[0] is None and chosen[1] is None else 1.):
                best = criterion
                chosen = (mem, drift, np.linalg.solve(mat, rhs), cv)
    return chosen


def solve_batch(dataset: dict) -> dict:
    tr, te = dataset['train'], dataset['test']
    pi = dataset.get('chamber_verification_trials')
    dt, t = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    xtr, xte = _features(tr, te)
    cal = _calibration(dataset['calibration_load_cells'])
    a, b = _inputs(tr, xtr, cal), _inputs(te, xte, cal)
    tau, theta, scale = _material(tr, a, dt, t)
    y, r = _predict(b, dt, t, tau, theta)
    if pi is not None:
        xp, = _features(pi, balanced=False)
        pin = _inputs(pi, xp, cal)
        mem, drift, coef, _ = _pilot_choice(pin, pi, dt, t, tau, theta, scale)
        if coef.size:
            correction = _chamber_design(b, dt, t, tau, theta, mem, drift) @ coef
            y, r = y + correction[:, :-1], r + correction[:, -1]
    return dict(estimated_taus=tau, hereditary_weights=theta,
                predicted_strain=y+te['strain_0'][:, None], initial_rate_pred=r)
