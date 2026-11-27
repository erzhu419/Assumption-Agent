import numpy as np
from scipy.optimize import least_squares


def basis(logtau, dt, t, past, hold, x, act, log_rows):
    tau = np.exp(logtau)
    h = dt[:, None] / tau
    a = np.exp(-h)
    b = -np.expm1(-h)
    z = np.zeros((len(hold), 2))
    dz = np.zeros_like(z)
    for s in range(len(dt)):
        u = past[:, s, None]
        dz = a[s] * (dz + h[s] * (z - u))
        z = a[s] * z + b[s] * u
    gap = hold[:, None] - z
    k = t[:, None] / tau
    phi = -np.expm1(-k)
    dphi = -k * np.exp(-k)
    flow = np.where(log_rows[:, None], np.log1p(t), t)
    drift = np.column_stack((np.ones(len(hold)), x))
    av = act * hold
    c = np.concatenate((gap[:, None] * phi, drift[:, None] * flow[:, :, None],
                        (av[:, None] * phi[:, 0])[:, :, None],
                        ((act * z[:, 0])[:, None] * flow)[:, :, None]), axis=2)
    r = np.column_stack((gap / tau, drift, av / tau[0], act * z[:, 0]))
    dc = -dz[:, None] * phi + gap[:, None] * dphi
    dr = (-dz - gap) / tau
    ec = np.stack((av[:, None] * dphi[:, 0], (act * dz[:, 0])[:, None] * flow), axis=-1)
    er = np.column_stack((-av / tau[0], act * dz[:, 0]))
    return c, r, dc, dr, ec, er


def fit(dt, t, past, hold, x, act, log_rows, yc, yr, robust=True):
    y = np.r_[yc.ravel(), yr]
    n = yc.size
    def design(logtau):
        c, r, dc, dr, ec, er = basis(logtau, dt, t, past, hold, x, act, log_rows)
        return (np.vstack((c.reshape(-1, 9), r)), np.vstack((dc.reshape(-1, 2), dr)),
                np.vstack((ec.reshape(-1, 2), er)))

    def run(p, sc, sr, loss):
        scale = np.r_[np.full(n, sc), np.full(len(yr), sr)]
        last = None
        def get(logtau):
            nonlocal last
            if last is None or not np.array_equal(logtau, last[0]):
                last = logtau.copy(), design(logtau)
            return last[1]
        def fun(p):
            a, _, _ = get(p[:2])
            return (np.einsum('ij,j->i', a, p[2:]) - y) / scale
        def jac(p):
            a, d, e = get(p[:2])
            j = d * p[2:4]
            j[:, 0] += e @ p[-2:]
            return np.column_stack((j, a)) / scale[:, None]
        low = np.log(min(dt.min(), t[t > 0].min()) / 20)
        high = np.log((dt.sum() + t.max()) * 20)
        return least_squares(fun, p, jac=jac, loss=loss, x_scale='jac', max_nfev=90,
                             bounds=(np.r_[[low]*2, [-np.inf]*9], np.r_[[high]*2, [np.inf]*9]))

    results = []
    for ts in ((.26, 1.4), (.55, 3.2), (.9, 5.5), (1.4, .26)):
        l = np.log(ts)
        a, _, _ = design(l)
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        results.append(run(np.r_[l,w], .04, .05, 'cauchy' if robust else 'linear'))
    p = min(results, key=lambda s: s.cost).x
    if robust:
        a, _, _ = design(p[:2])
        e = np.einsum('ij,j->i', a, p[2:]) - y
        def scale(v):
            return max(1e-5, 2.5 * 1.4826 * np.median(np.abs(v - np.median(v))))
        p = run(p, scale(e[:n]), scale(e[n:]), 'cauchy').x
    return p
