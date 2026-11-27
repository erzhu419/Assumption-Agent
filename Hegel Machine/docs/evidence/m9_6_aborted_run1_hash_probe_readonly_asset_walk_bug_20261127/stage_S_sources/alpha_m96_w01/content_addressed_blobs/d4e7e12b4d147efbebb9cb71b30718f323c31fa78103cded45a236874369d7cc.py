"""Estimate a shared two-branch sub-yield creep law from calibrated coupons."""
import numpy as np
from scipy.optimize import least_squares


def _load_calibration(checks, labs):
    calibration = {}
    for lab in labs:
        mask = np.asarray(checks['lab_id']) == lab
        x = np.asarray(checks['cmd_stress'], float)[mask]
        y = np.asarray(checks['true_stress'], float)[mask]
        i, j = np.triu_indices(len(x), 1)
        dx = x[i] - x[j]
        good = np.abs(dx) > max(1e-10, 0.08 * np.ptp(x)) if len(x) else dx.astype(bool)
        if not np.any(good):
            calibration[lab] = (1., np.median(y-x) if len(x) else 0.)
            continue
        i, j, dx = i[good], j[good], dx[good]
        gain = np.median((y[i] - y[j]) / dx)
        bias = np.median(y - gain*x)
        res = y - gain*x - bias
        keep = np.abs(res) <= max(5e-4, 4.5 * np.median(np.abs(res)))
        if keep.sum() >= 2 and np.ptp(x[keep]) > 1e-10:
            gain, bias = np.linalg.lstsq(np.column_stack((x[keep], np.ones(keep.sum()))), y[keep], rcond=None)[0]
        calibration[lab] = gain, bias
    return calibration


def _align(train, test):
    lab, stratum, raw, ref = (np.concatenate((train[k], test[k])) for k in
        ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for group in np.unique(lab):
        rows = lab == group
        panel = ref[rows]
        panel = panel[:, np.sum(np.isfinite(panel), axis=0) > 1]
        center = np.median(np.nanmean(panel, axis=0))
        widths = np.nanstd(panel, axis=0)
        scale = np.median(widths[widths > 0])
        aligned[rows] = (raw[rows] - center) / scale
    cells = [aligned[(lab == g) & (stratum == s)] for g, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    count = np.array([np.isfinite(c).sum(axis=0) for c in cells])
    sums = np.array([np.nansum(c, axis=0) for c in cells])
    means = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    fill = means.sum(axis=0) / np.maximum(1, (count > 0).sum(axis=0))
    aligned = np.where(np.isfinite(aligned), aligned, fill)
    return np.split(aligned, [len(train['lab_id'])])


def _physics(split, aligned, calibration, dt, time):
    gain, bias = np.array([calibration[g] for g in split['lab_id']]).T
    past = gain[:, None] * split['past_stress_cmd'] + bias[:, None]
    hold = gain * split['stress_hold_cmd'] + bias
    history_time = np.cumsum(dt[::-1])[::-1] - dt
    features = np.column_stack((np.ones(len(hold)), aligned))
    n, k = len(hold), len(time)
    drift = np.concatenate(((features[:, None, :] * time[None, :, None]).reshape(-1, 5), features))

    def branch(tau):
        # Unroll the exact zero-initial-state recurrence into history weights.
        decay = np.exp(-history_time / tau)
        step = -np.expm1(-dt / tau)
        z = np.einsum('ij,j->i', past, decay * step)
        dz = np.einsum('ij,j->i', past, decay * (history_time * step - dt * np.exp(-dt/tau))) / tau**2
        gap = hold - z
        curve = -np.expm1(-time / tau)
        dcurve = -time * np.exp(-time / tau) / tau**2
        return (np.r_[np.multiply.outer(gap, curve).ravel(), gap/tau],
                np.r_[(np.multiply.outer(gap, dcurve) - np.multiply.outer(dz, curve)).ravel(), -dz/tau-gap/tau**2])

    def design(taus):
        a, da = branch(taus[0])
        b, db = branch(taus[1])
        return np.column_stack((a, b, drift)), da, db
    return design, n, k


def _fit(design, y, nc):
    seeds = []
    # Rank the coarse times using lightly clipped least-squares residuals.
    scale = np.r_[np.full(nc, .025), np.full(len(y)-nc, .018)]
    grid = np.geomspace(.2, 8., 10)
    for i, a in enumerate(grid[:-1]):
        for b in grid[i+1:]:
            A = design((a, b))[0] / scale[:, None]
            yy = y / scale
            w = np.ones_like(y)
            for _ in range(4):
                co = np.linalg.solve(np.einsum('ij,ik,i->jk', A, A, w), np.einsum('ij,i,i->j', A, yy, w))
                res = np.einsum('ij,j->i', A, co) - yy
                w = 1 / (1 + res**2)
            seeds.append((np.log1p(res**2).sum(), np.r_[a, b, co]))

    def refine(p, scale):
        def fun(q):
            A, _, _ = design(q[:2])
            return (np.einsum('ij,j->i', A, q[2:]) - y) / scale

        def jac(q):
            A, da, db = design(q[:2])
            return np.column_stack((da*q[2], db*q[3], A)) / scale[:, None]

        lim = np.full(7, np.inf)
        return least_squares(fun, p, jac=jac, bounds=(np.r_[.2, .2, -lim], np.r_[8., 8., lim]),
                             x_scale='jac', loss='cauchy', max_nfev=100)

    best = min((refine(p, scale) for _, p in sorted(seeds, key=lambda s: s[0])[:3]), key=lambda r: r.cost)
    residual = best.fun * scale
    for section, floor in ((slice(None, nc), 1e-5), (slice(nc, None), 1e-5)):
        r = residual[section]
        scale[section] = max(floor, 1.4826 * np.median(np.abs(r-np.median(r)))) * 1.7
    result = refine(best.x, scale).x
    if result[0] > result[1]:
        result[:4] = result[[1, 0, 3, 2]]
    return result[:2], result[2:]


def solve_alpha_m96_w01_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    labs = np.union1d(train['lab_id'], test['lab_id'])
    calibration = _load_calibration(dataset['calibration_load_cells'], labs)
    x_train, x_test = _align(train, test)
    train_design, n, k = _physics(train, x_train, calibration, dt, time)
    y = np.r_[(train['strain_curves_obs'] - train['strain_0'][:, None]).ravel(), train['instant_rate_obs']]
    taus, weights = _fit(train_design, y, n*k)
    test_design, n, k = _physics(test, x_test, calibration, dt, time)
    pred = test_design(taus)[0] @ weights
    rate = pred[n*k:].copy()
    strain = pred[:n*k].reshape(n, k) + test['strain_0'][:, None]
    # If an active drift feature is missing, onset identifies its aggregate
    # contribution after removing the modeled hereditary rate. The resulting
    # drift correction acts on the hold curve as well as its onset derivative.
    missing = np.any(~np.isfinite(test['coupon_features_raw']) & (weights[3:] != 0), axis=1)
    onset = np.asarray(test['instant_rate_obs'])
    use_onset = missing & np.isfinite(onset)
    strain[use_onset] += (onset[use_onset] - rate[use_onset])[:, None] * time
    rate[use_onset] = onset[use_onset]
    return {'estimated_taus': taus, 'hereditary_weights': weights, 'viscoplastic_weights': np.zeros(2),
            'predicted_strain': strain, 'initial_rate_pred': rate}
