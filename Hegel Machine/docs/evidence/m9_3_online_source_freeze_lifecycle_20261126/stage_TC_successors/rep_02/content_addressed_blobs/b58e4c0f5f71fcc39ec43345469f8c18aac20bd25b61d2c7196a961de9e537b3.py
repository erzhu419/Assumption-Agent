"""Two-branch hereditary response with yield-gated viscoplastic strain."""

import numpy as np
from scipy.optimize import minimize


def _align(train, test):
    keys = ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')
    x, panel, lab, stratum = (np.concatenate((train[k], test[k])) for k in keys)
    x = x.astype(float)
    _, cell, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    weights = 1.0 / counts[cell.ravel()]
    for label in np.unique(lab):
        rows = lab == label
        refs = panel[rows]
        valid = np.isfinite(refs)
        keep = valid.sum(axis=0) >= 2
        refs = np.where(valid[:, keep], refs[:, keep], np.nan)
        offset = np.median(np.nanmean(refs, axis=0))
        gain = np.median(np.nanstd(refs, axis=0))
        x[rows] = (x[rows] - offset) / gain
    valid = np.isfinite(x)
    mass = np.sum(weights[:, None] * valid, axis=0)
    total = np.sum(weights[:, None] * np.where(valid, x, 0.0), axis=0)
    fill = np.divide(total, mass, out=np.zeros_like(mass), where=mass > 0)
    n = len(train['lab_id'])
    _, cell, counts = np.unique(np.column_stack((lab[:n], stratum[:n])), axis=0,
                               return_inverse=True, return_counts=True)
    fit_weights = n / (len(counts) * counts[cell.ravel()])
    x = np.where(valid, x, fill)
    return x[:n], x[n:], fit_weights


class _Response:
    def __init__(self, split, aligned, steps, times, yield_stress):
        self.history = np.asarray(split['past_stress'])
        self.hold = np.asarray(split['stress_hold_0'])
        self.steps = steps
        self.times = times
        self.tail = np.r_[np.cumsum(steps[:0:-1])[::-1], 0.0]
        self.overstress = np.maximum(np.abs(self.hold) - yield_stress, 0.0)**2 * np.sign(self.hold)
        work = (np.maximum(np.abs(self.history) - yield_stress, 0.0)**2 * steps).sum(axis=1)
        self.surge = work * self.hold
        self.drift = np.column_stack((np.ones(len(aligned)), aligned, self.overstress))

    def design(self, taus):
        kernel = -np.expm1(-self.steps[:, None] / taus) * np.exp(-self.tail[:, None] / taus)
        gap = self.hold[:, None] - np.einsum('ij,jk->ik', self.history, kernel)
        rise = -np.expm1(-self.times[:, None] / taus)
        curve = np.concatenate((gap[:, None, :] * rise[None, :, :],
                                self.drift[:, None, :] * self.times[None, :, None],
                                self.surge[:, None, None] * rise[None, :, :1]), axis=2)
        rate = np.column_stack((gap / taus, self.drift, self.surge / taus[0]))
        return curve, rate


def _weighted_fit(a, y, w):
    # The subproblem has only nine columns. Use the weighted normal equations
    # on normalized columns and fall back to QR/SVD if the basis loses rank.
    wa = w[:, None] * a
    gram = np.einsum('ni,nj->ij', a, wa)
    rhs = np.einsum('ni,n->i', wa, y)
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        root = np.sqrt(w)
        return np.linalg.lstsq(a * root[:, None], y * root, rcond=None)[0]


def _profile(response, train, specimen_weight, columns=slice(None)):
    obs = np.asarray(train['strain_curves_obs'])
    y = np.r_[(obs - np.asarray(train['strain_0'])[:, None]).ravel(), train['instant_rate_obs']]
    n_curve = obs.size
    base_weight = np.r_[np.repeat(specimen_weight, obs.shape[1]), specimen_weight]
    near = np.r_[np.full(n_curve, 0.065), np.full(len(obs), 0.065)]

    def make_profile(scale):
        ys = y / scale

        def evaluate(log_tau, return_residual=False):
            taus = np.exp(log_tau)
            curve, rate = response.design(taus)
            a = np.concatenate((curve.reshape(n_curve, -1), rate))[:, columns] / scale[:, None]
            norm = np.sqrt(np.mean(a * a, axis=0))
            norm[norm == 0] = 1.0
            a = a / norm
            coef = _weighted_fit(a, ys, base_weight)
            for _ in range(25):
                r = np.einsum('ni,i->n', a, coef) - ys
                updated = _weighted_fit(a, ys, base_weight / (1 + r*r))
                if np.max(np.abs(updated - coef)) < 1e-6 * (1 + np.max(np.abs(coef))):
                    coef = updated
                    break
                coef = updated
            r = np.einsum('ni,i->n', a, coef) - ys
            loss = np.sum(base_weight * np.log1p(r*r))
            if return_residual:
                return coef / norm, r * scale
            return loss
        return evaluate

    bounds = [(np.log(0.20), np.log(8.0))] * 2
    gap = 1e-4

    def refine(profile, start):
        def objective(q):
            return profile(q) / len(y)
        return minimize(objective, start, method='SLSQP', bounds=bounds,
                        constraints=[{'type': 'ineq', 'fun': lambda q: q[1]-q[0]-gap}],
                        options={'ftol': 1e-10, 'maxiter': 70}).x

    warm = make_profile(near)
    grid = np.linspace(*bounds[0], 9)
    coarse = sorted((warm((a, b)), (a, b))
                    for j, b in enumerate(grid) for a in grid[:j])
    starts = [refine(warm, q) for _, q in coarse[:3]]
    best = min(starts, key=warm)
    _, residual = warm(best, True)

    def mad(a):
        return 1.4826 * np.median(np.abs(a - np.median(a)))

    scale = np.r_[np.full(n_curve, max(0.010, 1.5 * mad(residual[:n_curve]))),
                  np.full(len(obs), max(0.010, 1.5 * mad(residual[n_curve:])))]
    profile = make_profile(scale)
    best = min((best, refine(profile, best)), key=profile)
    coef, _ = profile(best, True)
    return np.exp(best), coef


def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, x_test, weights = _align(train, test)
    extra = (np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times']),
             float(dataset['yield_stress']))
    taus, coef = _profile(_Response(train, x_train, *extra), train, weights)
    curves, rate = _Response(test, x_test, *extra).design(taus)
    prediction = np.asarray(test['strain_0'])[:, None] + np.einsum('nti,i->nt', curves, coef)
    # A specimen-level drift error changes the onset rate and accumulates
    # linearly in strain. Shrink the measured innovation at the noise scale.
    innovation = np.asarray(test['instant_rate_obs']) - rate @ coef
    drift_correction = np.sign(innovation) * np.maximum(np.abs(innovation) - 0.010, 0.0)
    prediction += drift_correction[:, None] * extra[1][None, :]
    return {'estimated_taus': taus, 'viscoplastic_weights': coef[-2:],
            'predicted_strain': prediction}
