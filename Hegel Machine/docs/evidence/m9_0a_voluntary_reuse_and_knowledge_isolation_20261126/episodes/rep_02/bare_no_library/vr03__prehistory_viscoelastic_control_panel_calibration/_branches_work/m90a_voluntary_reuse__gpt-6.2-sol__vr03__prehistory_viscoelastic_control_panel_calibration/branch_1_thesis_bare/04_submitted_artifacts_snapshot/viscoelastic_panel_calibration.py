"""Extract hereditary moduli and fit risk in the control-aligned coordinates."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(trace, taus):
    past_t = np.asarray(trace['past_timestamps'], dtype=float)
    past_u = np.asarray(trace['past_strain_history'], dtype=float)
    t = np.asarray(trace['timestamps'], dtype=float)
    u = np.asarray(trace['strain_u'], dtype=float)
    times = np.concatenate((past_t, t))
    strain = np.concatenate((past_u, u))
    fraction = -np.expm1(-np.diff(times)[:, None] / taus)
    memory = np.full(taus.size, past_u[0])
    design = np.column_stack((np.ones(t.size), u, np.empty((t.size, taus.size))))
    for k, (f, value) in enumerate(zip(fraction, strain[1:]), start=1):
        memory += f * (value - memory)
        if k >= past_t.size:
            design[k - past_t.size, 2:] = memory
    return design


def _robust_moduli(design, stress, starts):
    """Concentrate least-trimmed squares, then refit on all clean points."""
    def fit(rows):
        return np.linalg.lstsq(design[rows], stress[rows], rcond=None)[0]

    n = stress.size
    h = int(0.8 * n)
    best_loss = np.inf
    for start in (slice(None), *starts):
        coef = fit(start)
        previous = None
        for _ in range(12):
            error = (stress - design @ coef) ** 2
            rows = np.sort(np.argpartition(error, h - 1)[:h])
            if np.array_equal(rows, previous):
                break
            coef = fit(rows)
            previous = rows
        error = stress - design @ coef
        loss = np.sum(np.partition(error ** 2, h - 1)[:h])
        if loss < best_loss:
            best_loss, best_error, best_coef = loss, error, coef
        # The specified glitches are at least 4.5 stress units. A fit
        # with this much agreement already isolates the clean trace.
        if np.count_nonzero(np.abs(error) < 1.0) >= h:
            break
    clean = np.abs(best_error) < 2.25
    return (fit(clean) if np.count_nonzero(clean) >= h else best_coef)[1:]


def _aligned(split, taus, starts):
    moduli = np.array([
        _robust_moduli(_design(trace, taus), np.asarray(trace['stress_obs']), starts)
        for trace in split['traces']
    ])
    moduli[np.asarray(split['missing_channel_mask'], dtype=bool)] = np.nan
    centers = np.asarray(split['center_id'])
    controls = np.asarray(split['controls'], dtype=float)
    for center in np.unique(centers):
        rows = centers == center
        panel = controls[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        moduli[rows] = (moduli[rows] - offset) / gain
    return moduli


def _imputation_means(z, centers, strata):
    cells = np.unique(np.column_stack((centers, strata)), axis=0)
    means = []
    for c, s in cells:
        cell = z[(centers == c) & (strata == s)]
        count = np.isfinite(cell).sum(axis=0)
        means.append(np.divide(np.nansum(cell, axis=0), count,
                               out=np.full(z.shape[1], np.nan), where=count > 0))
    return np.nanmean(means, axis=0)


def _risk_design(z, strata):
    return np.column_stack((np.eye(3)[strata], z))


def _fit_risk(design, y):
    # A small slope penalty prevents an unstable fit without penalizing
    # the three stratum intercepts.
    def objective(coef):
        eta = design @ coef
        loss = np.logaddexp(0, (1 - 2 * y) * eta).sum()
        grad = design.T @ (expit(eta) - y)
        loss += 0.25 * (coef[3:] @ coef[3:])
        grad[3:] += 0.5 * coef[3:]
        return loss, grad

    return minimize(objective, np.zeros(design.shape[1]), jac=True,
                    method='L-BFGS-B', options={'gtol': 1e-8, 'ftol': 1e-12}).x


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Return imputed aligned query moduli, failure probabilities and slopes."""
    train, query = cohort['train'], cohort['query']
    taus = np.asarray(cohort['taus'], dtype=float)
    rng = np.random.default_rng(0)
    n = len(train['traces'][0]['timestamps'])
    starts = tuple(rng.choice(n, 8, replace=False) for _ in range(24))
    z_train = _aligned(train, taus, starts)
    z_query = _aligned(query, taus, starts)
    strata = np.asarray(train['stratum'], dtype=int)
    means = _imputation_means(z_train, np.asarray(train['center_id']), strata)
    z_train = np.where(np.isnan(z_train), means, z_train)
    z_query = np.where(np.isnan(z_query), means, z_query)
    coef = _fit_risk(_risk_design(z_train, strata), np.asarray(train['y'], dtype=float))
    eta_query = _risk_design(z_query, np.asarray(query['stratum'], dtype=int)) @ coef
    return {'aligned_query_moduli': z_query,
            'query_probabilities': np.clip(expit(eta_query), 1e-12, 1 - 1e-12),
            'modulus_weights': coef[3:]}
