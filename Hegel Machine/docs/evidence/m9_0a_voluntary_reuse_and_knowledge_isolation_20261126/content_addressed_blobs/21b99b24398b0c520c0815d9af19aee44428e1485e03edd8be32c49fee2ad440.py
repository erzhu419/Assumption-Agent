"""Fit hereditary moduli and calibrate failure risk across rheometry centers."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(trace, taus):
    t = np.concatenate((trace['past_timestamps'], trace['timestamps']))
    u = np.concatenate((trace['past_strain_history'], trace['strain_u']))
    start = len(trace['past_timestamps'])
    fraction = -np.expm1(-np.diff(t)[:, None] / taus)
    memory = np.full_like(taus, u[0])
    design = np.empty((len(t) - start, 4))
    design[:, 0] = 1
    design[:, 1] = u[start:]
    for k, f in enumerate(fraction, 1):
        memory += f * (u[k] - memory)
        if k >= start:
            design[k - start, 2:] = memory
    return design


def _robust_moduli(design, stress, samples):
    # Pick a least-median seed from minimal fits, then re-fit the consensus
    # so clean observations, not just the four seed points, determine moduli.
    x = design[samples]
    good = np.abs(np.linalg.det(x)) > 1e-12
    fits = np.linalg.solve(x[good], stress[samples[good], None]).squeeze(-1)
    ordinary = np.linalg.lstsq(design, stress, rcond=None)[0]
    fits = np.vstack((fits, ordinary))
    residuals = np.abs(stress[:, None] - design @ fits.T)
    fit = fits[np.argmin(np.median(residuals, axis=0))]
    for _ in range(4):
        residual = stress - design @ fit
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual - np.median(residual)) <= max(3.5 * mad, 1e-9)
        if np.linalg.matrix_rank(design[keep]) < design.shape[1]:
            break
        fit = np.linalg.lstsq(design[keep], stress[keep], rcond=None)[0]
    return fit[1:]


def _aligned(split, taus):
    rng = np.random.default_rng(2025)
    n_time = len(split['traces'][0]['timestamps'])
    samples = np.array([rng.choice(n_time, 4, replace=False) for _ in range(64)])
    z = np.array([_robust_moduli(_design(trace, taus), np.asarray(trace['stress_obs']), samples)
                  for trace in split['traces']])
    centers = np.asarray(split['center_id'])
    controls = np.asarray(split['controls'])
    for center in np.unique(centers):
        cell = centers == center
        panel = controls[cell]
        populated = np.sum(np.isfinite(panel), axis=0) >= 2
        panel = panel[:, populated]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        z[cell] = (z[cell] - offset) / gain
    z[np.asarray(split['missing_channel_mask'], dtype=bool)] = np.nan
    return z


def _risk_design(z, strata):
    # Reference-code stratum 0; the slopes are shared by all strata and centers.
    return np.column_stack((np.ones(len(z)), strata == 1, strata == 2, z))


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Return aligned query moduli, failure probabilities, and fitted slopes."""
    train, query = cohort['train'], cohort['query']
    taus = np.asarray(cohort['taus'], dtype=float)
    z_train, z_query = _aligned(train, taus), _aligned(query, taus)
    strata = np.asarray(train['stratum'])
    cells = np.column_stack((train['center_id'], strata))
    _, membership, counts = np.unique(cells, axis=0, return_inverse=True,
                                     return_counts=True)
    # Each training center–stratum cell contributes one mean per channel.
    means = np.array([np.nanmean(z_train[membership == k], axis=0)
                      for k in range(len(counts))])
    fill = np.nanmean(means, axis=0)
    z_train = np.where(np.isnan(z_train), fill, z_train)
    z_query = np.where(np.isnan(z_query), fill, z_query)

    design = _risk_design(z_train, strata)
    y = np.asarray(train['y'], dtype=float)
    weights = 1.0 / (len(counts) * counts[membership])
    ridge = 0.003  # Mild slope shrinkage for the finite, reweighted sample.

    def objective(coef):
        eta = design @ coef
        loss = weights @ (np.logaddexp(0, eta) - y * eta)
        grad = design.T @ (weights * (expit(eta) - y))
        slopes = coef[3:]
        grad[3:] += ridge * slopes
        return loss + 0.5 * ridge * (slopes @ slopes), grad

    coef = minimize(objective, np.zeros(design.shape[1]), jac=True,
                    method='BFGS', options={'gtol': 1e-9}).x
    probabilities = expit(_risk_design(z_query, np.asarray(query['stratum'])) @ coef)
    return {
        'aligned_query_moduli': z_query,
        'query_probabilities': np.clip(probabilities, np.finfo(float).eps,
                                       1.0 - np.finfo(float).eps),
        'modulus_weights': coef[3:],
    }
