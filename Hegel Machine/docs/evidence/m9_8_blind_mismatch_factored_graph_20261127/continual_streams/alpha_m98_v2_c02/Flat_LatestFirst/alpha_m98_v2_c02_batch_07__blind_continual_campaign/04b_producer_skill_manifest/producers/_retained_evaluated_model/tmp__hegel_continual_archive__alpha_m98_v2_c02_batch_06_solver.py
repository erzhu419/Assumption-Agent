"""Fit the shared relaxation spectrum and predict blind chamber coupons."""

from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _standardize(train, test):
    x, ref, lab, stratum = (np.concatenate((train[k], test[k]), axis=0)
                            for k in ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum'))
    for group in np.unique(lab):
        mask = lab == group
        controls = ref[mask]
        offset = np.nanmedian(np.nanmean(controls, axis=0))
        scale = np.nanmedian(np.nanstd(controls, axis=0))
        x[mask] = (x[mask] - offset) / scale
    _, inverse, counts = np.unique(np.column_stack((lab, stratum)), axis=0,
                                  return_inverse=True, return_counts=True)
    weight = (1.0 / counts[inverse]).reshape(-1, 1)
    observed = np.isfinite(x)
    fill = (np.sum(weight * np.where(observed, x, 0), axis=0)
            / np.sum(weight * observed, axis=0))
    x = np.where(observed, x, fill)
    n = len(train['lab_id'])
    return x[:n], x[n:]


terms=[(0,0,0,0)]
for k in (1,2,3):
 for ids in combinations_with_replacement(range(4),k): terms.append(tuple(ids.count(i) for i in range(4)))
terms=np.array(terms); deriv=np.maximum(terms[None,:,:]-np.eye(4,dtype=int)[:,None,:],0)
def basis(u): return np.prod(u[:,None,:]**terms,axis=2)
def invert(raw, b):
 z=raw.reshape(-1,4); u=np.linalg.solve(b[1:5].T,(z-b[0]).T).T
 for _ in range(32):
  r=basis(u)@b-z
  if np.max(abs(r))<1e-10: break
  jac=np.einsum('nik,ik,kj->nji', np.prod(u[:,None,None,:]**deriv,axis=-1), terms.T,b)
  step=np.linalg.solve(jac,r[...,None])[...,0]; h=np.ones((len(u),1)); nxt=u-step
  for __ in range(16):
   wrong=np.sum((basis(nxt)@b-z)**2,axis=1)>np.sum(r*r,axis=1)
   if not wrong.any(): break
   h[wrong]*=.5; nxt=u-h*step
  u=nxt
 return u.reshape(raw.shape)
def _stress_calibration(cal):
    x = cal['true_channel_loads']
    y = cal['bridge_raw']
    # Reject displaced sweep rows using forward residuals in bridge coordinates.
    a = basis(x)
    simple = a[:, :5]
    e = y - simple @ np.linalg.lstsq(simple, y, rcond=None)[0]
    magnitude = np.max(np.abs(e), axis=1)
    keep = magnitude < 6 * np.median(magnitude)
    full = np.linalg.lstsq(a[keep], y[keep], rcond=None)[0]
    # Select a sparse per-channel basis, then refit without coefficient shrinkage.
    b = np.zeros_like(full)
    for j in range(4):
        terms_j = np.abs(full[:, j]) > 0.004 * np.max(np.abs(full[:, j]))
        b[terms_j, j] = np.linalg.lstsq(a[keep][:, terms_j], y[keep, j], rcond=None)[0]
    return lambda raw: invert(raw, b).sum(axis=-1)


def _design(site, x, stress, dt, t):
    past, hold = stress(site['past_channel_raw']), stress(site['hold_channel_raw'])
    tail = np.cumsum(dt[::-1])[::-1] - dt
    features = np.column_stack((np.ones(len(x)), x))
    drift = features[:, None, :] * t[None, :, None]

    def matrices(taus):
        weights = -np.expm1(-dt[:, None] / taus) * np.exp(-tail[:, None] / taus)
        gap = hold[:, None] - past @ weights
        transient = gap[:, None, :] * -np.expm1(-t[:, None] / taus)
        return np.concatenate((transient, drift), axis=2), np.column_stack((gap / taus, features))

    return matrices


def _fit(matrices, strain, rate):
    target = np.r_[strain.ravel(), rate]
    n = strain.size

    def taus_of(p):
        first = np.exp(p[0])
        return np.array([first, first + (25.0 - first) * p[1]])

    def design(p):
        s, r = matrices(taus_of(p))
        return np.vstack((s.reshape(-1, 7), r))

    def mad(v):
        return max(1e-9, 1.4826 * np.median(np.abs(v - np.median(v))))

    def scales(residual):
        return np.r_[np.full(n, mad(residual[:n])), np.full(len(rate), mad(residual[n:]))]

    bounds = (np.r_[np.log(0.050001), 1e-6, np.full(7, -np.inf)],
              np.r_[np.log(24.999), 0.999999, np.full(7, np.inf)])
    initial = []
    for first, second in ((0.15, 0.9), (0.35, 2), (0.7, 5), (1.5, 10), (3, 18)):
        p = np.array([np.log(first), (second - first) / (25 - first)])
        a = design(p)
        beta = np.linalg.lstsq(a, target, rcond=None)[0]
        residual = a @ beta - target
        initial.append((np.r_[p, beta], residual))
    scale = scales(min(initial, key=lambda pair: np.median(np.abs(pair[1])))[1])

    def fit(p, scale, loss):
        return least_squares(lambda q: (design(q) @ q[2:] - target) / scale,
                             p, bounds=bounds, loss=loss, f_scale=2.0,
                             x_scale='jac', max_nfev=180)

    pilot = min((fit(p, scale, 'soft_l1') for p, _ in initial), key=lambda opt: opt.cost)
    scale = scales(pilot.fun * scale)
    solution = fit(pilot.x, scale, 'cauchy').x
    return taus_of(solution), solution[2:]


def _chamber(site, x, stress, dt, times, taus, theta):
    past = stress(site['past_channel_raw'])
    hold = stress(site['hold_channel_raw'])
    tail = np.cumsum(dt[::-1])[::-1] - dt
    # Preload states relax during the chamber equilibration wait.
    weights = -np.expm1(-dt[:, None] / taus) * np.exp(-(tail[:, None] + 7.2) / taus)
    amplitude = (hold[:, None] - past @ weights) * theta[:2]
    drift = theta[2] + x @ theta[3:]
    kernel = -np.expm1(-times[:, None] / taus)
    # Integrate a persistent drift component and an exponentially gated component.
    drift_integral = 0.22 * times - 0.79 * np.expm1(-times)
    return amplitude @ kernel.T + drift[:, None] * drift_integral, amplitude @ (1 / taus) + 1.01 * drift


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    x_train, x_test = _standardize(train, test)
    stress = _stress_calibration(dataset['calibration_load_cells'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    fit_design = _design(train, x_train, stress, dt, times)
    taus, theta = _fit(fit_design, train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    strain, rate = _chamber(test, x_test, stress, dt, times, taus, theta)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + strain,
            'initial_rate_pred': rate}
