"""Profile the linear constitutive coefficients at two relaxation times."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _lab_checks(checks, lab):
    x, y = (np.asarray(checks[k], float) for k in ('cmd_stress', 'true_stress'))
    take = (np.asarray(checks['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[take], y[take]


def _line(x, y, tol, prior=None):
    i, j = np.triu_indices(x.size, 1)
    usable = abs(x[j] - x[i]) >= 4 * tol
    i, j = i[usable], j[usable]
    gains = (y[j] - y[i]) / (x[j] - x[i])
    lines = np.c_[gains, y[i] - gains * x[i]]
    if prior is not None:
        lines = np.vstack((lines, prior))
    lines = lines[lines[:, 0] > 0]
    if not lines.size:
        return None, np.zeros(x.size, bool)
    r = abs(y - lines[:, :1] * x - lines[:, 1:])
    cost = np.minimum((r / tol)**2, 1).sum(axis=1)
    if prior is not None:
        cost += .025 * np.minimum(((lines - prior)**2).sum(axis=1) / tol**2, 1)
    mask = r[np.argmin(cost)] < tol
    if mask.sum() >= 2 and np.ptp(x[mask]) >= 4 * tol:
        fit = np.linalg.lstsq(np.c_[x[mask], np.ones(mask.sum())], y[mask], rcond=None)[0]
        return fit, mask
    return None, mask


def _calibrate(checks, sy):
def _features(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    feat = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        m = labs == lab
        r = ref[m]
        r = r[:, np.isfinite(r).sum(axis=0) >= 2]
        feat[m] = (feat[m] - np.median(np.nanmean(r, axis=0))) / np.median(np.nanstd(r, axis=0))
    _, cell = np.unique(np.c_[labs, strata], axis=0, return_inverse=True)
    cell = cell.ravel()
    finite = np.isfinite(feat)
    means = []
    for k in np.unique(cell):
        rows = cell == k
        count = finite[rows].sum(axis=0)
        total = np.where(finite[rows], feat[rows], 0.).sum(axis=0)
        means.append(np.divide(total, count, out=np.full(feat.shape[1], np.nan), where=count > 0))
    means = np.array(means)
    count = np.isfinite(means).sum(axis=0)
    fill = np.divide(np.nansum(means, axis=0), count, out=np.zeros(feat.shape[1]), where=count > 0)
    return np.c_[np.ones(len(feat)), np.where(finite, feat, fill)]


def _inputs(split, feat, cal, dt, sy):
    gain, bias = cal[np.asarray(split['lab_id'], int)].T
    past = gain[:, None] * split['past_stress_cmd'] + bias[:, None]
    hold = gain * split['stress_hold_cmd'] + bias
    ov = np.maximum(abs(hold) - sy, 0)**2 * np.sign(hold)
    work = np.sum(np.maximum(abs(past) - sy, 0)**2 * dt, axis=1) * hold
    return past, hold, np.c_[feat, ov], work


def _design(inputs, dt, t, tau):
    past, hold, drift, work = inputs
    end = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    mem = np.exp(-end[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    delta = hold[:, None] - past @ mem
    response = -np.expm1(-t[:, None] / tau)
    curve = np.concatenate((delta[:, None, :] * response,
                            drift[:, None, :] * t[None, :, None],
                            (work[:, None] * response[:, 0])[:, :, None]), axis=2)
    rate = np.c_[delta / tau, drift, work / tau[0]]
    return np.concatenate((curve, rate[:, None, :]), axis=1)


