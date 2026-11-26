import json
import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive')
_CAL_KEYS = ('lab_id', 'cmd_stress', 'true_stress')


def _calibrate(rows, prior=None):
    """Fit each cell's affine map, rejecting isolated EMI excursions."""
    labs, x, y = (np.asarray(rows[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    maps = np.tile([1., 0.], (4, 1))
    historical = np.zeros(4, dtype=bool) if prior is None else np.isfinite(prior).all(axis=1)
    if prior is not None:
        maps[historical] = prior[historical]
    for lab in range(4):
        use = (labs == lab) & np.isfinite(x) & np.isfinite(y)
        u, v = x[use], y[use]
        if not len(u):
            continue
        if historical[lab] and len(u) <= 2:
            residual = v - (maps[lab, 0]*u + maps[lab, 1])
            agrees = np.abs(residual) <= 1e-3
            if agrees.any():
                maps[lab, 1] += np.median(residual[agrees])
                continue
        design = np.column_stack((u, np.ones(len(u))))
        i, j = np.triu_indices(len(u), 1)
        delta = u[j] - u[i]
        good = np.abs(delta) > 1e-8
        if not good.any():
            maps[lab, 1] = np.median(v-maps[lab, 0]*u)
            continue
        slopes = (v[j[good]] - v[i[good]]) / delta[good]
        if len(u) <= 4:
            offsets = ((v[i[good]]-slopes*u[i[good]]) +
                       (v[j[good]]-slopes*u[j[good]])) / 2
            fits = offsets[:, None] + slopes[:, None]*u
            inlier = np.abs(v-fits) <= 1e-3
            deviation = np.mean((fits-(maps[lab, 0]*u+maps[lab, 1]))**2, axis=1)
            best = np.lexsort((deviation, -inlier.sum(axis=1)))[0]
            keep = inlier[best]
            maps[lab] = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
            continue
        offsets = np.median(v - slopes[:, None]*u, axis=1)
        residual = np.abs(v - slopes[:, None]*u - offsets[:, None])
        best = np.argmin(np.median(residual, axis=1))
        r = residual[best]
        keep = r <= max(6*np.median(r), 1e-3)
        if np.ptp(u[keep]) > 1e-8:
            maps[lab] = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
        else:
            maps[lab] = slopes[best], offsets[best]
    return maps


def _archive_rows(record):
    # Accept calibration tables and records wrapped like the public dataset.
    if isinstance(record, (list, tuple)):
        for item in reversed(record):
            yield from _archive_rows(item)
    elif isinstance(record, dict):
        table = record.get('dataset', record)
        if isinstance(table, dict):
            table = table.get('calibration_load_cells', table)
            if isinstance(table, dict) and all(key in table for key in _CAL_KEYS):
                yield table


def _load_maps(rows):
    current = _calibrate(rows)
    labs, u, v = (np.asarray(rows[k]) for k in _CAL_KEYS)
    valid = np.isfinite(u) & np.isfinite(v)
    need = set()
    for lab in range(4):
        take = valid & (labs == lab)
        x, y = u[take], v[take]
        slope, bias = current[lab]
        if len(x) <= 4 or np.ptp(x) <= 1e-8 or np.any(np.abs(y - (slope*x + bias)) > 1e-3):
            need.add(lab)
    if not need or not _ARCHIVE.is_dir():
        return current
    prior = np.full((4, 2), np.nan)
    files = (path for path in _ARCHIVE.rglob('*')
             if path.is_file() and path.suffix in ('.json', '.pkl', '.pickle'))
    for path in sorted(files, key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            with path.open('rb') as stream:
                record = json.load(stream) if path.suffix == '.json' else pickle.load(stream)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            continue
        for table in _archive_rows(record):
            a, x, y = (np.asarray(table[k]) for k in _CAL_KEYS)
            if a.ndim != 1 or a.shape != x.shape or a.shape != y.shape:
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            fitted = _calibrate(table)
            for lab in tuple(need):
                mask = finite & (a == lab)
                g, b = fitted[lab]
                inliers = mask & (np.abs(y - (g*x + b)) <= 1e-3)
                if (inliers.sum() >= 3 and inliers.sum() > mask.sum()/2
                        and np.ptp(x[inliers]) > 1e-8):
                    prior[lab] = fitted[lab]
                    need.remove(lab)
            if not need:
                return _calibrate(rows, prior)
    return _calibrate(rows, prior)


def _align(dataset):
    """Remove the shared lab offset and gain, then fill with cell-balanced means."""
    tr, te = dataset['train'], dataset['test']
    lab, strata, raw, refs = (np.concatenate([tr[k], te[k]]) for k in
                              ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab_id in np.unique(lab):
        mask = lab == lab_id
        panel = refs[mask]
        count = np.isfinite(panel).sum(axis=0)
        panel = panel[:, count >= 2]
        if panel.shape[1]:
            loc = np.nanmedian(np.nanmean(panel, axis=0))
            scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
            aligned[mask] = (raw[mask] - loc) / (scale if scale > 0 else 1.)
        else:
            aligned[mask] = raw[mask]
    _, cell, size = np.unique(np.column_stack((lab, strata)), axis=0,
                               return_inverse=True, return_counts=True)
    weight = (1. / size[cell])[:, None]
    finite = np.isfinite(aligned)
    den = (weight * finite).sum(axis=0)
    fill = np.divide((weight * np.where(finite, aligned, 0.)).sum(axis=0), den,
                     out=np.zeros(raw.shape[1]), where=den > 0)
    return np.where(finite, aligned, fill)


def _inputs(split, x, maps):
    gain, bias = maps[split['lab_id']].T
    return (split['past_stress_cmd']*gain[:, None] + bias[:, None],
            split['stress_hold_cmd']*gain + bias,
            np.column_stack((np.ones(len(x)), x)), np.asarray(split['strain_0']))


class _Law:
    def __init__(self, inputs, dt, times):
        self.hist, self.hold, self.x, self.e0 = inputs
        self.t = np.asarray(times)
        self.tail = np.r_[np.cumsum(np.asarray(dt)[:0:-1])[::-1], 0.]
        self.start = self.tail + dt

    def evaluate(self, p, jac=False):
        tau, a = p[:2], p[2:4]
        late = np.exp(-self.tail[:, None]/tau)
        early = np.exp(-self.start[:, None]/tau)
        delta = self.hold[:, None] - self.hist @ (late - early)
        et = np.exp(-self.t[:, None]/tau)
        shape = -np.expm1(-self.t[:, None]/tau)
        drift = self.x @ p[4:]
        strain = self.e0[:, None] + (delta*a) @ shape.T + drift[:, None]*self.t
        rate = delta @ (a/tau) + drift
        if not jac:
            return strain, rate
        dd = -(self.hist @ (late*self.tail[:, None] - early*self.start[:, None]))/tau**2
        dshape = -et*self.t[:, None]/tau**2
        base = delta[:, None, :]*shape[None, :, :]
        jcurve = np.concatenate(((dd[:, None, :]*shape + delta[:, None, :]*dshape)*a,
                                 base, self.x[:, None, :]*self.t[None, :, None]), axis=2)
        jrate = np.column_stack(((dd/tau-delta/tau**2)*a, delta/tau, self.x))
        return strain, rate, jcurve, jrate


def _fit(law, strain, rate):
    lo = np.r_[[.20, .20], np.full(7, -np.inf)]
    hi = np.r_[[8., 8.], np.full(7, np.inf)]
    def run(p, sc, sr, loss):
        def res(v):
            c, r = law.evaluate(v)
            return np.r_[((c-strain)/sc).ravel(), (r-rate)/sr]
        def jac(v):
            _, _, c, r = law.evaluate(v, True)
            return np.vstack((c.reshape(-1, 9)/sc, r/sr))
        return least_squares(res, p, jac=jac, bounds=(lo, hi), loss=loss,
                             x_scale='jac', max_nfev=100)
    starts = ([.42, 3.0], [.85, 5.8], [.25, 1.6])
    best = min((run(np.r_[tau, .5, .5, np.zeros(5)], .05, .08, 'soft_l1')
                for tau in starts), key=lambda fit: fit.cost)
    c, r = law.evaluate(best.x)
    def scale(residual):
        return max(1e-6, 1.4826*np.median(np.abs(residual-np.median(residual))))
    sc, sr = scale(c-strain), scale(r-rate)
    best = run(best.x, 2.0*sc, 2.0*sr, 'cauchy')
    p = best.x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p


def solve_alpha_w01_s00(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    maps = _load_maps(dataset['calibration_load_cells'])
    x = _align(dataset)
    n = len(train['lab_id'])
    dt, t = np.asarray(dataset['past_dt']), dataset['eval_times']
    model = _Law(_inputs(train, x[:n], maps), dt, t)
    p = _fit(model, train['strain_curves_obs'], train['instant_rate_obs'])
    curve, rate = _Law(_inputs(test, x[n:], maps), dt, t).evaluate(p)
    return dict(estimated_taus=p[:2], hereditary_weights=p[2:],
                viscoplastic_weights=np.zeros(2), predicted_strain=curve,
                initial_rate_pred=rate)
