import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/alpha_m97_c01_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c01_batch_07.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    d = item['dataset']
    tr = d['train']
    lab = np.asarray(tr['lab_id'])
    n = len(lab)
    p, h, x, start = s._inputs(d)
    m = s._Model(d, p, h, x)
    y = np.column_stack((tr['strain_curves_obs'] - start[:n, None], tr['instant_rate_obs']))
    complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
    gate = m.active[:n]
    errors = {name: np.empty_like(y) for name in ('exact', 'algebraic_extensions', 'centered_telemetry')}
    tele = np.asarray(tr['aux_chamber_telemetry'], float)
    tele = tele - np.vstack([np.median(tele[lab == c], axis=0) for c in lab])
    for c in np.unique(lab):
        fit = np.flatnonzero(complete & (lab != c))
        val = np.flatnonzero(lab == c)
        lt, w = s._fit(m, y, fit)
        a = m.design(lt, slice(n))
        err = a @ w - y
        errors['exact'][val] = err[val]
        gap = a[:, -1, 0] * np.exp(lt[0])
        flow = m.flow[:n]
        pairs = np.column_stack([(x[:n, i] - x[:n, j])**2 for i in range(4) for j in range(i + 1, 4)])
        triple = (x[:n, 0] * x[:n, 1] * x[:n, 2])[:, None]
        frac = (gap / (1 + h[:n]**2))[:, None]
        proposals = {'algebraic_extensions': np.column_stack((pairs, triple, frac)), 'centered_telemetry': tele}
        core = a[fit].reshape(-1, 9)
        print('held-out lab', c, 'core rank', np.linalg.matrix_rank(core), 'taus', np.exp(lt).round(5))
        for name, features in proposals.items():
            add = flow[:, :, None] * features[:, None, :]
            mtx = np.concatenate((a, add), axis=2)
            ef = err[fit]
            sd = np.maximum(1.4826 * np.median(abs(ef - np.median(ef, axis=0)), axis=0), 1e-7)
            root = (1 + (ef / (3 * sd))**2)**-0.5
            z = mtx[fit].reshape(-1, mtx.shape[-1]) * root.reshape(-1, 1)
            coeff = np.linalg.lstsq(z, (y[fit] * root).ravel(), rcond=1e-10)[0]
            errors[name][val] = mtx[val] @ coeff - y[val]
    for name, e in errors.items():
        for full in (True, False):
            rows = complete == full
            slices = [rows & (lab == c) & (gate == g) for c in np.unique(lab) for g in (False, True)]
            slice_scores = [(np.quantile(abs(e[k,:-1]), .9), np.quantile(abs(e[k,-1]), .9)) for k in slices if k.any()]
            if slice_scores:
                print(name, 'complete', full, 'worst curve/rate p90', np.max(slice_scores, axis=0).round(6), 'pooled MAE', np.array([abs(e[rows,:-1]).mean(), abs(e[rows,-1]).mean()]).round(6))
    base = errors['exact']
    for c in np.unique(lab):
        for g in (False, True):
            k = complete & (lab == c) & (gate == g)
            if k.any():
                print('exact signed lab/gate residual', c, g, k.sum(), 'last-horizon/rate', np.median(base[k][:, [-2, -1]], axis=0).round(6))
    gt = item['gt']
    stress_error = np.max(abs(p[n:] - gt['te_hist_true']))
    max_stress = np.max(abs(p[n:]), axis=1)
    threshold = float(d['crit_stress'])
    print('public stress reconstruction max error', stress_error, 'test coupons within that error of damage threshold', np.sum(abs(max_stress - threshold) <= stress_error))
