import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_w02_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

def subset(block, mask):
    return {k: v[mask] for k, v in block.items()}

def rmse(x):
    return np.sqrt(np.nanmean(x*x))

for case in pickle.load(open('/holdout_task/public_alpha_w02_batch_04.pkl', 'rb')):
    d, g = case['dataset'], case['gt']
    tr, te = d['train'], d['test']
    dt, t, y = d['past_dt'], d['eval_times'], d['yield_stress']
    cal = s._calibrate(d['calibration_load_cells'])
    n = len(tr['lab_id'])
    x = s._features(tr, te)
    tr_in = s._inputs(tr, cal, dt, y, x[:n])
    te_in = s._inputs(te, cal, dt, y, x[n:])
    obs = np.c_[tr['strain_curves_obs']-tr['strain_0'][:, None], tr['instant_rate_obs']]
    clean = np.c_[g['te_clean']-te['strain_0'][:, None], g['te_true_rate0']]
    miss = ~np.isfinite(tr['coupon_features_raw']).all(1)
    testmiss = ~np.isfinite(te['coupon_features_raw']).all(1)
    lab = tr['lab_id']
    yielded = (tr_in[3] != 0) | (tr_in[4] != 0)
    testyield = (te_in[3] != 0) | (te_in[4] != 0)
    fold = np.empty(n, int)
    rng = np.random.default_rng(318)
    for l in range(4):
        for j in (False, True):
            for m in (False, True):
                ids = rng.permutation(np.flatnonzero((lab == l) & (yielded == j) & (miss == m)))
                fold[ids] = (np.arange(len(ids))+rng.integers(4)) % 4
    errors = np.empty((2, *obs.shape))
    for k in [0, 1, 2, 3, None]:
        fit = np.ones(n, bool) if k is None else fold != k
        inp = tuple(z[fit] for z in tr_in)
        tau, c = s._fit(inp, dt, t, obs[fit])
        residual = obs[fit] - s._design(inp, dt, t, np.log(tau)) @ c
        slopes = np.nanmedian(residual[:, :-1]/t, axis=1)
        shifts = np.array([np.nanmedian(slopes[(lab[fit] == l) & ~miss[fit]]) for l in range(4)])
        block = te if k is None else subset(tr, ~fit)
        held = te_in if k is None else tuple(z[~fit] for z in tr_in)
        a = s._design(held, dt, t, np.log(tau)) @ c
        b = a + shifts[block['lab_id'], None]*np.r_[t, 1.]
        for q in (a, b):
            s._condition_drift(subset(tr, fit), block, inp, obs[fit], q, dt, t, tau, c)
        if k is None:
            independent = np.stack((a-clean, b-clean))
        else:
            errors[:, ~fit] = np.stack((a-obs[~fit], b-obs[~fit]))
        print('fold, fitted lab drifts', k, np.round(shifts, 6))
    for name,e,labs,regime,missing,metric in [
        ('real OOF curve observations', errors, lab, yielded, miss, lambda a: np.nanmedian(abs(a))),
        ('independent clean test', independent, te['lab_id'], testyield, testmiss, rmse)]:
        for group,use in [('all', np.ones(len(labs), bool)), ('complete', ~missing), ('incomplete', missing)]:
            results=[]
            for l in range(4):
                for r in (False, True):
                    z=use & (labs == l) & (regime == r)
                    if z.any():
                        results.append([metric(e[0,z,:-1]),metric(e[1,z,:-1]),metric(e[0,z,-1]),metric(e[1,z,-1])])
            print(name, group, 'worst curve base/centered, rate base/centered', np.round(np.max(results,axis=0), 7))
