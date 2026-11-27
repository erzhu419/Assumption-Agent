import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
s = importlib.util.spec_from_file_location('solver', root + 'alpha_m96_w03_s00_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
for case in pickle.load(open(root + 'public_alpha_m96_w03_s00.pkl', 'rb')):
    d = case['dataset']; tr, te = d['train'], d['test']
    lab = tr['lab_id']; dt, t = d['past_dt'], d['eval_times']
    g, b = m._load_cell_fit(d['calibration_load_cells'])
    p = tr['past_stress_cmd']*g[lab,None] + b[lab,None]
    h = tr['stress_hold_cmd']*g[lab] + b[lab]
    X = m._align(tr, te)[:len(lab)]
    y = tr['strain_curves_obs'] - tr['strain_0'][:,None]; r = tr['instant_rate_obs']
    complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
    metrics = {k: [] for k in ('exact','pair','rational')}
    for name, held in [(f'lab{i}', lab==i) for i in range(4)] + [('wingA', lab<2), ('wingB', lab>=2)]:
        fit = ~held
        ts, w = m._fit(dt, t, p[fit], h[fit], X[fit], y[fit], r[fit])
        A, B = m._design(ts, dt, t, p, h, X)
        u = h / d['yield_stress']
        extras = {'exact': np.empty((len(lab),0)), 'pair': np.column_stack([(X[:,i]+X[:,j])**2-(X[:,i]-X[:,j])**2 for i in range(4) for j in range(i+1,4)])/4, 'rational': np.column_stack((u**2/(1+u**2),u**3/(1+u**2)))}
        sc = 2.4*m._mad((A[fit]@w-y[fit]).ravel()); sr = 2.4*m._mad(B[fit]@w-r[fit]); q = held & complete
        for key, V in extras.items():
            a = np.concatenate((A,t[None,:,None]*V[:,None,:]),axis=2); z = np.c_[B,V]
            c = w
            if V.shape[1]:
                def res(c):
                    return np.r_[((a[fit]@c-y[fit])/sc).ravel(), (z[fit]@c-r[fit])/sr]
                c = least_squares(res, np.r_[w,np.zeros(V.shape[1])], loss='cauchy').x
            ec = np.abs(a[q]@c-y[q]); er = np.abs(z[q]@c-r[q]);
            v = [np.median(ec), np.max(np.median(ec,axis=0)), np.median(er)]
            metrics[key].append(v)
        print(name, 'median strain / worst horizon median / median rate:', {k: np.round(v[-1],5).tolist() for k,v in metrics.items()})
    print('worst held-out slices:', {k: np.round(np.max(v,axis=0),5).tolist() for k,v in metrics.items()})
