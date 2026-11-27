import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import trim_mean
from pathlib import Path
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('solver', root/'alpha_m97_c02_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_alpha_m97_c02_batch_06.pkl').open('rb') as f:
    items = pickle.load(f)
for item in items:
    d = item['dataset']; tr = d['train']; lab = tr['lab_id']; n = len(lab)
    t, dt = d['eval_times'], d['past_dt']
    p,h,x = [a[:n] for a in m._prepare(d)]
    C = tr['strain_curves_obs']-tr['strain_0'][:,None]; R = tr['instant_rate_obs']
    complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
    print('response fields', sorted(tr), 'regime index', d['shear_banding_index'])
    for l in np.unique(lab):
        use = lab != l; test = ~use
        tau,w = m._fit(p[use],h[use],x[use],dt,t,C[use],R[use]); A,B = m._design(p,h,x,dt,t,tau)
        curve,rate = C-A@w,R-B@w
        # Candidate omitted terms: a different saturation curve, stress-gated terminal flow, and branch contrast on the slow branch.
        phi = t/(1+.85*t)
        candidate_c = np.stack((x[:,-1,None]*(t/(1+.85*t*t)-phi), x[:,-1,None]*(phi/(1+h[:,None]**2)-phi), B[:,8,None]*tau[0]*(-np.expm1(-t/tau[1]))),axis=2)
        candidate_r = np.column_stack((np.zeros(n), x[:,-1]*(1/(1+h*h)-1), B[:,8]*tau[0]/tau[1]))
        # Refit nuisance coefficients only on the training fold; the held lab never supplies targets.
        aug_c = np.concatenate((A,candidate_c),axis=2); aug_r = np.column_stack((B,candidate_r))
        ec,er = curve[use].ravel(),rate[use]
        sc,sr = [max(1e-9,1.4826*np.median(abs(v-np.median(v)))) for v in (ec,er)]
        robust = least_squares(lambda v: np.r_[((aug_c[use]@v-C[use])/sc).ravel(), (aug_r[use]@v-R[use])/sr], np.r_[w,0.,0.,0.], loss='cauchy',f_scale=2.,x_scale='jac')
        alt_c=C-aug_c@robust.x; alt_r=R-aug_r@robust.x
        z=test&complete
        print('held lab',l,'complete N',z.sum(),'base/aug trimmed abs curve',*[round(float(trim_mean(np.abs(a[z]).ravel(),.10)),5) for a in (curve,alt_c)],'base/aug trimmed abs rate',*[round(float(trim_mean(np.abs(a[z]),.10)),5) for a in (rate,alt_r)],'base signed med complete/incomplete rate',*[round(float(np.median(rate[q])),5) for q in (z,test&~complete) if q.any()])
    o=m.solve_alpha_m97_c02_batch_06(d); g=item['gt']; print('public absolute parameter errors', abs(o['estimated_taus']-g['taus']).round(6),abs(o['hereditary_weights']-g['theta']).round(6),abs(o['regime_mode_weights']-g['vp_theta']).round(6))
    p,h,_=m._prepare(d); print('physical stress absolute-error maxima',np.max(abs(p[n:]-g['te_hist_true'])),np.max(abs(h[n:]-g['te_hold_true'])))
