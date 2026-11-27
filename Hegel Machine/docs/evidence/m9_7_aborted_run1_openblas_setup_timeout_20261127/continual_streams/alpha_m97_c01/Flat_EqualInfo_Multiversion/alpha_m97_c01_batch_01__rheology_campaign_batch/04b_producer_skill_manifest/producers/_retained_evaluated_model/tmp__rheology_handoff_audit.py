import pickle, runpy
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
m = runpy.run_path(str(root/'alpha_m97_c01_batch_01_solver.py'))
with (root/'public_alpha_m97_c01_batch_01.pkl').open('rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['gt']
out = m['solve_alpha_m97_c01_batch_01'](d)
(past, hold, x), n = m['_prepare'](d)
dt, t = d['past_dt'], d['eval_times']
tau, hw, rw = (out[k] for k in ('estimated_taus','hereditary_weights','regime_mode_weights'))
tail = dt.sum() - np.cumsum(dt)
z = past[n:] @ (np.exp(-tail[:,None]/tau) - np.exp(-(tail+dt)[:,None]/tau))
gap = hold[n:,None] - z
D = np.maximum(np.abs(past[n:])-d['crit_stress'],0)**2 @ dt
act = np.tanh(D)
amplitudes = gap*hw[:2]
amplitudes[:,0] += rw[0]*act*hold[n:]
drift = hw[2] + x[n:]@hw[3:7] + rw[1]*act*z[:,0]
eps = d['test']['strain_0'][:,None] + amplitudes @ (1-np.exp(-t[None,:]/tau[:,None])) + drift[:,None]*np.log1p(t)
rate = amplitudes @ (1/tau) + drift
for label, pred, actual in [('strain',eps,out['predicted_strain']),('rate',rate,out['initial_rate_pred'])]:
    for name, sel in [('zero damage',D==0),('positive damage',D>0)]:
        e = np.max(np.abs(pred[sel]-actual[sel])); print(label,name,'max independent handoff difference',e); assert e < 1e-10
s = d['test']; mask = np.isfinite(s['coupon_features_raw']).all(axis=1)
curve_e = eps-g['te_clean']; rate_e = rate-g['te_true_rate0']
print('Complete coupons only, mean signed strain/rate residual by lab and damage group:')
for lab in np.unique(s['lab_id']):
    for damage in (False, True):
        q = mask & (s['lab_id']==lab) & ((D>0)==damage)
        if q.any(): print(int(lab),damage,int(q.sum()),round(curve_e[q].mean(),6),round(rate_e[q].mean(),6))
print('Reference alignment comparison by lab, including cases with missing coupons:')
for lab in np.unique(s['lab_id']):
    q = s['lab_id']==lab
    print(int(lab),'max past/hold stress error',np.max(abs(past[n:][q]-g['te_past_true'][q])),np.max(abs(hold[n:][q]-g['te_hold_true'][q])))
