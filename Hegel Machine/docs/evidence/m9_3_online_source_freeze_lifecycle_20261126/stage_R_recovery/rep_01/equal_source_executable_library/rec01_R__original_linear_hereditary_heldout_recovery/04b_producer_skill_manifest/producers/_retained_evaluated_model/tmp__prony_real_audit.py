import importlib.util, pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'hereditary_prony_recovery_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r = pickle.load(open(root+'public_rec01_R_datasets.pkl','rb'))[0]
d, g = r['dataset'], r['gt']; tr, te = d['train'], d['test']; t, dt = d['eval_times'], d['past_dt']
o = m.solve_hereditary_prony_recovery(d); tau = o['estimated_taus']; w = o['hereditary_weights']
rmse = lambda e: float(np.sqrt(np.mean(e*e)))
for s in (tr,te):
    q = np.zeros((len(s['lab_id']),2))
    for j,h in enumerate(dt):
        q += (s['past_stress'][:,j,None]-q) * -np.expm1(-h/tau)
    gap = s['stress_hold_0'][:,None]-q
    curve, rate = m._basis(s,dt,t,tau)
    np.testing.assert_allclose(curve,gap[:,None,:]*-np.expm1(-t[:,None]/tau),atol=2e-14)
    np.testing.assert_allclose(rate,gap/tau,atol=2e-14)
curve, rate = m._basis(te,dt,t,tau)
np.testing.assert_allclose(o['predicted_strain']-te['strain_0'][:,None]-(o['initial_rate_pred']-rate@w[:2])[:,None]*t,curve@w[:2],atol=2e-14)
print('Independent variable-step ODE and shared curve/onset consistency: pass')
print('tau max relative error',max(abs(tau/g['taus']-1)),'weight max absolute error',max(abs(w-g['theta'])))
e = o['predicted_strain']-g['te_clean']; er = o['initial_rate_pred']-g['te_true_rate0']; valid = np.isfinite(te['coupon_features_raw']).all(axis=1)
for label,rows in [('complete',valid),('incomplete',~valid)]:
    print(label,rows.sum(),'strain',rmse(e[rows]),'last horizon',rmse(e[rows,-1]),'rate',rmse(er[rows]))
for lab in np.unique(te['lab_id']):
    rows = te['lab_id']==lab
    print('lab',lab,'curve/rate RMSE',rmse(e[rows]),rmse(er[rows]),'mean rate residual',round(er[rows].mean(),6))

# Nested residual probe: extract calibration and fit the solver and each optional
# correction using outer-training rows only. Score clean public held-out rows.
def take(s, rows):
    return {k:v[rows] for k,v in s.items()}
def features(s, reference):
    raw = s['coupon_features_raw']; x = np.empty_like(raw)
    for lab in np.unique(s['lab_id']):
        p = reference['ref_coupons'][reference['lab_id']==lab]; rows=s['lab_id']==lab
        x[rows]=(raw[rows]-np.nanmedian(np.nanmean(p,axis=0)))/np.nanmedian(np.nanstd(p,axis=0))
    a,b,c,f=x.T; pairs=[(a,b),(a,c),(a,f),(b,c),(b,f),(c,f)]
    quad=np.column_stack([u*v for u,v in pairs]+[u*u for u in x.T]); cubic=np.c_[a*b*c,a*b*f,a*c*f,b*c*f]
    ratios=np.column_stack([u/(1+v*v) for p in pairs for u,v in (p,p[::-1])]); diffs=np.column_stack([z for u,v in pairs for z in (u+v,u-v)])
    return np.c_[quad,cubic,ratios,diffs,(s['stratum']==1),(s['stratum']==2)]
for fold in range(4):
    rows = np.arange(len(tr['lab_id']))%4!=fold; target = valid & (np.arange(len(valid))%4==fold)
    a=take(tr,rows); b=take(te,target); fit=m.solve_hereditary_prony_recovery({**d,'train':a,'test':b})
    z=fit['estimated_taus']; ba,bb,_=m._coupon_design(a,b); branch,affine=m._fit(a,dt,t,ba)[1][0:2],m._fit(a,dt,t,ba)[1][2:]
    ai=ba.any(axis=1); _, ar=m._basis(a,dt,t,z); residual=a['instant_rate_obs']-ar@branch-ba@affine
    x,y=features(a,a)[ai],features(b,a)
    # Orthogonalize pairs and nonlinear features against the fitted lab-affine basis.
    proj=np.linalg.lstsq(ba[ai],x,rcond=None)[0]; x-=ba[ai]@proj; y-=bb@proj
    base=fit['initial_rate_pred']-g['te_true_rate0'][target]
    for label,cols in [('stratum',slice(-2,None)),('nonlinear',slice(None,-2))]:
        u,v=x[:,cols],y[:,cols]; coef=np.linalg.solve(u.T@u+10*np.eye(u.shape[1]),u.T@residual[ai]); adj=v@coef
        print('fold',fold,label,'held-out rate RMSE base/corrected',round(rmse(base),6),round(rmse(base+adj),6))
