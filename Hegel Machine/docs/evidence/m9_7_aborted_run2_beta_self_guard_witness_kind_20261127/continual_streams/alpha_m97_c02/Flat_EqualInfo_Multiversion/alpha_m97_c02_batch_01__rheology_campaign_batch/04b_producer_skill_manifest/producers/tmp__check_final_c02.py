import importlib.util, pickle, numpy as np
p='/holdout_task/alpha_m97_c02_batch_01_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_alpha_m97_c02_batch_01.pkl','rb'))[0]; d=i['dataset']; g=i['gt']; tr,te=d['train'],d['test']; n=len(tr['lab_id']); cal=m._calibration(d); x=m._align(d); a=m._inputs(d,te,cal,x[n:]); out=m.solve_alpha_m97_c02_batch_01(d)
assert np.all(np.isfinite(x)) and np.all(np.diff(out['estimated_taus'])>0)
for k,v in [('estimated_taus','taus'),('hereditary_weights','theta'),('regime_mode_weights','vp_theta')]: print(k,'max abs error',np.max(np.abs(out[k]-g[v])))
curve=out['predicted_strain']-g['te_clean']; rate=out['initial_rate_pred']-g['te_true_rate0']
missing=np.isnan(te['coupon_features_raw']).any(1)
for name,sel in [('all',np.ones(len(rate),bool)),('complete',~missing),('missing',missing)]+[(f'lab {lab} damage {dam}',(te['lab_id']==lab)&((a[-2]>0)==dam)) for lab in range(4) for dam in (False,True)]:
 if np.any(sel): print(name,sel.sum(),'curve/rate RMSE',np.round([np.sqrt(np.mean(v[sel]**2)) for v in (curve,rate)],5))
# Check the exact derivative independent of measured rate noise, including both damage states.
t=np.log(out['estimated_taus']); w=np.r_[out['hereditary_weights'],out['regime_mode_weights']]; eps=1e-6
for idx in (.12,.80):
 b=m._inputs(dict(d,shear_banding_index=idx),te,cal,x[n:]); c,r,*_=m._design(t,*b[:3],np.array([0.,eps]),*b[4:]); assert np.allclose((c[:,1]@w-c[:,0]@w)/eps,r@w,rtol=3e-6,atol=3e-6)
# One transient spike in a supported sweep must leave the in-batch affine fit intact.
c={k:v.copy() for k,v in d['calibration_load_cells'].items()}; c['true_stress'][0]+=9
assert np.max(np.abs(m._calibration(dict(d,calibration_load_cells=c))-cal))<1e-4
