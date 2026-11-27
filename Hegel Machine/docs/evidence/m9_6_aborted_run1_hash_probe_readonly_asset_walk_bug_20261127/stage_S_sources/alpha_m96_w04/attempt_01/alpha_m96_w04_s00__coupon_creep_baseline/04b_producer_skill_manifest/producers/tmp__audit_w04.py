import importlib.util, pickle
import numpy as np
root='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',root+'alpha_m96_w04_s00_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
z=pickle.load(open(root+'public_alpha_m96_w04_s00.pkl','rb'))[0]; d=z['dataset']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']
lab=tr['lab_id']; miss=np.isnan(tr['coupon_features_raw']).any(1); strata=tr['stratum']; e=np.empty_like(tr['strain_curves_obs']); er=np.empty(len(lab)); core=np.empty_like(e)
for c in np.unique(lab):
 q=lab==c; n=q.sum(); fit={k:v[~q] for k,v in tr.items()}; val={k:np.concatenate((tr[k][q],te[k])) for k in tr.keys()&te.keys()}
 o=m.solve_alpha_m96_w04_s00(dict(d,train=fit,test=val)); e[q]=tr['strain_curves_obs'][q]-o['predicted_strain'][:n]; er[q]=tr['instant_rate_obs'][q]-o['initial_rate_pred'][:n]
 a,b=m._calibration(d['calibration_load_cells']); hist,hold=m._stress(val,a,b); inv=1/o['estimated_taus']; zs=np.zeros((len(hold),2))
 for h,ds in zip(hist.T,dt):
  decay=np.exp(-ds*inv); zs=decay*zs+(1-decay)*h[:,None]
 gaps=hold[:,None]-zs; _,x=m._aligned(fit,val); drift=np.c_[np.ones(len(x)),x]; w=o['hereditary_weights']; exact_c=np.einsum('nm,km,m->nk',gaps,-np.expm1(-t[:,None]*inv),w[:2]); exact_r=(gaps*inv)@w[:2]; f,r=m._design(o['estimated_taus'],hist,hold,drift,dt,t)
 np.testing.assert_allclose(f@w,exact_c+(drift@w[2:])[:,None]*t,atol=1e-12,rtol=0); np.testing.assert_allclose(r@w,exact_r+drift@w[2:],atol=1e-12,rtol=0)
 np.testing.assert_allclose(o['predicted_strain']-val['strain_0'][:,None],exact_c+(o['initial_rate_pred']-exact_r)[:,None]*t,atol=1e-12,rtol=0)
 core[q]=e[q]-er[q,None]*t
 for j in np.unique(strata[q]):
  g=q&(strata==j)&~miss
  if g.any(): print('lab/stratum',c,j,'complete n',g.sum(),'rate bias/RMSE',np.median(er[g]),np.sqrt(np.mean(er[g]**2)),'curve median/Q85',np.quantile(abs(e[g]),[.5,.85]))
 for flag in (False,True):
  g=q&(miss==flag)
  if g.any(): print('lab/missing',c,flag,'n',g.sum(),'curve median/Q85',np.quantile(abs(e[g]),[.5,.85]),'drift-free median',np.median(abs(core[g])))
print('Independent recurrence, derivative and strain/rate handoff: passed at 1e-12')
