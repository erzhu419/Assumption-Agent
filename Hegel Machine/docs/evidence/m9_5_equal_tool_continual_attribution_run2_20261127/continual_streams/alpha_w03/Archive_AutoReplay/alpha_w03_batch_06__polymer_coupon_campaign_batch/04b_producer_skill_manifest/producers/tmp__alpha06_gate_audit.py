import importlib.util,pickle,numpy as np
from pathlib import Path
root=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('solver',root/'alpha_w03_batch_06_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=pickle.loads((root/'public_alpha_w03_batch_06.pkl').read_bytes())[0]['dataset']; tr,te=d['train'],d['test']; n=len(tr['lab_id'])
x,h,p=m._prep(d); dt,t,sy=[d[k] for k in ('past_dt','eval_times','yield_stress')]; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
# Independently construct the Prony history by sequential state updates.
out=m.solve_alpha_w03_batch_06(d); tau=out['estimated_taus']; C,R=m._design(tau,x,h,p,dt,t,sy)
state=np.zeros((len(h),2)); work=np.zeros(len(h))
for step,delta in enumerate(dt):
 decay=np.exp(-delta/tau); state=decay*state+(1-decay)*p[:,step,None]
 work+=delta*np.maximum(0,abs(p[:,step])-sy)**2
shape=1-np.exp(-t[:,None]/tau)
over=np.maximum(0,abs(h)-sy)**2*np.sign(h)
assert np.max(abs(C[:,:,:2]-(h[:,None]-state)[:,None,:]*shape))<1e-12
assert np.max(abs(C[:,:,2:7]-x[:,None,:]*t[None,:,None]))<1e-12
assert np.max(abs(C[:,:,7]-over[:,None]*t))<1e-12
assert np.max(abs(C[:,:,8]-(work*h)[:,None]*shape[:,0]))<1e-12
print('independent state recurrence and all nine columns agree')
# A stress-induced shape defect must survive removing the coupon's linear drift.
# Fit elsewhere, then compare within-lab shape residuals against nonlinear stress combinations.
from scipy.stats import spearmanr
lab=tr['lab_id']; complete=np.all(np.isfinite(tr['coupon_features_raw']),axis=1)
for half in ((0,1),(2,3)):
 train=np.flatnonzero(complete & ~np.isin(lab,half)); val=np.flatnonzero(complete & np.isin(lab,half))
 ta,co=m._fit(x[train],h[train],p[train],dt,t,sy,y[train],r[train]); B,_=m._design(ta,x[val],h[val],p[val],dt,t,sy)
 q=y[val]-B@co; drift=np.median(q/t,axis=1); q-=drift[:,None]*t
 early=np.median(q[:,:3],axis=1); late=np.median(q[:,-3:],axis=1)
 for a in half:
  j=lab[val]==a; v=val[j]; print('held wing, lab',a,'complete',len(v),'early/late median',np.round([np.median(early[j]),np.median(late[j])],6),'onset/late shape stress rank correlations',np.round([[spearmanr(z[j],u).statistic for u in (h[v],h[v]-state[v,0],h[v]/(1+work[v]**2))] for z in (early,late)],3))
