import importlib.util as iu, pickle
import numpy as np
from scipy.optimize import least_squares
s=iu.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_w01_batch_06_solver.py'); p=iu.module_from_spec(s); s.loader.exec_module(p)
c=pickle.load(open('/holdout_task/public_alpha_w01_batch_07.pkl','rb'))[0]; d=c['dataset']; tr,te=d['train'],d['test']; dt,t,sy=[d[k] for k in ['past_dt','eval_times','yield_stress']]; x,_,complete,_=p._align(tr,te); lab=tr['lab_id']; maps=p._calibrate(d['calibration_load_cells']); print('public gt schema',[(k,np.shape(v)) for k,v in c['gt'].items()] if isinstance(c['gt'],dict) else type(c['gt']),flush=True)
base_tau,base_th=p._fit(tr,x,complete,maps,dt,t,sy)
def arrays(tau,mp,mask):
 h,v=p._stress(tr,mp); a,b=p._design(tau,h,v,x,dt,t,*p._plastic(h,v,dt,sy)); return a[mask], b[mask]
yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']
a,b=arrays(base_tau,maps,np.ones(len(lab),bool)); ec=a@base_th-yc; er=b@base_th-yr
print('prior taus',base_tau,'weights',base_th,'history duration',sum(dt),'dt',dt.round(3),flush=True)
for l in range(4):
 use=lab==l; print('lab',l,'curve medabs by horizon',np.median(np.abs(ec[use]),axis=0).round(5),'rate abs quantiles',np.quantile(np.abs(er[use]),[.1,.5,.9,1]).round(5),'curve abs quantiles',np.quantile(np.abs(ec[use]),[.1,.5,.9,1]).round(5),flush=True)
rng=np.random.default_rng(7); val=np.zeros(len(lab),bool)
for l in range(4):
 ids=np.flatnonzero(lab==l); val[rng.permutation(ids)[:len(ids)//3]]=True
fit=~val&complete
for use_rate in [True,False]:
 for plastic in [False,True]:
  yc0=yc[fit]; yr0=yr[fit]; y=np.r_[yc0.ravel(),yr0] if use_rate else yc0.ravel()
  sc=np.r_[np.tile([max(.0003,p._mad(ec[fit,j])) for j in range(len(t))],fit.sum()), np.full(fit.sum(),max(.0003,p._mad(er[fit&(lab<2)])))] if use_rate else np.tile([max(.0003,p._mad(ec[fit,j])) for j in range(len(t))],fit.sum()); w=1/sc
  def profile(q,return_fit=False):
   tau=np.exp(q[:2]); mp=maps.copy(); mp[2:]=q[2:].reshape(2,2); a,b=arrays(tau,mp,fit); A=np.vstack([a.reshape(-1,9),b]) if use_rate else a.reshape(-1,9); A=A if plastic else A[:,:7]; A=A*w[:,None]; co=np.linalg.lstsq(A,y*w,rcond=1e-10)[0]; e=A@co-y*w
   if return_fit: return tau,mp,np.pad(co,(0,9-len(co))),e/w
   return e
  starts=[maps[2:].ravel(),[.604,4.569,1.246,-4.513],[1,0,1,0],[1.126,3.211,1.91,-3.25]] if plastic else [maps[2:].ravel()]
  for start in starts:
   q=np.r_[np.log(base_tau),start]; bounds=(np.r_[np.log([.10,1.05]),.15,-6,.15,-6],np.r_[np.log([1.049,24.9]),4.,6,4.,6]); q=np.clip(q,*bounds)
   for _ in range(5):
    q=least_squares(profile,q,bounds=bounds,max_nfev=55).x; tau,mp,th,e=profile(q,True); w[:]=1/sc/np.sqrt(1+(e/(2.5*sc))**2)
   a,b=arrays(tau,mp,val); e1=a@th-yc[val]; e2=b@th-yr[val]; lv=lab[val]; print('joint',use_rate,plastic,'start',np.round(start,2),'taus',tau.round(4),'maps',mp[2:].round(4),'weights',th.round(6),'val lab med curve/rate',[[round(np.median(np.abs(z[lv==l])),6) for l in range(4)] for z in (e1,e2)],flush=True)
   w[:]=1/sc
