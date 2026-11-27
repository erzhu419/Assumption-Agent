import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
b=m._bridge(d['calibration_load_cells']); t=d['eval_times']; el=np.r_[np.cumsum(d['past_dt'][::-1])[::-1],0.]
for name in ('train','chamber_verification_trials'):
 g=d[name]; A=m._prepare(g,b); p,h,v=A; lab=g['lab_id']; mis=np.isnan(g['coupon_features_raw']).any(1); y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; pilot=name!='train'; scales=np.r_[np.hypot(.004,.003*t) if pilot else np.hypot(.010,.007*t),.006 if pilot else .015]
 tags=['base','load_drift','pair_gaps','rational','cubic']+(['telemetry','centered_telemetry'] if 'aux_chamber_telemetry' in g else [])
 E={tag:np.empty_like(y) for tag in tags}
 for l in np.unique(lab):
  q=lab==l; fit={k:a[~q] for k,a in g.items()}; taus,th=m._fit(tuple(a[~q] for a in A),el,t,fit,pilot); F=m._design(A,el,t,taus); E['base'][q]=(F@th-y)[q]; z=F[:,-1,:2]*taus
  drift_t=np.r_[t,1.][None,:,None]; Xs={'load_drift':h[:,None],'pair_gaps':np.c_[h-p[:,-1],p[:,-1]-p[:,0]],'cubic':np.c_[v[:,1]**3,v[:,2]**3,v[:,1]*v[:,2]*v[:,3]]}
  if 'telemetry' in tags:
   a=g['aux_chamber_telemetry']; Xs['telemetry']=a; Xs['centered_telemetry']=a-np.array([a[lab==j].mean(0) for j in lab])
  for tag in tags[1:]:
   if tag=='rational':
    # Fit saturation inside the fold, with the linear spectrum as the nested a=0 case.
    H=np.c_[h,h]; den=lambda a: 1.+a[None,None,:]*H[:,None,:]**2
    def pred(w): return (F[:,:,:2]/den(w[7:]))@w[:2]+F[:,:,2:]@w[2:7]
    opt=least_squares(lambda w:((pred(w)[~q]-y[~q])/scales).ravel(),np.r_[th,[1e-5]*2],bounds=(np.r_[[-np.inf]*7,0,0],np.r_[[np.inf]*7,20,20]),loss='soft_l1',max_nfev=100); E[tag][q]=(pred(opt.x)-y)[q]
   else:
    G=np.concatenate((F,drift_t*Xs[tag][:,None,:]),axis=2)
    opt=least_squares(lambda w:((G[~q]@w-y[~q])/scales).ravel(),np.r_[th,np.zeros(G.shape[-1]-7)],loss='soft_l1',max_nfev=100); E[tag][q]=(G@opt.x-y)[q]
 print(name,'LOLO worst-lab curve/rate, missing-row curve/rate')
 for tag,e in E.items():
  mae=lambda a: np.array([abs(a[:,:-1]).mean(),abs(a[:,-1]).mean()]); print(tag,np.max([mae(e[lab==j]) for j in np.unique(lab)],0).round(5),mae(e[mis]).round(5))
