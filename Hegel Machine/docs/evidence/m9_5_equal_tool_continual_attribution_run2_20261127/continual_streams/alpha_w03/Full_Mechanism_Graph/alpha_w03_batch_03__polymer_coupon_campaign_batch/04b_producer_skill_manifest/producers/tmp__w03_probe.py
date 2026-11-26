import numpy as np,pickle
from scipy.optimize import least_squares
from w03_prior import _prep
D=pickle.load(open('/holdout_task/public_alpha_w03_batch_03.pkl','rb'))[0]['dataset']; tr=D['train']; n=len(tr['lab_id']); x,h,p=(a[:n] for a in _prep(D)); u=tr['initial_memory_states']; dt=D['past_dt']; t=D['eval_times']; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; target=np.c_[y,r]; lab=tr['lab_id']; end=np.cumsum(dt); lag=end[-1]-end

def bases(tau,kind,drift):
 tau=np.asarray(tau, dtype=float)
 e=np.exp(-dt[:,None]/tau); z=p@((1-e)*np.exp(-lag[:,None]/tau)); pre=u*np.exp(-end[-1]/tau); g=h[:,None]-z; q=-np.expm1(-t[:,None]/tau); shape=np.vstack((q,1/tau)); tm=np.r_[t,1.]
 if kind=='fixed': m=(g-pre,)
 elif kind=='free': m=(g,pre)
 else: m=(g,)
 mats=[v[:,None,:]*shape for v in m]; w=np.c_[np.ones(n),x]; mats.append(w[:,None,:]*tm[None,:,None]);
 if drift=='stress': mats.append(((h[:,None])*w)[:,None,:]*tm[None,:,None]);
 return np.concatenate(mats,axis=2)

def run(kind,drift,hold,rate_weight):
 fit=lab!=hold; test=~fit; scale=np.r_[np.ones(len(t)),rate_weight];
 def ls(tau):
  a=bases(tau,kind,drift); aa=(a[fit]*scale[None,:,None]).reshape(-1,a.shape[-1]); b=(target[fit]*scale).ravel(); c=np.linalg.lstsq(aa,b,rcond=None)[0]; return a,c,aa@c-b
 def fun(v): return ls([v[0],v[0]+v[1]])[2]
 best=min([least_squares(fun,v,bounds=([.051,.02],[5.,19.]),ftol=1e-8) for v in ((.35,2),(.75,4),(1.4,6))],key=lambda r:r.cost); tau=np.cumsum(best.x); a,c,_=ls(tau); pred=a@c; res=pred[test]-target[test]; return tau,c,np.sqrt(np.mean(res**2,axis=0))
for kind in ('fixed','free','empty'):
 for drift in ('plain','stress'):
  for rate_weight in (.35,1.,4.):
   out=[run(kind,drift,l,rate_weight) for l in range(4)]; err=np.array([a[2] for a in out]); print(kind,drift,rate_weight,'tau',np.round([a[0] for a in out],3),'coef',np.round(out[0][1],3),'worst curves',np.round(np.sqrt(np.mean(err[:,:-1]**2,axis=1)).max(),5),'worst rate',round(err[:,-1].max(),5),'split horizons',np.round(err.max(axis=0),3))
