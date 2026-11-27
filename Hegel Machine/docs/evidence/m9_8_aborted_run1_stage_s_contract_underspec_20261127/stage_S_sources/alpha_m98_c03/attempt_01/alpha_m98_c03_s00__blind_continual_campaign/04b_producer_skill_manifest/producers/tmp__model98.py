import numpy as np
from scipy.optimize import least_squares

def calibrate(d):
 c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; b=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]; return np.linalg.solve(b[:4],np.ones(4)),b[4]

def prep(s, stats=None, imp=None):
 lab=s['lab_id']; stratum=s['stratum']; ref=s['ref_coupons']; raw=s['coupon_features_raw'];
 if stats is None:
  stats={i:(np.median(ref[lab==i].mean(0)),np.median(ref[lab==i].std(0))) for i in np.unique(lab)}
 x=raw.copy();
 for i in np.unique(lab):
  z=ref[lab==i]; m,v=stats.get(i,(np.median(z.mean(0)),np.median(z.std(0)))); x[lab==i]=(x[lab==i]-m)/max(v,1e-8)
 if imp is None:
  _,iv,cnt=np.unique(np.c_[lab,stratum],axis=0,return_inverse=True,return_counts=True); w=1/cnt[iv]; imp=np.nansum(x*w[:,None],axis=0)/np.sum(np.isfinite(x)*w[:,None],axis=0)
 return np.where(np.isfinite(x),x,imp),stats,imp

def basis(tau,dt,t,u,h,x,kind):
 dt=np.asarray(dt); tail=np.cumsum(dt[::-1])[::-1]-dt
 g=-np.expm1(-dt[:,None]/tau)*np.exp(-tail[:,None]/tau)
 m=u@g
 if kind.startswith('free'): m=np.concatenate([np.broadcast_to(h[:,None],m.shape),m],axis=1); tau=np.tile(tau,2)
 elif kind.startswith('hold'): m=h[:,None]+m*0
 elif kind.startswith('plus'): m=h[:,None]+m
 elif kind.startswith('reverse'): m=h[:,None]-u[:,::-1]@g
 elif kind.startswith('steps'): m=h[:,None]-u@np.exp(-tail[:,None]/tau)
 elif kind.startswith('hist'): pass
 else: m=h[:,None]-m
 f=m[:,None,:]*(-np.expm1(-t[:,None]/tau)); r=m/tau
 if '_rate' in kind: f=f*tau; r=r*tau
 z=np.c_[np.ones(len(x)),x];
 if '_stress' in kind: z=z*h[:,None]
 return np.concatenate([f,z[:,None,:]*t[None,:,None]],axis=-1),np.c_[r,z]

def fit(d,s,kind,tx=None):
 t=d['eval_times']; dt=d['past_dt']; w,b=calibrate(d); x=prep(s)[0] if tx is None else tx; u=(s['past_channel_raw']-b)@w; h=(s['hold_channel_raw']-b)@w; y=s['strain_curves_obs']-s['strain_0'][:,None]; r=s['instant_rate_obs'];
 def fun(p):
  a,c=basis(np.exp(p[:2]),dt,t,u,h,x,kind); return np.r_[(a@p[2:]-y).ravel(),(c@p[2:]-r)*2]
 out=[]
 for ts in ((.2,2),(.45,3),(.8,5),(.12,12)):
  q=least_squares(fun,np.r_[np.log(ts),np.ones(4 if kind.startswith('free') else 2),np.zeros(5)],bounds=(np.r_[np.log([.05001]*2),[-np.inf]*(9 if kind.startswith('free') else 7)],np.r_[np.log([24.999]*2),[np.inf]*(9 if kind.startswith('free') else 7)]),loss='cauchy',f_scale=.03,max_nfev=250); out.append(q)
 q=min(out,key=lambda q:q.cost); e=fun(q.x); return q.x,q.cost,np.quantile(np.abs(e[:y.size]),[.5,.9,.95,.99]),np.quantile(np.abs(e[y.size:])/2,[.5,.9,.99])
if __name__=='__main__':
 import pickle
 d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; z,stats,imp=prep(d['train']); print('stats',stats,'impute',imp)
 for k in ['diff','diff_stress','plus','hold','hist','reverse','steps','free','free_stress']:
  for s in ['train','chamber_verification_trials']:
   r=fit(d,d[s],k,z if s=='train' else prep(d[s],stats,imp)[0]); print(k,s,np.round(np.r_[np.exp(r[0][:2]),r[0][2:]],5),round(r[1],5),r[2].round(5),r[3].round(5),flush=True)
