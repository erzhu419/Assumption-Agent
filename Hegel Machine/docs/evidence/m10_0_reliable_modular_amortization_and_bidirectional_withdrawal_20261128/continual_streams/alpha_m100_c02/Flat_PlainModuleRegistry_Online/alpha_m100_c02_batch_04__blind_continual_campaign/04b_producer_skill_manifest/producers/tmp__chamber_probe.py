import numpy as np, pickle, sys
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
v=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d=calibrate_upstream(dict(v['dataset'],chamber_verification_trials=v['campaign_diagnostics']))
a,b,c=[d[k] for k in ('train','test','chamber_verification_trials')]
dt,t=d['past_dt'],d['eval_times']
def features(a,b,c=None):
 labs=np.r_[a['lab_id'],b['lab_id']]; s=np.r_[a['stratum'],b['stratum']]; ref=np.vstack([a['ref_coupons'],b['ref_coupons']]); x=np.vstack([a['coupon_features_raw'],b['coupon_features_raw']]);
 z=c['coupon_features_raw'].copy() if c is not None else None
 for l in np.unique(labs):
  m=labs==l; p=ref[m]; u=np.median(p.mean(0)); sd=np.median(p.std(0)); x[m]=(x[m]-u)/sd
  if c is not None: n=c['lab_id']==l; z[n]=(z[n]-u)/sd
 _,i,n=np.unique(np.c_[labs,s],axis=0,return_inverse=True,return_counts=True); w=1/n[i,None]; mu=np.nansum(w*x,0)/np.sum(w*np.isfinite(x),0)
 x=np.c_[np.ones(len(x)),np.where(np.isfinite(x),x,mu)]
 return x[:len(a['lab_id'])],x[len(a['lab_id']):],None if c is None else np.c_[np.ones(len(z)),np.where(np.isfinite(z),z,mu)]
def load(s): return s['past_channel_raw'].sum(-1),s['hold_channel_raw'].sum(-1)
def gap(s,tau):
 p,h=load(s); z=np.zeros((len(h),len(tau)))
 for dd,pp in zip(dt,p.T): z=np.exp(-dd/tau)*z-np.expm1(-dd/tau)*pp[:,None]
 return h[:,None]-z

def data(s): return np.c_[s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']]
def basis(tau): return np.vstack([-np.expm1(-t[:,None]/tau),1/tau])
def design(s,x,tau): return np.concatenate([gap(s,tau)[:,None,:]*basis(tau),x[:,None,:]*np.r_[t,1][None,:,None]],axis=-1)
x,xb,xc=features(a,b,c); y=data(a); yc=data(c); unit=np.r_[np.ones(len(t)),0.5]
def fit(s=a,x=x):
 y=data(s); init=np.log([.45,3.]); A=design(s,x,np.exp(init))*unit[None,:,None]; beta=np.linalg.lstsq(A.reshape(-1,7),(y*unit).ravel(),rcond=None)[0]
 def fun(p): return ((design(s,x,np.exp(p[:2]))@p[2:]-y)*unit).ravel()
 opt=least_squares(fun,np.r_[init,beta],bounds=(np.r_[np.log([.05000001]*2),[-np.inf]*7],np.r_[np.log([24.999999]*2),[np.inf]*7]),loss='cauchy',f_scale=.012,max_nfev=150)
 order=np.argsort(opt.x[:2]); return np.exp(opt.x[:2][order]),np.r_[opt.x[2:4][order],opt.x[4:]]
def err(y,p):
 e=y-p; return np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1]))],6)
if __name__=='__main__':
 tau,th=fit(); print('material',tau,th,'train err',err(y,design(a,x,tau)@th),'diag baseline',err(yc,design(c,xc,tau)@th))
 def prof(q,full=False):
  B=np.c_[basis(np.exp(q)),np.r_[t,1]]; co=np.linalg.lstsq(B*unit[:,None],(yc*unit).T,rcond=None)[0].T; p=co@B.T
  return (co,p) if full else ((p-yc)*unit).ravel()
 for q in (tau,[.2,1.5],[.4,5.]):
  opt=least_squares(prof,np.log(q),loss='soft_l1',f_scale=.01,max_nfev=300,bounds=(np.log([.04,.04]),np.log([30,30]))); co,p=prof(opt.x,True); print('free taus',np.exp(opt.x),'err',err(yc,p),'amp first eight',np.round(co[:8],4)); print('amplitude correlations',np.round(np.corrcoef(np.column_stack([co,gap(c,tau),load(c)[1],xc[:,1:]]).T)[:3,3:],3)); print('baseline drift residual',np.round(co[:,2]-xc@th[2:],4))
 co,p=prof(np.log(tau),True); print('fixed taus, free amplitudes',err(yc,p)); print('fixed amplitude difference',np.round(co-np.c_[gap(c,tau)*th[:2],xc@th[2:]],3)); print('diag worst row errors',np.round(np.linalg.norm(yc-p,axis=1),4))
