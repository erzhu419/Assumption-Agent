import pickle,numpy as np
from model98 import calibrate,prep,basis
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; t=D['eval_times']; dt=D['past_dt']; w,b=calibrate(D); S=[D[k] for k in ('train','chamber_verification_trials','test')]
def data(s):
 return (s['past_channel_raw']-b)@w,(s['hold_channel_raw']-b)@w,s['strain_curves_obs']-s['strain_0'][:,None],s['instant_rate_obs']
def fit(s):
 u,h,y,r=data(s); n=len(h); z=np.r_[t,1.]; obs=np.c_[y,r]; scale=np.r_[np.full(len(t),.012),.009]; i=np.arange(n)
 def f(p,j=False):
  a,c=basis(np.exp(p[:2]),dt,t,u,h,np.empty((n,0)),'diff'); g=np.concatenate([a[:,:,:2],c[:,None,:2]],axis=1); err=(g@p[2:4]+p[4:,None]*z-obs)/scale
  if not j:return err.ravel()
  J=np.zeros((n,len(z),4+n)); J[:,:,:2]=np.stack([(np.concatenate([q[0][:,:,:2],q[1][:,None,:2]],axis=1)@p[2:4]-g@p[2:4])/1e-5 for q in [basis(np.exp(p[:2]+1e-5*v),dt,t,u,h,np.empty((n,0)),'diff') for v in np.eye(2)]],axis=-1); J[:,:,2:4]=g; J[i,:,4+i]=z; return (J/scale[:,None]).reshape(-1,4+n)
 out=[least_squares(f,np.r_[np.log(ts),[.8,.6],np.full(n,.35)],jac=lambda p:f(p,True),loss='cauchy',bounds=(np.r_[np.log([.050001]*2),[0.,0.],np.full(n,-np.inf)],np.r_[np.log([24.99999]*2),np.full(n+2,np.inf)]),max_nfev=140) for ts in ((.45,3),(.65,6),(.2,12))]; q=min(out,key=lambda q:q.cost); ix=np.argsort(q.x[:2]); return np.exp(q.x[:2])[ix],q.x[2:4][ix],q.x[4:],f(q.x).reshape(n,-1)*scale

def subset(s,mask): return {k:v[mask] for k,v in s.items()}
def combine(ss): return {k:np.concatenate([s[k] for s in ss]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
if __name__=='__main__':
 pstats=[prep(s)[1] for s in S]; trainstats=pstats[0]; trainimp=prep(S[0])[2]; allstats=prep(combine(S))[1]; otherstats=prep(combine([S[1],S[2]]))[1]; print('panel stats train chamber test pooled deployment',pstats,allstats,otherstats)
 for k,s in enumerate(S[:2]):
  taus,weights,drift,e=fit(s); m=np.isfinite(s['coupon_features_raw']).all(axis=1); print('FIT',k,taus,weights,'curve and rate quantiles',np.quantile(np.abs(e[:,:-1]),[.5,.8,.9,.95]),np.quantile(np.abs(e[:,-1]),[.5,.8,.9,.95]));
  for label,stats in [('train',trainstats),('self',pstats[k]),('pooled',allstats),('deployment',otherstats),('test',pstats[2])]:
   for impmode in ('fit','train'):
    x=prep(s,stats,None if impmode=='fit' else trainimp)[0]; a=np.c_[np.ones(len(x)),x]; sol=least_squares(lambda p:a@p-drift,np.linalg.lstsq(a[m],drift[m],rcond=None)[0],loss='cauchy',f_scale=.006); oof=np.empty(len(x));
    for lab in np.unique(s['lab_id']):
     mask=s['lab_id']!=lab; q=least_squares(lambda p:a[mask]@p-drift[mask],np.linalg.lstsq(a[mask],drift[mask],rcond=None)[0],loss='soft_l1',f_scale=.012); oof[~mask]=a[~mask]@q.x-drift[~mask]
    print(label,impmode,sol.x.round(5),'complete resid',np.quantile(abs(sol.fun[m]),[.5,.9,1]).round(4),'missing',np.quantile(abs(sol.fun[~m]),[.5,.9,1]).round(4),'worst lab mean oof',max(np.mean(abs(oof[s['lab_id']==i])) for i in np.unique(s['lab_id'])).round(4))
  if k==0:
   print('full lab drift params:');
   for lab in np.unique(s['lab_id']):
    sel=(s['lab_id']==lab)&m; raw=s['coupon_features_raw'][sel]; A=np.c_[np.ones(len(raw)),raw]; print(lab,sel.sum(),np.linalg.lstsq(A,drift[sel],rcond=None)[0].round(5))
