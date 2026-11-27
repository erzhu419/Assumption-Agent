import pickle, numpy as np
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; dt=D['past_dt']; t=np.r_[0.,D['eval_times']]; cal=D['calibration_load_cells']; B=np.linalg.lstsq(np.c_[np.ones(len(cal['bridge_raw'])),cal['bridge_raw']],cal['true_channel_loads'],rcond=None)[0]
def prep(a):
 g=a['lab_id']; s=a['stratum']; x=a['coupon_features_raw'].copy(); ref=a['ref_coupons']; _,inv,c=np.unique(np.c_[g,s],axis=0,return_counts=True,return_inverse=True); w=1/c[inv]; w/=w.mean()
 for lab in np.unique(g):
  j=g==lab; x[j]=(x[j]-np.median(ref[j].mean(0)))/np.median(ref[j].std(0))
 m=np.isfinite(x); fill=(np.where(m,x,0)*w[:,None]).sum(0)/(m*w[:,None]).sum(0); x=np.where(m,x,fill)
 p=a['past_channel_raw']@B[1:].sum(1)+B[0].sum(); h=a['hold_channel_raw']@B[1:].sum(1)+B[0].sum()
 y=np.column_stack((a['instant_rate_obs'],a['strain_curves_obs']-a['strain_0'][:,None])); return p,h,np.c_[np.ones(len(g)),x],y,w

def design(a,tau,mode='forward',drift='plain'):
 p,h,x,y,w=a; d=dt if mode!='reverse' else dt[::-1]; p=p if mode!='reverse' else p[:,::-1]; tail=np.cumsum(d[::-1])[::-1]-d
 k=np.exp(-tail[:,None]/tau)*(-np.expm1(-d[:,None]/tau)); q=p@k
 if mode=='normalized': q/=k.sum(0)
 if mode=='none': q*=0
 if mode=='last': q=p[:,-1,None]
 if mode=='negative': q=-q
 if mode=='instant': q=h[:,None]*0 + p[:,-1,None]-q
 amp=h[:,None]-q; basis=-tau*np.expm1(-t[:,None]/tau); basis[0]=1
 if drift=='load': x=x*h[:,None]
 if drift=='intercept_load': x=x.copy(); x[:,0]*=h
 return np.concatenate((amp[:,None,:]*basis[None,:,:],x[:,None,:]*np.r_[1.,t[1:]][None,:,None]),axis=2)

def fit(a,mode,drift):
 y=a[3]; w=a[4]; sw=np.sqrt(w)[:,None]; scale=np.r_[0.025,np.full(len(t)-1,.025)]
 def fun(z):
  tau=np.sort(np.exp(z[:2])); return ((design(a,tau,mode,drift)@z[2:]-y)*sw/scale).ravel()
 out=[]
 for init in ([.23,1.6],[.7,4],[.11,8]):
  f=design(a,np.array(init),mode,drift); coef=np.linalg.lstsq((f*sw[:,:,None]).reshape(-1,7),(y*sw).ravel(),rcond=None)[0]; z=least_squares(fun,np.r_[np.log(init),coef],bounds=(np.r_[[np.log(.05001)]*2,[-np.inf]*7],np.r_[[np.log(24.999)]*2,[np.inf]*7]),loss='soft_l1',max_nfev=130); out.append(z)
 z=min(out,key=lambda z:z.cost); tau=np.sort(np.exp(z.x[:2])); err=design(a,tau,mode,drift)@z.x[2:]-y
 return z.cost,tau,z.x[2:],err

if __name__=='__main__':
 for name in ('train','chamber_verification_trials'):
  a=prep(D[name]); print(name)
  results=[]
  for mode in ('forward','reverse','normalized','none','last','negative','instant'):
   for drift in ('plain','load','intercept_load'):
    c,tau,coef,e=fit(a,mode,drift); results.append((c,mode,drift,tau.round(4),coef.round(4),round(np.median(abs(e[:,0])),4),round(np.median(abs(e[:,1:])),4)))
  for v in sorted(results,key=lambda v:v[0]): print(v)
