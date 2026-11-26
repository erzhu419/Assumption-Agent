import pickle,numpy as np
from scipy.optimize import least_squares
p=pickle.load(open('/holdout_task/public_suc01_T_datasets.pkl','rb'))[0];d=p['dataset'];q=d['train']; t=d['eval_times']; dt=d['past_dt']; n=len(q['lab_id']); l=q['lab_id']; s=q['stratum']; h=q['stress_hold_0']; y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; f=q['coupon_features_raw']; ref=q['ref_coupons']; u=np.empty_like(f)
for a in np.unique(l):
 ix=l==a; rr=ref[ix]; u[ix]=(f[ix]-np.nanmedian(np.nanmean(rr,axis=0)))/np.nanmedian(np.nanstd(rr,axis=0))
w=np.array([1/sum((l==a)&(s==b)) for a,b in zip(l,s)]); fill=np.nansum(w[:,None]*u,axis=0)/np.sum(w[:,None]*np.isfinite(u),axis=0); u=np.where(np.isfinite(u),u,fill); Z=np.c_[np.ones(n),u];
def gaps(tau):
 z=q['initial_memory_states'].copy()
 for j,step in enumerate(dt):
  a=np.exp(-step/tau); z=a*z+(1-a)*q['past_stress'][:,j,None]
 return h[:,None]-z

def design(tau,mode):
 g=gaps(tau); e=-np.expm1(-t[:,None]/tau); T=np.r_[0,t]; A=np.empty((n,len(T),2)); A[:,0,:]=g/tau; A[:,1:,:]=g[:,None,:]*e
 U=np.empty((n,len(T),Z.shape[1])); U[:,0,:]=Z*h[:,None]; U[:,1:,:]=T[None,1:,None]*U[:,0:1,:]
 if mode==0:return np.concatenate((A,U),axis=2)
 if mode==1:return np.concatenate((A,U,np.tile(U[:,:,[0]],(1,1,4))*Z[:,None,1:]**2),axis=2)
 if mode==2:return np.concatenate(((A[:,:,:,None]*Z[:,None,None,:]).reshape(n,len(T),-1),U),axis=2)
 if mode==3:return np.concatenate(((A[:,:,:,None]*Z[:,None,None,:]).reshape(n,len(T),-1),U,np.ones_like(U[:,:,:1])*T[None,:,None]),axis=2)
 if mode==4:return np.concatenate((A,U,np.ones_like(U)*Z[:,None,:]*T[None,:,None],A*tau*T[None,:,None]),axis=2)
Y=np.c_[r,y];
for mode in range(5):
 def fun(v):return ((design(v[:2],mode)@v[2:]-Y)*np.r_[1,np.ones(len(t))]).ravel()
 A=design(np.array([.55,3.]),mode); fit=least_squares(fun,np.r_[.55,3.,np.linalg.lstsq(A.reshape(-1,A.shape[-1]),Y.ravel(),rcond=None)[0]],loss='cauchy',f_scale=.04,bounds=(np.r_[.2,.22,np.full(A.shape[-1],-np.inf)],np.r_[8.9,9.,np.full(A.shape[-1],np.inf)]),max_nfev=180); E=fun(fit.x).reshape(Y.shape); print(mode,np.round(fit.x,4),np.round(np.quantile(np.abs(E),[.5,.8,.9],axis=0),3),flush=True)
