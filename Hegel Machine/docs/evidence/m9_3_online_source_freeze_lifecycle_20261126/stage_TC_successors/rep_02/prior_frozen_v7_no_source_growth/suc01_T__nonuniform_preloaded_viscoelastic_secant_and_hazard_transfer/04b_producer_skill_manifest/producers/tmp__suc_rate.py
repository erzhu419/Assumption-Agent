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
def prep(q, cal=None):
 l=q['lab_id']; s=q['stratum']; f=q['coupon_features_raw']; ref=q['ref_coupons']; u=np.empty_like(f)
 if cal is None:
  cal={}
  for a in np.unique(l):
   rr=ref[l==a]; cal[a]=(np.nanmedian(np.nanmean(rr,axis=0)),np.nanmedian(np.nanstd(rr,axis=0)))
 for a in np.unique(l):
  ix=l==a; mu,sigma=cal[a]; u[ix]=(f[ix]-mu)/sigma
 return u,cal

def fit_predict(tr,te,wt,lam=1):
 u,cal=prep(tr); v,_=prep(te,cal); l=tr['lab_id']; s=tr['stratum']; _,idx,counts=np.unique(np.c_[l,s],axis=0,return_inverse=True,return_counts=True); w=1/counts[idx]; w/=w.mean(); fill=np.nansum(w[:,None]*u,axis=0)/(np.isfinite(u).T@w); X=np.c_[np.ones(len(u)),np.where(np.isfinite(u),u,fill)]; V=np.c_[np.ones(len(v)),np.where(np.isfinite(v),v,fill)]; Y=np.c_[tr['instant_rate_obs'],tr['strain_curves_obs']-tr['strain_0'][:,None]]; good=np.isfinite(u).all(1)
 def basis(q,tau):
  z=q['initial_memory_states'].copy()
  for j,step in enumerate(dt):
   a=np.exp(-step/tau); z=a*z+(1-a)*q['past_stress'][:,j,None]
  g=q['stress_hold_0'][:,None]-z
  return np.concatenate(((g/tau)[:,None,:],g[:,None,:]*-np.expm1(-t[:,None]/tau)),1)
 T=np.r_[1,t]; W=(0.35+0.65*good)[:,None]**lam*np.r_[wt,np.ones(len(t))]
 def fun(b):return ((basis(tr,b[:2])@b[2:4]+(X@b[4:])[:,None]*T-Y)*W).ravel()
 A=basis(tr,np.array([.55,3.])); A=np.concatenate((A,X[:,None,:]*T[None,:,None]),2); b0=np.linalg.lstsq((A*W[:,:,None]).reshape(-1,7),(Y*W).ravel(),rcond=None)[0]
 opt=least_squares(fun,np.r_[.55,3.,b0],loss='cauchy',f_scale=.025,bounds=(np.r_[.2,.22,np.full(7,-np.inf)],np.r_[8.9,9.,np.full(7,np.inf)]),max_nfev=100); b=opt.x; B=basis(te,b[:2])@b[2:4]; D=V@b[4:]; Dr=te['instant_rate_obs']-B[:,0];
 print('fit',wt, np.round(b,5), 'train med',np.round(np.median(np.abs(fun(b).reshape(Y.shape)/W),0),4),'D vs Dr',np.round(np.quantile(np.abs(D-Dr),[.25,.5,.75,.9]),4)); return b,B,D,Dr

def hazards(te,P):
 P=np.c_[te['strain_0'],P]; T=np.r_[0,t]; crit=te['critical_strain']; reach=P>=crit[:,None]; j=reach.argmax(1); rr=np.arange(len(j)); ans=np.full(len(j),t[-1]); ans[reach[:,0]]=0.; ix=reach.any(1)&(j>0); rr=rr[ix]; k=j[ix]; ans[ix]=T[k-1]+(T[k]-T[k-1])*(crit[ix]-P[rr,k-1])/(P[rr,k]-P[rr,k-1]); return ans
te=d['test']; gt=p['gt']; good=np.isfinite(te['coupon_features_raw']).all(1); groups=[np.ones(len(good),bool),good,~good]; den=np.maximum(abs(te['stress_hold_0']),.5)
for wt in [.5,1,2,4]:
 b,B,D,Dr=fit_predict(q,te,wt)
 Dg=np.median((gt['te_clean']-te['strain_0'][:,None]-B[:,1:])/t,axis=1); print('drift errors quantiles',np.round(np.quantile(np.abs(D-Dg),[.5,.8,.95]),4),np.round(np.quantile(np.abs(Dr-Dg),[.5,.8,.95]),4))
 for mix in [0,.25,.5,.75,1]:
  m=np.where(good,mix,1); P=te['strain_0'][:,None]+B[:,1:]+((1-m)*D+m*Dr)[:,None]*t; E=np.c_[np.abs(P-gt['te_clean']).mean(1),np.abs((P[:,-1]-te['strain_0'])/den-gt['te_secant']),np.abs(hazards(te,P)-gt['te_cross'])];print(mix, np.round([E[g].mean(0) for g in groups],4).tolist())
