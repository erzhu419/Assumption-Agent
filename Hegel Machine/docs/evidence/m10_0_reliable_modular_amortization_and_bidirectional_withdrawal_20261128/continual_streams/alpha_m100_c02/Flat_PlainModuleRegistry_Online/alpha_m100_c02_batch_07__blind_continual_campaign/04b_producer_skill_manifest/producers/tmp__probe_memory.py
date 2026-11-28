import sys
sys.path.insert(0,'/holdout_task')
import downstream_solver as m
from _verified_upstream import calibrate_upstream
import pickle,numpy as np
from scipy.optimize import least_squares
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=item['dataset']; diag=item['campaign_diagnostics']; d=calibrate_upstream(raw); g=calibrate_upstream(dict(raw,test=diag))['test']; tr,te=d['train'],d['test']; t=d['eval_times']; dt=d['past_dt']; x,xt=m._pooled_features(tr,te); tau,th=m._fit_intrinsic(tr,x,dt,t); print('material',tau,th)
ll=np.r_[tr['lab_id'],te['lab_id']]; rr=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xx=m._standardize(np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]),ll,rr); mu=np.nanmean(xx,axis=0); mu=np.array([xt[np.isnan(te['coupon_features_raw'][:,j]),j+1][0] if np.isnan(te['coupon_features_raw'][:,j]).any() else (x[np.isnan(tr['coupon_features_raw'][:,j]),j+1][0] if np.isnan(tr['coupon_features_raw'][:,j]).any() else mu[j]) for j in range(4)])
gx=np.array(g['coupon_features_raw'],copy=True)
for lab in np.unique(ll):
 p=rr[ll==lab]; gx[g['lab_id']==lab]=(gx[g['lab_id']==lab]-np.median(p.mean(0)))/np.median(p.std(0))
gx=m._with_drift_intercept(gx,mu); print('diag missing',np.argwhere(np.isnan(g['coupon_features_raw'])))
past,hold=m._loads(g); drift=gx@th[2:]; y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; w=np.r_[np.ones(len(t)),.5]; old=m._predict_chamber(g,gx,tau,th,t); print('old median/max',np.median(abs(old-y),axis=0),np.max(abs(old-y),axis=0));
def report(name,a,b):
 e=a-b; print(name,'strain med/rms/max',np.round([np.median(abs(e[:,:-1])),np.mean(e[:,:-1]**2)**.5,np.max(abs(e[:,:-1]))],5),'rate',np.round([np.median(abs(e[:,-1])),np.mean(e[:,-1]**2)**.5,np.max(abs(e[:,-1]))],5))
def robust(a,b):
 return least_squares(lambda p:(a@p-b)/.012,np.linalg.lstsq(a,b,rcond=None)[0],jac=lambda p:a/.012,loss='soft_l1',max_nfev=120).x
bas=np.c_[m._basis(tau,t),np.r_[t,1.]]
ind=np.array([robust(bas*w[:,None],row*w) for row in y]); report('individual',ind@bas.T,y); print('amp sample',np.round(np.c_[hold,drift,ind][:12],5))
loads=np.c_[hold,past]; decay=np.exp(-dt[:,None]/tau); z= (1-decay)*np.vstack([np.prod(decay[i+1:],axis=0) for i in range(len(dt))]); mem=past@z
base=(hold[:,None]-mem)*th[:2] @ bas[:,:2].T
A=drift[:,None,None]*bas[None,:,:]; p=robust((A*w[None,:,None]).reshape(-1,3),((y-base)*w).ravel()); pred=base+A@p
report('intrinsic history + drift',pred,y); print('drift coefficients',p)
print('rows lab missing hold drift gapfast gapslow residual strain rms/rate, amplitude residual'); print(np.round(np.c_[g['lab_id'],np.isnan(g['coupon_features_raw']).any(axis=1),hold,drift,hold[:,None]-mem,np.sqrt(np.mean((pred-y)[:,:-1]**2,axis=1)),(y-pred)[:,-1],ind-(np.c_[(hold[:,None]-mem)*th[:2],np.zeros(len(y))]+drift[:,None]*p)],4)); print('dt',dt)
def hist(ts):
 ts=np.atleast_1d(ts); a=np.exp(-dt[:,None]/ts); return past @ ((1-a)*np.vstack([np.prod(a[i+1:],axis=0) for i in range(len(dt))]))
def models(name,q):
 g0=hold[:,None]-mem; r=drift[:,None]; cross=mem[:,::-1]; h=hold[:,None]; delta=np.diff(past,axis=1,prepend=past[:,:1]);
 if name=='exact_gap': return base,(r,r,r)
 if name=='mix_gaps': return base,(np.c_[r,g0],np.c_[r,g0],r)
 if name=='mix_memory': return base,(np.c_[r,mem],np.c_[r,mem],r)
 if name=='memory_and_last': return base,(np.c_[r,mem[:,0],past[:,-1]],np.c_[r,mem[:,1],past[:,-1]],r)
 if name=='interactions': return base,(np.c_[r,mem[:,0]*r[:,0],g0[:,0]*r[:,0],r[:,0]**2],np.c_[r,mem[:,1]*r[:,0],g0[:,1]*r[:,0],r[:,0]**2],r)
 if name=='new_memory': return (hold[:,None]-hist(np.exp(q)))*th[:2] @ bas[:,:2].T,(r,r,r)
 if name=='new_memory_free':
  mm=hist(np.exp(q)); return np.zeros_like(base),(np.c_[r,h,mm[:,0]],np.c_[r,h,mm[:,1]],r)
 if name=='late_history':
  a=np.exp(-dt[:,None]/tau); au=1-a; zz=au*np.vstack([np.prod(a[i+1:],axis=0) for i in range(len(dt))]); mm=past[:,int(q[0]):]@zz[int(q[0]):]; return (hold[:,None]-mm)*th[:2] @ bas[:,:2].T,(r,r,r)
 raise ValueError(name)
def fit(name,rows):
 def build(q):
  b,f=models(name,q); return b,np.concatenate([f0[:,None,:]*bas[None,:,k,None] for k,f0 in enumerate(f)],axis=2)
 def fit_given(q):
  b,Z=build(q); n=Z.shape[-1]; U=(Z[rows]*w[None,:,None]).reshape(-1,n); yy=((y-b)[rows]*w).ravel(); return robust(U,yy)
 if name.startswith('new_memory'):
  def ff(q):
   b,Z=build(q); return (((b+Z@fit_given(q)-y)[rows])*w/.008).ravel()
  obj=least_squares(ff,np.log(tau),bounds=(np.log(.06),np.log(24.)),loss='soft_l1',max_nfev=110); q=obj.x
 elif name=='late_history':
  def cost(j):
   b,Z=build([j]); u=(b+Z@fit_given([j])-y)[rows]*w/.012; return np.sum(np.hypot(1,u)-1)
  q=[min(range(len(dt)),key=cost)]
 else: q=[]
 b,Z=build(q); p=fit_given(q); return b+Z@p,p,q
for name in ['exact_gap','mix_gaps','mix_memory','memory_and_last','interactions','new_memory','new_memory_free','late_history']:
 a,p,q=fit(name,np.ones(len(y),bool)); cv=np.zeros_like(y); groups=g['lab_id']; worst=[]
 for lab in np.unique(groups):
  s=groups==lab; cv[s]=fit(name,~s)[0][s]; e=cv[s]-y[s]; worst.append([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))])
 print(name,np.round(p,5),np.round(q,5),'worst lab rmse',np.max(worst,axis=0)); report('full',a,y); report('LOLO',cv,y)
