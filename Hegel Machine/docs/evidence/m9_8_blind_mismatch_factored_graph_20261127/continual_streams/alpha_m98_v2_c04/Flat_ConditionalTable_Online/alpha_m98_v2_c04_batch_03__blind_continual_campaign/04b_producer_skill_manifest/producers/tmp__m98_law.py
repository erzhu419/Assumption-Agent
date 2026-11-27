import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/alpha_m98_v2_c04_batch_02_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; tr,te=d['train'],d['test']; cd=i['campaign_diagnostics']; dt,t=d['past_dt'],d['eval_times']; b=m._bridge(d['calibration_load_cells']); xt=m._features(tr,te); n=len(tr['lab_id']); tau,th=m._fit(xt[:n],*m._loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('material',tau,th)
p,h=m._loads(cd,b); y=cd['strain_curves_obs']-cd['strain_0'][:,None]; r=cd['instant_rate_obs']; Y=np.c_[y,r]; print('chamber first row',Y[0].round(5)); print('chamber ranges',np.ptp(Y,axis=0).round(5),'svd',np.linalg.svd(Y,compute_uv=False).round(5)); print('diagnostic labs',np.unique(cd['lab_id'],return_counts=True),'missing',np.isnan(cd['coupon_features_raw']).sum(0))
merged={k:np.concatenate((te[k],cd[k])) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}; xd={'train+diag':m._features(tr,cd)[n:],'all':m._features(tr,merged)[len(xt):]}
labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack([tr['ref_coupons'],te['ref_coupons']]); x=cd['coupon_features_raw'].copy()
for lab in np.unique(cd['lab_id']):
 q=refs[labs==lab]; mask=cd['lab_id']==lab; x[mask]=(x[mask]-np.nanmedian(np.nanmean(q,0)))/np.nanmedian(np.nanstd(q,0))
_,ix,cnt=np.unique(np.c_[labs,np.r_[tr['stratum'],te['stratum']]],axis=0,return_inverse=True,return_counts=True); raw=np.vstack([tr['coupon_features_raw'],te['coupon_features_raw']]); w=1/cnt[ix,None]; fill=(xt*w*(~np.isnan(raw))).sum(0)/(w*(~np.isnan(raw))).sum(0); xd['train+test norm']=np.where(np.isnan(x),fill,x)
later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
def memory(u):
 return p@(-np.expm1(-dt[:,None]/np.atleast_1d(u))*np.exp(-later[:,None]/np.atleast_1d(u)))
def design(q,X,mode):
 u=np.sort(np.exp(q[:2])); v=np.exp(q[2:]); dr=th[2]+X@th[3:]; one=np.ones(len(h)); gaps=h[:,None]-memory(tau)
 T=np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1]))
 if mode=='intrinsic_gap': blocks=[np.c_[gaps[:,0],dr],np.c_[gaps[:,1],dr],dr[:,None]]
 if mode=='chamber_gap':
  g=h[:,None]-memory(u); blocks=[np.c_[g[:,0],dr],np.c_[g[:,1],dr],dr[:,None]]
 if mode=='mix_gap': blocks=[np.c_[gaps,dr]]*2+[dr[:,None]]
 if mode=='free_gap':
  g=h[:,None]-memory(v); blocks=[np.c_[g[:,0],dr],np.c_[g[:,1],dr],dr[:,None]]
 if mode=='free_memory':
  z=memory(v); blocks=[np.c_[h,z[:,0],dr],np.c_[h,z[:,1],dr],np.c_[h,dr]]
 if mode=='full_past': blocks=[np.c_[h,p,dr]]*2+[np.c_[h,p,dr]]
 if mode=='free_features':
  z=memory(v); blocks=[np.c_[h,z[:,0],one,X],np.c_[h,z[:,1],one,X],np.c_[h,one,X]]
 if mode=='intrinsic_plus_fast': blocks=[np.c_[gaps[:,0],dr],np.c_[gaps[:,1],dr],np.c_[h,dr]]
 return np.concatenate([B[:,None,:]*T[None,:,j,None] for j,B in enumerate(blocks)],axis=2)
def fit(X,mode,mask):
 def func(q):
  A=design(q,X,mode)[mask].reshape(mask.sum()*Y.shape[1],-1); b=Y[mask].ravel(); w=np.ones(len(b)); c=None
  for _ in range(3):
   c=np.linalg.lstsq(A*w[:,None],b*w,rcond=1e-9)[0]; w=(1+((A@c-b)/.007)**2)**(-.25)
  return A@c-b,c
 guesses=[[.465,1.55],[.45,4.5]] if mode not in ('free_gap','free_memory','free_features') else [[.465,1.55,.5,4.5],[.45,3.,.3,1.5],[.45,1.5,1.5,.5]]
 out=min((least_squares(lambda q:func(q)[0],np.log(g),bounds=(np.log(.05),np.log(24.9)),loss='soft_l1',f_scale=.006,max_nfev=100) for g in guesses),key=lambda a:a.cost)
 return out.x,func(out.x)[1]
def stats(e):return np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))],5)
from itertools import combinations
from scipy.optimize import minimize
labs=cd['lab_id']; drift=th[2]+xd['all']@th[3:]; X=xd['all']; g=h[:,None]-memory(tau)
B0=np.c_[drift,h,p]; B=[np.c_[g[:,0],h,p,drift],np.c_[g[:,1],h,p,drift],B0]
pen=[np.array([.005]+[.2]*7+[.01])]*2+[np.array([.01]+[.2]*7)]
ch=b if False else None
# Remove total-load information from raw channel contrasts.
rawh=cd['hold_channel_raw']; rawp=cd['past_channel_raw']; Dh=rawh-rawh.mean(1,keepdims=True); Dp=rawp-rawp.mean(2,keepdims=True)
E={ 'base':None, 'feature':X, 'missing':np.isnan(cd['coupon_features_raw'])[:,:2], 'stratum':np.eye(3)[cd['stratum']][:,:2], 'channel':np.c_[Dh[:,:3],Dp[:,-1,:3]], 'nonlinear_memory':np.c_[(h[:,None]-p)**2,p*p], 'load_drift':np.c_[(h-h.mean())*drift,g*drift[:,None],drift**2], 'interaction':np.c_[g[:,0]*g[:,1],g*g,(h-h.mean())[:,None]*g], 'feature_interaction':np.c_[X**2,*[X[:,i]*X[:,j] for i,j in combinations(range(4),2)]] }
def fit(mask,key,scale):
 ex=E[key]; BB=B; pp=pen
 if ex is not None:
  e=ex-ex[mask].mean(0); e=e/np.maximum(e[mask].std(0),.01); BB=[np.c_[a,e] for a in B]; pp=[np.r_[q,np.full(e.shape[1],scale)] for q in pen]
 reg=np.diag(np.concatenate(pp)); y=Y[mask].ravel()
 def calc(q,ret=False):
  u=np.exp(q); T=np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1])); D=np.concatenate([a[:,None,:]*T[None,:,j,None] for j,a in enumerate(BB)],2); A=D[mask].reshape(len(y),-1); c=np.linalg.lstsq(np.vstack([A,reg]),np.r_[y,np.zeros(len(reg))],rcond=None)[0]; return D@c if ret else np.r_[A@c-y,reg@c]
 q=least_squares(calc,np.log([.466,1.488]),bounds=(np.log([.15,.7]),np.log([.7,12.])),max_nfev=60).x
 return calc(q,True),np.exp(q)
from itertools import product
# Single-mode extensions preserve identifiability and avoid giving every added term three coefficients.
Z=drift-th[2]; z=h[:,None]-g
F={'none':None,'drift2':drift**2,'drift3':Z**3,'drift_sat':np.tanh(2*Z)-2*Z,'hold_drift':(h-3.6)*Z,'fastgap_drift':g[:,0]*Z,'slowgap_drift':g[:,1]*Z,'hold2':(h-3.6)**2,'fastgap2':g[:,0]**2,'slowgap2':g[:,1]**2,'mem0_sq':z[:,0]**2,'mem1_sq':z[:,1]**2,'signed_past0':np.maximum(-p[:,0],0.),'mem_abs':np.abs(p)@(-np.expm1(-dt/tau[1])*np.exp(-later/tau[1])),'mem_sq':(p*p)@(-np.expm1(-dt/tau[1])*np.exp(-later/tau[1]))}
order=np.argsort(h); folds=[labs!=k for k in np.unique(labs)]+[~np.isin(np.arange(len(h)),order[:4]),~np.isin(np.arange(len(h)),order[-4:])]; allrows=np.ones(len(h),bool)
def fit1(mask,key,mode,regscale):
 BB=B.copy(); pp=[a.copy() for a in pen]; v=F[key]; pp=[x*regscale for x in pp]
 if v is not None:
  v=(v-v[mask].mean())/max(v[mask].std(),.001); BB[mode]=np.c_[BB[mode],v]; pp[mode]=np.r_[pp[mode],.3]
 reg=np.diag(np.concatenate(pp)); y=Y[mask].ravel()
 def calc(q, ret=False):
  u=np.exp(q); T=np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1.])); D=np.concatenate([a[:,None,:]*T[None,:,j,None] for j,a in enumerate(BB)],2); A=D[mask].reshape(len(y),-1); c=np.linalg.lstsq(np.vstack([A,reg]),np.r_[y,np.zeros(len(reg))],rcond=None)[0]; return (D@c,c) if ret else np.r_[A@c-y,reg@c]
 q=least_squares(calc,np.log([.466,1.488]),bounds=(np.log([.15,.7]),np.log([.7,12.])),max_nfev=60).x
 return calc(q,True)
def score(cand,fs):
 a=np.array([stats(fit1(mask,*cand)[0][~mask]-Y[~mask]) for mask in fs]); return a[:4].max(0),a[4:].max(0)
cands=[('none',0,s) for s in [.5,1.,2.,4.,8.]]+[(k,j,1.) for k in F if k!='none' for j in (0,1,2)]
res=[]
for cand in cands:
 a,b=score(cand,folds); res.append((np.mean(a+b),cand,a,b))
for _,c,a,b in sorted(res,key=lambda x:x[0])[:9]: print(c,'worst lab',a,'worst load tail',b)
print('saved baseline',score(('none',0,1.),folds))
