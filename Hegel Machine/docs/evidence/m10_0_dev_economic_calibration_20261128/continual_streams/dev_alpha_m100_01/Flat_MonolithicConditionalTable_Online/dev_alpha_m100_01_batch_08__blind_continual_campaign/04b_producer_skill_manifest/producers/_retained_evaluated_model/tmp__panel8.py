import sys,pickle,importlib.util,numpy as np
from scipy.optimize import least_squares
root='/tmp/hegel_continual_archive/'
def load(i):
 s=importlib.util.spec_from_file_location('solver',root+f'dev_alpha_m100_01_batch_{i:02d}_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load(7); import cubic8 as cb; m._bridge=cb._bridge; m._loads=cb._loads; item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; tr,te,d=ds['train'],ds['test'],item['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; n=len(tr['lab_id']); x=m._features(tr,te); xd=m._features(tr,d)[n:]; b=m._bridge(ds['calibration_load_cells']);  p,h=m._loads(tr,b)
def obs(q):return q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']
c,r=obs(tr); T,w=m._fit(p,h,x[:n],dt,t,c,r); print('TAUS',T,'THETA',w,'bridge',b[0],b[1].round(5));
def stats(label,curve,rate,c,r):
 e=curve-c; f=rate-r; print(label,'rmse',np.sqrt(np.mean(e**2)),np.sqrt(np.mean(f**2)),'median',np.median(abs(e)),np.median(abs(f)),'curve by time',np.sqrt(np.mean(e**2,axis=0)).round(4))
A,B=m._design(T,p,h,x[:n],dt,t);stats('train',A@w,B@w,c,r)
p,h=m._loads(d,b); c,r=obs(d)
K=np.vstack((np.column_stack((-np.expm1(-t[:,None]/T),t)),np.r_[1/T,1.]))
y=np.column_stack((c,r))-(h[:,None]*w[:2])@K[:,:2].T
u,s,v=np.linalg.svd(y,full_matrices=False); k=v[0]*np.sign(v[0,-1]); k=k/k[-1]; coeff=np.linalg.lstsq(K,k,rcond=None)[0]; print('rank spectrum',s[:5], 'kernel relative to rate',coeff,'rank one RMSE',np.sqrt(np.mean((y-(y@k/(k@k))[:,None]*k)**2)))
from itertools import combinations
allq={'train':tr,'test':te,'diag':d}
raw=d['coupon_features_raw']; labs=d['lab_id']; complete=np.isfinite(raw).all(1)
for count in [1,2,3]:
 for keys in combinations(allq,count):
  q=[allq[z] for z in keys]; xl=np.concatenate([z['lab_id'] for z in q]); xr=np.concatenate([z['ref_coupons'] for z in q]); xf=np.concatenate([z['coupon_features_raw'] for z in q]); xs=np.concatenate([z['stratum'] for z in q]); xd0=raw.copy()
  for lab in np.unique(labs):
   a=xr[xl==lab]; mu=np.median(a.mean(0)); sd=np.median(a.std(0)); xd0[labs==lab]=(xd0[labs==lab]-mu)/sd; xf[xl==lab]=(xf[xl==lab]-mu)/sd
  _,idx,ct=np.unique(np.column_stack((xl,xs)),axis=0,return_inverse=True,return_counts=True); weights=1/ct[idx,None]; imp=np.nansum(xf*weights,axis=0)/(np.isfinite(xf)*weights).sum(0); xd0=np.where(np.isfinite(xd0),xd0,imp); g=w[2]+xd0@w[3:]; a=g[:,None,None]*K
  for mask in [complete,np.ones(len(g),bool)]:
   out=np.full_like(y,np.nan); qs=[]
   for lab in [*np.unique(labs),None]:
    fit=mask if lab is None else mask & (labs!=lab)
    p=least_squares(lambda p:(a@p-y)[fit].ravel(),[.14,2.8,.2],loss='soft_l1',f_scale=.004).x; qs.append(p)
    if lab is not None:out[labs==lab]=(a@p-y)[labs==lab]
   print(keys,'all' if mask.all() else 'complete',qs[-1].round(5),'worst lab RMSE',max(np.sqrt(np.mean(out[mask & (labs==j),:-1]**2)) for j in np.unique(labs)),max(np.sqrt(np.mean(out[mask & (labs==j),-1]**2)) for j in np.unique(labs)),'latent errors',((y@k/(k@k))/(g+1e-20))[mask].round(3) if keys==('diag',) else '')
