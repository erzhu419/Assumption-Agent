import pickle,runpy,numpy as np
from scipy.optimize import least_squares
p='/tmp/hegel_continual_archive/alpha_m99_c04_s00_solver.py'; a=runpy.run_path(p)
batch=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('batch length',len(batch)); i=batch[0]; ds=i['dataset']; tr,te,d=ds['train'],ds['test'],i['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; z=np.load('/tmp/chamber_probe.npz'); tau,b,past,hold,y,sc=[z[k] for k in ['tau','b','past','hold','y','sc']]; labs=d['lab_id']; n,k=y.shape
keys=['lab_id','stratum','coupon_features_raw','ref_coupons']
def stack(*groups): return {k:np.concatenate([g[k] for g in groups]) for k in keys}
def build(ref, imp):
 def align(g):
  x=g['coupon_features_raw'].copy()
  for l in np.unique(g['lab_id']):
   r=ref['ref_coupons'][ref['lab_id']==l]; s=g['lab_id']==l
   x[s]=(x[s]-np.median(r.mean(0)))/np.median(r.std(0))
  return x
 x=align(imp); _,j,c=np.unique(np.c_[imp['lab_id'],imp['stratum']],axis=0,return_inverse=True,return_counts=True); w=(1/c[j])[:,None]; m=np.isfinite(x); mean=(np.where(m,x,0)*w).sum(0)/(m*w).sum(0); xd=align(d); return np.where(np.isfinite(xd),xd,mean)
G={'train':tr,'test':te,'diag':d,'train_test':stack(tr,te),'train_diag':stack(tr,d),'test_diag':stack(te,d),'all':stack(tr,te,d)}
age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; mem=lambda q: past@(-np.expm1(-dt[:,None]/q)*np.exp(-age[:,None]/q)); h=mem(tau); w=1/np.r_[np.repeat(.0025,k-1),.0005]; miss=np.any(np.isnan(d['coupon_features_raw']),axis=1)
def fit(x,idx,kind):
 dr=b[2]+x@b[3:]; O=np.ones(n); H=np.c_[O,hold]; C=np.c_[H,h];
 def design(q):
  B=np.vstack([np.c_[-np.expm1(-t[:,None]/q),t],np.r_[1/q,1]])
  if kind==0: F=[np.c_[H,dr],np.c_[C,dr],np.c_[O,dr]]
  elif kind==1: F=[np.c_[H,dr],np.c_[C,dr,hold*dr,h*dr[:,None]],np.c_[O,dr]]
  elif kind==2: F=[np.c_[C,x],np.c_[C,x],np.c_[O,x]]
  else: F=[np.c_[H,dr],np.c_[C,dr,dr**2],np.c_[O,dr]]
  return np.concatenate([D[:,None,:]*B[None,:,j,None] for j,D in enumerate(F)],axis=-1)
 def prof(p,ret=False):
  A=design(np.exp(p)); W=(A[idx]*w[None,:,None]).reshape(len(idx)*k,-1); Y=(y[idx]*w).ravel(); c=np.linalg.lstsq(W,Y,rcond=None)[0]; return (A@c,c) if ret else W@c-Y
 f=least_squares(prof,np.log(tau),bounds=(np.log([.4,1.3]),np.log([.8,8.])),diff_step=1e-4); return prof(f.x,True),np.exp(f.x)
def metric(e): return np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))],5)
rows=[]
for r in G:
 for im in ('diag','train_test','all',r):
  x=build(G[r],G[im]);
  for kind in [0,1,2,3]:
   (yhat,c),q=fit(x,np.arange(n),kind); cv=np.zeros_like(y)
   for l in np.unique(labs):
    m=labs==l; cv[m]=fit(x,np.flatnonzero(~m),kind)[0][0][m]-y[m]
   rows.append((sum(metric(cv)),r,im,kind,metric(yhat-y),metric(cv),metric(cv[~miss]),metric(cv[miss]),max(metric(cv[labs==l])[0] for l in np.unique(labs)),q,c))
for row in sorted(rows,key=lambda r:r[0])[:15]: print('ref imp kind train cv complete missing worst_curve poles',row[1:-1]); print('coef',np.round(row[-1],5))
for row in sorted([r for r in rows if r[3]==0 and r[1]==r[2]],key=lambda r:r[0]): print('same-panel',row[1:-1])
print('missing coupons',np.flatnonzero(miss),'sweeps',[(k,np.shape(v)) for k,v in ds['calibration_load_cells']['single_channel_sweeps'].items()])
