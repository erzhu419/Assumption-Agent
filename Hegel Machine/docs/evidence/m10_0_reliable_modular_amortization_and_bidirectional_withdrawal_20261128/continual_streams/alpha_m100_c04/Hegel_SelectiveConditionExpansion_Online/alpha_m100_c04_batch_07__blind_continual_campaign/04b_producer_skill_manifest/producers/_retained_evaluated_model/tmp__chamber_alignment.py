import sys, json, pickle, importlib.util
import numpy as np
sys.path.insert(0,'/holdout_task')
from _verified_upstream import calibrate_upstream
base='/tmp/hegel_continual_archive/'
def load(name):
 s=importlib.util.spec_from_file_location(name,base+name+'_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m=load('alpha_m100_c04_batch_04'); a=load('alpha_m100_c04_batch_01')
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; raw=it['dataset']; pay=json.load(open('/holdout_task/_upstream_param_payload.json')); ds=calibrate_upstream(raw,pay); dg=calibrate_upstream(dict(raw,test=it['campaign_diagnostics']),pay)['test']; tr,te=ds['train'],ds['test']; dt,t=ds['past_dt'],ds['eval_times']; n=len(tr['lab_id']); x=m._features(tr,te)
def obs(p): return p['strain_curves_obs']-p['strain_0'][:,None],p['instant_rate_obs']
taus,w=m._fit(m._inputs(tr,x[:n]),dt,t,*obs(tr)); print('fit',taus,w)
def metric(y,r,p,q):
 for v in (p-y,q-r): print('rmse/mae/median/max',np.round([np.sqrt(np.mean(v*v)),np.mean(abs(v)),np.median(abs(v)),np.max(abs(v))],5),end=' ')
 print()
from itertools import product
labs=np.r_[tr['lab_id'],te['lab_id']]; strata=np.r_[tr['stratum'],te['stratum']]; panel=np.vstack([tr['ref_coupons'],te['ref_coupons']]); xd=dg['coupon_features_raw'].copy(); missing=~np.isfinite(xd).all(axis=1)
for lab in np.unique(labs):
 ref=panel[labs==lab]; mask=dg['lab_id']==lab; xd[mask]=(xd[mask]-np.nanmedian(np.nanmean(ref,axis=0)))/np.nanmedian(np.nanstd(ref,axis=0))
_,cell,cnt=np.unique(np.c_[labs,strata],axis=0,return_inverse=True,return_counts=True); xd=np.where(np.isfinite(xd),xd,np.average(x,axis=0,weights=1/cnt[cell]))
past,h,c=m._inputs(dg,xd); y,r=obs(dg); A,B=m._design(taus,(past,h,c),dt,t); gap=B[:,:2]*taus; z=h[:,None]-gap; d=c@w[2:]; F=np.c_[np.ones(len(d)),h,z,d,c[:,1:],past]; names=['1','h','z0','z1','d','x0','x1','x2','x3','p0','p1','p2','p3','p4','p5']; M=np.r_[np.c_[-np.expm1(-t[:,None]/taus),t],np.r_[1/taus,1][None,:]]; Y=np.c_[y,r]; coef=Y@np.linalg.pinv(M).T
print('diagnostic',len(d),'complete',sum(~missing),'corr for coefficients and corrections'); print(names); print(np.round(np.corrcoef(np.c_[coef,coef-np.c_[gap*w[:2],d],F[:,1:]].T)[:6,6:],2))
def fit(subs,mask,base):
 J=np.concatenate([F[:,ids,None]*M[None,None,:,j] for j,ids in enumerate(subs)],axis=1).transpose(0,2,1); off=base@M.T; b=np.linalg.lstsq(J[mask].reshape(-1,J.shape[-1]),(Y-off)[mask].ravel(),rcond=None)[0];return off+J@b,b
b0=np.c_[gap*w[:2],np.zeros(len(d))]; zero=np.zeros_like(b0)
candidates=[('drift coupling',[[4],[4],[4]],b0),('affine drift coupling',[[0,4],[0,4],[0,4]],b0),('own branch gap + drift',[[1,2,4],[1,3,4],[4]],zero),('own branch gap + affine drift',[[0,1,2,4],[0,1,3,4],[0,4]],zero),('cross-branch',[[1,2,3,4],[1,2,3,4],[4]],zero),('affine cross-branch',[[0,1,2,3,4],[0,1,2,3,4],[0,4]],zero),('all linear',[[0,1,5,6,7,8,9,10,11,12,13,14]]*3,zero)]
print('Rows and reference shapes', [(len(p['lab_id']),p['ref_coupons'].shape) for p in [tr,te,dg]])
for name, xt in [('joint',x[:n]),('train-only',m._features(tr,tr)[:n])]:
 u,v=m._fit(m._inputs(tr,xt),dt,t,*obs(tr)); print('\nMATERIAL',name,u,v)
 for lab in np.unique(tr['lab_id']):
  test=tr['lab_id']==lab; mask=~test; ti=m._inputs(tr,xt); yt,rt=obs(tr); ul,vl=m._fit(tuple(f[mask] for f in ti),dt,t,yt[mask],rt[mask]); A,B=m._design(ul,tuple(f[test] for f in ti),dt,t); print(lab,ul,'MAE strain/rate',np.mean(abs(A@vl-yt[test])),np.mean(abs(B@vl-rt[test])))
 dy,dr=obs(dg); yy=np.c_[dy,dr]; rmat=np.r_[np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1][None,:]]
 for mode, xx in [('cohort only',m._features(dg,dg)[:len(d)]),('joint with train',m._features(tr,dg)[n:]),('deployment panel',xd)]:
  inp=m._inputs(dg,xx); _,bd=m._design(u,inp,dt,t); gaps=bd[:,:2]*u; base=np.c_[gaps*v[:2],np.zeros(len(d))]@rmat.T; driver=inp[2]@v[2:]; J=driver[:,None,None]*rmat
  for label, msk in [('complete',~missing),('all',np.ones(len(d),bool))]:
   def solve(mask):return np.linalg.lstsq(J[mask].reshape(-1,3),(yy-base)[mask].ravel(),rcond=None)[0]
   err=base-yy; L=dg['lab_id']; stats=[]
   for l in np.unique(L):
    val=L==l; rr=err[val]+J[val]@solve(msk&~val); stats.append([np.sqrt(np.mean(rr[:,:-1]**2)),np.sqrt(np.mean(rr[:,-1]**2)),np.max(abs(rr[:,-1]))])
   print(mode,label,solve(msk), 'worst-lab',np.max(stats,axis=0))
