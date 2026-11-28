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
for name,subs,base in candidates:
 p,b=fit(subs,~missing,base); print(name,'coeff',np.round(b,5)); scores=[]
 for lab in np.unique(dg['lab_id']):
  val=dg['lab_id']==lab; err=fit(subs,~missing&~val,base)[0][val]-Y[val]; scores.append([np.sqrt(np.mean(err[:,:-1]**2)),np.sqrt(np.mean(err[:,-1]**2))]);
 print('complete fit',np.sqrt(np.mean((p[~missing]-Y[~missing])**2,axis=0)).round(4),'LOLO',np.round(scores,4));
 if name=='affine cross-branch':print('mean errors by coupon',np.round(np.c_[np.mean(p[:,:-1]-y,axis=1),p[:,-1]-r,missing],4))
