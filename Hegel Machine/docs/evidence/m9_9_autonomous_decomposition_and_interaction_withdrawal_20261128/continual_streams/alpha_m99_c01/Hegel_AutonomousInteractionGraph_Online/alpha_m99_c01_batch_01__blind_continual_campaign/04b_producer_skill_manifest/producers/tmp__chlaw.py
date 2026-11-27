exec(open('/tmp/chbasis.py').read().split('C,E=profile(tau)')[0])
import json
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for row in log:
 for k,v in row.items():
  if isinstance(v,dict): print('LOG',k,[(kk,np.shape(vv),str(vv)[:105]) for kk,vv in v.items()])
  else: print('LOG',k,np.shape(v),str(v)[:250])
A=m._design(tau,xd,pd,hd,dt,t)[1]; r=A[:,2:]@beta[2:]; z=hd[:,None]-A[:,:2]*tau
# Fit curve and rate responses together so an unstable split between slow amplitude and linear drift cannot drive model selection.
from scipy.linalg import qr

def assess(name,F):
 F=np.c_[np.ones(len(F)),F]; B=np.linalg.lstsq(F,Y,rcond=None)[0]; U,s,_=np.linalg.svd(F,full_matrices=False); H=(U[:,s>1e-9]**2).sum(1); E=Y-F@B; L=E/np.maximum(1-H[:,None],1e-7); C=np.linalg.lstsq(basis(tau),B.T,rcond=None)[0].T
 print(name, 'coef',np.round(C.T,4).tolist() if F.shape[1]<10 else '-', 'rms', np.round([np.std(E[:,:-1]),np.std(E[:,-1]),np.std(L[:,:-1]),np.std(L[:,-1])],5),'max',np.round(np.max(abs(L),axis=0)[[0,5,11,12]],4)); return E

def cat(*parts): return {k:np.concatenate([s[k] for s in parts],axis=0) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
_,xall=m._features(cat(tr,te),di); _,xself=m._features(di,di)
for name,xj in [('train+diag',xd),('all',xall),('diag-only',xself),('raw',np.where(np.isfinite(di['coupon_features_raw']),di['coupon_features_raw'],0))]:
 rj=beta[2]+xj@beta[3:]; print('ALIGN',name, np.round(rj-r,4)); assess(name,np.c_[hd,rj]); assess(name+'+z',np.c_[hd,z,rj]); assess(name+'+z+allfeatures',np.c_[hd,z,xj])
rj=beta[2]+xself@beta[3:];
for name,F in [('hold+drift+nonlin',np.c_[hd,rj,rj**2,rj**3,hd*rj]),('hist',np.c_[hd,pd,rj]),('bridge hold component',np.c_[di['hold_channel_raw'],rj]),('hist x drift',np.c_[hd,z,rj,rj[:,None]*z]),('lab offset',np.c_[hd,r,np.eye(4)[di['lab_id']][:,1:]]),('stratum offset',np.c_[hd,r,np.eye(3)[di['stratum']][:,1:]]),('hold and drift difference',np.c_[hd,r,rj-r])]: assess(name,F)
E=assess('reference align baseline',np.c_[hd,r]); print('residual rows early/mid/late/rate',np.round(E[:,[0,5,11,12]],4)); print('drift perturbation from raw missing',np.where(~np.isfinite(di['coupon_features_raw']))); print('baseline residual singular values',np.linalg.svd(E,compute_uv=False).round(5))
