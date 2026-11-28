exec(open('/tmp/chamber_probe.py').read().split('for tt in (ta,')[0])
import downstream_solver as s
xd=s._features(d,{k:v[:0] for k,v in d.items()})
full=np.isfinite(d['coupon_features_raw']).all(axis=1)
v=np.c_[np.ones(len(h)),xd]@th[2:]; f=F(ta)
base=h[:,None]*(f[:,:2]@th[:2]); resid=Y-base
A=v[:,None,None]*f[None,:,:]
for lab in np.unique(d['lab_id']):
 m=d['lab_id']==lab; keep=~m & full
 b=np.linalg.lstsq(A[keep].reshape(-1,3),resid[keep].ravel(),rcond=None)[0]
 E=base+v[:,None]*(f@b)-Y
 for name,idx in [('complete',m&full),('all',m),('imputed',m&~full)]:
  if idx.any():
   e=E[idx]; print('lab',lab,name,'max-time-RMSE/rate-RMSE',np.max(np.sqrt(np.mean(e[:,:-1]**2,axis=0))),np.sqrt(np.mean(e[:,-1]**2)),'max',np.max(abs(e[:,:-1])),np.max(abs(e[:,-1])))
 print('estimated coupling',b)
print('test missing by lab',[(int(l), np.sum(~np.isfinite(te['coupon_features_raw'][te['lab_id']==l]).all(axis=1)),np.sum(te['lab_id']==l)) for l in np.unique(te['lab_id'])]); print('time range',t)
