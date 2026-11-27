exec(open('/tmp/chamber98.py').read().split('np.set_printoptions')[0])
import json
np.set_printoptions(precision=6,suppress=True,linewidth=180)
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('log summaries',[(v['completed_short_name'],np.asarray(v['facility_telemetry_trace']).shape, np.mean(v['facility_telemetry_trace'],axis=0)) for v in log]); print('telemetry stats',[(k,np.mean(d[k]['aux_chamber_telemetry'],axis=0),np.std(d[k]['aux_chamber_telemetry'],axis=0)) for k in ('train','test')]); print('amp PCA relative to theta H',np.linalg.svd(A-H[:,None]*w[:2],compute_uv=False)); print('amp contrasts',np.c_[H,R,A-H[:,None]*w[:2]])
for group in [allsite,tr,te,ck]:
 print('ref',[(i,np.median(np.mean(group['ref_coupons'][group['lab_id']==i],axis=0)),np.median(np.std(group['ref_coupons'][group['lab_id']==i],axis=0))) for i in np.unique(group['lab_id'])])
# Profile chamber against candidate state constructions in measurement space.
_,xx_diag=m._standardize(ck,ck); _,xx_with_tr=m._standardize(tr,ck)
raw=cal['bridge_raw']; coeff=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],cal['true_channel_loads'],rcond=None)[0]; true=lambda u:u@coeff[:4]+coeff[4]; pc=true(ck['past_channel_raw']); hc=true(ck['hold_channel_raw']);
for name, Q in [('H+X',np.c_[H,F]),('H+Xdiag',np.c_[H,np.ones(len(H)),xx_diag]),('H+Xtrain',np.c_[H,np.ones(len(H)),xx_with_tr]),('H+P+X',np.c_[H,P,F]),('H+decay+X',np.c_[H,P@decay,F]),('H+reverse decay+X',np.c_[H,P[:,::-1]@decay,F]),('H+HC+R',np.c_[H,hc[:,:3],R]),('H+HC+P+R',np.c_[H,hc[:,:3],P,R]),('H+R+P0 mean',np.c_[H,R,P[:,0],P.mean(1)]),('H+R+history bend',np.c_[H,R,P[:,0],P[:,1]-P[:,-2],P[:,2]+P[:,3],P[:,-1]])]:
 u=np.linalg.lstsq(Q,A,rcond=None)[0]; err=(Q@u-A)@B.T; lev=(Q*np.linalg.pinv(Q).T).sum(1); loo=(err+A@B.T-Y)/np.maximum(1-lev[:,None],1e-6); print(name,'train',np.sqrt(np.mean(err**2,axis=0))[[0,5,11,12]],'LOO',np.sqrt(np.mean(loo**2,axis=0))[[0,5,11,12]],'coef',u.T)
# If both modes share a drift-like latent, profiling out each coupon's latent is informative about the orthogonal contrast.
for base in ['hold','gap','linear']:
 def resid(v,out=False):
  zz=np.exp(v[:2]); bb=np.vstack([-np.expm1(-t[:,None]/zz),1/zz]);
  if base=='hold': amp=H[:,None]*w[:2]; j=2
  elif base=='gap': amp=G*w[:2]; j=2
  else: amp=H[:,None]*v[2:4]; j=4
  u=np.array([v[j],1.]); pred=amp@bb.T; f=bb@u; s=(Y-pred)@f/(f@f); e=pred+s[:,None]*f-Y
  return (zz,u,s,e) if out else e.ravel()
 init=np.r_[np.log(z),w[:2],.13] if base=='linear' else np.r_[np.log(z),.13]; opt=least_squares(resid,init,max_nfev=90); zz,u,s,e=resid(opt.x,True); print('shared route',base,opt.x,zz,u,np.sqrt(np.mean(e**2,axis=0))[[0,5,11,12]],'latent',s)
