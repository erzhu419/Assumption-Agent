from chamber_probe import *
from scipy.optimize import least_squares

def feat(mode):
 if mode=='joint': return xq
 if mode=='diag': return prior._features(tr,q)[1]
 if mode=='frozen':
  pool=[tr,te]; lab=np.concatenate([g['lab_id'] for g in pool]); strata=np.concatenate([g['stratum'] for g in pool]); ref=np.concatenate([g['ref_coupons'] for g in pool]); raw=np.concatenate([g['coupon_features_raw'] for g in pool]).copy(); z=q['coupon_features_raw'].copy()
  for l in np.unique(lab):
   a=lab==l; b=q['lab_id']==l; mu=np.median(ref[a].mean(0)); sig=np.median(ref[a].std(0)); raw[a]=(raw[a]-mu)/sig; z[b]=(z[b]-mu)/sig
  _,ix,n=np.unique(np.c_[lab,strata],axis=0,return_counts=True,return_inverse=True); w=1/n[ix]; mu=np.nansum(raw*w[:,None],axis=0)/(np.isfinite(raw).T@w)
  return np.where(np.isfinite(z),z,mu)
 if mode=='self': return prior._features(q,q)[1]

# Physically motivated amplitude maps, followed by a wider linear alternative.
patterns=['hold+drift','all-drift','hold+X','history','interaction']
def basis(z,pattern):
 dr=beta[2]+z@beta[3:]; o=np.ones(len(z)); h=hq
 if pattern=='hold+drift': return [np.c_[o,h],np.c_[o,h,dr],np.c_[o,dr]]
 if pattern=='all-drift': return [np.c_[o,h,dr]]*3
 if pattern=='hold+X': return [np.c_[o,h,z]]*3
 if pattern=='history': return [np.c_[o,h,dr,prior._gap(tau,pq,hq,dt)]]*3
 if pattern=='interaction': return [np.c_[o,h,dr,h*dr]]*3

def fit(B,rows,clock):
 bs=[b.shape[1] for b in B]
 def unpack(l): return tau if clock=='bench' else [tau[0],np.exp(l[0])] if clock=='slow' else np.exp(l)
 def mat(l):
  K=modes(np.asarray(unpack(l))); return np.concatenate([b[:,None,:]*K[None,:,j,None] for j,b in enumerate(B)],axis=2)
 def project(l):
  X=mat(l); A=X[rows].reshape(-1,sum(bs)); yy=Y[rows].ravel(); c=np.linalg.lstsq(A,yy,rcond=None)[0]; return (A@c-yy),c,X
 lo,hi=([np.log(1.2)],[np.log(20)]) if clock=='slow' else (np.log([.3,1.2]),np.log([1.1,20])); start=np.log([tau[1]]) if clock=='slow' else np.log(tau)
 l=start if clock=='bench' else least_squares(lambda l:project(l)[0],start,bounds=(lo,hi),diff_step=1e-4,max_nfev=45).x
 _,c,X=project(l); return X@c,unpack(l),np.split(c,np.cumsum(bs)[:-1])
if __name__=='__main__':
 for fm in ['frozen','diag','joint','self']:
  z=feat(fm)
  for pat in patterns:
   for clock in ['bench','slow','free']:
    B=basis(z,pat); pred=np.zeros_like(Y); worst=[]
    for lab in np.unique(q['lab_id']):
     test=q['lab_id']==lab; a,_,_=fit(B,~test,clock); pred[test]=a[test]; worst.append(np.sqrt(np.mean((a[test]-Y[test])**2,axis=0)))
    e=pred-Y; a,ts,co=fit(B,np.ones(len(Y),dtype=bool),clock); print(fm,pat,clock,'worst lab curve/rate',np.round([np.max(np.sqrt(np.mean(np.array(worst)[:,:-1]**2,axis=1))),np.max(np.array(worst)[:,-1])],5),'oof',np.round([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))],5),'train',np.round(np.sqrt(np.mean((a-Y)**2)),5),'ts',np.round(ts,4));
    if fm=='frozen' and pat=='hold+drift': print('coef', [np.round(v,5) for v in co])
