exec(open('/tmp/baseline.py').read().split('for name,b,xx')[0])
from scipy.optimize import least_squares
from itertools import product
from functools import lru_cache
N,K=yd.shape; y=np.c_[yd,rd]; labs=di['lab_id']; xraw=di['coupon_features_raw']; later=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
benchmem=pd@(-np.expm1(-dt[:,None]/tau)*np.exp(-later[:,None]/tau));
def prep(kind):
    ls=np.r_[tr['lab_id'],te['lab_id']]; refs=np.vstack((tr['ref_coupons'],te['ref_coupons'])); refs0=di['ref_coupons']; xx=np.vstack((tr['coupon_features_raw'],te['coupon_features_raw'])); zz=xraw.copy()
    for lab in np.unique(ls):
        ref={'diag':refs0[labs==lab],'jointdiag':np.vstack((refs[ls==lab],refs0[labs==lab])),'train':refs[:len(tr['lab_id'])][tr['lab_id']==lab]}.get(kind,refs[ls==lab]); a=np.median(ref.mean(0)); b=np.median(ref.std(0)); xx[ls==lab]=(xx[ls==lab]-a)/b; zz[labs==lab]=(zz[labs==lab]-a)/b
    cell=np.c_[ls,np.r_[tr['stratum'],te['stratum']]]; _,ii,counts=np.unique(cell,axis=0,return_inverse=True,return_counts=True); w=1/counts[ii]; fill=np.nansum(w[:,None]*xx,axis=0)/np.sum(w[:,None]*np.isfinite(xx),axis=0)
    return np.where(np.isfinite(zz),zz,fill)
def basis(q):
 u=np.exp(q); return np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1.]))
def run(X, indices, fixed=False):
 mask=np.ones(N,bool); mask[indices]=False; w=np.r_[np.ones(K),1.]
 def solve(q,out=False):
  B=basis(q); A=np.column_stack([(z[:,None,:]*B[None,:,j,None]).reshape(N*(K+1),-1) for j,z in enumerate(X)]).reshape(N,K+1,-1); D=(A[mask]*w[None,:,None]).reshape(mask.sum()*(K+1),-1); coef=np.linalg.lstsq(D,(y[mask]*w).ravel(),rcond=None)[0]; pred=A@coef
  return (pred,coef,np.exp(q)) if out else ((pred[mask]-y[mask])*w).ravel()
 q=np.log(tau) if fixed else least_squares(solve,np.log([tau[0],.55*tau[1]]),bounds=(np.log([.25,1.]),np.log([.75,10.])),max_nfev=45).x
 return solve(q,True)
for kind in ('standard','diag','jointdiag','train'):
 zz=prep(kind); r=th[2]+zz@th[3:]; base=np.c_[np.ones(N),hd,r];
 models={'a':base,'x':np.c_[base[:,:2],zz],'a+z':np.c_[base,benchmem],'a+p':np.c_[base,pd],'a+q':np.c_[base,r*r,r*hd,hd*hd],'a+cross':np.c_[base,r*hd],'a+lab':np.c_[base,np.eye(4)[labs][:,1:]],'a+stratum':np.c_[base,np.eye(3)[di['stratum']][:,1:]],'a+tanh':np.c_[base,np.tanh(r)],'a+prod':np.c_[base,r*r,r*hd,benchmem,benchmem*r[:,None]]}
 for name,X in models.items():
  for fixed in ([False,True] if name in ('a','x') else [False]):
   pred,c,u=run([X]*3,[],fixed); er=pred-y; oo=np.empty_like(y)
   for lab in np.unique(labs):
    ix=np.flatnonzero(labs==lab); oo[ix]=(run([X]*3,ix,fixed)[0]-y)[ix]
   metric=lambda e:np.round([np.sqrt(np.mean(e[:,:K]**2)),np.sqrt(np.mean(e[:,-1]**2)),max(np.sqrt(np.mean(e[labs==i,:K]**2)) for i in np.unique(labs)),max(np.sqrt(np.mean(e[labs==i,-1]**2)) for i in np.unique(labs))],4); print(kind,name,fixed,'tau',u.round(3),'fit',metric(er),'labcv',metric(oo),'c',c.round(4) if name=='a' else '')
   if kind=='standard' and name=='a' and not fixed:
    print('residual rows by feature missing and r:',np.c_[labs,np.isnan(xraw).any(axis=1),r,er[:,0],er[:,-2:]].round(4)); print('residual svd',np.linalg.svd(er,compute_uv=False).round(4))
