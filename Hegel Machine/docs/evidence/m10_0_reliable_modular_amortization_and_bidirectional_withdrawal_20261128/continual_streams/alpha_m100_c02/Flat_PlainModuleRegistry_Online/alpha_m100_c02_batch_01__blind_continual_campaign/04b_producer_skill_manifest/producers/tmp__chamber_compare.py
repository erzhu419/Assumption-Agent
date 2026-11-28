from chamber_probe import *
from scipy.optimize import least_squares
one=n.ones(len(h));

def feats(x, mode):
 dr=x@th[2:]; basic=n.c_[one,h,dr]; plain=n.c_[one,h,x[:,1:]]
 if mode=='reset': return (basic,basic,n.c_[one,dr])
 if mode=='reset_x': return (plain,plain,x)
 if mode=='structured': return (n.c_[basic,z[:,0]],n.c_[basic,z[:,1]],n.c_[one,dr])
 if mode=='linear':
  all=n.c_[plain,past]; return (all,all,all)
 if mode=='reset_constant_drift': return (basic,basic,one[:,None])
 if mode=='reset_nonlinear': return (n.c_[basic,dr**2,h*dr],n.c_[basic,dr**2,h*dr],n.c_[one,dr,dr**2])

def solve(F,mask,free=False):
 dims=[f.shape[1] for f in F]; sel=n.repeat(mask,y.shape[1]);
 def matrix(u):
  b=basis(u);return n.column_stack([(f[:,None,:]*b[None,:,j,None]).reshape(y.size,-1) for j,f in enumerate(F)])
 A=matrix(tau); q=n.linalg.lstsq(A[sel],y.ravel()[sel],rcond=None)[0];
 if free:
  opt=least_squares(lambda p:matrix(n.exp(p[:2]))[sel]@p[2:]-y.ravel()[sel],n.r_[n.log(tau),q],loss='soft_l1',f_scale=.004,max_nfev=120)
  u=n.exp(opt.x[:2]);q=opt.x[2:]; A=matrix(u)
 else:
  u=tau; q=least_squares(lambda q:A[sel]@q-y.ravel()[sel],q,jac=lambda _:A[sel],loss='soft_l1',f_scale=.004).x
 return (A@q).reshape(y.shape),u,n.split(q,n.cumsum(dims)[:-1])

align={ 'combined': x, 'diag_only':m._features(g,g)[:len(h)] }
for al,v in align.items():
 for mode in ('reset','reset_x','structured','linear','reset_constant_drift','reset_nonlinear'):
  for free in (False,True):
   F=feats(v,mode); o,u,a=solve(F,one.astype(bool),free); q=n.empty_like(y)
   for l in n.unique(g['lab_id']):
    ix=g['lab_id']==l;q[ix]=solve(F,~ix,free)[0][ix]-y[ix]
   worst=n.max([n.sqrt(n.mean(q[g['lab_id']==i]**2,axis=0))[[0,4,11,12]] for i in range(4)],axis=0)
   print(al,mode,free,'tau',u.round(4),'fit',n.sqrt(n.mean((o-y)**2,axis=0))[[0,4,11,12]].round(4),'LOLO',worst.round(4),'coef',[w.round(4).tolist() for w in a] if mode in ('reset','reset_x','reset_constant_drift') else '')
print('refs means',[(n.mean(s['ref_coupons'],axis=0).round(2),n.isnan(s['coupon_features_raw']).sum(0)) for s in [tr,te,g]])
