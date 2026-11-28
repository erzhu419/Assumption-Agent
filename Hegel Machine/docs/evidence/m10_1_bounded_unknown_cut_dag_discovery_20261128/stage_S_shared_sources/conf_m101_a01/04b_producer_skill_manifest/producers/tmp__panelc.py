from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.optimize import least_squares
p=s['panel_c']; u=p['u']; perm=np.random.default_rng(19).permutation(len(u)); splits=[(np.arange(len(u)),np.arange(len(u)))]+[(np.setdiff1d(np.arange(len(u)),v),v) for v in np.array_split(perm,4)]
P=poly(u,2); links={'sin':np.sin,'tanh':np.tanh,'cos':np.cos,'atan':np.arctan,'sqrat':lambda q:q*q/(1+q*q),'sincos':lambda q:np.c_[np.sin(q),np.cos(q)]}
for k in ('v1','v2'):
 res=[]; target=p[k]
 for base in ('1','aff','diag'):
  for name,g in links.items():
   for npar in (1,4):
    if g==links['sincos'] and base=='diag': continue
    def features(z,f):
     a={'1':np.ones((len(z),1)),'aff':aff(z),'diag':np.c_[aff(z),z*z]}[base]
     return np.c_[a,g(z*f)]
    errors=[]; params=[]
    for it,iv in splits:
     def err(f):
      a=features(u[it],f); return (a@ls(a,target[it])-target[it]).ravel()
     opt=min([least_squares(err,np.full(npar,h),bounds=(.12,8),max_nfev=100) for h in (.65,1.8,3.,5.)],key=lambda o:o.cost)
     a=features(u,opt.x);errors.append(np.mean((a[iv]@ls(a[it],target[it])-target[iv])**2)**.5);params.append(opt.x)
    res.append((max(errors[1:]),base,name,npar,np.round(errors,6),np.round(params[0],5)))
 print(k)
 for r in sorted(res)[:12]:print(r)
print('panel inputs ranges',*[np.quantile(s[k]['u'],[0,.5,1],axis=0) for k in ('panel_a','panel_b','panel_c')]);print('panel C v1 / v2 covariance',np.corrcoef(np.c_[p['v1'],p['v2']].T)[:4,4:])
