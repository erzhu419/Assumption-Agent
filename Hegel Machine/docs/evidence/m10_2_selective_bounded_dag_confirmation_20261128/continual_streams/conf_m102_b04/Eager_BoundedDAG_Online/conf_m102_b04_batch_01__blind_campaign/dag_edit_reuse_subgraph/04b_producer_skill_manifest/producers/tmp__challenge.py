exec(open('/tmp/probe.py').read().split('res=[]')[0])
from scipy.spatial.distance import cdist
p=s['panel_c']; u=p['u']; v1=p['v1']; v2=p['v2']; c=Q['caldiag']; Wb=matrices['panel_bv']; D=lambda x:np.c_[affine(x),x*x,x**3,x**4]; W=np.linalg.lstsq(D(u),v2,rcond=None)[0]
for k,V in [('v1',v1),('v1-v2',v1-v2)]:
 for name,func in [('cubic',lambda u:np.c_[affine(u),u*u,u**3]),('joint',lambda u:np.c_[affine(u),u*u,u**3,u**4])]:
  A=func(u); w=np.linalg.lstsq(A,V,rcond=None)[0]; h=np.sum(A*np.linalg.pinv(A).T,axis=1); print(k,name,np.mean(abs((A@w-V)/(1-h[:,None]))))
R={}; a=lambda f:f@f.T
for tag,U in [('raw',x),('a',c),('b',affine(c)@Wb),('c1',affine(c)@matrices['panel_cv1']),('c2',D(c)@W)]:
 for mag in [0,.09,.3,.7]:
  T=U/np.sqrt(1+mag*U*U)
  for d in [2,3,4,5]:
   for pure in ([False,True] if d>=3 else [False]):
    f=basis(T,d) if not pure else np.c_[basis(T,2),*[T**k for k in range(3,d+1)]]
    R[(tag,mag,d,pure)]=a(f)
 for l in [.7,1.3,2.2,3.5]:
  d=cdist(U,U)/l; R[(tag,'se',l)]=np.exp(-d*d/2); R[(tag,'m52',l)]=(1+np.sqrt(5)*d+5*d*d/3)*np.exp(-np.sqrt(5)*d)
base=R[('b',.3,3,False)]; linear=a(basis(c,2));
for name,K in list(R.items()):
 if name[0]=='c2' or name[1] in ['se','m52']:
  R[('base+',*name)]=base+K
  R[('quad+',*name)]=linear+K
params=[(name,lam) for name in R for lam in [.0001,.001,.003,.01,.03,.1]]
def errors(tr,te,Y):
 out=[]
 for name,lam in params:
  K=R[name]; out.append(np.mean(abs(K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),Y[tr],assume_a='pos')-Y[te])))
 return np.array(out)
E=np.array([errors(tr,te,y) for tr,te in splits]); idx=np.argsort(E.mean(0)); print('top mean:',[(params[i],np.round([E[:,i].mean(),E[0,i],E[:,i].max()],5)) for i in idx[:12]]); idx=np.argsort(E.max(0));print('top worst:',[(params[i],np.round([E[:,i].mean(),E[0,i],E[:,i].max()],5)) for i in idx[:12]])
nested=[]; preds=[]
for tr,te in splits:
 rng=np.random.default_rng(103); errs=np.array([errors(fit,val,y) for _ in range(3) for val in np.array_split(rng.permutation(tr),4) for fit in [np.setdiff1d(tr,val)]]).mean(0); idx=np.argsort(errs)[:8]; nested.append(E[len(nested),idx[0]]); tmp=[]
 for i in idx:
  name,lam=params[i];K=R[name];tmp.append(K[np.ix_(te,tr)]@solve(K[np.ix_(tr,tr)]+lam*np.eye(len(tr)),y[tr],assume_a='pos'))
 preds.append(np.mean(abs(np.mean(tmp,axis=0)-y[te])))
 print('nested/all',params[idx[0]],round(nested[-1],5),round(preds[-1],5))
print('nested results',stat(nested),stat(preds))
