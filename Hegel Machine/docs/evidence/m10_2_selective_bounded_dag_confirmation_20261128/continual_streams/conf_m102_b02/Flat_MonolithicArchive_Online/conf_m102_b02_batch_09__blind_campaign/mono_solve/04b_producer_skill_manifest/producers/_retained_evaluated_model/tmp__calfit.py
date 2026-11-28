from panel import *
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize_scalar
u,v=s['panel_a'].values(); keep=(np.abs(v).max(1)<3);u,v=u[keep],v[keep]
for d in [1,2,3]:
 pf=PolynomialFeatures(d); X=pf.fit_transform(u); c=ls(X,v); print('A poly',d,np.sqrt(np.mean((X@c-v)**2,axis=0))); print(list(zip(pf.get_feature_names_out(),c.round(3))))
for j in range(4):
 print('channel',j)
 cand=[]
 for name,fn in funcs.items():
  for k in range(4):
   for mode in (0,1,2):
    def D(a,p):
     b=fn(p*a[:,k]); return I(np.c_[a,b if mode==0 else b*a[:,j] if mode==1 else b*a[:,(j+1)%4]])
    def err(p):
     d=D(u,p);return np.mean((d@ls(d,v[:,j])-v[:,j])**2)
    r=min((minimize_scalar(err,bounds=b,method='bounded') for b in [(.08,1.4),(1.4,3),(3,5)]),key=lambda z:z.fun);cand.append((r.fun**.5,name,k,mode,r.x))
 print(sorted(cand)[:8])
