import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; i,j=np.triu_indices(4,1)
I=lambda u: np.c_[np.ones(len(u)),u]
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-11)[0]
pair=lambda u:u[:,i]*u[:,j]
def cv(a,v):
 p=a@np.linalg.pinv(a); return np.sqrt(np.mean(((v-p@v)/(1-p.diagonal()[:,None]))**2))
u=s['panel_a']['u']; v=s['panel_a']['v']; keep=np.ones(len(u),bool)
for k in range(10):
 a=I(u); c=ls(a[keep],v[keep]); r=np.linalg.norm(a@c-v,axis=1); keep=r<max(.005,3*np.median(r))
print('cal kept',sum(keep),'error',np.sort(r).round(5),'coeff',c.round(5));
for name,d in [('affine',u),('pairs',np.c_[u,pair(u)]),('cubic',np.c_[u,pair(u),u**3]),('quad',np.c_[u,pair(u),u**2]),('full',np.c_[u,pair(u),u**2,u**3])]:
 a=I(d); c=ls(a[keep],v[keep]); print('A',name,cv(a[keep],v[keep]),c.round(4).tolist())
for p,key in [('panel_b','v'),('panel_c','v2')]:
 u=s[p]['u'];v=s[p][key];
 for fn in [np.sin,np.tanh,np.arcsinh,lambda z:z/(1+z*z),np.cos,np.sinh,np.arctan]:
  for base in [u,np.c_[u,pair(u)],np.c_[u,pair(u),u**2]]:
   def d(q):return I(np.c_[base,fn(u*q)])
   def score(q):
    a=d(q); P=a@np.linalg.pinv(a); return ((v-P@v)/np.maximum(1-P.diagonal()[:,None],.001)).ravel()
   sol=min([least_squares(score,[q],bounds=(.1,7),max_nfev=100) for q in [.6,1.25,2.15,3.5,5.5]],key=lambda a:a.cost)
   print(p,fn.__name__,base.shape[1],sol.x.round(4),np.std(sol.fun).round(6))
 if p=='panel_c':
  a=I(np.c_[u,pair(u)]); print('C1',ls(a,s[p]['v1']).round(5)); print('C2',ls(a,v).round(5))
