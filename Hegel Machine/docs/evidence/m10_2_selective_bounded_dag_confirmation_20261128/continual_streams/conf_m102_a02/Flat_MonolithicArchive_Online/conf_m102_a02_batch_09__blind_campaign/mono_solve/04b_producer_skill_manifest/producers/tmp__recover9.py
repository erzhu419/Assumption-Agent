import pickle,numpy as np
from itertools import combinations_with_replacement as comb
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; L=lambda f,y:np.linalg.lstsq(f,y,rcond=None)[0]; A=lambda x:np.c_[np.ones(len(x)),x]
np.set_printoptions(precision=5,suppress=True,linewidth=150)
u,v=(s['panel_a'][k] for k in ['u','v']); e=np.linalg.norm(v-A(u)@L(A(u),v),axis=1); keep=np.argsort(e)[:-3]; print('removed',np.argsort(e)[-3:],np.sort(e)[-6:]);u,v=u[keep],v[keep]
a=A(u); terms=[i for d in range(2,5) for i in comb(range(4),d)]; f=np.array([np.prod(u[:,i],axis=1) for i in terms]).T
for j in range(4):
 ids=[]
 for step in range(8):
  vals=[]
  for k in set(range(f.shape[1]))-set(ids):
   z=np.c_[a,f[:,ids+[k]]]; c=L(z,v[:,j]); vals.append((np.std(z@c-v[:,j]),k,c))
  e,k,c=min(vals,key=lambda a:a[0]); ids.append(k); print('A',j,step,terms[k],e,c if step in [0,2,7] else '')
funcs={'rat':(lambda z,p:z/(1+p[0]*abs(z)**p[1]),[2.5,2.],([.01,.3],[30,6])), 'soft':(lambda z,p:z/(1+p[0]*abs(z)**p[1])**.5,[3.,2.],([.01,.3],[100,6])), 'tanh':(lambda z,p:np.tanh(p[0]*z),[2.],([.01],[30.])), 'sin':(lambda z,p:np.sin(p[0]*z),[2.],([.01],[20.])), 'gauss':(lambda z,p:np.exp(-p[0]*z*z),[.3],([.001],[20.])), 'oddgauss':(lambda z,p:z*np.exp(-p[0]*z*z),[1.],([.001],[30.]))}
for panel,key,w,z in [('A','v',u,v),*[('C',k,s['panel_c']['u'],s['panel_c'][k]) for k in ('v1','v2')]]:
 for name,(fn,start,bd) in funcs.items():
  for extra in (False,True):
   def des(p):return np.c_[A(w) if extra else np.ones(len(w)),fn(w,p)]
   def res(p):
    F=des(p);return (F@L(F,z)-z).ravel()
   p=least_squares(res,start,bounds=bd,max_nfev=100).x; F=des(p); c=L(F,z); rr=res(p).reshape(z.shape); print(panel,key,name,extra,p,np.std(rr),np.std(rr/(1-np.sum(F*np.linalg.pinv(F).T,axis=1))[:,None]));
   if np.std(rr)<.001:print(c)
