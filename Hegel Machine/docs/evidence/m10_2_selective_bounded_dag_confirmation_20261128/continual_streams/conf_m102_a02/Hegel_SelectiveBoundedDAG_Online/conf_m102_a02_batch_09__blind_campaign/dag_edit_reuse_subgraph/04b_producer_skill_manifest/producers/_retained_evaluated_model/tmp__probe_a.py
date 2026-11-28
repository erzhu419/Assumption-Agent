import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); ok=np.max(abs(v-u),axis=1)<2; u,v=u[ok],v[ok]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
for (i,j) in [(0,2),(1,3)]:
 a,b=u[:,i],u[:,j]; y=v[:,[i,j]]; print('PAIR',i,j)
 for d in [1,2,3,4,5]:
  terms=[t for k in range(d+1) for t in combinations_with_replacement(range(2),k)]; x=u[:,[i,j]]; f=np.stack([np.prod(x[:,t],axis=1) for t in terms],1); w=ls(f,y); print(d,np.mean(abs(f@w-y)),[(t,np.round(c,3)) for t,c in zip(terms,w) if d==3]);
 for label,fun in [('tanh',lambda x,p:np.tanh(p*x)),('rational',lambda x,p:x/(1+p*x*x)),('sqrt',lambda x,p:x/np.sqrt(1+p*x*x)),('arctan',lambda x,p:np.arctan(p*x)),('soft',lambda x,p:x/(1+abs(x)**p)),('exp',lambda x,p:x*np.exp(-p*x*x))]:
  for kind in [0,1,2]:
   def design(p):
    z=[a,b] if kind==0 else ([a,a+p[1]*b] if kind==1 else [a,b,a+p[1]*b,a-p[1]*b]); return np.column_stack([np.ones(len(a)),a,b]+[fun(x,p[0]) for x in z]);
   def r(p):
    f=design(p); return (f@ls(f,y)-y).ravel()
   fits=[least_squares(r,[k,c],bounds=([.01,-3],[25,3]),max_nfev=150) for k,c in [(1,1),(3,-1),(8,1)]]; p=min(fits,key=lambda f:f.cost); print(label,kind,np.round(p.x,5),np.mean(abs(p.fun)))
