import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[k]['u']; v=s[k][j]; a=aff(u); lin=a@ls(a,v); r=v-lin
 print(k,j,'range',v.min(0),v.max(0),'rSVD',np.linalg.svd(r,compute_uv=False)); print('lin-v-rn:',np.c_[lin[:,0], v[:,0], np.linalg.norm(u,axis=1),np.linalg.norm(r,axis=1)])
 fs={'linear':a,'quadratic':np.c_[a, *[u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]], 'diag cubic':np.c_[a,u**3], 'row radial cubic':np.c_[a,u*np.sum(u*u,1)[:,None]], 'sin':aff(np.sin(1.25*u)), 'tanh':aff(np.tanh(1.25*u)), 'asinh':aff(np.arcsinh(1.25*u)), 'atan':aff(np.arctan(1.25*u)),'u+sin':np.c_[a,np.sin(1.25*u)],'u+tanh':np.c_[a,np.tanh(1.25*u)]}
 for name,f in fs.items():
  d=f@ls(f,v)-v; h=(np.linalg.svd(f,full_matrices=False)[0]**2).sum(1); print(name, 'fit/loo',np.sqrt(np.mean(d*d)),np.sqrt(np.mean((d/(1-h)[:,None])**2)))
 for name,fun in {'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'sin':np.sin,'softsign':lambda x:x/(1+abs(x)),'sqrt':lambda x:x/np.sqrt(1+x*x),'inv3':lambda x:np.sinh(np.arcsinh(x)/3)}.items():
  for offset in (False,True):
   def pred(q,A):
    z=fun(A@q[:5]); return z*np.exp(q[5])+(q[6] if offset else 0)
   outs=[]; loo=[]
   for t in v.T:
    p=least_squares(lambda q:pred(q,a)-t,np.r_[ls(a,t)/(3 if name=='inv3' else 1),0., *([0.] if offset else [])],max_nfev=500,bounds=([-np.inf]*5+[-6]+([-np.inf] if offset else []),[np.inf]*5+[6]+([np.inf] if offset else [])))
    outs.append(np.mean(p.fun**2)); h=(np.linalg.svd(p.jac,full_matrices=False)[0]**2).sum(1); loo.append(np.mean((p.fun/(1-h))**2))
   print(name,offset,np.sqrt(np.mean(outs)),np.sqrt(np.mean(loo)))
