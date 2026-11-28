import pickle,numpy as np,sys,itertools
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task/_frozen_nodes');import node_00 as f
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; A=f._affine(u); x=f._inverse(s['train']['x'],f._fit_sensor(s['panel_a'])); y=s['train']['y']; tr=s['train']['traj']; U,D,V=np.linalg.svd(tr,full_matrices=False); H=np.c_[y,U[:,:2]*D[:2]]
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,k):
 return np.column_stack([np.prod(x[:,j],axis=1) for n in range(k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),n)])
def cv(A,H,ids=None):
 if ids is None:ids=np.arange(len(A))
 i=ids[:-20];j=ids[-20:];return np.round(abs(A[j]@fit(A[i],H[i])-H[j]).mean(0),5)
from scipy.optimize import minimize_scalar
from itertools import combinations
pairs=list(combinations(range(4),2))
def cross(u):return np.column_stack([u[:,i]*u[:,j] for i,j in pairs])
def B(u):return np.c_[np.ones(len(u)),u,u*u]
def v1(u,k,fun):return np.c_[np.ones(len(u)),u,fun(k*u)]
mods={}
for key,n in [('panel_c','v1'),('panel_c','v2'),('panel_b','v')]:
 p=s[key];u=p['u'];v=p[n];print('\n',key,n)
 def evalb(a):
  c=fit(a,v); r=a@c-v
  lev=np.sum(np.linalg.qr(a,mode='reduced')[0]**2,axis=1)
  return np.sqrt(np.mean(r*r)),np.mean(abs(r/(1-lev[:,None])))
 print('quadratic coefficients',np.round(fit(poly(u,2),v),3))
 for name,fun in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('erf',__import__('scipy.special',fromlist=['erf']).erf),('cos',np.cos),('gauss',lambda z:np.exp(-z*z)),('soft',lambda z:z/(1+z*z)**.5)]:
  for i,bas in enumerate([lambda q,k:v1(q,k,fun),lambda q,k:np.c_[v1(q,k,fun),cross(q)],lambda q,k:np.c_[B(q),fun(q*k)]]):
   res=minimize_scalar(lambda k:evalb(bas(u,k))[0],bounds=(.15,8),method='bounded');k=res.x
   if res.fun<.2:print(name,i,round(k,4),np.round(evalb(bas(u,k)),5),'down',cv(bas(x,k),H),cv(poly(bas(x,k)[:,1:5],2),H))
 print('sq',evalb(B(u)), 'poly',evalb(poly(u,2)))
 if n=='v1':
  for k in [.5,1,1.25,1.5,2,2.15,2.5,3,4]:
   for fn in [np.tanh,np.sin,np.arctan]:
    bas=lambda q: v1(q,k,fn); c=fit(bas(u),v); z=bas(x)@c; print('C1',fn.__name__,k,np.round(evalb(bas(u)),5),cv(np.c_[poly(z,2),x],H),cv(poly(z,2),H))
