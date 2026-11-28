import pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def aff(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,n=2):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for d in range(1,n+1) for j in combinations_with_replacement(range(x.shape[1]),d)])
a=s['panel_a']; u,v=a['u'],a['v']; D=aff(u); rng=np.random.default_rng(1)
for typ in ('diag','full'):
 best=None
 for i in range(1200):
  j=rng.choice(len(u),5,False)
  if typ=='diag':
   coeff=np.array([fit(D[j][:,[0,k+1]],v[j,k]) for k in range(4)]); pred=coeff[:,0]+u*coeff[:,1]
  else: coeff=fit(D[j],v[j]); pred=D@coeff
  e=np.linalg.norm(pred-v,axis=1); val=np.sort(e)[:33]@np.sort(e)[:33]
  if best is None or val<best[0]:best=(val,coeff,e)
 print('A',typ,best[0],best[1],np.sort(best[2])); j=np.argsort(best[2])[:35]; print('inlier diff corr',np.corrcoef(np.c_[u[j],(v-u)[j]].T)[:4,4:])
for p in 'bc':
 a=s['panel_'+p]; u=a['u']
 for key,v in a.items():
  if key=='u':continue
  funcs={'linear':lambda a:u,'atan':lambda a:np.arctan(a*u),'tanh':lambda a:np.tanh(a*u),'sin':lambda a:np.sin(a*u),'rational':lambda a:u/np.sqrt(1+a*u*u),'quadratic':lambda a:np.c_[u,u*u/(1+a*u*u)],'sq+atan':lambda a:np.c_[np.arctan(a*u),u*u]}
  print('PANEL',p,key)
  for name,fun in funcs.items():
   def loss(a):
    d=aff(fun(a));return np.sqrt(np.mean((d@fit(d,v)-v)**2))
   r=minimize_scalar(loss,bounds=(.02,4),method='bounded');print(name,round(r.x,4),round(r.fun,6))
  print('poly2 rms',np.mean((poly(u)@fit(poly(u),v)-v)**2)**.5)
x,y,z=[s['train'][k] for k in ('x','y','traj')];t=s['eval_times']; U,S,V=np.linalg.svd(z,full_matrices=False);print('V',V[:2]);
fs={'2exp':lambda p:np.array([-np.expm1(-p[0]*t),-np.expm1(-p[1]*t)]),'exp+t':lambda p:np.array([-np.expm1(-p[0]*t),t]),'exp+t*exp':lambda p:np.array([-np.expm1(-p[0]*t),t*np.exp(-p[1]*t)]),'exp+t/(1+t)':lambda p:np.array([-np.expm1(-p[0]*t),t/(1+p[1]*t)]),'exp+sin':lambda p:np.array([-np.expm1(-p[0]*t),np.sin(p[1]*t)])}
for name,fn in fs.items():
 def res(p):
  f=fn(p);return (fit(f.T,z.T).T@f-z).ravel()
 r=least_squares(res,[1.5,.4],bounds=(.001,8)); print(name,r.x,np.sqrt(np.mean(r.fun**2)))
 if name=='2exp':
  f=fn(r.x); amps=fit(f.T,z.T).T;print('amp corr',np.corrcoef(np.c_[x,y,amps].T)[:5,5:]);print('amp rows',np.c_[x,y,amps][:12]);
