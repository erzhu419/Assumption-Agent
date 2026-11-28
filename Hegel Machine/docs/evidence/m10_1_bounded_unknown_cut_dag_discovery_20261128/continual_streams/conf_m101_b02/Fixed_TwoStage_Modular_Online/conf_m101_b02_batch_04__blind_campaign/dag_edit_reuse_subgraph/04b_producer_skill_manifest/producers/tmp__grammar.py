import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def base(u):return np.column_stack((np.ones(len(u)),u))
def quad(u):return np.column_stack([u[:,i]*u[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
def fit(X,Y):
 c=ls(X,Y);r=Y-X@c;h=np.sum(np.linalg.qr(X,mode='reduced')[0]**2,axis=1);return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-h[:,None]))**2)),c
for k,v in [('panel_b','v'),('panel_c','v2')]:
 u=s[k]['u'];y=s[k][v];a=base(u);q=quad(u);pw=q[:,[0,4,7,9]];pa=q[:,[1,3,5,8]];print(k)
 for label,Q in [('none',q[:,:0]),('squares',pw),('cycle',pa),('all',q)]:
  for ty in ['sin','cos','sc']:
   def calc(f):
    H={'sin':np.sin(f*u),'cos':np.cos(f*u),'sc':np.c_[np.sin(f*u),np.cos(f*u)]}[ty]
    return fit(np.c_[a,Q,H],y)
   if Q.shape[1]==10 and ty=='sc':continue
   grid=np.linspace(.3,12,118); costs=[calc(f)[1] for f in grid]; ix=np.argsort(costs)[:1];
   for i in ix:
    res=minimize_scalar(lambda f:calc(f)[1],bounds=(grid[max(0,i-1)],grid[min(i+1,len(grid)-1)]),method='bounded');print(label,ty,round(res.x,4),np.round(calc(res.x)[:2],6))
    if res.fun<.012: print(np.round(calc(res.x)[2],4))
