import numpy as np
from scipy.optimize import least_squares

def fit_a(panel):
    u, v = panel['u'], panel['v']
    n = np.roll(u, -1, axis=1)
    D = np.stack((np.ones_like(u), u, n*(u-1), u**3), axis=-1)
    w = np.ones(len(u))
    for _ in range(24):
        c = np.array([np.linalg.lstsq(D[:,j]*w[:,None], v[:,j]*w,rcond=None)[0] for j in range(4)]).T
        e = np.linalg.norm(np.einsum('njk,kj->nj',D,c)-v, axis=1)
        w = 1/np.sqrt(1+(e/max(0.001, 2*np.median(e)))**2)
    keep = np.argsort(e)[:-3]
    return np.array([np.linalg.lstsq(D[keep,j], v[keep,j],rcond=None)[0] for j in range(4)]).T

def inv_a(x,c):
    b,d,q,g = c
    A=np.diag(d); A[np.arange(4),(np.arange(4)+1)%4]=-q
    u=np.linalg.solve(A,(x-b).T).T
    for i, row in enumerate(x):
        u[i]=least_squares(lambda y:b+d*y+q*(y-1)*np.roll(y,-1)+g*y**3-row,u[i],gtol=1e-12,ftol=1e-12,xtol=1e-12).x
    e=b+d*u+q*(u-1)*np.roll(u,-1,axis=1)+g*u**3-x
    print('inverse max residual',np.max(abs(e)))
    return u

if __name__=='__main__':
 import itertools,pickle
 p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
 c=fit_a(p['panel_a']); print('A',c)
 t=inv_a(p['train']['x'],c); s=inv_a(p['test']['x'],c); print('latent ranges',t.min(0),t.max(0),s.min(0),s.max(0))
 u,v=p['panel_b']['u'],p['panel_b']['v']; y=p['train']['traj'];
 def bases(x):
  A=np.c_[np.ones(len(x)),x]; s=lambda f:np.sin(f*x); co=lambda f:np.cos(f*x)
  Q=np.column_stack([x[:,i]*x[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)])
  R=np.roll(x,-1,axis=1); e=x*R
  yield 'aff',A
  yield 'Q',np.c_[A,Q]
  yield 'Q+cubic',np.c_[A,Q,x**3]
  for f in (1,1.5,2,2.5,3):
   yield 'Q+sin '+str(f),np.c_[A,Q,s(f)]
   yield 'Q+cos '+str(f),np.c_[A,Q,co(f)]
   yield 'cyc+sin/cos '+str(f),np.c_[A,e,x**2,s(f),co(f)]
 def probe(x,Y,name):
  print(name)
  for k,X in bases(x):
   C=np.linalg.lstsq(X,Y,rcond=None)[0]; E=X@C-Y; h=(np.linalg.qr(X)[0]**2).sum(1)
   print(k,'fit/LOO',np.mean(abs(E)),np.mean(abs(E/(1-h)[:,None])))
   if k=='Q': print(C if Y.shape[1]==4 else C[:,-1])
 probe(u,v,'B forward'); probe(v,u,'B inverse'); probe(t,y,'traj after A')
