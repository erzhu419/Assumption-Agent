import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
ls=lambda a,y:np.linalg.lstsq(a,y,rcond=1e-10)[0]
one=lambda z:np.c_[np.ones(len(z)),z]
u=s['panel_a']['u'];v=s['panel_a']['v']; raw=np.vstack([s['train']['x'],s['test']['x']]); init=np.linalg.norm(v,axis=1)<4
names=['1']+['x'+str(i) for i in range(4)]+['p'+str(i)+str(j) for i in range(4) for j in range(i,4)]+['c'+str(i) for i in range(4)]
def poly(x):return np.c_[one(x),np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]),x**3]
def test(fn,q0=None, verbose=False):
 m=init.copy(); q=q0
 for _ in range(5):
  def solve(q):
   f=fn(u,q); w=np.array([ls(f[m,j],v[m,j]) for j in range(4)]);return w,(f*w).sum(-1)-v
  if q0 is not None:q=least_squares(lambda q:solve(q)[1][m].ravel(),q,bounds=(.05,7),max_nfev=100).x
  w,r=solve(q); e=np.linalg.norm(r,axis=1);mm=e<max(.001,5*np.median(e));
  if np.array_equal(mm,m) or mm.sum()<25:break
  m=mm
 if verbose:print('m',np.flatnonzero(~m),'w',np.round(w,4))
 return np.std(r[m]),m.sum(),q,w,m
D={}
D['fullpoly']=(lambda x,q:np.repeat(poly(x)[:,None,:],4,axis=1),None)
for k in (1,2,3):
 for name,f in {'poly':lambda a,b,q:[a,b,a*b,a**3], 'poly2':lambda a,b,q:[a,b,a*b,a**2,b**2,a**3,b**3], 'trig':lambda a,b,q:[a,b,np.tanh(q[0]*a),np.sin(q[1]*b)],'rat':lambda a,b,q:[a,b,a/(1+(q[0]*a)**2),np.sin(q[1]*b)],'mix':lambda a,b,q:[a,b,a*b,np.sin(q[0]*a),np.sin(q[1]*b)],'ratmix':lambda a,b,q:[a,b,a*b,np.sin(q[1]*b),a/(1+(q[0]*a)**2)]}.items():
  D[f'{name}{k}']=(lambda x,q,k=k,f=f:np.stack([np.ones_like(x)]+f(x,np.roll(x,-k,axis=1),q),axis=-1),None if name.startswith('poly') else [1.5,2.3])
for name,f in {'fulltanh':lambda x,q:np.c_[one(x),np.tanh(q[0]*x)],'fullsin':lambda x,q:np.c_[one(x),np.sin(q[0]*x)],'fullsinrat':lambda x,q:np.c_[one(x),np.sin(q[0]*x),x/(1+(q[1]*x)**2)],'fullpolytrig':lambda x,q:np.c_[poly(x)[:,:15],np.sin(q[0]*x),np.tanh(q[1]*x)]}.items():D[name]=(lambda x,q,f=f:np.repeat(f(x,q)[:,None,:],4,axis=1),[2.2,1.5])
for name,(fn,q0) in D.items():
 e,n,q,w,m=test(fn,q0);print(name,round(e,7),n,q)
 if name in ('fullpoly','fullsin','fulltanh','fullsinrat','fullpolytrig'):
  print(np.round(w,3))
