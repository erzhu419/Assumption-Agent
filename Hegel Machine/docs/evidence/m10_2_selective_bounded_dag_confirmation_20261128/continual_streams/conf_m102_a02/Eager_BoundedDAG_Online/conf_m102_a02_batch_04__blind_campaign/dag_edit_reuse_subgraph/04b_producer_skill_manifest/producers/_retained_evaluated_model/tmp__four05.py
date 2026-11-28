import numpy as np,pickle
from itertools import combinations_with_replacement as cw
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; U,V=[S['panel_a'][k] for k in ['u','v']]; g=np.max(abs(U-V),axis=1)<3; u,v=U[g],V[g]
inds=[i for d in (1,2,3,4) for i in cw(range(4),d)]
def pol(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for i in inds if len(i)<=d])
def fit(z,y):return np.linalg.lstsq(z,y,rcond=1e-10)[0]
c=fit(pol(u,3),v); print('cubic coefficients:'); print([(i,np.round(w,5).tolist()) for i,w in zip([()]+inds,c) if np.max(abs(w))>.004]); c[np.abs(c)<.001]=0
x=S['train']['x']; Y=np.column_stack([S['train']['y'],S['train']['traj']]); ca=fit(pol(u,1),v); i0=(x-ca[0])@np.linalg.inv(ca[1:]); xi=least_squares(lambda t:(pol(t.reshape(-1,4),3)@c-x).ravel(),i0.ravel(),jac_sparsity=np.kron(np.eye(len(x)),np.ones((4,4))),max_nfev=60).x.reshape(-1,4); print('inverse error/range',np.max(abs(pol(xi,3)@c-x)),np.max(abs(xi),axis=0)); rng=np.random.default_rng(915); folds=[(i[:54],i[54:]) for i in (rng.permutation(len(x)) for _ in range(12))]
def cv(z,L=0,y=Y):
 out=[]
 for a,b in folds:
  zt=z[a]; coef=fit(zt,y[a]) if L==0 else solve(zt.T@zt+L*np.eye(z.shape[1]),zt.T@y[a],assume_a='pos'); er=abs(z[b]@coef-y[b]); out.append([er[:,0].mean(),er[:,1:].mean()]);
 return np.round(np.mean(out,axis=0),5),np.round(np.max(out,axis=0),5)
mae=lambda a: np.mean(np.abs(a))
np.set_printoptions(precision=5,suppress=True,linewidth=140)
for base in (0,1,2):
 for trig in ('s','c','sc'):
  def design(w,f):
   return np.column_stack([pol(w,base)]+[fun(w*f) for k,fun in [('s',np.sin),('c',np.cos)] if k in trig]);
  for y in (Y[:,:1],Y[:,1:]):
   def obj(f):
    z=design(xi,f); return (z@fit(z,y)-y).ravel()
   r=min((least_squares(obj,np.full(1,f),bounds=(.4,8)) for f in (1,2,3,4,5)),key=lambda a:a.cost)
   q=least_squares(obj,np.full(4,r.x[0]),bounds=(.4,8)); z=design(xi,r.x); zz=design(xi,q.x)
   def ev(z):return np.mean([mae(z[b]@fit(z[a],y[a])-y[b]) for a,b in folds]),mae(y-z@fit(z,y))
   print(base,trig,y.shape[1],r.x,q.x,ev(z),ev(zz));
   if base==2 and trig=='sc':print('coef',fit(zz,y)[:,[0,-1]].T)
