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
for name,w in [('raw',x),('forward',pol(x,3)@c),('inverse',xi)]:
 print(name,'ranges',np.max(abs(w),axis=0));
 for d in (1,2,3,4):
  z=pol(w,d); rr=Y-z@fit(z,Y); print('poly',d,'fit',np.round([abs(rr[:,0]).mean(),abs(rr[:,1:]).mean()],5),*[cv(z,a) for a in (0,.03,.2)])
 iq,jq=np.triu_indices(4); q=w[:,iq]*w[:,jq];
 for f in (.5,1,1.5,2,2.5,3):
  for ftype,z in [('indiv',np.c_[pol(w,2),np.sin(f*w),np.cos(f*w)]),('pair',np.c_[pol(w,1),np.sin(f*w),np.cos(f*w),np.sin(f*q),np.cos(f*q)]),('sat',np.c_[pol(w,1),np.tanh(f*w),np.tanh(f*q)]),('rational',np.c_[pol(w,1),w/(1+f*w*w),q/(1+f*q*q)])]: print(ftype,f,cv(z,.04))
 for f in (.07,.17,.33,.65,1.3,2.6):
  K=np.exp(-f*cdist(w,w,'sqeuclidean')); out=[]
  for a,b in folds:
   pred=K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+.001*np.eye(len(a)),Y[a],assume_a='pos'); er=abs(pred-Y[b]); out.append([er[:,0].mean(),er[:,1:].mean()]);
  print('rbf',f,np.round(np.mean(out,axis=0),5))
