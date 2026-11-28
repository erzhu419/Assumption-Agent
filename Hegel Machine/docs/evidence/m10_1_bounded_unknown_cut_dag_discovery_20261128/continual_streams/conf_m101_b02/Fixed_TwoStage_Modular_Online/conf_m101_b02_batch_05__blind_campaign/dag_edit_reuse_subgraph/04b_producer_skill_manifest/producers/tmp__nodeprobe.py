import pickle
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def abasis(u,j):
 return np.column_stack((np.ones(len(u)),u,u[:,j]**3,u[:,j]*u[:,(j+1)%4],u[:,j]/(1+u[:,j]**2),np.cos(u[:,0]),np.cos(u[:,j]))) if j else np.column_stack((np.ones(len(u)),u,u[:,0]**3,u[:,0]*u[:,1],u[:,0]/(1+u[:,0]**2),np.cos(u[:,0])))
def fit_a(p):
 u,v=p['u'],p['v']; aa=[abasis(u,j) for j in range(4)]; w=np.ones(len(u))
 for it in range(24):
  q=np.sqrt(w); bb=[np.linalg.lstsq(a*q[:,None],v[:,j]*q,rcond=None)[0] for j,a in enumerate(aa)]; r=np.linalg.norm(np.column_stack([a@b for a,b in zip(aa,bb)])-v,axis=1); w=1/(1+(r/max(.001,1.5*np.median(r)))**2)
 keep=np.argsort(r)[:-3]; bb=[np.linalg.lstsq(a[keep],v[keep,j],rcond=None)[0] for j,a in enumerate(aa)]; print('A core residual and coefs',[(round(np.sqrt(np.mean((a[keep]@b-v[keep,j])**2)),5),np.round(b,3).tolist()) for j,(a,b) in enumerate(zip(aa,bb))]); return lambda u: np.column_stack([abasis(u,j)@b for j,b in enumerate(bb)])
A=fit_a(s['panel_a']); x=s['train']['x']; n=len(x); u=np.asarray([least_squares(lambda w: (A(w[None,:])[0]-v), v, max_nfev=80).x for v in x]); print('A inverse residual and range',np.max(abs(A(u)-x)),u.min(0),u.max(0)); print('input means std',x.mean(0),x.std(0),u.mean(0),u.std(0))
y=np.column_stack((s['train']['y'],s['train']['traj'])); rng=np.random.default_rng(15); order=rng.permutation(n); splits=[(np.arange(52),np.arange(52,n))]+[(np.setdiff1d(np.arange(n),k),k) for k in np.array_split(order,4)]
def score(g):
 errs=[]
 for tr,te in splits:
  h=g[tr]; pen=.0003*np.eye(h.shape[1]); pen[0,0]=0; d=abs(g[te]@np.linalg.solve(h.T@h+pen,h.T@y[tr])-y[te]); errs.append([d[:,0].mean(),d[:,1:].mean()]);
 return np.round([errs[0],np.mean(errs[1:],axis=0),np.max(errs[1:],axis=0)],5).tolist()
def basis(z,t):
 d=[np.ones(n),z];
 if t in ('q','cub','rich','sep'): d.append(np.column_stack([np.prod(z[:,ind],axis=1) for ind in combinations_with_replacement(range(4),2)]));
 if t in ('cub','rich'): d.append(np.column_stack([np.prod(z[:,ind],axis=1) for ind in combinations_with_replacement(range(4),3)]));
 if t in ('sep','rich'): d += [np.sin(z),np.cos(z),np.tanh(z),z/(1+z*z),np.sin(2*z),np.cos(2*z)];
 return np.column_stack(d)
for label,z in [('raw',x),('invA',u)]:
 for t in ['lin','q','cub','sep','rich']: print(label,t,score(basis(z,t)))
np.savez('/tmp/probe.npz',u=u,x=x)
