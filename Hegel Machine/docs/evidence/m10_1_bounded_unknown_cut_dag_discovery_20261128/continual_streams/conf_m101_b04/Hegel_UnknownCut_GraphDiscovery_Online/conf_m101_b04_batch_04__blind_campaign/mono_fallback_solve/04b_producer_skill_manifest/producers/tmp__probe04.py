import pickle,numpy as np
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=130)
a=s['panel_a']; u,v=a['u'],a['v']; A=lambda x:np.c_[np.ones(len(x)),x]
def mon(x,d):return np.column_stack([np.prod(x[:,i],axis=1) for i in cwr(range(4),d)])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
rng=np.random.default_rng(14)
fs={'aff':A(u),'diag3':np.c_[A(u),u**2,u**3],'poly2':np.c_[A(u),mon(u,2)],'odd-cubic':np.c_[A(u),mon(u,3)]}
for k,X in fs.items():
 best=None
 for _ in range(400):
  ix=rng.choice(len(u),X.shape[1]+1,replace=False)
  for __ in range(4):
   w=fit(X[ix],v[ix]); e=np.linalg.norm(v-X@w,axis=1); ix=np.argsort(e)[:39]
  loss=np.linalg.norm(e[ix]);
  if best is None or loss<best[0]:best=(loss,w,e)
 print(k,'trim loss',best[0],'residuals',np.sort(best[2])[::2]); print('coeffs',best[1][:13])
 if k=='aff':wa=best[1];ea=best[2]
b=s['panel_b']; wb=fit(A(b['u']),b['v']); x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; print('x range',x.min(0),x.max(0),'calibrated',((x-wa[0])@np.linalg.inv(wa[1:])).std(0))
from scipy.optimize import least_squares
T=s['eval_times'];
for k,f in {'exp':lambda t,r:-np.expm1(-t[:,None]*r),'negsin':lambda t,r:np.c_[np.sin(t*r[0]),-np.expm1(-t*r[1])],'expsin':lambda t,r:np.c_[-np.expm1(-t*r[0]),np.sin(t*r[1])], 'exp-int':lambda t,r:np.c_[-np.expm1(-t*r[0]),t]}.items():
 def res(r):
  B=f(T,r); return (B@fit(B,tr.T)-tr.T).ravel()
 r=least_squares(res,[1.5,.4],bounds=(.005,8)).x; print('time',k,r,np.sqrt(np.mean(res(r)**2)))
 for lab,z in {'raw':x,'a':(x-wa[0])@np.linalg.inv(wa[1:]),'ab':A((x-wa[0])@np.linalg.inv(wa[1:]))@wb}.items():
  B=f(T,r); M=np.c_[y,fit(B,tr.T).T]
  for n,F in {'aff':A(z),'sep':np.c_[A(z),z**2,z**3],'quad':np.c_[A(z),mon(z,2)],'cub':np.c_[A(z),mon(z,2),mon(z,3)]}.items():
   P=F[52:]@fit(F[:52],M[:52]); print(k,lab,n,np.round([abs(P[:,0]-y[52:]).mean(),abs(P[:,1:]@B.T-tr[52:]).mean()],5))
 if k=='exp':print('mode vs ab',fit(A(A((x-wa[0])@np.linalg.inv(wa[1:]))@wb),fit(f(T,r),tr.T).T))
