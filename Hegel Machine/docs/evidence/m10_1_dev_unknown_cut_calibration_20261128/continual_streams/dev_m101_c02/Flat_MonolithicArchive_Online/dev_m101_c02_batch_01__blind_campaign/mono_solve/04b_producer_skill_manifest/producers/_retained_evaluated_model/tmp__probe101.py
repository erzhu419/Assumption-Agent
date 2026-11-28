import pickle,numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['eval_times']; tr=s['train']; x=tr['x']; z=tr['traj']; y=tr['y']; np.set_printoptions(precision=6,suppress=True)
F=lambda u:np.c_[np.ones(len(u)),u]
L=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
p=s['panel_a']; o=F(p['u']); v=p['v']; keep=np.ones(len(v),bool)
for _ in range(8):
 a=L(o[keep],v[keep]); e=np.linalg.norm(v-o@a,axis=1); keep=e<3*np.median(e)
xc=F(x)@a
_,sv,V=np.linalg.svd(z,full_matrices=False); print('temporal',V[:2]); shapes={'t':t,'t/(1+t)':t/(1+t),'exp1':-np.expm1(-t),'log1p':np.log1p(t),'t/(.5+t)':t/(.5+t),'exp2':-np.expm1(-2*t),'sqrt':np.sqrt(t),'t2':t*t,'sin':np.sin(t)}
for name,q in shapes.items():print('curve projection',name,np.linalg.norm(q-V[:2].T@(V[:2]@q))/np.linalg.norm(q))
from itertools import combinations
for err,n,m in sorted((np.linalg.norm(z-z@np.linalg.pinv(np.array([q,r]))@np.array([q,r]))/np.sqrt(z.size),n,m) for (n,q),(m,r) in combinations(shapes.items(),2))[:8]: print('pair',err,n,m)
def features(u,kind):
 a=F(u); pairs=[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]
 if kind=='aff':return a
 if kind=='quad':return np.c_[a,u*u,*pairs]
 if kind=='odd':return np.c_[a,u**3]
 if kind=='even':return np.c_[a,u*u,*pairs,u**4]
 if kind=='mix':return np.c_[a,u*u,*pairs,u**3]
 if kind=='trig':return np.c_[a,np.sin(u),np.cos(u),*pairs]
 if kind=='tanh':return np.c_[a,np.tanh(u),u*u,*pairs]
rng=np.random.default_rng(45); folds=[(np.arange(56),np.arange(56,72))]+[(i[16:],i[:16]) for i in [rng.permutation(72) for _ in range(5)]]
for pn in ('panel_b','panel_c'):
 p=s[pn]; u=p['u'];
 for vn,v in p.items():
  if vn=='u':continue
  for un,U in [('raw',u),('cal',F(u)@a)]:
   for k in ('aff','quad','odd','even','mix','trig','tanh'):
    Q=features(U,k); C=L(Q,v); h=np.sum(np.linalg.svd(Q,full_matrices=False)[0]**2,1); e=v-Q@C
    print(pn,vn,un,k,'LOO',np.mean(np.abs(e/(1-h[:,None]))),'resid',np.mean(abs(e)))
for nm,U in [('raw',x),('cal',xc)]:
 for k in ('aff','quad','odd','mix','trig','tanh'):
  A=features(U,k); metrics=[]
  for i,j in folds:
   pred=A[j]@L(A[i],np.c_[y,z][i]); er=abs(pred-np.c_[y,z][j]); metrics.append([er[:,0].mean(),er[:,1:].mean()])
  print('train regression',nm,k,np.round(metrics[0],5),np.round(np.max(metrics,axis=0),5))
