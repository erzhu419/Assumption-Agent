import sys, pickle, numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
sys.path.insert(0,'/holdout_task')
from batch_solver import _panel_b,_quadratic,_relaxation,_fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; t=s['eval_times']; z=_panel_b(s['panel_b'],x); n=len(x)
p=_quadratic(z); q=_quadratic(x)[:,1:]; dx=cdist(x,x,'sqeuclidean'); dz=cdist(z,z,'sqeuclidean')
xt=s['test']['x']; print('test outside train marginal ranges:', ((xt<x.min(0))|(xt>x.max(0))).sum(0).tolist()); print('test in coordinate tails:', ((xt<np.sort(x,axis=0)[7])|(xt>np.sort(x,axis=0)[-8])).sum(0).tolist())
rng=np.random.default_rng(19530); random=[np.arange(n),*(rng.permutation(n) for _ in range(23))]; tails=[]
for j in range(x.shape[1]):
 ix=x[:,j].argsort()
 for b in (ix[:8],ix[-8:]): tails.append((np.setdiff1d(np.arange(n),b),b))
folds=[(i[:52],i[52:]) for i in random]+tails
specs=[(0,.075,.0003,np.inf),(0,.035,.0003,np.inf)]+[(w,.035,r,l) for w,r in [(1,.002),(0,.015)] for l in [0,.03,.15,.75,4,20,100]]
errs=[]
for a,b in folds:
 modes,B=_relaxation(T[a],t,[1.8,.41]); target=np.c_[y[a],modes]; row=[]
 for w,width,ridge,lam in specs:
  d=dx+w*dz; k=np.exp(-width*d[np.ix_(a,a)])+ridge*np.eye(len(a)); kt=np.exp(-width*d[np.ix_(b,a)])
  pa,pb=p[a],p[b]
  if np.isfinite(lam):
   scale=q[a].std(0); pa=np.c_[pa,q[a]/scale]; pb=np.c_[pb,q[b]/scale]
  L=cho_factor(k,lower=True); aa=solve_triangular(L[0],pa,lower=True); yy=solve_triangular(L[0],target,lower=True)
  if np.isfinite(lam) and lam>0:
   pen=np.zeros((q.shape[1],pa.shape[1])); pen[:,p.shape[1]:]=np.sqrt(lam)*np.eye(q.shape[1]); aa=np.vstack((aa,pen)); yy=np.vstack((yy,np.zeros((q.shape[1],3))))
  beta=_fit(aa,yy); pred=pb@beta+kt@cho_solve(L,target-pa@beta); e=np.abs(pred[:,1:]@B.T-T[b]); row.append([np.abs(pred[:,0]-y[b]).mean(),e.mean(),e[:,-1].mean()])
 errs.append(row)
e=np.array(errs); print('metrics per row: documented split, random mean, random max, tail mean, tail max; each y/state/terminal')
for j,sp in enumerate(specs): print(sp,np.round([e[0,j],e[:24,j].mean(0),e[:24,j].max(0),e[24:,j].mean(0),e[24:,j].max(0)],5).tolist())
