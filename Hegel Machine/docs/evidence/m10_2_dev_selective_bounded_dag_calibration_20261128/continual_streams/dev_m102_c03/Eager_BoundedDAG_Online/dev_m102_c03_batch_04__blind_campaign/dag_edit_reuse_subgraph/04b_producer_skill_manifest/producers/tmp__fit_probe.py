import numpy as np,pickle
from itertools import combinations,combinations_with_replacement
from scipy.linalg import solve
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X=tr['x']; Y=np.column_stack([tr['y'],tr['traj']]); t=s['eval_times']; Q=-np.expm1(-t[:,None]*[.425,1.832]); nr=len(X)
P=s['panel_a']; u=P['u']; v=P['v']; keep=(abs(v-u).max(1)<1); A=np.linalg.lstsq(np.c_[np.ones(sum(keep)),u[keep]],v[keep],rcond=None)[0]; print('robust panel A',keep.sum(),A,'resid',np.quantile(np.linalg.norm(v- np.c_[np.ones(len(u)),u]@A,axis=1),[0,.25,.5,.7,.8,1])); Z=(X-A[0])@np.linalg.inv(A[1:]);
for k in ('panel_b','panel_c'):
 p=s[k]; u=p['u']; F=np.c_[np.ones(len(u)),u,u*u,np.sin(u),np.cos(u),*[u[:,i]*u[:,j] for i,j in combinations(range(4),2)]]
 for v in p.keys()-{'u'}:
  print(k,v,'corr u',np.corrcoef(u.T,p[v].T)[:4,4:]); print('in sample wide poly rms',np.std(p[v]-F@np.linalg.lstsq(F,p[v],rcond=None)[0],axis=0))
print('x moments',X.mean(0),X.std(0),np.quantile(X,[0,.05,.5,.95,1],axis=0)); print('z moments',Z.mean(0),Z.std(0),np.quantile(Z,[0,.05,.5,.95,1],axis=0))
folds=[(np.arange(52),np.arange(52,72))]; folds.extend((np.setdiff1d(np.arange(nr),j),j) for j in np.array_split(np.random.default_rng(134).permutation(nr),4))
def evaluate(K,rg):
 out=[]
 for a,b in folds:
  yh=Y[a].mean(0); ka=K[np.ix_(a,a)]; kb=K[np.ix_(b,a)]; an=ka.mean(0); kn=ka-an[None,:]-an[:,None]+an.mean(); kv=kb-kb.mean(1)[:,None]-an[None,:]+an.mean(); p=kv@solve(kn+rg*np.eye(len(a)),Y[a]-yh,assume_a='pos')+yh; p[:,1:]=p[:,1:]@np.linalg.pinv(Q).T@Q.T
  e=abs(p-Y[b]); out.append([e[:,0].mean(),e[:,1:].mean()])
 return np.array(out)
R=[]
for tag,z in [('x',X),('z',Z),('tanhz',np.tanh(Z)),('sinz',np.sin(Z))]:
 D=cdist(z,z,'sqeuclidean'); D1=cdist(z,z,'cityblock')
 kernels={}
 for g in (.025,.05,.1,.2,.4,.8,1.6):
  kernels[f'rbf{g}']=np.exp(-g*D); h=np.sqrt(5*g*D); kernels[f'mat{g}']=(1+h+h*h/3)*np.exp(-h); kernels[f'add{g}']=sum(np.exp(-g*(zz[:,None]-zz[None,:])**2) for zz in z.T)
 for d in range(1,6):
  kernels[f'poly{d}']=(1+z@z.T/4)**d
 for name,K in kernels.items():
  for rg in (1e-7,1e-5,1e-4,.001,.01,.1):
   p=evaluate(K,rg); R.append((tag,name,rg,p[0],p[1:].mean(0),p[1:].max(0)))
for j in (0,1):
 for typ in (3,4):
  print('best',j,typ); print(*sorted(R,key=lambda r:r[typ][j])[:8],sep='\n')
