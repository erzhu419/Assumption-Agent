import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; u,v=a['u'],a['v']; keep=np.max(abs(v-u),axis=1)<1
ls=lambda A,b:np.linalg.lstsq(A,b,rcond=None)[0]
D=np.array([ls(np.c_[np.ones(keep.sum()),u[keep,j]],v[keep,j]) for j in range(4)])
tr=s['train']; X=(tr['x']-D[:,0])/D[:,1]; T=tr['traj']; Y=tr['y']; t=s['eval_times'];
def poly(x,d):
 return np.column_stack([np.ones(len(x))]+[x[:,i].prod(1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def cv(A,H):
 return np.mean(abs(A[52:]@ls(A[:52],H[:52])-H[52:]),axis=0)
from sklearn.linear_model import lasso_path, MultiTaskLasso, OrthogonalMatchingPursuit
inds=[i for d in range(1,5) for i in combinations_with_replacement(range(4),d)]; A=poly(X,4)[:,1:]; names=np.array([''.join(str(j+1) for j in i) for i in inds]);
F=lambda p:-np.expm1(-t[:,None]*np.exp(p)); p=least_squares(lambda p:(F(p)@ls(F(p),T[:52].T)-T[:52].T).ravel(),[-.8,.5]).x; Z=ls(F(p),T.T).T; H=np.c_[Y,Z]; print('rates',np.exp(p),'correlation',np.corrcoef(H.T));
sc=A[:52].std(0); B=(A-A[:52].mean(0))/sc; hx=H-H[:52].mean(0)
for j in range(3):
 print('OUT',j)
 al,co,_=lasso_path(B[:52],hx[:52,j],alphas=np.geomspace(.5,.00005,100),max_iter=20000,tol=1e-8); results=[]
 for a,c in zip(al,co.T):
  k=abs(c)>1e-7; C=np.c_[np.ones(72),A[:,k]]; v=ls(C[:52],H[:52,j]); e=abs(C@v-H[:,j]); results.append((e[52:].mean(),e[:52].mean(),a,k.sum(),k,v))
 for r in sorted(results,key=lambda r:r[0])[:3]+[min([r for r in results if r[3]>=m],key=lambda r:r[3]) for m in [5,12,20,35]]:
  print(np.round(r[:4],6),list(zip(names[r[4]],r[5][1:].round(4))),round(r[5][0],4))
for a in [.1,.05,.02,.01,.005,.002,.001,.0003]:
 m=MultiTaskLasso(alpha=a,tol=1e-8,max_iter=20000).fit(B[:52],hx[:52]); k=np.linalg.norm(m.coef_,axis=0)>1e-7; C=np.c_[np.ones(72),A[:,k]]; v=ls(C[:52],H[:52]); out=C@v; print('joint',a,k.sum(),np.mean(abs(out[52:]-H[52:]),0),np.mean(abs(out[52:,1:]@F(p).T-T[52:])), names[k])
