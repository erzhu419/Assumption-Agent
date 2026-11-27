import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; R=c['bridge_raw']; L=c['true_channel_loads']; N=len(L)
idx=[j for k in range(4) for j in combinations_with_replacement(range(4),k)]
X=np.column_stack([np.prod(L[:,j],axis=1) if j else np.ones(N) for j in idx]); A=np.linalg.lstsq(X,R,rcond=None)[0]; v=np.linalg.svd(R-X@A,full_matrices=False)[2][0]; v*=np.sign(v[0]); P=np.eye(4)-np.outer(v,v)
np.set_printoptions(precision=5,suppress=True,linewidth=145)
print('nuisance direction',v,'sum',v.sum()); print('linear projected'); print(A[:5]@P)
for j,a in zip(idx[5:],A[5:]@P):
 b=a@P; co=b@P/np.diag(P); er=np.linalg.norm(b-co[:,None]*P,axis=1); k=np.argmin(er)
 if np.linalg.norm(b)>.0006: print('projected term',j,'fit output',k,'coefficient',co[k],'error',er[k], 'full',b)
y=R@v
for k in np.argsort(y): print('nuisance row',y[k].round(4),L[k].round(4))
for w in [v,np.ones(4)/2,np.array([1.,-1,1,-1])/2,*np.eye(4)]:
 q=L@w; Z=np.c_[np.ones(N),L,q*q,q**3,np.tanh(q)]; print('coordinate',w,'corr',np.corrcoef(q,y)[0,1],'approx rmse',np.std(y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]))
