import sys,pickle,itertools
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
sys.path.insert(0,'/holdout_task');import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];tr=s['train'];A=m._coordinates(s,tr['x']);Y=np.c_[tr['y'],tr['traj']];N=len(Y)
folds=[(np.arange(52),np.arange(52,72))]+[(q[:56],q[56:]) for q in [np.random.default_rng(i).permutation(N) for i in range(10)]]
for a,b in folds[:5]:
 p,q=m._predict(A[np.r_[a,b]],len(a),Y[a,0],Y[a,1:]);e=abs(np.c_[p,q]-Y[b]);print('fallback',np.round([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()],4))
S=A.std(0);X=A/S; i,j=np.triu_indices(12); prod=X[:,i]*X[:,j];K0=np.ones((N,N));K1=X@X.T;Q=prod@prod.T
D=cdist(X,X,'sqeuclidean')/12
res=[]
def score(K,l,tag):
 es=[]
 for a,b in folds:
  p=K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+l*np.eye(len(a)),Y[a],assume_a='pos');e=abs(p-Y[b]);es.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
 E=np.array(es);res.append((tag,E))
for h,l in itertools.product([0,.01,.05,.2,1,3],[.0001,.001,.004,.01,.03]):
 R=np.exp(-D/(2*1.8**2));score(K0+K1+Q*.08+h*R,l,('hyb',h,l))
for z,c,d in itertools.product([.35,1,2],repeat=3):
 scale=np.repeat([z,c,d],4); V=X*scale; P=V[:,i]*V[:,j]; dist=cdist(V,V,'sqeuclidean')/12; L=V@V.T; Q=P@P.T
 for l in [.0001,.001,.004,.015]:
  score(K0+L+.1*Q+.4*np.exp(-dist/5),l,('weighted',z,c,d,l))
for out in range(3):
 print('OUT',out)
 for k,e in sorted(res,key=lambda kv: .55*kv[1][:,out].mean()+.3*kv[1][:,out].max()+.15*kv[1][0,out])[:8]:print(k,np.round(e[0],4),np.round(e.mean(0),4),np.round(e.max(0),4))
# Determine whether scalar and trajectory need the same correction beyond the low-order fit.
F=np.c_[np.ones(N),X,prod]; G=F@F.T; p=solve(G+.004*np.eye(N),Y,assume_a='pos')/np.diag(solve(G+.004*np.eye(N),np.eye(N),assume_a='pos'))[:,None];print('LOO residual corr',np.round(np.corrcoef(p.T)[0],3));print('large scalar errors',np.c_[A, Y[:,0],p[:,0]][np.argsort(abs(p[:,0]))[-8:]].round(3))
