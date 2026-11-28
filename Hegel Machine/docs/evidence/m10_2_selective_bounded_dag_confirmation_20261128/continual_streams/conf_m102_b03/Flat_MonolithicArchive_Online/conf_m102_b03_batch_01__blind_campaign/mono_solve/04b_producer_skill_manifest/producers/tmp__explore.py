import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=135)
def F(x):return np.c_[np.ones(len(x)),x]
def fit(X,Y):return np.linalg.lstsq(X,Y,rcond=None)[0]
a=s['panel_a']; u=a['u']; v=a['v']; ids=np.arange(len(u))
for _ in range(5):
 A=fit(F(u[ids]),v[ids]); r=np.linalg.norm(v-F(u)@A,axis=1);ids=np.argsort(r)[:45]
print('A',A,'resid',np.sort(r));print('A residual samples', (v-F(u)@A)[ids[-8:]])
b=s['panel_b']; B=fit(F(b['u']),b['v']);print('B sv',np.linalg.svd(B[1:])[1])
c=s['panel_c'];print('C',[(k,np.round(v[:10],4)) for k,v in c.items()]);print('times',s['eval_times']);t=s['eval_times'];tr=s['train']['traj'];print('traj SVD',np.linalg.svd(tr)[1]);
for n in [1,2,3]:
 def h(p):return -np.expm1(-t[:,None]*p)
 def fun(p):
  H=h(p);return (H@fit(H,tr.T)-tr.T).ravel()
 q=least_squares(fun,np.geomspace(.4,2,n),bounds=(.001,20));print('rates',q.x,'MAE',np.mean(abs(q.fun)))
y=np.c_[s['train']['y'],tr]; x=s['train']['x']; z=(x-A[0])@np.linalg.inv(A[1:]); w=F(z)@B
for name,X in [('raw',x),('aligned',z),('B',w)]:
 for d in range(1,5):
  P=PolynomialFeatures(d).fit_transform(X); err=abs(P[52:]@fit(P[:52],y[:52])-y[52:]);print(name,'poly',d,err[:,0].mean(),err[:,1:].mean())
 D=cdist(X,X,'sqeuclidean');I=np.eye(52);P=PolynomialFeatures(1).fit_transform(X)
 for g in [.01,.04,.12,.35,1.]:
  K=np.exp(-g*D); best=[]
  for l in [1e-6,1e-4,1e-3,.01]:
   core=np.block([[K[:52,:52]+l*I,P[:52]],[P[:52].T,np.zeros((P.shape[1],P.shape[1]))]]); coef=np.linalg.solve(core,np.r_[y[:52],np.zeros((P.shape[1],y.shape[1]))]);err=abs(np.c_[K[52:,:52],P[52:]]@coef-y[52:]);best.append((err[:,0].mean(),err[:,1:].mean()))
  print(name,'kernel',g,best)
