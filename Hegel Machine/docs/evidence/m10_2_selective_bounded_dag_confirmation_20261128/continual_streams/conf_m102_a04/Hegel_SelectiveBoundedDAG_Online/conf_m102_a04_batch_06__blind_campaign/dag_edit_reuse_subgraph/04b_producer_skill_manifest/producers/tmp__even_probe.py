from dag_probe import *
from scipy.spatial.distance import cdist
z=mods[0]._align(s['panel_a'],s['train']['x']); t=s['eval_times']; b=mods[1]._time_basis(t,T[:,1:]); q=np.linalg.qr(b)[0]; Y=np.c_[T[:,0],T[:,1:]@q]; E=np.c_[z*np.roll(z,-1,axis=1),np.cos(2.4*z)];
for k in range(4): print('symmetry coordinate',k,'positive frac',np.mean(z[:,k]>0),'mean',z[:,k].mean())
print('singular b projection residual',np.linalg.svd(T[:,1:]-(T[:,1:]@q)@q.T,compute_uv=False)); print('basis',b[[0,-1]]);
F1=np.c_[np.ones(n),z]; F2=np.c_[np.ones(n),E]; F3=np.c_[F2,z]; c=lst(F3,Y); print('joint even/odd fit coefficients',c.round(3)); print('joint residual',np.mean(abs(F3@c-Y),0));
for name,U in [('mixed',W),('separated',E),('full even quadratic Z',np.c_[E,z[:,0]*z[:,2],z[:,1]*z[:,3]]),('linear signed Z',z),('separated and signed',np.c_[E,z])]:
 from sklearn.model_selection import RepeatedKFold
 A=np.c_[np.ones(n),U]; splits=[(np.arange(52),np.arange(52,n))]+list(RepeatedKFold(n_splits=4,n_repeats=5,random_state=11).split(T));
 for lam in (.0,3,12):
  es=[]
  for tr,va in splits:
   a=A.copy(); a[:,1:]=(a[:,1:]-a[tr,1:].mean(0))/np.maximum(.1,a[tr,1:].std(0)); coef=np.linalg.solve(a[tr].T@a[tr]+np.diag(np.r_[0,np.full(a.shape[1]-1,lam)]),a[tr].T@Y[tr]); e=a[va]@coef-Y[va]; es.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@q.T))]);
  print(name,lam,'official',np.round(es[0],4),'mean',np.round(np.mean(es[1:],0),4),'max',np.round(np.max(es[1:],0),4))
