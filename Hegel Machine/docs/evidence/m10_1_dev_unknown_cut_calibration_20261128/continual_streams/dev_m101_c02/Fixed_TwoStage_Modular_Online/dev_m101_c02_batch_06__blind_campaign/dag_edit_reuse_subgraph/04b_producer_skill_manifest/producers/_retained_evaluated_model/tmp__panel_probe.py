import numpy as np,pickle,itertools
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True)
u,v=(p['panel_a'][k] for k in ['u','v']); print('ranges',u.min(0),u.max(0),p['train']['x'].min(0),p['train']['x'].max(0))
A=np.c_[np.ones(len(u)),u]; Q=np.array([u[:,i]*u[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]).T
for name,X in [('aff',A),('cubic-diag',np.c_[A,u**3]),('poly-diag',np.c_[A,u**2,u**3]),('quadratic',np.c_[A,Q]),('quad-cubic',np.c_[A,Q,u**3]),('tanh',np.c_[A,np.tanh(u)]),('sin',np.c_[A,np.sin(u)])]:
 def fit(X,v):
  w=np.ones(len(X));
  for _ in range(30):
   c=np.linalg.lstsq(X*w[:,None],v*w[:,None],rcond=None)[0]; e=np.linalg.norm(X@c-v,axis=1); w=1/np.sqrt(1+(e/max(.008,np.median(e)*2))**2)
  return c
 c=fit(X,v); e=np.linalg.norm(X@c-v,axis=1); keep=np.argsort(e)[:-3]; loo=[]
 for i in keep:
  m=np.delete(keep,np.where(keep==i)); loo.append(np.linalg.norm(X[i]@np.linalg.lstsq(X[m],v[m],rcond=None)[0]-v[i]))
 print(name,'r train/LOO median/max',np.median(e),np.median(loo),max(loo),'out',np.argsort(e)[-3:]);
 if name in ('quad-cubic','cubic-diag'): print(c)
x=p['train']['x']; Y=p['train']['traj']; print('traj svd',np.linalg.svd(Y-Y.mean(0),compute_uv=False)); X=np.c_[np.ones(len(x)),x]; print('traj raw mae',np.mean(abs(X@np.linalg.lstsq(X,Y,rcond=None)[0]-Y)))
