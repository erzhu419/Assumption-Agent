import pickle,numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=6,suppress=True,linewidth=160)
fit=lambda X,Y: lstsq(X,Y)[0]
u,v=[s['panel_a'][k] for k in ('u','v')]
cols=[]; cs=[]
for i in range(4):
 j=(i+1)%4; a,b=u[:,i],u[:,j]; F=np.c_[np.ones(len(a)),a,b,a*b]; c=fit(F,v[:,i]);
 for _ in range(20):
  e=F@c-v[:,i]; good=np.argsort(abs(e))[:36]; c=fit(F[good],v[good,i]);
 e=v[:,i]-F@c; cols.append(e); cs.append(c); print('A cyclic',i,c,np.sort(abs(e))[[0,10,20,30,34,35,36,37,38,39,40,41,42,43,44,45,46,47]])
E=np.array(cols).T; good=np.linalg.norm(E,axis=1)<.25; print('A residual',np.c_[u[good],E[good]][:14]);print('A excluded',np.where(~good)[0]);
for k in ['panel_a','panel_b','panel_c']:
 d=s[k]; u=d['u']; base=np.c_[np.ones(len(u)),u]; cross=np.array([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]).T
 for name,y in d.items():
  if name=='u':continue
  keep=good if k=='panel_a' else np.ones(len(u),bool)
  errs=[]
  for tag,B in [('squares',u*u),('allquad',np.c_[u*u,cross]),('cube',np.c_[u*u,u**3]),('sin1',np.sin(u)),('tanh',np.tanh(u)),('cyc',np.array([u[:,i]*u[:,(i+1)%4] for i in range(4)]).T),('sin+cos',np.c_[np.sin(u),np.cos(u)])]+[(f'cos{f}',np.cos(f*u)) for f in [.5,1,1.25,1.5,2,2.5]]+[(f'rat{f}',u/(1+f*u*u)) for f in [.22,.5,1]]:
   X=np.c_[base,B][keep]; z=y[keep]; c=fit(X,z); e=X@c-z; h=np.sum(X*np.linalg.pinv(X).T,axis=1); errs.append((np.mean(abs(e/(1-h[:,None]))),np.mean(abs(e)),tag))
  print(k,name,sorted(errs)[:9])
  if k!='panel_a':
   for activation in ['identity','asinh','atanh']:
    if activation=='atanh' and np.max(abs(y))>=1:continue
    z={'identity':lambda y:y,'asinh':np.arcsinh,'atanh':np.arctanh}[activation](y)
    C=fit(np.c_[base,u*u],z);print(activation,'base+squares',np.mean(abs(np.c_[base,u*u]@C-z)),C[5:])
