import sys,pickle,itertools as it,importlib.util
sys.path.insert(0,'/holdout_task')
import numpy as np
import node_00 as n
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; c=s['panel_c']; b=s['panel_b']; t=s['train']['traj']; y=s['train']['y']; raw=s['train']['x']; p=n._fit(a['u'],a['v']); u=n._invert(raw,p)
def aff(x):return np.c_[np.ones(len(x)),x]
def X(x,terms):return np.c_[aff(x),*[np.prod(x[:,q],axis=1) for q in terms]]
q=list(it.combinations_with_replacement(range(4),2)); cube=list(it.combinations_with_replacement(range(4),3)); sq=[(j,j) for j in range(4)]; adj=[(j,(j+1)%4) for j in range(4)]; cu=[(j,)*3 for j in range(4)]
B=np.linalg.lstsq(aff(b['u']),b['v'],rcond=None)[0]; A=lambda x:aff(x)@B; AI=lambda z:np.linalg.solve(B[1:].T,(z-B[0]).T).T
sp=importlib.util.spec_from_file_location('r','/holdout_task/_frozen_nodes/node_02.py'); r=importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
def score(name,z):
 out=r.run_node({'node_01':{'z_tr':z[:52],'z_te':z[52:],'y':y[:52],'traj':t[:52]}}); print(name, np.mean(abs(out['y_hat']-y[52:])),np.mean(abs(out['state_pred']-t[52:])),flush=True)
print('keys',list(s),'telemetry',s.get('telemetry_vec')); print('A fit',p)
for k in ('v1','v2'):
 for name,ts in [('squares',sq),('adj',adj),('own3',cu),('adj+own3',adj+cu),('sq+own3',sq+cu),('q+own3',q+cu)]:
  D=X(c['u'],ts); C=np.linalg.lstsq(D,c[k],rcond=None)[0]; h=np.sum(np.linalg.qr(D)[0]**2,1); err=np.mean(abs((D@C-c[k])/(1-h)[:,None])); print('panel',k,name,round(err,6));
  if name in ('adj+own3','sq+own3'):
   print(np.round(C[5:],4))
   F=lambda x:X(x,ts)@C
   for key,x in [('u',u),('z',A(u))]:
    score(k+name+key+' direct',F(x)); score(k+name+key+' affine out',A(F(x)))
# Check whether the dominant rank-one trajectory term is the only nonlinear part of the trajectory.
z=A(u); D=X(z,q); P=np.linalg.lstsq(D[:52],t[:52],rcond=None)[0]; U,S,V=np.linalg.svd(P[5:],full_matrices=False); print('trajectory quadratic singular values',S,'lead coefs',U[:,0]);
