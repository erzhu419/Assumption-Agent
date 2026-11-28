import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x,y,T=(s['train'][k] for k in ('x','y','traj')); t=s['eval_times']; np.set_printoptions(precision=6,suppress=True,linewidth=135)
lst=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def feat(x): return np.column_stack([np.ones(len(x)),x,*[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)]])
F=feat(x); beta=lst(F,y); residual=y-F@beta; cov=np.linalg.inv(F.T@F)*np.sum(residual**2)/(len(x)-15); print('raw beta/std',np.c_[beta,np.sqrt(cov.diagonal())]); print('y resid rms',np.std(residual))
Q=np.zeros((4,4))
for (i,j),v in zip(combinations_with_replacement(range(4),2),beta[5:]): Q[i,j]=Q[j,i]=v/(1+(i!=j))
print('eig yQ',np.linalg.eigh(Q))
print('traj affine coef',lst(F[:,:5],T)); print('resid full quadratic coef',lst(F,T)[5:]); print('traj affine residual rms',np.std(T-F[:,:5]@lst(F[:,:5],T),axis=0))
print('times',t)
for kind in ['twoexp','step_ramp','step_pulse','offset_twoexp']:
 def basis(p):
  a,b=p[:2]; t0=t+(p[2] if len(p)==3 else 0); f=-np.expm1(-a*t0); g=-np.expm1(-b*t0)
  if kind=='step_ramp': return np.array([f,t0-f/a])
  if kind=='step_pulse': return np.array([f,g-f])
  return np.array([f,g])
 def loss(p):
  b=basis(p); return (lst(b.T,T.T).T@b-T).ravel()
 p=least_squares(loss,[.8,2.5,0] if kind=='offset_twoexp' else [.8,2.5],max_nfev=1000).x
 print(kind,p,'rms',np.std(loss(p)), 'basis',basis(p))
