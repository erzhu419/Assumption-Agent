from chamber_data import *
from scipy.optimize import least_squares
z=h[:,None]-g; l=di['lab_id']; one=np.ones(len(h)); a,Bs,d=B.T; base=c.copy();base[:,0]=th[0]*h
Y=y-base@B.T
V={'hold':h,'zslow':z[:,1],'s*zslow':s*z[:,1],'strain0':di['strain_0']}
# Verification records omit coupon telemetry.
# Verification records omit coupon telemetry.
# Verification records omit coupon telemetry.
V.update({f'x{j}*s':q*s for j,q in enumerate(xx.T)})
V.update({f'p{j}*s':q*s for j,q in enumerate(p.T)})
V.update({f'ref{j}':q for j,q in enumerate(di['ref_coupons'].T)})
def ev(keys, constant=True, slope=True):
 cols=[s[:,None]*a,s[:,None]*d]+[q[:,None]*Bs for q in ([one] if constant else [])+([s] if slope else [])+[V[k] for k in keys]]
 A=np.stack(cols,axis=-1); e=np.empty_like(Y)
 for j in np.unique(l):
  m=l!=j; co=np.linalg.lstsq(A[m].reshape(-1,A.shape[-1]),Y[m].ravel(),rcond=None)[0];e[~m]=Y[~m]-A[~m]@co
 co=np.linalg.lstsq(A.reshape(-1,A.shape[-1]),Y.ravel(),rcond=None)[0];return np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),co
for baseline in [[],['s*zslow']]:
 rows=sorted((ev(baseline+[k])[0],k,ev(baseline+[k])) for k in V if k not in baseline)
 for row in rows[:12]:print(baseline,row)
