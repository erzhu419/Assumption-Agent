import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.max(abs(v),axis=1)<4; u,v=u[keep],v[keep]; np.set_printoptions(precision=5,suppress=True,linewidth=180)
for j in range(4):
 a,b=u[:,j],u[:,(j+1)%4]; t=v[:,j];
 for deg in range(1,6):
  powers=[(d-q,q) for d in range(deg+1) for q in range(d+1)]; d=np.column_stack([a**p*b**q for p,q in powers]); beta=np.linalg.lstsq(d,t,rcond=None)[0]; r=t-d@beta; h=np.sum(d*np.linalg.pinv(d).T,axis=1); print(j,deg,'RMSE',np.std(r),'LOO',np.std(r/(1-h)),'beta',dict(zip(powers,beta.round(5))) if deg<=4 else '');
 print('univariate cubic + cross'); d=np.c_[np.ones(len(a)),a,a*a,a*a*a,b,a*b,a*a*b,a*a*a*b]; beta=np.linalg.lstsq(d,t,rcond=None)[0]; print(np.std(t-d@beta),beta)
