import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; X=np.c_[np.ones(len(u)),u]; np.set_printoptions(precision=4,suppress=True)
for k in ['v1','v2']:
 Y=p[k]; print(k,'ranges',Y.min(0),Y.max(0));
 for deg in (1,2,3):
  F=PolynomialFeatures(deg).fit_transform(u)
  pred=np.array([F[i]@np.linalg.lstsq(np.delete(F,i,0),np.delete(Y,i,0),rcond=None)[0] for i in range(len(u))]);print('poly LOO',deg,np.sqrt(np.mean((Y-pred)**2,axis=0)))
  if deg==2: print('quadratic coeffs',np.linalg.lstsq(F,Y,rcond=None)[0].T)
 for nm,g in [('sin',np.sin),('tanh',np.tanh),('softsign',lambda t:t/np.sqrt(1+t*t)),('rational',lambda t:t/(1+t*t)),('square',lambda t:t*t),('softplus',lambda t:np.logaddexp(0,t))]:
  fits=[]
  for y in Y.T:
   w=np.linalg.lstsq(X,y,rcond=None)[0]; best=None
   for a in [.3,.7,1.3]:
    for shift in [0,.5]:
     def fn(q,x):return q[0]+q[1]*g(x@q[2:]);
     q=least_squares(lambda q:fn(q,X)-y,np.r_[0,1,shift+a*w[0],a*w[1:]],max_nfev=300)
     if best is None or q.cost<best.cost:best=q
   J=best.jac; h=np.sum(J*np.linalg.pinv(J).T,axis=1); fits.append((np.sqrt(np.mean(best.fun**2)),np.sqrt(np.mean((best.fun/np.maximum(1-h,.03))**2)),best.x.round(3).tolist()))
  print(nm,fits)
