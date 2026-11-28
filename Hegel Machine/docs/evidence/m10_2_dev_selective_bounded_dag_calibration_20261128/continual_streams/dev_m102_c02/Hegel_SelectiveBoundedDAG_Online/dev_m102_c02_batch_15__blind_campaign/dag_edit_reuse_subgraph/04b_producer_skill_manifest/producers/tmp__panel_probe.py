import pickle, numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for k in ['panel_a','panel_b']:
 u,v=(s[k][i] for i in ['u','v']); print(k,'ranges',u.min(0),u.max(0),v.min(0),v.max(0))
 if k.endswith('a'):
  for j in range(4):
   x,y=u[:,j],v[:,j]; i,l=np.triu_indices(len(x),1); ok=abs(x[l]-x[i])>.15*np.ptp(x); a=np.median((y[l]-y[i])[ok]/(x[l]-x[i])[ok]); b=np.median(y-a*x); r=y-b-a*x
   print(j,'affine',b,a,'median',np.median(abs(r)),'resid abs sorted',np.round(np.sort(abs(r)),3))
  for terms in ['z','zn','zp','znp','full']:
   a=[];rr=[]
   rng=np.random.default_rng(42)
   for j in range(4):
    x,y=u[:,j],v[:,j]; w=u[:,(j+1)%4]; p=u[:,(j-1)%4]; d=np.c_[np.ones(len(u)),x];
    if 'n' in terms or terms=='full': d=np.c_[d,w,x*w]
    if 'p' in terms or terms=='full': d=np.c_[d,p,x*p]
    if terms=='full': d=np.c_[d,x*x,x**3]
    best=None
    for _ in range(300):
     ids=rng.choice(len(u),d.shape[1]+1,replace=False); beta=np.linalg.lstsq(d[ids],y[ids],rcond=None)[0]; r=abs(y-d@beta); loss=np.mean(np.sort(r)[:40]);
     if best is None or loss<best[0]: best=loss,beta
    a.append(best[1]); rr.append(abs(y-d@best[1]));
   print(terms,'median',np.round(np.median(rr,axis=1),5),'coeff',np.round(a,4))
 else:
  for mode in range(6):
   def des(a):
    c=np.cos(a*u); d=[c,[u,c],[c,np.sin(a*u)], [u,c,np.sin(a*u)], [u,c,np.cos(2*a*u)], [u,c,u**2]][mode]; return np.c_[np.ones(len(u)),*d] if isinstance(d,list) else np.c_[np.ones(len(u)),d]
   def err(a,loo=False):
    D=des(a); r=v-D@np.linalg.lstsq(D,v,rcond=None)[0]; h=np.sum(np.linalg.qr(D)[0]**2,axis=1); return np.mean(abs(r/(1-h[:,None]) if loo else r))
   z=minimize_scalar(err,bounds=(1.4,3),method='bounded'); print(mode,'at telemetry fit/LOO',err(2.15),err(2.15,True),'opt',z.x,z.fun,err(z.x,True));
   if mode in (0,1): print(np.linalg.lstsq(des(2.15),v,rcond=None)[0].round(4))
