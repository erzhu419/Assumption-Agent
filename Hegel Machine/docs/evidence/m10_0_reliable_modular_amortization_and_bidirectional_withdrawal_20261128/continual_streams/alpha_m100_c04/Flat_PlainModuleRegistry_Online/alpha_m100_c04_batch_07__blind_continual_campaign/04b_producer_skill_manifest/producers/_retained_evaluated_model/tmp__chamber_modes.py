import runpy,contextlib,io,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/probe_chamber.py')
z=np.load('/tmp/chamber_probe.npz'); globals().update(dict(z)); Y=np.c_[y,yr]; w=np.r_[np.ones(len(t)),.75];
A=lambda tau: np.vstack([-np.expm1(-t[:,None]/np.atleast_1d(tau)),1/np.atleast_1d(tau)])
L=np.r_[t,1.]; tail=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; st=past@(-np.expm1(-dt[:,None]/taus)*np.exp(-tail[:,None]/taus)); G=(hold[:,None]-st)*theta[:2]
xqonly=p['m']._features(p['q'],p['q'])[:len(y)]
for name,x in [('deployment-ref',cov[:,1:]),('train-diagnostic-ref',xd),('diagnostic-only-ref',xqonly)]:
 d=theta[2]+x@theta[3:]; H=np.c_[G,d,hold,np.ones(len(d))];
 for j in [3,5]:
  h=H[:,:j]; beta=np.linalg.lstsq(h,Y,rcond=None)[0]; loo=(Y-h@beta)/(1-(h*np.linalg.pinv(h).T).sum(1))[:,None]; print(name,j,'LOO',np.sqrt(np.mean(loo**2,0))[[0,4,-2,-1]].round(5)); print('loadings',beta[:,[0,2,5,8,11,12]].round(4))
  for kind in ['fixed+one','free2','free2+linear']:
   def design(u):
    if kind=='fixed+one': return A(np.r_[taus,np.exp(u)])
    a=A(np.exp(u)); return np.c_[a,L] if kind.endswith('linear') else a
   def fit(u):
    a=design(u); b=np.linalg.lstsq(np.kron(h,a*w[:,None]),(Y*w).ravel(),rcond=None)[0].reshape(j,-1); return b,h@b@a.T-Y
   s=least_squares(lambda u:(fit(u)[1]*w).ravel(),np.log([1.6] if kind=='fixed+one' else [.7,1.6]),bounds=(np.log(.08),np.log(20)),max_nfev=60); b,e=fit(s.x); print(kind,np.exp(s.x).round(4),np.sqrt(np.mean(e**2,0))[[0,4,-2,-1]].round(5),b.round(4).tolist())
