from m99_probe import *
from scipy.optimize import least_squares
n=len(tr['lab_id']); bridge=m._calibration(cal); X=m._features(tr,te); ct=m._coupon_inputs(tr,X[:n],bridge); taus,theta=m._fit(ct,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']);
G=m._features(tr,g)[n:]; cg=m._coupon_inputs(g,G,bridge); F,R=m._design(taus,cg,dt,t); obs=g['strain_curves_obs']-g['strain_0'][:,None]; rate=g['instant_rate_obs']; Y=np.c_[obs,rate]; base=np.c_[F@theta,R@theta]; P,H,D=cg; drift=D@theta[2:]; gap=R[:,:2]*taus; AB=gap*theta[:2]; n=len(H);
def basis(u): return np.vstack((np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1.]))
def report(name,e): print(name,'rms',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),'max_lab',max(np.sqrt(np.mean(e[g['lab_id']==lab,:-1]**2)) for lab in np.unique(g['lab_id'])))
report('baseline',Y-base)
for ratescale in (0.3,1.,3.):
 w=np.r_[np.ones(len(t)),ratescale]; y=Y*w
 def project(u):
  b=basis(u)*w[:,None]; q=np.linalg.lstsq(b,y.T,rcond=None)[0].T; return q,(y-q@b.T)
 for start in (taus,[.3,1.5],[1.,8.]):
  f=least_squares(lambda u:project(u)[1].ravel(),start,bounds=(.05,25),max_nfev=150); q,e=project(f.x); report(('free',f.x),e/w)
 q,e=project(taus); report('fixed',e/w)
 if ratescale==1.:
  print('effective amplitudes and errors by coupon: drift hold fast slow dA dB dD');print(np.c_[drift,H,AB,q-np.c_[AB,drift]])
  Z=np.c_[H,P,G,drift,drift**2,drift**3,AB,H*drift]
  for k in range(3): print('correlation parameter',k,np.array([np.corrcoef(z,(q-np.c_[AB,drift])[:,k])[0,1] for z in Z.T]).round(3))
print('direct residual decompositions [early late rate]')
E=(Y-base)[:,[0,-2,-1]]
for label,Z in [['drift',np.c_[drift]],['drift poly',np.c_[drift,drift**2,drift**3]],['features',G],['stress',np.c_[H,P]],['stress + drift',np.c_[H,P,drift]],['gap + drift',np.c_[AB,drift]],['gap + drift poly',np.c_[AB,drift,drift**2,drift**3]],['gap + drift + pair',np.c_[AB,drift,AB*drift[:,None],np.prod(AB,axis=1)]],['gap + features',np.c_[AB,G]]]:
 Z=np.c_[np.ones(n),Z]; c=np.linalg.lstsq(Z,E,rcond=None)[0]; r=E-Z@c; h=np.sum(np.linalg.qr(Z)[0]**2,axis=1); print(label,'LOO',np.sqrt(np.mean((r/(1-h[:,None]))**2,axis=0)).round(5),'coef',c.round(4).tolist())
