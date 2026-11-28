exec(open('/tmp/explore08.py').read().split('for name in')[0])
from scipy.optimize import minimize
from itertools import combinations_with_replacement as cwr
print('cal',m._calibration(s['panel_a'])); print('latent limits',z.min(0),z.max(0))
for q in (p,[.42,1.25],[.35,1.35]):
 h=m._warp(z,q); print('q',q,'state h',probe(aff(h),s['train']['traj']),'state B',probe(aff(aff(h)@b),s['train']['traj']))
 A=m._mixed(z,h); print('y scalar coefficients',np.linalg.lstsq(A[:52],y[:52],rcond=None)[0].round(3)); print('y all coeff',np.linalg.lstsq(A,y,rcond=None)[0].round(3))
 def ridge_profile(q):
  a=m._mixed(z,m._warp(z,q))[:52]; scale=np.sqrt(np.mean(a*a,0)).clip(.000001); a=a/scale; g=a.T@a; rhs=a.T@y[:52]; c=np.linalg.solve(g+.03*np.eye(len(g)),rhs)
  for _ in range(8):
   pen=.003/(np.abs(c)+.02)**2;pen[:9]*=.1;c=np.linalg.solve(g+np.diag(pen),rhs)
  return np.linalg.norm(a@c-y[:52]),c/scale
 r=minimize(lambda q:ridge_profile(q)[0],q,bounds=[(.01,.75),(.55,3.)],method='Nelder-Mead'); a=m._mixed(z,m._warp(z,r.x)); c=ridge_profile(r.x)[1]; print('adaptive',r.x,c.round(3),'mae',np.abs(a@c-y)[:52].mean(),np.abs(a@c-y)[52:].mean())
print('trajectory svd',np.linalg.svd(s['train']['traj'],compute_uv=False)); print('state coefficients extrap',np.linalg.lstsq(aff(m._warp(z,p)),s['train']['traj'],rcond=None)[0].round(3))
