from wallstudy import *
from scipy.optimize import root
c=fit(0,good); print('cal rmse',np.sqrt(np.mean((pred(ref[good],0,c)-raw[good])**2)))
y=np.array([[root(lambda x: pred(x,0,c)-r,r).x for r in p] for p in yy]).reshape(3,-1)
Z=s['Z_nuisance']; Q=np.linalg.qr(Z)[0]; print('Z characteristics',Z[:6].round(4),np.linalg.svd(Z,compute_uv=False)); print('aux mean',s['aux_manifold_telemetry'].mean(0)); print('sensor coords',ij.tolist()); print('pulses keys',[list(p) for p in pulses]);
def resid(z):return w@forward(z[:4],z[4:] if len(z)>4 else (1,)*4).T-y
def project(r):return r-(r@Q)@Q.T
def rm(r):return np.sqrt(np.mean(r*r,axis=-1)).round(8)
for z in [(0,)*4,(1,)*4]:
 r=resid(z); print('wall',z,'plain',rm(r),'profiled',rm(project(r)),'Z coeff',np.linalg.lstsq(Z,r.T,rcond=None)[0].T.round(5)); print('plain shift',rm(r.reshape(3,3,-1))); print('projected extremes',np.max(np.abs(project(r)),1));
for n in (4,8):
 for test in (-1,0,1,2):
  take=np.arange(3)!=test
  sol=least_squares(lambda z:project(resid(z))[take].ravel(),np.r_[[.1]*4,[1.]*(n-4)],bounds=(np.r_[[-.5]*4,[.1]*(n-4)],np.r_[[5.]*4,[8.]*(n-4)]),gtol=1e-12,ftol=1e-12,xtol=1e-12)
  print('profile fit',n,test,sol.x.round(7),rm(project(resid(sol.x))))
