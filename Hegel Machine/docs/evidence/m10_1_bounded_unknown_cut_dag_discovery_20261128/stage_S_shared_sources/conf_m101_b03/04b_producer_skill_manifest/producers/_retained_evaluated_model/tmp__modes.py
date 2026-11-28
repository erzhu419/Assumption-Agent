import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe import *
from scipy.optimize import least_squares
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
q=base['ai']
def temporal(r,t=t): return -np.expm1(-t[:,None]*np.asarray(r))
def profile(r):
 h=temporal(r); return (h@ls(h,T[:n].T)-T[:n].T).ravel()
r=least_squares(profile,[.5,2],bounds=(.01,10)).x; H=temporal(r); M=ls(H,T.T).T; Z=np.c_[y,M]
if __name__=='__main__':
 print('rates',r,'Z quantiles',np.quantile(Z,[0,.05,.25,.5,.75,.95,1],axis=0)); print('Z data',np.c_[q,Z][::3]); print('Z correlations',np.corrcoef(np.c_[q,Z].T)); print('linear coefficients',ls(aff(q)[:n],Z[:n])); print('linear residual correlations',np.corrcoef((Z-aff(q)@ls(aff(q)[:n],Z[:n])).T));
 for name,p in [('ai',q),('aib',base['aib']),('x',x)]:
  for d in (1,2,3):
   pf=PolynomialFeatures(d); F=pf.fit_transform(p); b0=ls(F[:n],Z[:n]); print(name,d,'test',np.mean(abs(F[n:]@b0-Z[n:]),axis=0),'train',np.mean(abs(F[:n]@b0-Z[:n]),axis=0));
   if d==2: print(np.c_[pf.get_feature_names_out(),np.round(b0,4)])
 for j in (0,1,2):
  for d in (1,2,3):
   pf=PolynomialFeatures(d); F=pf.fit_transform(np.c_[q,np.delete(Z,j,axis=1)]); b0=ls(F[:n],Z[:n,j]); print('given other outputs',j,d,np.mean(abs(F[n:]@b0-Z[n:,j])))
