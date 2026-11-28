exec(open('/tmp/probe.py').read().split('U,D,V=')[0])
from scipy.special import erf
import itertools
fs={
 'sin':(lambda x,k:np.sin(k*x),[2.85,-.40]), 'tanh':(lambda x,k:np.tanh(k*x),[1.45,-.75]), 'erf':(lambda x,k:erf(k*x),[1.47,-.65]), 'atan':(lambda x,k:np.arctan(k*x),[2.,-.6]), 'rational':(lambda x,k:x/(1+k*x*x),[1.,-1.]), 'rational2':(lambda x,k:x/np.sqrt(1+k*x*x),[1.,-1.]),'sinplusatan':(lambda x,k:np.sin(k*x),[2.85,-.40]), 'cubic':(lambda x,k:x**3,[1.,.5]),'exponential':(lambda x,k:x*np.exp(-k*x*x),[1.,-1.])}
idx=[(0,2),(1,3)]
def ry(w):return np.c_[w,w[:,0]*w[:,2]+w[:,1]*w[:,3]]
def relax(d):
 q=np.linalg.qr(d[:52])[0]; tar=q.T@Y[:52,1:]
 def basis(p):return -np.expm1(-np.outer(t,p))
 def res(p):
  f=basis(p);return (f@fit(f,tar.T)-tar.T).ravel()
 f=basis(least_squares(res,[.42,1.8],bounds=(.03,8.)).x);return d[52:]@fit(d[:52],Y[:52,1:])@f@np.linalg.pinv(f)
for name,(fn,p0) in fs.items():
 for tied in ([True,False] if name!='sinplusatan' else [True]):
  def des(x,p):
   if name=='sinplusatan':return aff(x+p[1]*fn(x,p[0])+p[2]*np.arctan(p[3]*x))
   if tied:return aff(x+p[1]*fn(x,p[0]))
   return np.c_[aff(x),fn(x,p[0])]
  def rr(p):
   d=des(u,p);return (d@fit(d,v)-v).ravel()
  if name=='sinplusatan':p0=p0+[.1,1.25]
  if not tied:p0=p0[:1]
  sol=least_squares(rr,p0,max_nfev=500);p=sol.x; d=des(u,p);w=des(z,p)@fit(d,v)
  levels=[ry(w),aff(ry(w)),poly(w,2)]; errs=[abs(g[52:]@fit(g[:52],Y[:52,0])-Y[52:,0]).mean() for g in levels]; print(name,tied,p,'panel',np.std(rr(p)), 'y',np.round(errs,5),'traj',*[np.mean(abs(relax(g)-Y[52:,1:])) for g in [w,aff(w)]], 'y coef',fit(levels[0][:52],Y[:52,0]))
