exec(open('/tmp/probe.py').read().split("print('A row error'")[0])
from scipy.optimize import least_squares
from scipy.special import expit
from itertools import combinations_with_replacement

def affine(z): return np.c_[np.ones(len(z)),z]
def quad(z): return np.column_stack((affine(z),*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(z.shape[1]),2)]))
def err(f,g,Y):
 b=np.linalg.lstsq(f,Y[:len(f)],rcond=None)[0]; return np.abs(g@b-Y[len(f):]).mean(0)
for k in ['v1','v2']:
 Z=pc['u']; V=pc[k]
 f=quad(Z); B=np.linalg.lstsq(f,V,rcond=None)[0]; print('panel',k,'quadratic',np.round(B,3));
 for name,F in [('affine',affine(Z)),('odd3',np.c_[affine(Z),Z**3]),('full diagonal3',np.c_[affine(Z),Z**2,Z**3]),('radial3',np.c_[affine(Z),Z*np.sum(Z**2,axis=1)[:,None]]),('diagonal tanh',np.c_[affine(Z),np.tanh(1.25*Z)]),('quadratic',f)]:
  pin=np.linalg.pinv(F); R=F@(pin@V)-V; print(k,name,np.abs(R/(1-(F*pin.T).sum(1))[:,None]).mean())
# Compare a shared smooth switch, a saturating single index, and a quadratic baseline.
for zn,z in [('c',c),('u',u),('x',x),('v1quad',quad(u)@np.linalg.lstsq(quad(pc['u']),pc['v1'],rcond=None)[0]),('v2quad',quad(u)@np.linalg.lstsq(quad(pc['u']),pc['v2'],rcond=None)[0])]:
 lm=affine(z); D=quad(z); YY=np.c_[y,t[:,-1]]; raw=[]
 for fm in [0,1,2,3]:
  def feat(w):
   S=expit(lm@w)[:,None]; return [np.c_[lm,S],np.c_[lm,S*lm],np.c_[D,S],np.c_[D,S*lm]][fm]
  def resid(w):
   F=feat(w)[:52]; R=F@np.linalg.lstsq(F,YY[:52],rcond=None)[0]-YY[:52]; return np.r_[R.ravel(),0.002*w]
  rng=np.random.default_rng(23); opts=[least_squares(resid,w,max_nfev=125) for w in [np.zeros(5)]+list(rng.normal(0,3,(15,5)))]; best=min(opts,key=lambda a:a.cost); F=feat(best.x); E=err(F[:52],F[52:],Y); print(zn,fm,'train',round(best.cost,4),'val',np.round([E[0],E[1:].mean()],4),'w',np.round(best.x,3),flush=True)
