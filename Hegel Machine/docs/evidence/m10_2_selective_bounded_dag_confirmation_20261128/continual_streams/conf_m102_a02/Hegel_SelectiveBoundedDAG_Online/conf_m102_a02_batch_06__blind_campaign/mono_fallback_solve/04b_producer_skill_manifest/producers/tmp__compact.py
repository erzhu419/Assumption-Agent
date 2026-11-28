exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from sklearn.linear_model import Lasso
W=dict(np.load('/tmp/discover.npz')); N=len(y); I=np.arange(52); J=np.arange(52,N)

def fit(F,alpha):
 s=np.maximum(F[I].std(0),.03); F=F/s
 K=F@F.T+100.; return kr(K,alpha)
def kr(K,a):
 B=np.linalg.inv(K[np.ix_(I,I)]+a*np.eye(len(I))); c=B@y[I]; return np.mean(abs(c/np.diag(B))),np.mean(abs(K[np.ix_(J,I)]@c-y[J]))
R=[]
for nm in ['qb','bc','qbc','qbd','b','q','bd','bcd']:
 A=np.column_stack([W[k] for k in nm]); names=[k+str(i) for k in nm for i in range(4)]
 for dg in [1,2,3,4]:
  P=PolynomialFeatures(dg,include_bias=False); F=P.fit_transform(A)
  for a in np.geomspace(1e-4,30,14):R.append((*fit(F,a),nm,dg,a))
 for mix in [.25,1,4]:
  S=A.copy(); S[:,4:]*=mix; d=cdist(S,S,'sqeuclidean'); P=PolynomialFeatures(2,include_bias=False).fit_transform(A); P/=np.maximum(P[I].std(0),.1); K0=P@P.T+100
  for h in [.35,.6,1,1.8,3,5]:
   for w in [0.01,.1,1,10,100]:
    for a in [.001,.03,.3]:R.append((*kr(K0+w*np.exp(-d/(2*h*h)),a),nm,'gp',mix,h,w,a))
print('BEST REG LOO',sorted(R)[:12]); print('BEST REG HOLD',sorted(R,key=lambda a:a[1])[:12]);
# Profile low-dimensional smooth modulation of an affine/quadratic readout.
def pol(a):return PolynomialFeatures(2,include_bias=True).fit_transform(a)
def prof(f):
 X=f[I]; return np.linalg.lstsq(X,y[I],rcond=1e-9)[0]
from scipy.optimize import minimize
R=[]
for nm in ['q','b','c','d','qb','bc']:
 A=np.column_stack([W[k] for k in nm]);
 for gn in ['q','b','c','d']:
  G=W[gn]; G=np.c_[np.ones(N),G];
  for fm in ['rational','gate','addtanh','sine','addrat']:
   for deg in [1,2]:
    base=np.c_[np.ones(N),A] if deg==1 else pol(A)
    if base.shape[1]>20:continue
    def ff(p):
     g=G@p
     if fm=='rational':return base/(1+g[:,None]**2)
     if fm=='gate':return np.c_[base,(G[:,1:])/(1+g[:,None]**2)]
     if fm=='addtanh':return np.c_[base,np.tanh(g)]
     if fm=='addrat':return np.c_[base,1/(1+g*g)]
     return np.c_[base,np.sin(g)]
    def err(p):
     f=ff(p);return f[I]@prof(f)-y[I]
    rng=np.random.default_rng(4)
    for p in [np.zeros(5)+.1,*rng.normal(size=(4,5))]:
     o=least_squares(err,p,bounds=(-8,8),max_nfev=120); F=ff(o.x); E=F@prof(F)-y; R.append((np.mean(abs(E[I])),np.mean(abs(E[J])),nm,gn,fm,deg,o.x.tolist()))
print('BEST LAW TRAIN',sorted(R)[:14]);print('BEST LAW HOLD',sorted(R,key=lambda a:a[1])[:14]);
