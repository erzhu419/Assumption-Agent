exec(open('/tmp/probe.py').read().split("print('A row error'")[0])
from scipy.optimize import minimize_scalar,least_squares
from scipy.special import expit
N=52

def aff(z): return np.c_[np.ones(len(z)),z]
def prod(z): return np.column_stack([z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(z.shape[1]),2)])
def score(f):
 B=np.linalg.lstsq(f[:N],Y[:N],rcond=1e-8)[0];e=abs(f@B-Y); return [e[:N,0].mean(),e[N:,0].mean(),e[N:,1:].mean()]

def candidates(z):
 L=aff(z); Q=np.c_[L,prod(z)];
 for k in [0.25,0.5,0.75,1,1.25,1.5,2,3,4,5]:
  for name,g in [('exp',z*np.exp(-k*z*z)),('tanh',np.tanh(k*z)),('sin',np.sin(k*z)),('rat',z/(1+k*z*z)),('cauchy',1/(1+k*z*z)),('hinge',np.maximum(abs(z)-k,0)*np.sign(z)),('power',np.sign(z)*abs(z)**k),('soft',np.logaddexp(0,k*z)/k)]:
   for j,f in enumerate([np.c_[Q,g],np.c_[L,g,prod(g)],np.c_[L,g,prod(z+g)],np.c_[Q,g,z*g],np.c_[L,prod(z),prod(g),g]]):
    yield score(f),name,k,j
print('UNIV',sorted(candidates(c),key=lambda a:max(a[0][1:]))[:12],flush=True)
Z=b['u']; V=b['v']; maps=[]
for name,g in [('exp',lambda z,k:z*np.exp(-k*z*z)),('tanh',lambda z,k:np.tanh(k*z)),('sin',lambda z,k:np.sin(k*z)),('rat',lambda z,k:z/(1+k*z*z)),('erf',lambda z,k:__import__('scipy.special',fromlist=['erf']).erf(k*z)),('atan',lambda z,k:np.arctan(k*z)),('softsign',lambda z,k:z/np.sqrt(1+k*z*z))]:
 def f(z,k): return np.c_[aff(z),g(z,k)]
 def loss(k):
  F=f(Z,k); B=np.linalg.lstsq(F,V,rcond=None)[0]; return np.std(F@B-V), B
 k=minimize_scalar(lambda k:loss(k)[0],bounds=(.05,4),method='bounded').x; sd,B=loss(k); C=f(u,k)@B; print('B',name,k,sd,score(np.c_[aff(C),prod(C)]),flush=True); maps.append((sd,C))
L=aff(c); Q=np.c_[L,prod(c)]; A=np.c_[y,t[:,-1]]; rng=np.random.default_rng(617)
for kind in ['tanhq','clipq','expq','ratq','softq','hingeq','squareq','cubeq','tanhmix','expmix','ratmix','power']:
 def make(w):
  h=Q@w[:15]; g={ 'tanh':lambda:np.tanh(h), 'clip':lambda:np.clip(h,-1,1), 'exp':lambda:np.exp(-h*h), 'rat':lambda:1/(1+h*h),'soft':lambda:np.logaddexp(0,h),'hinge':lambda:np.maximum(h,0),'square':lambda:h*h,'cube':lambda:h*h*h,'power':lambda:np.sign(h)*abs(h)**1.5}[kind.removesuffix('mix').removesuffix('q')](); return np.c_[Q,g] if not kind.endswith('mix') else np.c_[Q,g[:,None]*L]
 def fun(w):
  F=make(w)[:N]; return np.r_[(F@np.linalg.lstsq(F,A[:N],rcond=1e-8)[0]-A[:N]).ravel(),.005*w]
 w0=np.linalg.lstsq(Q[:N],y[:N],rcond=None)[0]; opts=[least_squares(fun,w,max_nfev=110) for w in [w0,2*w0]+list(rng.normal(0,1,(10,15)))]; best=min(opts,key=lambda a:a.cost); print(kind,round(best.cost,5),np.round(score(make(best.x)),5),np.round(best.x,3),flush=True)
