exec(open('/tmp/probe.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
from scipy.special import expit
z=(x-a[0])/a[1]; L=A(z); Q=P(z,2); rng=np.random.default_rng(98)
folds=[(np.arange(52),np.arange(52,72))]+[(v[:52],v[52:]) for v in (rng.permutation(72) for _ in range(5))]
funcs={'tanh':np.tanh,'rat':lambda g:g/(1+g*g),'sin':np.sin,'atan':np.arctan,'erf':None,'softsign':lambda g:g/np.sqrt(1+g*g),'sig-aff':expit,'rat-aff':lambda g:1/(1+g*g)}
from scipy.special import erf
funcs['erf']=erf
for mode,f in funcs.items():
 vals=[]
 for i,j in folds:
  def design(h,idx):
   v=f(L[idx]@h)[:,None]; return np.c_[Q[idx],v*L[idx] if mode.endswith('aff') else v]
  q,r=np.linalg.qr(Q[i]); R=np.eye(len(i))-q@q.T; ztar=R@y[i]
  def err(h):
   v=f(L[i]@h)[:,None]; v=R@(v*L[i] if mode.endswith('aff') else v); return v@fit(v,ztar)-ztar
  init=np.array([0.,.09,.26,-.02,1.]); starts=[k*init for k in (.4,.8,1.5,2.5,4,6)]+[np.r_[0.,h] for h in rng.normal(0,1,(5,4))]; sols=[least_squares(err,h,bounds=(-12,12),max_nfev=120) for h in starts]; h=min(sols,key=lambda o:o.cost).x; coef=fit(design(h,i),y[i]); vals.append((np.mean(abs(design(h,i)@coef-y[i])),np.mean(abs(design(h,j)@coef-y[j]))));
  if i[0]==0 and i[1]==1: print(mode,'h',h,'coef',coef,flush=True)
 print(mode,'folds',np.array(vals),flush=True)
