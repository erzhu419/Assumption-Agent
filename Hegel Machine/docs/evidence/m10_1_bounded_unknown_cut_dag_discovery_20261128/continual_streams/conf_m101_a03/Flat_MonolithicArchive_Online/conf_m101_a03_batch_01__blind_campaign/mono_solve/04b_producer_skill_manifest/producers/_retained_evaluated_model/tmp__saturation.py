import pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=S['train']['x']; y=S['train']['y']
def fit(A,b): return np.linalg.lstsq(A,b,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
pa=S['panel_a']; A=aff(pa['u']); h=fit(A,pa['v']); keep=np.argsort(np.linalg.norm(A@h-pa['v'],axis=1))[:41]; h=fit(A[keep],pa['v'][keep]); Z=(X-h[0])/np.diag(h[1:]); pb=S['panel_b']; B=fit(aff(pb['u']),pb['v']); W=aff(Z)@B
A=aff(Z); P=PolynomialFeatures(3).fit_transform(Z); P2=P[:,:15]; rng=np.random.default_rng(451); splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))];
for _ in range(5):
 q=rng.permutation(72); splits.append((q[:52],q[52:]))
def models(i,j,verbose=False):
 a=A[i]; target=y[i]; out={}
 def record(name,p,pars,b=None):
  out[name]=(np.mean(abs(p[j]-y[j])),np.std(p[i]-target));
  if verbose: print(name,np.round(out[name],6),np.round(pars,4),'' if b is None else np.round(b,3))
 d0=fit(a,target)*.2
 for k in (2,3):
  F=P2 if k==2 else P
  for shape in ('abs','square','smooth'):
   def den(d):
    return 1+(abs(d) if shape=='abs' else d*d if shape=='square' else np.sqrt(1+d*d)-1)
   def profile(c):
    G=F/den(A@c)[:,None]; b=fit(G[i],target); return G@b,b
   r=least_squares(lambda c: (profile(c)[0][i]-target),d0 if shape=='abs' else d0*2,max_nfev=200); p,b=profile(r.x); record(f'den-{k}-{shape}',p,r.x,b if k==2 else None)
  for shape in ('softsign','sqrt','tanh','atan'):
   def link(t,k):
    v=t*k
    if shape=='softsign': return t/(1+abs(v))
    if shape=='sqrt': return t/np.sqrt(1+v*v)
    if shape=='atan': return np.arctan(v)/k
    return np.tanh(v)/k
   def fun(c): return link(F@c[:-1],c[-1])
   r=least_squares(lambda c:fun(c)[i]-target,np.r_[fit(F[i],target),.13],bounds=(np.r_[np.full(F.shape[1],-np.inf),.00001],np.r_[np.full(F.shape[1],np.inf),.8]),max_nfev=180); record(f'link-{k}-{shape}',fun(r.x),r.x[-1:],r.x[:-1] if k==2 else None)
 for shape in ('abs','softsign','tanh','sqrt','sin'):
  def profile(c):
   v=A@c
   g=v*abs(v) if shape=='abs' else v/(1+abs(v)) if shape=='softsign' else np.tanh(v) if shape=='tanh' else v/np.sqrt(1+v*v) if shape=='sqrt' else np.sin(v)
   G=np.c_[P2,g]; b=fit(G[i],target); return G@b,b
  r=least_squares(lambda c:profile(c)[0][i]-target,d0*(3 if shape!='abs' else 1),max_nfev=200); p,b=profile(r.x); record(f'add-{shape}',p,r.x,b)
 for k in (2,3):
  F=P2 if k==2 else P; D=1+.2*abs(A@fit(a,target)); G=F/D[:,None]; record(f'fixed-{k}',G@fit(G[i],target),[])
 return out
G=[models(i,j,n==0) for n,(i,j) in enumerate(splits)]; print('MAE across splits; worst first:')
for name in sorted(G[0],key=lambda m:max(g[m][0] for g in G)):
 print(name,np.round([g[name][0] for g in G],5))
