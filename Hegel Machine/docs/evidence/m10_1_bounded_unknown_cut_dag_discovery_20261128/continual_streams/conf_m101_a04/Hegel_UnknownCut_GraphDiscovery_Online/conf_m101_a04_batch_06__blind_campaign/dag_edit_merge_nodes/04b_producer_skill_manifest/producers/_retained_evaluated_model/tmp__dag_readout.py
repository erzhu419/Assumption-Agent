exec(open('/tmp/probe.py').read().split('np.set_printoptions')[0])
from scipy.linalg import qr
from scipy.optimize import least_squares
np.set_printoptions(precision=4,suppress=True,linewidth=170)
def fit(A,Y):return np.linalg.lstsq(A,Y,rcond=1e-9)[0]
def cb(X):return np.c_[np.ones(len(X)),np.cos(2.4*X)]
c=cb(x)@fit(cb(u),p['v2']); x,b,c=x[:n],b[:n],c[:n]; Y=np.c_[y,tr]; k=52; T=s['eval_times']
base=np.c_[np.ones(n),b,c]; e=Y-base@fit(base,Y)
print('svd raw/base',np.linalg.svd(Y,compute_uv=False),np.linalg.svd(e,compute_uv=False)); print('temporal vectors',np.linalg.svd(tr,full_matrices=False)[2][:4]);print('base coef',fit(base,Y)[:,[0,1,4,8]])
D={}; Q=np.c_[x,b,c]; names=[f'{s}{i}' for s in 'xbc' for i in range(4)]
for i,(nm,w) in enumerate(zip(names,Q.T)):
 D[nm]=w
 for j in range(i+1):D[nm+'*'+names[j]]=w*Q[:,j]
 for f in (np.tanh,np.sin,np.cos):
  for g in (.7,1.,1.5,2.,2.4,3.):D[f'{f.__name__}({g}{nm})']=f(w*g)
 for g in (.5,1,2):D[f'{nm}/(1+{g}{nm}²)']=w/(1+g*w*w)
 for j in range(i):
  v=Q[:,j];m=names[j]
  for g in (1,2):
   for sgn in (-1,1):
    h=w+sgn*v
    for f in (np.tanh,np.sin,np.cos):D[f'{f.__name__}({g}({nm}{sgn:+d}{m}))']=f(g*h)
   for a,aa,bb in [(w,nm,m),(v,m,nm)]:D[f'{aa}/(1+{g}{bb}²)']=a/(1+g*(v if aa==nm else w)**2)
  D['tanh('+nm+'*'+m+')']=np.tanh(w*v)
  D['sin('+nm+'*'+m+')']=np.sin(w*v)
X=np.stack(list(D.values()),axis=1); names=list(D)
for label,Z in [('joint',Y),('scalar',Y[:,:1]),('mid',Y[:,[4]]),('late',Y[:,[-1]])]:
 S=np.ones((n,1)); selected=[]; A=Z[:k]/Z[:k].std(axis=0)
 for t in range(20):
  q=qr(S[:k],mode='economic')[0]; v=X[:k]-q@(q.T@X[:k]); vv=np.linalg.norm(v,axis=0); score=np.linalg.norm(v.T@A,axis=1)/np.maximum(vv,1e-8); score[vv<1e-7]=0; j=np.argmax(score); S=np.c_[S,X[:,j]];selected.append(names[j]); E=np.abs(S@fit(S[:k],Z[:k])-Z)
  if t in (2,5,9,14,19):print(label,t+1,np.round(np.r_[E[:k].mean(axis=0),E[k:].mean(axis=0)][[0,Z.shape[1]]] if label!='joint' else [E[:k,0].mean(),E[k:,0].mean(),E[:k,1:].mean(),E[k:,1:].mean()],4),selected[-5:])
# Test additive nonlinear ridge projections in calibrated v1/v2 blocks.
for tag,P in [('bc',np.c_[b,c]),('b',b),('c',c),('cos',np.cos(2.4*x))]:
 A=np.c_[np.ones(n),b,c]; W=P/P[:k].std(axis=0); d=W.shape[1]; A0=A[:k]; q=qr(A0,mode='economic')[0]; W0=W[:k]; rng=np.random.default_rng(8)
 for f in (np.sin,np.tanh,np.cos):
  best=[]
  for _ in range(8):
   def feat(z,w):return f(w@z[:-1]+z[-1]);
   def res(z):
    H=feat(z,W0);H=H-q@(q.T@H);return ((Y[:k]-q@(q.T@Y[:k]))-np.outer(H,(H@Y[:k])/max(H@H,1e-12))).ravel()
   r=least_squares(res,rng.normal(0,.55,d+1),max_nfev=100); B=np.c_[A,feat(r.x,W)];E=abs(B@fit(B[:k],Y[:k])-Y);best.append((np.linalg.norm(r.fun),E[k:,0].mean(),E[k:,1:].mean(),r.x))
  ans=min(best,key=lambda v:v[0]);print('ridge',tag,f.__name__,np.round(ans[:3],4),np.round(ans[3],3))
