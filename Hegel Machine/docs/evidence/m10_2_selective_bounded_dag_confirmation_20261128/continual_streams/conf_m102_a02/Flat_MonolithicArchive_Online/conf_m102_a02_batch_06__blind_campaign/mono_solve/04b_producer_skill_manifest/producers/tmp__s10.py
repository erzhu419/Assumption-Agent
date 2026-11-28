exec(open('/tmp/discover.py').read().split('forms=')[0])
from scipy.special import erf
u,v=s['panel_b']['u'],s['panel_b']['v']; Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; rng=np.random.default_rng(13); splits=[np.arange(72)]+[rng.permutation(72) for _ in range(4)];
def rat(z,p,q): return z*abs(z)**q/(1+p*z*z)
forms={'rat':rat,'rats':lambda z,p,q:z*abs(z)**q/(1+p*abs(z)**q),'tanhdiff':lambda z,p,q:z-np.tanh(p*z)/p,'sindiff':lambda z,p,q:z-np.sin(p*z)/p,'expdiff':lambda z,p,q:z*(-np.expm1(-p*abs(z)**q)),'tanhcube':lambda z,p,q:np.tanh(p*z*abs(z)**q),'atan':lambda z,p,q:np.arctan(p*z*abs(z)**q),'erf':lambda z,p,q:erf(p*z*abs(z)**q),'cubexp':lambda z,p,q:z*abs(z)**q*np.exp(-p*z*z),'poly':lambda z,p,q:z*abs(z)**q,'ratps':lambda z,p,q:z**3/(1+p*abs(z)**q),'rat3p':lambda z,p,q:z**3/(1+p*z*z)**q,'cubsin':lambda z,p,q:z**3*np.sinc(p*z/np.pi)**q}
results=[]
for n,g in forms.items():
 for base in [False,True]:
  def F(z,p,g=g,base=base):return np.c_[aff(z) if base else np.ones(len(z)),g(z,*p)]
  def e(p):
   A=F(u,p);return (A@ls(A,v)-v).ravel()
  p=min((least_squares(e,[a,b],bounds=([.005,.15],[5,6]),max_nfev=100) for a,b in [(1.25,2),(.22,2),(2.5,3)]),key=lambda r:r.cost).x; A=F(u,p);w=ls(A,v); E=A@w-v; H=np.einsum('ij,ji->i',A,np.linalg.pinv(A)); results.append((np.mean(abs(E)/(1-H[:,None])),n,base,p,lambda z,F=F,p=p,w=w:F(z,p)@w))
for r in sorted(results,key=lambda r:r[0])[:12]:print('B',r[:-1])
B=min(results,key=lambda r:r[0])[-1]; h=B(x); print('B at zero',B(np.zeros((1,4)))); print('sv',np.linalg.svd(Y,compute_uv=False));print('corr temporal',np.corrcoef(Y.T)[0]);print('times',t)
U=s['panel_c']['u']; cf=lambda q:np.c_[aff(q),q*np.exp(-1.0223*abs(q)**.9585)]; df=lambda q:np.c_[aff(q),q*q,np.exp(-.7109*q*q)]; C=lambda q:cf(q)@ls(cf(U),s['panel_c']['v1']); D=lambda q:df(q)@ls(df(U),s['panel_c']['v2']);
def score(A,lam):
 R=[]
 for idx in splits:
  tr,te=idx[:52],idx[52:]; S=np.maximum(np.linalg.norm(A[tr],axis=0),1e-9); F=A/S; w=ls(np.vstack([F[tr],np.sqrt(lam)*np.eye(F.shape[1])]),np.vstack([Y[tr],np.zeros((F.shape[1],9))])); E=abs(F[te]@w-Y[te]);R.append([E[:,0].mean(),E[:,1:].mean()]);
 return np.array(R)
R=[]
for bn,bb in [(r[1]+str(r[2]),r[-1]) for r in sorted(results,key=lambda r:r[0])[:5]]:
 h=bb(x); X={'x':x,'b':h,'c':C(x),'d':D(x),'bc':bb(C(x)),'bd':bb(D(x)),'cb':C(h),'db':D(h)}
 for n,z in X.items():
  for nn,w in X.items():
   if nn<n:continue
   q1,q2=quad(z),quad(w); m=(z[:,:,None]*w[:,None,:]).reshape(len(x),16); m0=m[:,::5]; cyc=np.c_[m0,z*np.roll(w,-1,axis=1),w*np.roll(z,-1,axis=1)]; sp=np.c_[q1,q2[:,1:],z**3,w**3];
   for k,F in [('add',np.c_[q1,q2[:,1:]]),('mix',np.c_[q1,q2[:,1:],m]),('cyc',np.c_[q1,q2[:,1:],cyc]),('sep3',sp),('ratform',np.c_[q1,q2[:,1:],z*w,z/(1+.22*w*w),w/(1+.22*z*z),z/(1+1.25*w*w),w/(1+1.25*z*z)]),('squareratform',np.c_[q1,q2[:,1:],z/(1+.22*w*w),w/(1+.22*z*z),z*z/(1+.22*w*w),w*w/(1+.22*z*z)])]:
    for l in [.00001,.0003,.003]:R.append((score(F,l),bn,n,nn,k,l))
for col in (0,1):
 for mode in (0,1):
  print('rank',col,mode,[(r[0][:,col].round(4),r[1:]) for r in sorted(R,key=lambda r:r[0][:,col].mean() if mode else r[0][0,col])[:5]])
