exec(open('/tmp/explore06.py').read().split('for nm,F')[0]); from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from scipy.special import erf
C=s['panel_c']; i,j=np.triu_indices(4,1)
def af(x):return np.c_[np.ones(len(x)),x]
def pair(x):return x[:,i]*x[:,j]
def calib(U,V,D):
 def basis(x,p,kind):
  k=np.exp(p[:4]); h=p[4:]; g=x*k+h
  if kind==0: return np.c_[af(np.tanh(g)),x*x]
  if kind==1: return np.c_[af(x),np.tanh(g)]
  if kind==2: return np.c_[af(np.tanh(g)),pair(x)]
  if kind==3: return np.c_[af(x),x*x,np.sin(g)]
  if kind==4: return af(np.tanh(g))
 best=None
 for kind in range(5):
  def resid(p):
   F=basis(U,p,kind); return (F@fit(F,V)-V).ravel()
  sol=min((least_squares(resid,np.r_[np.full(4,k),np.zeros(4)],bounds=(np.r_[np.full(4,-2),np.full(4,-2)],np.r_[np.full(4,2),np.full(4,2)]),max_nfev=90) for k in [-.3,.4,1]),key=lambda r:r.cost)
  F=basis(U,sol.x,kind); coef=fit(F,V); rms=np.sqrt(np.mean((F@coef-V)**2)); score=rms*(1+F.shape[1]/20); print('panel fit',kind,round(rms,7),np.round(sol.x,3),flush=True)
  if best is None or score<best[0]:best=score,basis(D,sol.x,kind)@coef
 return best[1]
Bv=calib(b['u'],b['v'],z); w1=calib(C['u'],C['v1'],z); w2=np.c_[af(z),z*z]@fit(np.c_[af(C['u']),C['u']**2],C['v2']);
np.savez('/tmp/coord06.npz',z=z,B=Bv,C=w1,D=w2)
fold=[np.arange(52,72),*np.array_split(np.random.default_rng(101).permutation(n),4)]; targets=np.c_[y,T];
def gram(F):return F@F.T
Fs={'z':z,'B':Bv,'C':w1,'D':w2,'BC':np.c_[Bv,w1],'BCD':np.c_[Bv,w1,w2],'all':np.c_[z,Bv,w1,w2]}; Ks={}
for name,F in Fs.items():
 F=(F-F.mean(0))/F.std(0); G=gram(F)/F.shape[1]; d=cdist(F,F,'sqeuclidean')/F.shape[1];
 for k,K in [('lin',1+G),('quad',(1+G)**2),('cube',(1+G)**3),*((f'rbf{l}',1+G+np.exp(-d*l)) for l in [.05,.2,.5,1.]),*((f'mat{l}',1+G+(1+np.sqrt(d*l))*np.exp(-np.sqrt(d*l))) for l in [.5,2.])]: Ks[name+k]=K
for l in [.05,.2,.5]:Ks['add'+str(l)]=sum(Ks[nm+'rbf'+str(l)] for nm in ['z','B','C','D'])/4
for name, F in [('z',z),('B',Bv),('C',w1),('D',w2)]:
 for h in [.6,1.2,2.5,4.]:
  grid=F[:,None,:]-F[None,:,:]; Ks[name+'separable'+str(h)]=1+gram(F)+np.exp(-h*grid**2).mean(2)
rec=[]
for nm,K in Ks.items():
 for r in [1e-5,1e-3,.01,.1]:
  E=[]
  for j in fold:
   ii=np.setdiff1d(np.arange(n),j); err=abs(K[np.ix_(j,ii)]@solve(K[np.ix_(ii,ii)]+r*np.eye(len(ii)),targets[ii],assume_a='pos')-targets[j]); E.append([err[:,0].mean(),err[:,1:].mean()])
  E=np.array(E);rec.append((nm,r,E))
for col in [0,1]:
 print('BEST',col)
 for name,r,E in sorted(rec,key=lambda a:.35*a[2][0,col]+.65*a[2][:,col].mean())[:18]:print(name,r,np.round(E[:,col],5))
