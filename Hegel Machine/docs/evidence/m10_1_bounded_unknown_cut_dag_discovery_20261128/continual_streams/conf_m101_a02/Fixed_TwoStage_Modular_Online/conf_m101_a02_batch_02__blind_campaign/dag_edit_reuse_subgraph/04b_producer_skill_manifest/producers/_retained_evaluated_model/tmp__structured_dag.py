import numpy as np, pickle, runpy, itertools
from scipy.optimize import least_squares
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; m=runpy.run_path(P+'_frozen_nodes/node_00.py'); z=m['run_node']({'__input__':s})['z_tr']; y=s['train']['y']; n=len(y)
rng=np.random.default_rng(11)
groups=[np.arange(52,72)]+[rng.permutation(n)[:20] for _ in range(8)]+[np.argsort(abs(a))[-12:] for a in z.T]; allidx=np.arange(n)
def folds(g): return np.setdiff1d(allidx,g),g
def design(terms): return np.column_stack([np.ones(n)]+[z[:,k].prod(1) for k in terms])
terms=[k for d in (1,2,3) for k in itertools.combinations_with_replacement(range(4),d)]
A=design(terms); coef=np.linalg.lstsq(A,y,rcond=None)[0]; print('cubic coefficients',list(zip(terms,np.round(coef[1:],4))))
Q=A[:,:15]; L=A[:,:5]
def fit(F,tr,te):
 c=np.linalg.lstsq(F[tr],y[tr],rcond=None)[0]; return np.mean(abs(F[te]@c-y[te]))
def cv(F):
 e=[fit(F,*folds(g)) for g in groups]; return np.round([e[0],np.mean(e[1:9]),max(e[1:9]),max(e[9:])],5)
for name,F in [('cubic',A),('mixed cubic',A[:,[0]+[i+1 for i,k in enumerate(terms) if len(k)<3 or len(set(k))>1]]),('pure cubic',np.column_stack([Q,z**3])),('triple',np.column_stack([Q,*[z[:,k].prod(1) for k in itertools.combinations(range(4),3)]]))]: print(name,cv(F))
dirs=[(str(i),z[:,i]) for i in range(4)]+[(f'{i}{sgn:+d}{j}',z[:,i]+sgn*z[:,j]) for i,j in itertools.combinations(range(4),2) for sgn in (-1,1)]
library=[]; names=[]
for name,u in dirs:
 for f in (.5,1.,1.5,2.,2.5,3.):
  for kind,v in [('sin',np.sin(f*u)-f*u),('cos',np.cos(f*u)-1+(f*u)**2/2),('tanh',np.tanh(f*u)-f*u),('rat',u/(1+f*u*u)-u),('gauss',np.exp(-f*u*u)-1+f*u*u)]: names.append((name,kind,f)); library.append(v)
 for j,w in enumerate(z.T):
  for f in (.5,1.,2.):
   for kind,v in [('crossrat',w/(1+f*u*u)-w),('crosssin',w*(np.sin(f*u)-f*u)),('crosscos',w*(np.cos(f*u)-1))]: names.append((name,kind,j,f)); library.append(v)
for k in terms[14:]: names.append(k); library.append(z[:,k].prod(1))
D=np.column_stack(library)
def greedy(B,tr,te,k):
 X=B.copy(); sel=[]
 for _ in range(k):
  U=np.linalg.qr(X[tr],mode='reduced')[0]; G=D[tr]-U@(U.T@D[tr]); norm=np.linalg.norm(G,axis=0); scores=abs(G.T@(y[tr]-U@(U.T@y[tr])))/np.maximum(norm,1e-9); scores[sel]=-1; j=scores.argmax(); sel.append(j); X=np.column_stack((X,D[:,j]))
 return fit(X,tr,te),sel
for B in (L,Q):
 for k in (1,2,3,4,6,8):
  e=[greedy(B,*folds(g),k)[0] for g in groups]; a,sel=greedy(B,allidx,allidx,k); print('greedy',B.shape[1],k, np.round([a,e[0],np.mean(e[1:9]),max(e[1:9]),max(e[9:])],5),[names[j] for j in sel])
def projected(kind,B,tr,te,k=1):
 def f(t):
  u=z@t.reshape(k,5)[:,:4].T+t.reshape(k,5)[:,4];
  if kind=='sin': H=np.sin(u)
  if kind=='tanh': H=np.tanh(u)
  if kind=='atan': H=np.arctan(u)
  if kind=='rat': H=u/(1+u*u)
  if kind=='gauss': H=np.exp(-u*u)
  return np.column_stack((B,H))
 def res(t):
  X=f(t)[tr]; return X@np.linalg.lstsq(X,y[tr],rcond=None)[0]-y[tr]
 gen=np.random.default_rng(62); best=(np.inf,None)
 for i in range(10):
  t=gen.normal(0,.9,size=5*k)
  r=least_squares(res,t,bounds=(-4,4),max_nfev=65)
  if r.cost<best[0]: best=(r.cost,r.x)
 return fit(f(best[1]),tr,te),best[1]
for kind in ('sin','tanh','atan','rat','gauss'):
 for B in (L,Q):
  ae,p=projected(kind,B,allidx,allidx); e=[projected(kind,B,*folds(g))[0] for g in groups]; print('ridge term',kind,B.shape[1],np.round([ae,e[0],np.mean(e[1:9]),max(e[1:9]),max(e[9:])],5),p.round(3))
