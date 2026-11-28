import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task');import batch_solver as bs
from scipy.optimize import least_squares,minimize_scalar
from scipy.linalg import solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]; N=52; D=bs._coordinates(s,t['x']); X=D['x']; L=bs._ls; k,h=s['telemetry_vec'][:2]
def report(p):
 e=np.mean(abs(p-Y[N:]),0); return [round(e[0],5),round(e[1:].mean(),5),round(e[-1],5)]
# Fit shared nonlinear panel B coordinates; look for independently identifiable scalar harmonics.
u,v=s['panel_b']['u'],s['panel_b']['v']; a=[]
for name,fun in [('sin',lambda x,f:np.sin(f*x)),('tanh',lambda x,f:np.tanh(f*x)),('rat',lambda x,f:x/(1+f*x*x)),('soft',lambda x,f:x/(1+f*abs(x))),('sinh',lambda x,f:np.sinh(f*x)),('gauss',lambda x,f:x*np.exp(-f*x*x))]:
 for ext in [0,1,2]:
  f=lambda x,p:bs._aff(np.c_[x,fun(x,p),x**3] if ext==1 else np.c_[x,fun(x,p),np.sin(h*x)] if ext==2 else np.c_[x,fun(x,p)])
  def loss(p):
   z=f(u,p); r=v-z@L(z,v); q=np.linalg.qr(z)[0]; return np.mean(abs(r/np.maximum(.001,1-(q*q).sum(1,keepdims=True))))
  g=np.linspace(.05,4,80); i=np.argmin([loss(p) for p in g]); p=minimize_scalar(loss,bounds=(g[max(i-1,0)],g[min(i+1,len(g)-1)]),method='bounded').x;a.append((loss(p),name,ext,p))
print('panel B',sorted(a)[:10]);
# Anisotropic additive kernels with independent penalization of calibrand blocks and interaction order.
ks=[]
for nm,z in D.items():
 z=z/np.maximum(z[:N].std(0),1e-6); dot=z@z.T/4; ds=(z[:,None,:]-z[None,:,:])**2
 bank={'lin':dot,'quad':(1+dot)**2/4,'cub':(1+dot)**3/8,'add3':np.mean((1+z[:,None,:]*z[None,:,:])**3,axis=2)/8}
 for l in [.55,1.,2.,4.]:
  U=np.exp(-ds/(2*l*l)); A=U.mean(2); B=(U.sum(2)**2-(U*U).sum(2))/12
  bank['add'+str(l)]=A;bank['pair'+str(l)]=(A+B)/2;bank['full'+str(l)]=np.prod(U,2)
 for mode,K in bank.items():ks.append((nm,mode,K/np.mean(np.diag(K)[:N])))
# Nested random splits for each composition, plus prescribed split reported separately.
rng=np.random.default_rng(110); splits=[(np.setdiff1d(np.arange(N),j),j) for j in np.array_split(rng.permutation(N),4)]; splits.append((np.arange(N),np.arange(N,len(Y))))
base=[(a+':'+b,c) for a,b,c in ks]; d={(a,b):c for a,b,c in ks}; modes=sorted({m for a,m,c in ks}); opts=base.copy()
for pair in [('x','b'),('x','c'),('x','d'),('x','bc'),('b','bc'),('b','c'),('b','d'),('bc','bd')]:
 for a in modes:
  for b in modes:
   for w in [.2,.8]: opts.append((str((pair,a,b,w)),w*d[pair[0],a]+(1-w)*d[pair[1],b]))
res=[]; I=np.eye(len(Y));
for nm,K in opts:
 for lam in [1e-6,1e-4,.003,.03]:
  E=[]
  for i,j in splits:
   Mu=Y[i].mean(0); P=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+lam*I[:len(i),:len(i)],Y[i]-Mu,assume_a='pos')+Mu; E.append(np.mean(abs(P-Y[j]),0))
  e=np.array(E);res.append((nm,lam,*e[:-1].mean(0)[[0,8]],*e[-1][[0,8]],e[-1,1:].mean(),K))
for c in [2,3,4,5,6]: print('top',c,[r[:-1] for r in sorted(res,key=lambda r:r[c])[:6]])
# Search sparse groups around univariate cubic effects, with separate ridge shrinkage for interactions.
for nm,Z in D.items():
 if nm not in ['x','b','c','bc','bd']:continue
 A=np.c_[X,Z] if nm!='x' else X
 for deg in [3,4,5,6]:
  uni=np.c_[*[A**j for j in range(1,deg+1)]]; poly=np.column_stack([A[:,i]*A[:,j] for i in range(A.shape[1]) for j in range(i+1,A.shape[1])]); P=np.c_[uni,poly]; P=(P-P[:N].mean(0))/np.maximum(P[:N].std(0),1e-6)
  for u in [.0001,.01,.1,1.]:
   K=P@P.T/100; K+=P[:,:uni.shape[1]]@P[:,:uni.shape[1]].T*u; y=Y[:N].mean(0); p=K[N:,:N]@solve(K[:N,:N]+.0005*np.eye(N),Y[:N]-y,assume_a='pos')+y
   if report(p)[0]<.105 or report(p)[1]<.06: print('poly',nm,deg,u,report(p))
