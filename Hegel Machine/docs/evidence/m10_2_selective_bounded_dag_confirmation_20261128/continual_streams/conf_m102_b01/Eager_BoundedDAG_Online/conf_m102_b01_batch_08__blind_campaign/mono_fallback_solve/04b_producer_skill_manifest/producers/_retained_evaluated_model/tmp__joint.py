import sys,pickle,numpy as np
from scipy.optimize import least_squares
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
sys.path.insert(0,'/holdout_task');import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; Y=np.c_[y,T]; a,z,c,d,b,f,p=m._routes(s,x); n=52; ls=m._ls
U,S,V=np.linalg.svd(T[:n]-T[:n].mean(0),full_matrices=False); amp=(T-T[:n].mean(0))@V[0];print('S',S,'y/amp',np.corrcoef(y,amp)[0,1]);print('y residual from all traj',np.mean(abs(m._aff(T[n:])@ls(m._aff(T[:n]),y[:n])-y[n:])),ls(m._aff(T[:n]),y[:n])); print('amp examples',np.round(np.c_[z,amp,y][np.argsort(amp)[::6]],3));
R={'x':x,'z':z,'c':c,'d':d,'b':b,'f':f,'p':p}; u,v=(s['panel_b'][j] for j in ('u','v')); k=s['panel_c']; u2=k['u'];
for tag,fn in [('tan',lambda x:np.arctan(2*x**3)),('tanh',lambda x:np.tanh(1.25*x**3)),('rat',lambda x:x**3/(1+.55*x**4)),('asinh',lambda x:np.arcsinh(1.25*x**3))]:
 F=lambda x:m._aff(fn(x)); q=ls(F(u),v); R['B'+tag]=F(z)@q; R['I'+tag]=(z-q[0])@np.linalg.pinv(q[1:]); print(tag,np.std(F(u)@q-v))
def std(w):return (w-w[:n].mean(0))/np.maximum(w[:n].std(0),1e-9)
def dot(w):return w@w.T/w.shape[1]
def poly(w,k):return std(np.column_stack([np.prod(w[:,ii],axis=1) for ii in combinations_with_replacement(range(w.shape[1]),k)]))
def group(w):
 w=std(w); W=dot(w); Q=dot(poly(w,2)); C=dot(poly(w,3)); D=cdist(w,w,'sqeuclidean')/w.shape[1]; return W,Q,C,D
K={}
for k,w in R.items():
 W,Q,C,D=group(w); K[k+'lin']=W;K[k+'q']=W+Q;K[k+'p3']=W+.5*Q+C;K[k+'int']=Q+C
 for l in [.35,.7,1.4,2.8,5.6]:
  d0=np.sqrt(5*D)/l; K[k+'m'+str(l)]=(1+d0+d0*d0/3)*np.exp(-d0);K[k+'r'+str(l)]=np.exp(-.5*D/l**2)
for k in [q for q in R if q.startswith(('B','I'))]+['b','c','d','p']:
 for l in [.7,1.4,2.8]:
  K['sum'+k+str(l)]=K['zm'+str(l)]+K[k+'m'+str(l)]; K['prod'+k+str(l)]=K['zm'+str(l)]*K[k+'m'+str(l)]; K['poly'+k+str(l)]=K['zp3']+K[k+'q']+.25*K['zlin']*K[k+'lin']
P=[];L=[]; names=[]
for k,G in K.items():
 for r in (1e-6,.0001,.001,.01,.04):
  g=G[:n,:n]+5+np.eye(n)*r; ff=cho_factor(g); inv=cho_solve(ff,np.eye(n)); B=inv@Y[:n]; L.append(Y[:n]-B/np.diag(inv)[:,None]); P.append((G[n:,:n]+5)@B);names.append((k,r))
L=np.array(L);P=np.array(P); ve=np.mean(abs(P-Y[n:]),axis=1); le=np.mean(abs(L-Y[:n]),axis=1)
for j in [0,1]:
 v=ve[:,0] if j==0 else ve[:,1:].mean(1); l=le[:,0] if j==0 else le[:,1:].mean(1); print('best',j,[(names[i],round(v[i],5),round(l[i],5)) for i in np.argsort(v)[:9]]);print('LOO',[(names[i],round(v[i],5),round(l[i],5)) for i in np.argsort(l)[:9]]);
 for k in [4,12,32,80]:
  i=np.argsort(l)[:k]; print('pool',j,k,np.mean(abs(P[i,:,0].mean(0)-y[n:])) if j==0 else np.mean(abs(P[i,:,1:].mean(0)-T[n:])));
# Fit a common nonlinear drive by sparse projection pursuit, and look for extrapolation failure separately.
from scipy.optimize import minimize
rng=np.random.default_rng(1334)
for key,w in [('z',z),('x',x),('zb',np.c_[z,b]),('zc',np.c_[z,c]),('zcd',np.c_[z,c,d]),('zB',np.c_[z,R['Btan']])]:
 w=std(w);D=w.shape[1]; target=np.c_[y,amp]; target=target/target[:n].std(0)
 vals=[]
 for fun in [np.tanh,np.arcsinh,np.sin]:
  for rank in (2,3,4):
   def F(t):return m._aff(np.c_[w,fun(w@t.reshape(D+1,rank)[:-1]+t.reshape(D+1,rank)[-1])]);
   def loss(t):
    a=F(t)[:n]; q=np.linalg.solve(a.T@a+.015*np.eye(a.shape[1]),a.T@target[:n]);return np.r_[(a@q-target[:n]).ravel(),.025*t,.05*q.ravel()]
   best=None
   for i in range(5):
    f0=least_squares(loss,rng.normal(0,.45,(D+1)*rank),max_nfev=100)
    if best is None or f0.cost<best.cost: best=f0
   a=F(best.x);q=ls(a[:n],Y[:n]);e=np.mean(abs(a[n:]@q-Y[n:]),0);vals.append((round(e[0],4),round(e[1:].mean(),4),fun.__name__,rank,round(best.cost,4)))
 print('PPR',key,vals,flush=True)
