from newgraph import *
from scipy.linalg import solve
D0=np.load('/tmp/refine.npz'); z,b,c,d=(D0[q] for q in ['z','b','c','d']); Q={'z':z,'b':b,'c':c,'d':d,'cz':C(z),'dz':D(z)}; Y=np.c_[tr['y'],tr['traj']]; ids=np.random.default_rng(223).permutation(52).reshape(4,13); idx=np.arange(52)
def ev(H,reg,j):
 H=(H-H[:52].mean(0))/np.maximum(H[:52].std(0),1e-8); P=np.empty_like(Y[:52]);
 for i in ids:
  k=np.setdiff1d(idx,i); X=H[k]; mu=X.mean(0); y=Y[k]; G=X-mu; coef=solve(G.T@G+reg*np.eye(G.shape[1]),G.T@(y-y.mean(0)),assume_a='pos'); P[i]=(H[i]-mu)@coef+y.mean(0)
 X=H[:52]; co=solve(X.T@X+reg*np.eye(X.shape[1]),X.T@(Y[:52]-Y[:52].mean(0)),assume_a='pos'); E=np.abs(H[52:]@co+Y[:52].mean(0)-Y[52:]); R=abs(P-Y[:52]); return (R[:,j].mean(),E[:,j].mean())
L=[]
for name,x in Q.items():
 extra={'q':a._quad(x),'cub':x**3,'ss':x*abs(x),'rat':x/(1+.22*x*x),'sin':np.sin(2.15*x),'tanh':np.tanh(2.15*x),'opp':opp(x)[:,None]}
 for n,y in Q.items():
  if name==n:continue
  extra[n]=y; extra[n+'q']=np.c_[y,a._quad(y)];extra[n+'cross']=np.c_[y,a._quad(y),x*y,x*np.roll(y,1,axis=1),x*np.roll(y,2,axis=1)];extra[n+'g']=x/(1+.22*y*y)
 for keys in [(),*((k,) for k in extra),*((k,h) for k,h in combinations(extra,2) if k in ('q','cub','ss','rat') or h in ('q','cub','ss','rat'))]:
  H=np.column_stack([x]+[extra[k] for k in keys]);
  for lam in [.0002,.01,.2]:
   for j in [0,slice(1,None)]:L.append((str(j),*ev(H,lam,j),name,keys,lam))
for j in ['0','slice(1, None, None)']:
 ll=[e[1:] for e in L if e[0]==j];print(j,'nested',sorted(ll)[:12]);print(j,'holdout',sorted(ll,key=lambda r:r[1])[:8])
