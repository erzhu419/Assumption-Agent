exec(open('/tmp/dag.py').read().split('ops=')[0])
from itertools import product
from scipy.optimize import least_squares
N=52
Y=np.c_[y,ts@np.linalg.pinv(H.T)]
F0={'z':z,'L':L(z),'Q':Q(z),'R':R(z),'QL':Q(L(z)),'RL':R(L(z)),'QR':Q(R(z)),'RQ':R(Q(z))}
F0.update({'QR+':np.c_[F0['Q'],F0['R']], 'zr':np.c_[z,F0['R']], 'zq':np.c_[z,F0['Q']], 'Q*R':F0['Q']*F0['R']})
for k,y0 in enumerate(Y.T):
 res=[]
 for n,w in F0.items():
  for skip in [False,True] if k==0 else [False]:
   base=ones(Y[:,1:]) if skip else np.ones((len(y),1)); ry=y0[:N]-base[:N]@ls(base[:N],y0[:N]);
   ww=(w-w[:N].mean(0))/w[:N].std(0); w0=ww[:N]; d=w0.shape[1]; u=ls(w0,ry);u/=max(np.linalg.norm(u),1e-8)
   rng=np.random.default_rng(42)
   for fun,g in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('soft',lambda e:e/(1+abs(e))),('rat',lambda e:1/(1+e*e)),('lin',lambda e:e),('sq',lambda e:e*e)]:
    def ff(p,w0,b):return np.c_[b,g(w0@p[:-1]+p[-1])]
    def rr(p):
     f=ff(p,w0,base[:N]);return f@ls(f,y0[:N])-y0[:N]
    out=[]
    for u0 in [np.r_[u*sc,off] for sc,off in product([.55,1.6,3.5],[0,.75])] + [np.r_[rng.normal(0,sc,d),rng.uniform(-2,2)] for sc in [.5,1.25] for _ in range(3)]:
     a=least_squares(rr,u0,max_nfev=85);out.append((np.linalg.norm(a.fun),a.x))
    _,p0=min(out,key=lambda e:e[0]);ff0=ff(p0,ww,base);be=ls(ff0[:N],y0[:N]);e=abs(ff0@be-y0); coef=p0[:-1]/w[:N].std(0);res.append((e[:N].mean(),e[N:].mean(),n,fun,skip,np.r_[coef,p0[-1]-w[:N].mean(0)@coef],be))
 print('target',k,'best train',*sorted(res,key=lambda v:v[0])[:8],sep='\n');print('best test',*sorted(res,key=lambda v:v[1])[:5],sep='\n')
