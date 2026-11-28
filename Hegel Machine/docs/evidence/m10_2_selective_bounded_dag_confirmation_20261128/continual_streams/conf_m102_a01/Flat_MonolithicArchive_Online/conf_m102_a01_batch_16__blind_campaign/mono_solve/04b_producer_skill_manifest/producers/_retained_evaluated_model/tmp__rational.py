exec(open('/tmp/grammar.py').read().split('_,_,V=')[0])
from scipy.optimize import least_squares
from scipy.linalg import qr
n=52
T=s['eval_times']; print('TIME',T)
V=np.linalg.svd(t[:n],full_matrices=False)[2]; a=t@V[0]; b=t@V[1]; print('mode2',V[1]); print('rho y,b,a',np.corrcoef(np.c_[y,a,b].T))
def pol(w,d):return np.stack([np.prod(w[:,i],1) for i in combinations_with_replacement(range(w.shape[1]),d)],1)
def metrics(label,p,y):return (label,*[round(np.mean(abs(p[i]-y[i])),5) for i in [slice(n),slice(n,None)]])
for wkey in ['x','z','h','bz','zc0','zc1']:
 w=W[wkey]; X=A(w); Q=np.c_[X,pol(w,2)]; print('FRAME',wkey)
 for r,tag in [(y,'y'),(a,'state')]:
  out=[]
  for d in [1,2,3]:
   P=np.concatenate([X]+[pol(w,i) for i in range(2,d+1)],1)
   for D in [X[:,1:],Q[:,1:]]:
    F=np.c_[P,-r[:,None]*D]; c=ls(F[:n],r[:n]); out.append(metrics((tag,d,D.shape[1],'lin'),(P@c[:P.shape[1]])/(1+D@c[P.shape[1]:]),r))
   if d==3:continue
   for mode in ['square','radial','positive','exp']:
    B=X if mode in ['square','exp'] else X[:,1:]; sq=mode!='exp'
    def design(p):
     q=B@p if mode in ['square','exp'] else (B*B)@(p*p)
     den=1+q*q if mode=='square' else 1+q if mode in ['radial','positive'] else np.exp(np.clip(q,-12,12)); return P/den[:,None]
    def fun(p):
     f=design(p)[:n]; return f@ls(f,r[:n])-r[:n]
    rng=np.random.default_rng(7381); starts=np.r_[np.zeros((1,B.shape[1])),rng.normal(0,.5,size=(12,B.shape[1]))]; rs=min((least_squares(fun,p,bounds=(-4,4),max_nfev=90) for p in starts),key=lambda f:f.cost); f=design(rs.x); out.append((*metrics((tag,d,mode),f@ls(f[:n],r[:n]),r),np.round(rs.x,2)))
  print(sorted(out,key=lambda q:q[1])[:5])
# Check dependence of residuals on the temporal modes, including saturation, normalization, and analytic products.
Q=np.c_[A(x),pol(x,2)]; ev=[]
for j in range(3):
 u=t@V[j]
 for F,k in [(u[:,None],'lin'),(np.c_[u,u*u,u*u*u,np.tanh(1.25*u)],'nonlin'),(A(x)*u[:,None],'interaction'),(Q*u[:,None],'all')]:
  P=np.c_[Q,F]; ev.append(metrics((j,k),P@ls(P[:n],y[:n]),y))
print('time readout',ev)
