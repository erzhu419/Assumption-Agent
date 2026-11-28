from sink_probe import *
from scipy.optimize import least_squares

folds=[np.arange(52,72)]+list(np.array_split(np.random.default_rng(30).permutation(N),6))

def stats(F,reg=None):
    out=[]
    for v in folds:
        a=np.setdiff1d(np.arange(N),v); X=F[a]; scale=np.maximum(X.std(0),0.1); X=X/scale
        c=np.linalg.lstsq(X,Y[a],rcond=1e-9)[0] if reg is None else np.linalg.solve(X.T@X+np.diag(reg),X.T@Y[a]);out.append(np.mean(abs(F[v]/scale@c-Y[v]),0))
    return np.array(out)

rows=[]
for k in ('xa','xab','z','raw'):
 x=d.get(k,s['train']['x']); B=poly(x,1); Q=poly(x,2)
 for freq in (0.8,1.2,1.33,1.6,2.,2.45,3.,4.):
  T=np.tanh(freq*x); C=np.cos(freq*x); S=np.sin(freq*x)
  for name,F in [('T/C lin',np.c_[B,T,C]),('T/C quad',np.c_[Q,T,C]),('T quadratic',np.c_[B,poly(T,2)[:,1:]]),('S quadratic',np.c_[B,poly(S,2)[:,1:]]),('T mixed',np.c_[B,poly(T,2)[:,1:],*[T[:,i]*x[:,j] for i in range(4) for j in range(4)]]),('S mixed',np.c_[B,poly(S,2)[:,1:],*[S[:,i]*x[:,j] for i in range(4) for j in range(4)]])]:
   for l in (.0001,.005,.05):
    m=stats(F,np.r_[np.full(5,l*.05),np.full(F.shape[1]-5,l)]); score=m[1:,0].mean()+.35*m[1:,0].max(); rows.append((score,k,freq,name,l,m[0,0],m[1:,0].mean(),m[1:,0].max(),m[0,1:].mean(),m[1:,1:].mean(),m[1:,1:].mean(1).max()))
for row in sorted(rows)[:10]: print('scalar',row[:5],np.round(row[5:],5))
for row in sorted(rows,key=lambda a:a[-2]+.35*a[-1])[:6]: print('traj',row[1:5],np.round(row[5:],5))

# Inspect the residual spectrum and fit a small number of free-direction waves
# against the scalar on training rows only.
x=s['train']['x']; B=poly(x,2); n=52; rng=np.random.default_rng(3)
for deg in (1,2):
 B=poly(x,deg)
 for family in ('tanh','cos','sin'):
  f={'tanh':np.tanh,'cos':np.cos,'sin':np.sin}[family]
  for m in (1,2,3):
   def matrix(q):
    q=q.reshape(m,5); return np.c_[B,f(x@q[:,:4].T+q[:,4])]
   def obj(q):
    F=matrix(q)[:n]; return F@np.linalg.lstsq(F,Y[:n,0],rcond=1e-8)[0]-Y[:n,0]
   ans=min((least_squares(obj,rng.normal(0,1.25,5*m),bounds=(-8.,8.),max_nfev=85) for _ in range(8)),key=lambda fit: np.mean(fit.fun**2)); F=matrix(ans.x); c=np.linalg.lstsq(F[:n],Y[:n,0],rcond=1e-8)[0]; print('free',deg,family,m,round(np.std(ans.fun),4),round(abs(F[n:]@c-Y[n:,0]).mean(),4),np.round(ans.x.reshape(m,5),2),np.round(c[-m:],3))
print('traj mode geometry',np.round(np.linalg.svd(Y[:,1:],full_matrices=False)[1],4),np.round(np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2],4))
