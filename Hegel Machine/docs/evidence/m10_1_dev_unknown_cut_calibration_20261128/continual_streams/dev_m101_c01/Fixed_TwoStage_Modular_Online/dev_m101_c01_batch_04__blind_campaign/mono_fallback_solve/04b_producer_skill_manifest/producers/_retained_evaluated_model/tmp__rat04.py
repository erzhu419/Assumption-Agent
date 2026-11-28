from discover04 import *
from scipy.linalg import solve
from scipy.special import expit
A,B,C,D=fit_all(s); x=s['train']['x']; z=s['train']['traj']; y=s['train']['y']; t=s['eval_times']; u=aff(x)@B; rng=np.random.default_rng(91); splits=[(np.arange(56),np.arange(56,72))]+[(p[:56],p[56:]) for p in (rng.permutation(72) for _ in range(7))];
def ridge(F,Y,pen):
 return solve(F.T@F+np.diag(np.broadcast_to(pen,(F.shape[1],))**2),F.T@Y,assume_a='pos')
def mult(f,g):return (f[:,:,None]*g[:,None,:]).reshape(len(f),-1)
H=poly(x,2); M=poly(u,3)[:,15:]; G=np.c_[C(u),D(u)]; L=np.c_[H,G,mult(u,G),M]; J=np.c_[H,D(u),mult(u,D(u))]; K=np.c_[H,np.tanh(u),np.sign(u)*(-np.expm1(-1.8*abs(u))),np.sqrt(1+u*u)-1,M];
rows=[]
for i,j in splits:
 vals=[]
 for name,F,pen in [('base',L,.24),('alt',K,.14),('quad',H,.02),('mix',J,.015)]:
  vals.append((name,np.mean(abs(F[j]@ridge(F[i],y[i],pen)-y[j]))))
 for name,F,pen in [('Q',H,.015),('P',np.c_[H,M],.17),('L',L,.23),('K',K,.15)]:
  for kind in ['rank1','rank2','diag','poly','soft','asym']:
   W=aff(u); n={'rank1':5,'rank2':10,'diag':4,'poly':14,'soft':5,'asym':5}[kind]
   def den(p):
    if kind.startswith('rank'): return 1+np.sum((W@p.reshape(-1,5).T)**2,axis=1)
    if kind=='diag': return 1+(u*u)@(p*p)
    if kind=='poly': return .3+np.exp(np.clip(H[:,1:]@p,-4,4))*.7
    if kind=='soft': return 1+abs(W@p)
    return 1+np.maximum(W@p,0)**2
   def res(p,pred=False):
    d=den(p); T=F[i]/d[i,None]; w=ridge(T,y[i],pen); return F[j]@w/d[j] if pred else np.r_[T@w-y[i],pen*w,(.035 if kind=='poly' else .022)*p]
   starts=np.r_[np.zeros((1,n)),np.random.default_rng(4).normal(0,.45,(6 if name=='Q' else 3,n))]; b=min((least_squares(res,p,max_nfev=125,diff_step=1e-5) for p in starts),key=lambda a:a.cost); yh=res(b.x,True); vals.append((name+kind,np.mean(abs(yh-y[j]))));
   if len(rows)==0 and name=='Q': print('Q',kind,'coef',b.x.round(3),'fit',np.mean(abs(res(b.x)[:len(i)])))
 rows.append(dict(vals))
for k in sorted(rows[0],key=lambda k:np.mean([r[k] for r in rows])): print(k,np.round([r[k] for r in rows],5),'mean/worst',np.round([np.mean([r[k] for r in rows]),max(r[k] for r in rows)],5))
