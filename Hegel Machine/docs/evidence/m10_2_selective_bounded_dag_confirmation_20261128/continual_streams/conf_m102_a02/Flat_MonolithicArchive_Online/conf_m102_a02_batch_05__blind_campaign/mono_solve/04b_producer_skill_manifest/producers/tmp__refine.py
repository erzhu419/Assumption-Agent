from paths import *
from scipy.special import erf
np.set_printoptions(precision=5,suppress=True,linewidth=150)
u,v=[S['panel_b'][k] for k in ('u','v')]; z=cal(T['x']); tr=T['traj']; y=T['y']; t=S['eval_times']; print('t=',t)
funcs={'monom':lambda x,p:x**3, 'cubic5':lambda x,p:x**3+p[0]*x**5, 'cubic57':lambda x,p:x**3+p[0]*x**5+p[1]*x**7,'cubic-rat':lambda x,p:x**3/(1+p[0]*x*x),'cubic-root':lambda x,p:x**3/(1+p[0]*x*x)**p[1],'sincubic':lambda x,p:np.sin(p[0]*x)**3,'tanhcubic':lambda x,p:np.tanh(p[0]*x)**3,'gausscubic':lambda x,p:x**3*np.exp(-p[0]*x*x),'gauss':lambda x,p:x*np.exp(-p[0]*x*x),'sin':lambda x,p:np.sin(p[0]*x),'sin3pow':lambda x,p:np.sin(p[0]*x**3),'tanh3pow':lambda x,p:np.tanh(p[0]*x**3),'rat3pow':lambda x,p:x**3/(1+p[0]*x**6),'radial':lambda x,p:x**3/(1+p[0]*np.sum(x*x,axis=1)[:,None]),'cubirad':lambda x,p:x**3*np.exp(-p[0]*np.sum(x*x,axis=1)[:,None])}
models={};
for name,fn in funcs.items():
 p0={'cubic57':[-.7,.2],'cubic-root':[1,1],'cubic5':[-.4],'cubic-rat':[.5],'sincubic':[1],'tanhcubic':[1],'gausscubic':[.5],'gauss':[1],'sin':[2.8]}.get(name,[.6]);
 for mode in ('free','tied'):
  def model(x,p):
   return af(fn(x,p[:-1])+p[-1]*x) if mode=='tied' else np.c_[af(x),fn(x,p)]
  def residual(p):
   F=model(u,p); return (F@ls(F,v)-v).ravel()
  opt=least_squares(residual,p0+([0.] if mode=='tied' else []),max_nfev=150,gtol=1e-11);p=opt.x;F=model(u,p);M=ls(F,v); h=model(z,p)@M; A=af(h);Q=np.linalg.qr(F)[0];E=(F@M-v)/(1-(Q*Q).sum(1))[:,None]; e=abs(A[52:]@ls(A[:52],tr[:52])-tr[52:]).mean(); print(name,mode,p,'panel rmse/loo',np.sqrt(np.mean(opt.fun**2)),np.sqrt(np.mean(E**2)),'traj',e)
  models[name+mode]=(h,e,np.sqrt(np.mean(E**2)))
for fn in [lambda x:np.c_[af(x),x**2,x**3],lambda x:np.c_[af(x),x**2,x**3,x**4,x**5],lambda x:np.c_[af(x),x**3,x**5,x**7,x**9]]:
 F=fn(u);C=ls(F,v);print('polynomial',F.shape,np.sqrt(np.mean((F@C-v)**2)));print(C)
from itertools import product
for name,(h,e,ep) in sorted(models.items(),key=lambda kv:kv[1][1])[:9]:
 a=af(h); i,j=np.triu_indices(4); qq=h[:,i]*h[:,j]; cubic=np.column_stack([np.prod(h[:,k],axis=1) for k in combinations_with_replacement(range(4),3)]);print('readout',name)
 rng=np.random.default_rng(886);splits=[np.arange(72)]+[rng.permutation(72) for _ in range(10)]
 for tag,F in [('qx',np.c_[a,qq[:,i!=j]]),('q',np.c_[a,qq]),('q+d3',np.c_[a,qq,h**3]),('q+3',np.c_[a,qq,cubic])]:
  M=F.shape[1]; scores=[]
  for l in [0,.03,.15,.7]:
   es=[]
   for p in splits:
    it,iv=p[:52],p[52:]; R=np.diag(np.r_[np.zeros(5),np.full(min(10,M-5),l),np.full(max(0,M-15),l*2)]);w=np.linalg.solve(F[it].T@F[it]+R,F[it].T@y[it]);es.append(abs(F[iv]@w-y[iv]).mean());
   scores.append(np.round([es[0],np.mean(es[1:]),max(es[1:])],5))
  print(tag,scores)
