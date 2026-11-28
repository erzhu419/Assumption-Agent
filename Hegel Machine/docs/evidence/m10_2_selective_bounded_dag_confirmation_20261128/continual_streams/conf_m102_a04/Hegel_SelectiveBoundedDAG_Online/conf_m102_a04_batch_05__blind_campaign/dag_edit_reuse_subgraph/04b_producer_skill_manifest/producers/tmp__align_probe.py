exec(open('/tmp/probe_dag.py').read().split('for d in')[0])
def cyc(z): return z[:,[1,2,3,0]]
def bas(z): return np.c_[one(z),z*cyc(z),z**3]
f=bas(u); B=lin(f[keep],v[keep]); print('A sparse cubic',np.sqrt(np.mean((f[keep]@B-v[keep])**2)));print(np.round(B,5));
from scipy.optimize import root
Z=np.stack([root(lambda zi: (bas(zi[None])@B)[0]-xi,xi).x for xi in x]); print('inverse max',abs(bas(Z)@B-x).max(),Z.min(0),Z.max(0));np.save('/tmp/latent.npy',Z)
p=s['panel_c'];u=p['u'];v=p['v2'];w=u*u
for name,fn,bounds in [('rational',lambda w,q:w/(1+q*w),(.01,3)),('sin2', lambda w,q:np.sin(q*w),(.05,3)),('cos', lambda w,q:np.cos(q*np.sqrt(w)),(.05,4)),('log',lambda w,q:np.log1p(q*w),(.01,3)),('exp',lambda w,q:np.exp(-q*w),(.01,3)),('rational-diff',lambda w,q:w*(1-q*w)/(1+q*w),(.01,3))]:
 for add in (False,True):
  def F(w,q):return np.c_[np.ones(len(w)),fn(w,q),w] if add else np.c_[np.ones(len(w)),fn(w,q)]
  def r(q):
   a=F(w,q);return (a@lin(a,v)-v).ravel()
  q=least_squares(r,[.3],bounds=bounds,gtol=1e-13).x;print(name,add,q,np.mean(r(q)**2)**.5)
print('times',s['eval_times']); print('targets summary',np.quantile(s['train']['y'],[0,.1,.5,.9,1]), np.quantile(s['train']['traj'][:,-1],[0,.1,.5,.9,1]));print('traj singular vals',np.linalg.svd(s['train']['traj'],compute_uv=False))
