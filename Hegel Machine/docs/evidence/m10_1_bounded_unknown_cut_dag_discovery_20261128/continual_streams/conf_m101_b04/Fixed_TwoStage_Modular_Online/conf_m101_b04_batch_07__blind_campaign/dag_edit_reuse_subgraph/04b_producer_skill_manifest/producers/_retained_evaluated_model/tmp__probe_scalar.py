import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})
z,q,y,T=a['z_tr'],a['z_te'],a['y'],a['traj']; u,v=(a['panel_b'][k] for k in ('u','v')); t=a['eval_times']
from scipy.optimize import least_squares
from itertools import combinations_with_replacement

def fit(X,Y): return np.linalg.lstsq(X,Y,rcond=None)[0]
def basis(x,f): return np.column_stack((np.ones(len(x)),f))
def form(k,x,kind):
 s=x*k[0]
 fs={'sin':np.sin,'tanh':np.tanh,'rat':lambda s:s/(1+s*s),'atan':np.arctan}
 return fs[kind](s)+k[1]*x

def panel(kind,shared):
 def des(k,x): return basis(x,form(k,x,kind) if shared else np.column_stack((x,form([k[0],0],x,kind))))
 def resid(k):
  U=des(k,u); return (U@fit(U,v)-v).ravel()
from itertools import combinations
from scipy.optimize import minimize_scalar
opt=least_squares(lambda k: ( (D:=basis(u,np.sin(k[0]*u)+k[1]*u))@fit(D,v)-v).ravel(),[2.86,-2.52]); k=opt.x
D=lambda x: basis(x,np.sin(k[0]*x)+k[1]*x)
G=D(z)@fit(D(u),v)
print('t',t,'panel',k,'residual',np.sqrt(np.mean(opt.fun**2)))
s=52
for base_name,B in [('op',basis(z,G)),('free',np.c_[basis(z,G),z])]:
 A=B[:s]; Q=np.linalg.qr(A)[0]; rd=lambda F:F-Q@(Q.T@F)
 r=rd(y[:s]); cand=[]
 def score(F,nm):
  H=np.c_[B,F]; H1=H[:s]; w=fit(H1,y[:s]); e=y[:s]-H1@w; h=np.sum(np.linalg.qr(H1)[0]**2,axis=1)
  cand.append((np.mean(abs(e/np.maximum(1-h,1e-8))),np.mean(abs(H[s:]@w-y[s:])),np.std(e),nm,w[-F.shape[1]:]))
 for nm,X in [('G',G),('z',z),('both',np.c_[G,z])]:
  pool={}
  def add(name,a):
   if np.linalg.norm(rd(a[:s]))>1e-6: pool[name]=a
  for ij in combinations(range(X.shape[1]),2):
   i,j=ij; add(f'{i}*{j}',X[:,i]*X[:,j]);
  for i in range(X.shape[1]): add(f'{i}^2',X[:,i]**2)
  for ij in combinations(range(X.shape[1]),3): add(str(ij),np.prod(X[:,ij],axis=1))
  if nm!='both':
   for i in range(4):
    for j in range(i+1,4):
     for c in (-2,-1,-.5,.5,1,2):
      x=X[:,i]+c*X[:,j]
      for f in (np.sin,np.tanh):
       for scale in (.8,1.5,2.5,3.5): add(f'{f.__name__}({i}+{c}*{j})*{scale}',f(scale*x))
    for j in range(4):
     for c in (.5,1,2): add(f'{i}/(1+{c}*{j}^2)', X[:,i]/(1+c*X[:,j]**2))
  best=sorted(pool,key=lambda n:-abs(rd(pool[n][:s])@r)/np.linalg.norm(rd(pool[n][:s])))[:24]
  for i,n in enumerate(best):
   score(pool[n][:,None],(nm,n));
   for m in best[:i]:score(np.c_[pool[n],pool[m]],(nm,n,m))
 print(base_name,'dictionary ranked',sorted(cand,key=lambda a:a[0])[:10])
 for name,X in [('G',G),('z',z)]:
  for fn in ('sin','cos','tanh','rat','square','cube'):
   def f(p,x):
    w=x@p[:4]+p[4]; return {'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'rat':lambda w:1/(1+w*w),'square':np.square,'cube':lambda w:w**3}[fn](w)
   def err(p):
    v=rd(f(p,X[:s])); return v*(v@r/max(v@v,1e-16))-r
   rng=np.random.default_rng(2); opts=[least_squares(err,p,max_nfev=150) for p in rng.normal(size=(12,5))]; o=min(opts,key=lambda p:np.linalg.norm(p.fun)); score(f(o.x,X)[:,None],(name,fn,np.round(o.x,3))); print(base_name,'index',cand[-1])
