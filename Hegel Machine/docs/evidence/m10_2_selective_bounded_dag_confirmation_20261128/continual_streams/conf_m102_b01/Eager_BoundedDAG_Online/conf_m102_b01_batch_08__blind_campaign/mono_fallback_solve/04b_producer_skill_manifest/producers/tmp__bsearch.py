import numpy as np,pickle,itertools,importlib.util
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=np.load('/tmp/operators.npz'); u,v=(s['panel_b'][k] for k in ('u','v')); X=d['X']; Z=d['Z']; Y=d['Y']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; aff=lambda a:np.c_[np.ones(len(a)),a]; roll=lambda a:np.roll(a,-1,axis=1)
lib={}
def f(u,k):
 a=aff(u)
 return {'linear':a,'sep3':np.c_[a,u*u,u**3],'odd3':np.c_[a,u**3],'odd5':np.c_[a,u**3,u**5],'cyc3':np.c_[a,u*u,u**3,u*roll(u),u*u*roll(u),u*roll(u)**2],'cyc2':np.c_[a,u*u,u*roll(u)],'cross3':np.c_[a,u**3,u*roll(u),u*u*roll(u),u*roll(u)**2],'pairs3':np.c_[a,u*u,u**3,*[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]],'triple':np.c_[a,u*u,u**3,*[np.prod(u[:,p],axis=1) for p in itertools.combinations(range(4),3)]]}[k]
for k in ['linear','sep3','odd3','odd5','cyc3','cyc2','cross3','pairs3','triple']:
 a=f(u,k);q=ls(a,v);h=(a*np.linalg.pinv(a).T).sum(1);e=a@q-v; print(k,'err',np.linalg.norm(e)/np.sqrt(e.size),'loo',np.mean(abs(e/(1-h)[:,None])),'sorted',np.sort(np.linalg.norm(e,axis=1)) if k in ('linear','sep3') else '');lib[k]=(lambda p,k=k,q=q:f(p,k)@q)
 if k=='sep3': print('qsep3',q)
for tag,fn in [('sinh',np.sinh),('asinh',np.arcsinh),('cub',lambda w:w+w**3),('sin',np.sin),('atan',np.arctan),('tanh',np.tanh)]:
 for k in ('linear','sep3','odd3','cyc2'):
  a=f(u,k);h=(a*np.linalg.pinv(a).T).sum(1); P=np.linalg.pinv(a)
  def loss(g):
   w=fn(v*g)/g; return np.mean(((a@(P@w)-w)/(1-h)[:,None])**2)/np.var(w)
  opt=minimize_scalar(loss,bounds=(.001,2.5),method='bounded');
  if opt.fun<.025: print('inv',tag,k,opt.x,opt.fun)
for tag,fn in [('cubic',lambda u,t:u+t*u**3),('sinh',lambda u,t:np.sinh(u*t)/t),('tanh',lambda u,t:np.tanh(u*t)/t),('sin',lambda u,t:np.sin(u*t)/t),('exp',lambda u,t:np.expm1(u*t)/t),('pair',lambda u,t:u+t*roll(u)*u),('square',lambda u,t:u+t*u*u)]:
 def err(t):
  a=aff(fn(u,t));e=a@ls(a,v)-v;return e.ravel()
 op=min([least_squares(err,np.full(4,t),bounds=(-5,5)) for t in [-1.,.1,1.,2.]],key=lambda a:a.cost); t=op.x; q=ls(aff(fn(u,t)),v); print('input',tag,'rms',np.sqrt(np.mean(op.fun**2)),t);lib[tag]=(lambda p,q=q,t=t,fn=fn:aff(fn(p,t))@q)
# Prior operator and alternate routes: no labels enter panel fitting.
sp=importlib.util.spec_from_file_location('prior','/tmp/hegel_continual_archive/conf_m102_b01_batch_00_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m); W={'prior':m._operator(s,X),'z':Z,'x':X};
for n,ff in lib.items():
 for nn,p in [('x',X),('z',Z),('c1',d['C1']),('c2',d['C22'])]:W[n+nn]=ff(p)
for n in ['C1','C22']: W[n]=d[n]
R=[]
for k,w in W.items():
 for fn in ['atan','asinh','inv3']:
  for typ in (0,1):
   for j,Yj in enumerate([Y[:,0],Y[:,1:]]):
    yh=m._predict(w[:72],Yj[:52],typ,fn); R.append((np.mean(abs(yh-Yj[52:])),j,k,fn,typ))
for j in (0,1): print('CV',j,sorted(r for r in R if r[1]==j)[:12]); print('prior',sorted(r for r in R if r[1]==j and r[2]=='prior')[:4])
np.savez('/tmp/routes.npz',**W)
