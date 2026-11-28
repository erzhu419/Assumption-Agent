import numpy as np,pickle,sys,importlib.util,itertools
from scipy.optimize import minimize_scalar,differential_evolution,least_squares
p='/tmp/hegel_continual_archive/conf_m102_b01_batch_05_solver.py';spec=importlib.util.spec_from_file_location('prev',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; train=s['train']; x=train['x']; Y=np.c_[train['y'],train['traj']]; A=m._aff; L=m._ls
np.set_printoptions(precision=5,suppress=True,linewidth=165)
def poly(x,k):
 return np.column_stack([np.prod(x[:,j],axis=1) for d in range(k+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),d)])
def cross(x): return m._cross(x)
def dead(x,a,b): return np.sign(x)*np.maximum(abs(x)-a,0)**b
ca=m._calibration(s['panel_a']); z=m._inverse(ca,x);print('calibration',ca);print('A errors',np.sort(np.linalg.norm((m._cal_basis(s['panel_a']['u'])*ca).sum(-1)-s['panel_a']['v'],axis=1)));print('trajectory singular',np.linalg.svd(Y[:,1:],compute_uv=False));print('y trajectory last corr',np.corrcoef(Y[:,[0,8]].T));print('cal latent extrema',z.min(0),z.max(0))
b=s['panel_b']; c=s['panel_c']; bu=b['u']; cu=c['u']; bv=b['v'];
for key,panel,arrs in [('B',b,['v']),('C',c,['v1','v2'])]:
 u=panel['u']
 for vkey in arrs:
  v=panel[vkey]
  for name,f in [('aff',A),('cross',cross),('aff+cross',lambda x:np.c_[A(x),cross(x)[:,1:]]),('dead',lambda x:A(dead(x,.35,1.5))),('quad',lambda x:poly(x,2))]:
   d=f(u); C=L(d,v); e=d@C-v; h=np.sum(np.linalg.qr(d)[0]**2,axis=1); print(key,vkey,name,'RMSE, LOO',np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/(1-h[:,None]))**2))); 
   if np.sqrt(np.mean(e**2))<.003:print(C)
def er(par):
 d=A(dead(bu,*par)); return np.mean((d@L(d,bv)-bv)**2)
fit=differential_evolution(er,[(0,.75),(.5,3)],seed=1,tol=1e-9); print('B dead fit',fit.x,fit.fun)
def B(z):return A(dead(z,*fit.x))@L(A(dead(bu,*fit.x)),bv)
w=B(z)
fs={'aff':A,'cross':cross,'aff+cross':lambda x:np.c_[A(x),cross(x)[:,1:]],'quad':lambda x:poly(x,2)}
lat={'raw':x,'cal':z,'B':w,'dead':dead(z,*fit.x)}
for v in ('v1','v2'):
 for n,f in fs.items():
  lat[v+n]=f(w)@L(f(cu),c[v]);lat['solo'+v+n]=f(z)@L(f(cu),c[v]);
for k,q in lat.items():
 print('CV',k,[(np.round(np.mean(abs(e[:,0])),5),np.round(np.mean(abs(e[:,1:])),5)) for e in [d[52:]@L(d[:52],Y[:52])-Y[52:] for d in (A(q),poly(q,2),np.c_[A(q),q[:,0]*q[:,2],q[:,1]*q[:,3]])]])
for v in ('v1','v2'):
 print('C coefs',v,L(poly(cu,2),c[v]));
