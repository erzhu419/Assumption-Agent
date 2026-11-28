import numpy as np,pickle,importlib.util,itertools as it
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
p='/tmp/hegel_continual_archive/dev_m101_c04_batch_03_solver.py'; sp=importlib.util.spec_from_file_location('arch',p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; ls=m._ls; aff=m._affine
a,g=m._calibrate(s['panel_a']); u=(t['x']-a)/g; z=m._operator(s['panel_b'])(u); rng=np.random.default_rng(912); folds=[np.arange(72)]+[rng.permutation(72) for _ in range(5)]
np.set_printoptions(precision=4,suppress=True)
P=PolynomialFeatures(4); F=P.fit_transform(z); d=P.powers_.sum(1); names=P.get_feature_names_out(); B=F[:,d<=1]; Q=F[:,d==2]; print('quartic',[(k,round(v,3)) for k,v in zip(names,ls(F,y))]); print('poly singular tail',np.linalg.svd(F,compute_uv=False)[-8:]);
def test(fun):
 return np.round([np.mean(abs(fun(ix[:56])[ix[56:]]-y[ix[56:]])) for ix in folds],5)
for deg in [1,2,3]:
 for dd in [1,2,3]:
  N=F[:,d<=deg]; H=F[:,(d>0)&(d<=dd)];
  def pred(tr,ret=False):
   c=ls(np.c_[N,-y[:,None]*H][tr],y[tr]); den=1+H@c[N.shape[1]:]; return (c,den) if ret else N@c[:N.shape[1]]/den
  c,den=pred(np.arange(72),True); print('ratio',deg,dd,test(pred),'fit',np.std(pred(np.arange(72))-y),'den',den.min(),den.max(),'coef',c[N.shape[1]:])
# Variable projection: a single nonlinear index plus the archived affine/quadratic terms.
cross=np.c_[z[:,0]*z[:,2],z[:,1]*z[:,3]]
for label,h in [('tanh',np.tanh),('sin',np.sin),('rational',lambda w:w/(1+w*w)),('softsign',lambda w:w/np.sqrt(1+w*w)),('cubic',lambda w:w**3),('sigmoid',lambda w:np.tanh(w/2))]:
 for add in [cross,Q]:
  U=np.c_[B,add]; starts=np.r_[np.c_[np.zeros(4),np.eye(4)],np.c_[np.zeros(8),rng.normal(size=(8,4))]]
  def pred(tr,ret=False):
   def des(p): return np.c_[U,h(B@p)]
   def res(p):
    X=des(p)[tr]; return X@ls(X,y[tr])-y[tr]
   opt=min((least_squares(res,p,max_nfev=90,bounds=(-12,12)) for p in starts),key=lambda o:o.cost); X=des(opt.x); c=ls(X[tr],y[tr]); return (opt.x,c,np.std(X@c-y)) if ret else X@c
  print(label,add.shape[1],test(pred),pred(np.arange(72),True))
# Fixed ray grammars with a fitted scale, testing both additive and divisive placement.
rays=[]
for i in range(4):
 rays.append(np.eye(4)[i]);
 for j in range(i+1,4):
  for sg in [-1,1]: rays.append(np.eye(4)[i]+sg*np.eye(4)[j])
rays.extend([ls(B,y)[1:]/np.linalg.norm(ls(B,y)[1:])]); W=z@np.asarray(rays).T
for deg in [1,2,3]:
 N=F[:,d<=deg]; out=[]
 for j,w in enumerate(W.T):
  def calc(tr,ret=False):
   def des(p): return N/(1+(w*p[0]+p[1])[:,None]**2)
   def r(p):
    A=des(p)[tr];return A@ls(A,y[tr])-y[tr]
   opt=least_squares(r,[.6,0],bounds=([-8,-4],[8,4]),max_nfev=100); A=des(opt.x); return (np.sqrt(2*opt.cost/len(tr)),opt.x) if ret else A@ls(A[tr],y[tr])
  out.append((*calc(np.arange(72),True),j))
 print('ray ratios',deg,sorted(out,key=lambda r:r[0])[:5])
