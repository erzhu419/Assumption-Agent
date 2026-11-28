import pickle,runpy,itertools
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; old=runpy.run_path('/tmp/hegel_continual_archive/dev_m101_c04_batch_00_solver.py'); tr=s['train']; y=tr['y']; z=old['_coordinates'](tr['x'],old['_calibration'](s)); c=old['_cross'](z); n=len(y)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
w=ls(np.c_[z,c],y); d=z@w[:4]; pairs=np.column_stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)]); one=np.ones(n); rng=np.random.default_rng(125); folds=[(np.arange(56),np.arange(56,n))]+[(p[:56],p[56:]) for p in [rng.permutation(n) for _ in range(4)]]; p=np.argsort(abs(d));folds.append((p[:56],p[56:]));
def tanh(x):return np.tanh(x)
def rat(x):return x/(1+x*x)
def sqrt(x):return x/np.sqrt(1+x*x)
def atan(x):return np.arctan(x)
for name,f in [('tanh',tanh),('rat',rat),('sqrt',sqrt),('atan',atan),('sin',np.sin)]:
 for stage in ['through','single','gated','general','shared','sharedplus']:
  out=[]; cp=[]
  for a,b in folds:
   w0=ls(np.c_[z[a],c[a]],y[a]); v=w0[:4]; norm=np.linalg.norm(v)
   def design(p,i):
    if stage.startswith('shared'):
     h=z[i]@p[:4]+p[4]*c[i]; return np.c_[h,f(h+p[5]),c[i]] if stage=='sharedplus' else np.c_[h,f(h+p[5])]
    h=z[i]@p[:4]+p[4]; F=f(h+(p[5]*c[i] if stage in ('through','general') else 0)); B=np.c_[one[i],z[i],c[i],F];
    if stage=='gated':B=np.c_[B,c[i]/(1+h*h),c[i]*np.tanh(h)]
    if stage=='general':B=np.c_[B,pairs[i]]
    return B
   def obj(p):
    D=design(p,a);return D@ls(D,y[a])-y[a]
   fits=[]
   for scale in (0.65,1.3,2.0):
    for gate in (-.35,0,.35) if stage in ('through','general') else (0.25,):
     p0=np.r_[v*scale,gate if stage.startswith('shared') else 0.,0. if stage.startswith('shared') else gate] if stage!='single' and stage!='gated' else np.r_[v*scale,0.];fits.append(least_squares(obj,p0,bounds=(-8,8),max_nfev=120))
   r=min(fits,key=lambda x:x.cost); w1=ls(design(r.x,a),y[a]); out.append(np.mean(abs(design(r.x,b)@w1-y[b])));cp.append((np.linalg.norm(r.fun)/np.sqrt(len(a)),np.round(r.x,4),np.round(w1,4)))
  print(name,stage,np.round(out,5),cp[0],flush=True)
