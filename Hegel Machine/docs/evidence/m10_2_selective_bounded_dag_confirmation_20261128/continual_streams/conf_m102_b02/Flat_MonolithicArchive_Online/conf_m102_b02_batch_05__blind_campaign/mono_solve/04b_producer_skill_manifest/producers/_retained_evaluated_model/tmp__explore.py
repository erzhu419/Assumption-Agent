import pickle,itertools
import numpy as np
from scipy.special import erf
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def I(u):return np.column_stack((np.ones(len(u)),u))
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def poly(u,d):return np.column_stack([np.prod(u[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),d)])
fns={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'erf':erf,'asinh':np.arcsinh,'atan':np.arctan,'exp':np.exp,'abs':np.abs,'sq':np.square,'cube':lambda x:x**3,'rational':lambda x:x/(1+x*x),'softabs':lambda x:np.sqrt(1+x*x),'logquad':lambda x:np.log1p(x*x)}
operators={}
for key,out in [('a','v'),('b','v'),('c','v1'),('c','v2')]:
 p=s['panel_'+key]; u=p['u']; v=p[out]; a=I(u); take=np.arange(len(u));
 if key=='a':
  r=np.linalg.norm(a@ls(a,v)-v,axis=1); take=np.sort(np.argsort(r)[:-3]); print('a excluded',np.setdiff1d(np.arange(len(u)),take));u,v=u[take],v[take];a=I(u)
 def score(d):return np.sqrt(np.mean((d@ls(d,v)-v)**2))
 print(key,out,'linear',score(a),'poly2',score(np.c_[a,poly(u,2)]),'diag cubic',score(np.c_[a,u*u,u**3]),'pure linear fit',ls(a,v))
 funcs=[]
 for name,fn in fns.items():
  def D(w,q,fn=fn):return np.c_[I(w),fn(w*np.exp(q))]
  def obj(q):d=D(u,q);return (d@ls(d,v)-v).ravel()
  fs=[least_squares(obj,np.full(4,np.log(g)),bounds=(-2,2),max_nfev=75) for g in (.5,1.3,2.8,4.5)]
  f=min(fs,key=lambda f:f.cost); funcs.append((score(D(u,f.x)),name,np.exp(f.x),D,f.x))
 funcs.sort(key=lambda x:x[0]); print('component nonlinear',[(r,n,q) for r,n,q,*_ in funcs[:6]])
 r,n,q,D,par=funcs[0]; co=ls(D(u,par),v); operators[key+out]=lambda w,D=D,par=par,co=co:D(w,par)@co
 print('best coef',co)
 if key=='a':
  ac=ls(a,v); z=(s['train']['x']-ac[0])@np.linalg.inv(ac[1:]); print('cal residual max',np.max(np.abs(a@ac-v)),'z limits',z.min(0),z.max(0)); print('z quantiles',np.quantile(z,[0,.1,.5,.9,1],axis=0))
tr=s['train'];T=tr['traj'];Y=np.c_[tr['y'],T@np.linalg.svd(T,full_matrices=False)[2][:2].T];n=52
for nm,w in [('raw',tr['x']),('cal',z),('b',operators['bv'](z)),('c1',operators['cv1'](z)),('c2',operators['cv2'](z))]:
 for k,d in [('lin',I(w)),('p2',np.c_[I(w),poly(w,2)]),('p3',np.c_[I(w),poly(w,2),poly(w,3)])]:
  yh=d[n:]@ls(d[:n],Y[:n]); print(nm,k,np.mean(np.abs(yh-Y[n:]),axis=0))
