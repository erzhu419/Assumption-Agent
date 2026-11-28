import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s})['c']; y=s['train']['y']; a,b,d,e=c.T;p=a*d+b*e; X=np.c_[np.ones(len(c)),c,p]; n=len(y); tr=np.arange(52); va=np.arange(52,n); folds=np.array_split(np.random.default_rng(724).permutation(tr),5)
funcs={'tanh':np.tanh,'sin':np.sin,'rational':lambda x:x/(1+x*x),'atan':np.arctan,'softsign':lambda x:x/np.sqrt(1+x*x),'cube':lambda x:x**3,'erf':__import__('scipy').special.erf}
def fit(kind, mode, train, val):
 f=funcs[kind]; shifted='shift' in mode; inner='inner' in mode; pure='pure' in mode
 def design(z):
  q=c@z[:4]+(z[4] if shifted else 0)+(z[-1]*p if inner else 0)
  return np.c_[X[:,[0]] if pure and inner else X[:,[0,5]] if pure else X,f(q)]
 def proj(z):
  D=design(z)[train]; return D@np.linalg.lstsq(D,y[train],rcond=None)[0]-y[train]
 def go(scale):
  start=np.r_[np.array([1.3,1.25,-.96,-.96])*scale,np.zeros(shifted+inner)]; return least_squares(proj,start,bounds=(-5,5),max_nfev=130,gtol=2e-10)
 opt=min((go(scale) for scale in [.22,.45,.8]),key=lambda z:z.cost); D=design(opt.x); coef=np.linalg.lstsq(D[train],y[train],rcond=None)[0]; err=np.abs(D@coef-y); return err[train].mean(),err[val].mean(),opt.x,coef
for kind in funcs:
 for mode in ('add','addshift','addinner','pure','pureshift','pureinner','pureinnershift'):
  t,v,w,z=fit(kind,mode,tr,va)
  if t<.035 or v<.035:
   cv=[fit(kind,mode,np.setdiff1d(tr,j),j)[1] for j in folds]; print(kind,mode,'train/public/inner worst',np.round([t,v,max(cv)],6),'params',np.round(w,4),np.round(z,4),flush=True)
