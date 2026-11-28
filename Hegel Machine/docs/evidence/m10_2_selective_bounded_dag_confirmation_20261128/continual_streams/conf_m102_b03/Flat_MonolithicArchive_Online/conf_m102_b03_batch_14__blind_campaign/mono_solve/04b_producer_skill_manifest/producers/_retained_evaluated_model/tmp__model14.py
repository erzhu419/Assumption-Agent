import numpy as np
from scipy.optimize import least_squares

def D(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
fun={'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'atan':np.arctan,'sat':lambda a:a/(1+a*a),'abs':lambda a:a/(1+abs(a)),'soft':lambda a:np.logaddexp(0,a),'sq':lambda a:a*a,'cube':lambda a:a**3}
def calibration(panel,verbose=False):
 u,v=panel['u'],panel['v']; A=D(u); b=ls(A,v)
 for _ in range(8):
  r=np.linalg.norm(A@b-v,axis=1); mask=r<4*np.median(r);b=ls(A[mask],v[mask])
 u,v=u[mask],v[mask]; w=np.roll(u,2,axis=1)
 def design(u,w,j,f,g,k):return np.c_[np.ones(len(u)),u[:,j],w[:,j],fun[f](k[0]*u[:,j]),fun[g](k[1]*w[:,j])]
 results=[]
 for f in fun:
  for g in fun:
   def residual(k,ret=False):
    a=[design(u,w,j,f,g,k) for j in range(4)]; b=[ls(a[j],v[:,j]) for j in range(4)];return np.stack(b) if ret else np.column_stack([a[j]@b[j]-v[:,j] for j in range(4)]).ravel()
   opt=least_squares(residual,[1.3,2.15],bounds=(.15,5),max_nfev=35);results.append((np.std(opt.fun),f,g,opt.x,residual(opt.x,True)))
 results.sort(key=lambda a:a[0]); err,f,g,k,co=results[0]
 if verbose:
  for r in results[:7]:print('cal',r)
 def predict(x):
  z=np.linalg.solve(b[1:].T,(x-b[0]).T).T
  def forward(z):return np.column_stack([design(z,np.roll(z,2,axis=1),j,f,g,k)@co[j] for j in range(4)])
  for _ in range(18):
   R=forward(z)-x
   if np.max(abs(R),initial=0)<1e-10:break
   jacs=np.stack([(forward(z+e)-forward(z-e))/(2e-5) for e in np.eye(4)*1e-5],axis=2)
   delta=np.linalg.solve(jacs,R[...,None])[...,0];z-=np.clip(delta,-.7,.7)
  return z
 return predict

def operators(panel,verbose=False):
 u=panel['u']; funcs={**fun,'adj':lambda x:x*np.roll(x,1,axis=1),'opposite':lambda x:x*np.roll(x,2,axis=1)}; outs=[]
 for key,v in panel.items():
  if key=='u':continue
  res=[]
  for name,f in funcs.items():
   for lin in (False,True):
    def design(x,k):return D(np.c_[x,f(k*x)] if lin else f(k*x))
    def score(k):
     a=design(u,k[0]);return (a@ls(a,v)-v).ravel()
    for initial in (1,2.15,3.1):
     opt=least_squares(score,[initial],bounds=(.15,5),max_nfev=35); k=opt.x[0]; a=design(u,k);e=(a@ls(a,v)-v)/(1-np.diag(a@np.linalg.pinv(a)))[:,None];res.append((np.std(e),name,lin,k))
  res.sort();err,name,lin,k=res[0]
  if verbose:print(key,res[:9])
  def fit(name,lin,k):
   f=funcs[name]
   def design(x):return D(np.c_[x,f(k*x)] if lin else f(k*x))
   co=ls(design(u),v);return lambda x:design(x)@co
  outs.append(fit(name,lin,k))
 return outs

if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; pred=calibration(s['panel_a'],True);z=pred(s['train']['x']);print('z range',z.min(),z.max()); operators(s['panel_c'],True)
