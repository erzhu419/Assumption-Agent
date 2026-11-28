import numpy as np,pickle,glob,importlib.util
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(d,y):return np.linalg.lstsq(d,y,rcond=None)[0]
def pairs(x):return np.column_stack([x[:,i]*x[:,j] for i,j in cwr(range(x.shape[1]),2)])
def cal(panel,x):
 def design(z):
  q=np.roll(z,-1,axis=-1);return np.stack([np.ones_like(z),z,q,z*q,z**3],axis=-1)
 u,v=panel['u'],panel['v']; d=design(u); co=[]
 for j in range(4):
  a=d[:,j]; b=v[:,j]; ids=np.arange(len(a));
  for _ in range(15):
   w=ls(a[ids],b[ids]); e=abs(a@w-b);ids=np.argsort(e)[:40]
  ids=e<max(.001,8*np.median(e));co.append(ls(a[ids],b[ids]))
 co=np.array(co)
 from scipy.optimize import root
 def inv(row):
  fun=lambda z: (design(z)*co).sum(-1)-row
  r=root(fun,row,tol=1e-10)
  return r.x if max(abs(r.fun))<1e-8 else least_squares(fun,row,gtol=1e-12).x
 return np.array([inv(row) for row in x]).reshape(-1,4)
def dead(x):return np.sign(x)*np.maximum(abs(x)-.35,0)**1.5
b=s['panel_b']; c=s['panel_c'];u=c['u']; z=cal(s['panel_a'],np.r_[tr['x'],s['test']['x']]); w=aff(dead(z))@ls(aff(dead(b['u'])),b['v']); print('ranges',z.min(0),z.max(0),w.min(0),w.max(0))
from scipy.special import erf
from itertools import combinations
u=c['u'];v=c['v1'];y=tr['y'];p=tr['traj']
def opp(x):return np.c_[x[:,0]*x[:,2],x[:,1]*x[:,3]]
def score(q):
 a=aff(np.c_[q,w,opp(q)]);d=aff(np.c_[q,w]);return [np.mean(abs(m[52:72]@ls(m[:52],b[:52])-b[52:])) for m,b in [(a,y),(d,p)]]
def rms(r):return np.sqrt(np.mean(r*r))
def evalmodel(name,f):
 d=f(u);j=np.linalg.pinv(d);e=d@(j@v)-v;lev=(d*j.T).sum(1); print(name,round(rms(e),6),round(rms(e/np.maximum(1-lev[:,None],1e-8)),6),np.round(score(f(w)@(j@v)),5))
for name,g in [('odd',lambda x,k:np.tanh(k*x)),('oddas',lambda x,k:np.arcsinh(k*x)),('oddsq',lambda x,k:x/np.sqrt(1+k*x*x)),('odderf',lambda x,k:erf(k*x))]:
 for k in [1.25,1.65,1.75,2.15]:
  for tag,h in [('quad',lambda x:x*x),('cross',lambda x:x*np.roll(x,-1,axis=1)),('opp',opp),('cub',lambda x:x**3),('fin',lambda x:np.sin(2.15*x)),('none',lambda x:x[:,:0])]:
   evalmodel((name,k,tag),lambda x:aff(np.c_[x,g(x,k),h(x)]))
for funname,fun in [('tanh',np.tanh),('asinh',np.arcsinh),('erf',erf),('soft',lambda z:z/np.sqrt(1+z*z)),('atan',np.arctan)]:
 for tag in ['bias','skip','square','fixedskip','fixedquad']:
  e=[];pred=[];el=[]
  def fit(u,v):
   init=np.r_[ls(aff(u),v),1.] if 'fixed' in tag else ls(aff(u),v)
   def design(u,b):
    a=fun(aff(u)@b[:5]);return {'bias':aff(a[:,None]),'skip':aff(np.c_[u,a]),'square':aff(np.c_[u*u,a]),'fixedskip':aff(np.c_[a]),'fixedquad':aff(np.c_[a])}[tag]
   def off(u,b):return b[5]*(u[:,col] if tag=='fixedskip' else u[:,col]**2) if len(b)>5 else 0
   def res(b):
    d=design(u,b);r=v-off(u,b);return d@ls(d,r)-r
   opt=least_squares(res,init,max_nfev=200);b=opt.x;d=design(u,b);return lambda w: design(w,b)@ls(d,v-off(u,b))+off(w,b)
  for col in range(4):
   f=fit(u,v[:,col]);e.append(f(u)-v[:,col]);pred.append(f(w))
   el.append([fit(np.delete(u,i,0),np.delete(v[:,col],i))(u[i:i+1])[0]-v[i,col] for i in range(len(u))])
  print('mixed',funname,tag,round(rms(np.array(e)),6),round(rms(np.array(el)),6),np.round(score(np.array(pred).T),5))
