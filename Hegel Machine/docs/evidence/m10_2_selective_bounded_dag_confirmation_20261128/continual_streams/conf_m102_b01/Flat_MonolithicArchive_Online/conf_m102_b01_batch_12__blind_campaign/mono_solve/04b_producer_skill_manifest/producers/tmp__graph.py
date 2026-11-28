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
features={}
for name,f in {'tanh':lambda x,k:np.tanh(k*x),'atan':lambda x,k:np.arctan(k*x),'asinh':lambda x,k:np.arcsinh(k*x),'sin':lambda x,k:np.sin(k*x),'softabs':lambda x,k:x/(1+k*abs(x)),'softsquare':lambda x,k:x/(1+k*x*x),'sqrt':lambda x,k:x/np.sqrt(1+k*x*x),'erf':lambda x,k:__import__('scipy').special.erf(k*x)}.items():
 for plus in [False,True]:
  def d(x,k):return aff(np.c_[x,f(x,k)] if plus else f(x,k))
  v=c['v1']; opt=minimize_scalar(lambda k:np.mean((d(u,k)@ls(d(u,k),v)-v)**2),bounds=(.03,5),method='bounded');k=opt.x
  features[name+str(plus)]=(np.sqrt(opt.fun),k,d(w,k)@ls(d(u,k),v),d(z,k)@ls(d(u,k),v))
print('C1',[(a,round(q[0],7),round(q[1],5)) for a,q in sorted(features.items(),key=lambda it:it[1][0])]);
for name,f in {'quad':lambda x:np.c_[x,pairs(x)],'cyc':lambda x:np.c_[x,x*np.roll(x,-1,axis=1)],'opp':lambda x:np.c_[x,x[:,0]*x[:,2],x[:,1]*x[:,3]],'tanhquad':lambda x:np.c_[x,pairs(x),np.tanh(1.25*x)],'pure':lambda x:x*np.roll(x,-1,axis=1)}.items():
 d=aff(f(u));v=c['v2'];co=ls(d,v); print('C2',name,np.sqrt(np.mean((d@co-v)**2))); 
 if name in ('quad','cyc','opp'):print(np.round(co,4))
 features['v2_'+name]=(0,0,aff(f(w))@co,aff(f(z))@co)
def score(d,y,n=52):
 return np.mean(abs(d[n:72]@ls(d[:n],y[:n])-y[n:72]))
coords={'z':z,'w':w,'dead_z':dead(z),'dead_w':dead(w)}
for k,(err,t,g,h) in features.items():coords[k+'(w)']=g;coords[k+'(z)']=h
for key in ['y','traj']:
 ranks=[];y=tr[key]
 for name,x in coords.items():
  for suffix,d in [('lin',aff(x)),('quad',aff(np.c_[x,pairs(x)])),('opp',aff(np.c_[x,x[:,0]*x[:,2],x[:,1]*x[:,3]])),('cyc',aff(np.c_[x,x*np.roll(x,-1,axis=1)])),('tanh',aff(np.tanh(1.25*x))),('atan',aff(np.arctan(1.25*x))),('asinh',aff(np.arcsinh(1.25*x))),('mix',aff(np.c_[w,x])),('xz',aff(np.c_[x,x*np.roll(z,-1,axis=1)])),('xw',aff(np.c_[x,x*np.roll(w,-1,axis=1)]))]:
   ranks.append((score(d,y),name,suffix))
 print(key,sorted(ranks)[:22])
