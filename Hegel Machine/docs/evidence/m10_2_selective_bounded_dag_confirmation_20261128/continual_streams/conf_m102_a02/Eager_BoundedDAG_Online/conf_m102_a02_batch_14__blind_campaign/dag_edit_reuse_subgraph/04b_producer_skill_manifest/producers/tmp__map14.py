import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=np.c_[s['train']['y'],s['train']['traj']]; xt=np.r_[x,s['test']['x']]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-11)[0]
fun={'tanh':np.tanh,'erf':erf,'rat':lambda x:x/(1+x*x),'sat':lambda x:x/np.hypot(1,x),'atan':np.arctan,'sin':np.sin,'logcosh':lambda x:np.logaddexp(x,-x),'softplus':lambda x:np.logaddexp(0,x)}
p=s['panel_a'];m=np.max(abs(p['v']),1)<4;u=p['u'][m];v=p['v'][m]
def ades(a,i,f,w,mode):
 p,q=a[:,i],a[:,(i+2)%4]; base=[np.ones(len(a)),p,p**3];return np.column_stack(base+([q] if mode&1 else [])+[f(w*q)]+([p*q] if mode&2 else []))
results=[]
for name,f in fun.items():
 for mode in (0,1,2,3):
  def obj(w):
   return np.mean([np.mean((a@ls(a,v[:,i])-v[:,i])**2) for i in range(4) for a in [ades(u,i,f,w,mode)]])
  opt=minimize_scalar(obj,bounds=(.2,5),method='bounded');results.append((opt.fun**.5,name,mode,opt.x))
print('A',sorted(results)[:15]); _,nm,mo,w= min(results); f=fun[nm]; c=[ls(ades(u,i,f,w,mo),v[:,i]) for i in range(4)]; print('A coeff',np.array(c));
def fw(z):return np.column_stack([ades(z,i,f,w,mo)@c[i] for i in range(4)])
def inv(x):return np.array([least_squares(lambda z: (fw(z[None])-b).ravel(),np.clip(b,-1.28,1.28),bounds=(-1.3,1.3),gtol=1e-12).x for b in x])
z=inv(xt);print('A roundtrip',np.quantile(abs(inv(v)-u),[.5,.9,1]), 'data inverse residual',np.quantile(abs(fw(z)-xt),[.5,.9,1]),'z range',z.min(),z.max())
def panels(p,key):
 u,v=p['u'],p[key]; res=[]
 for name,f in fun.items():
  for mode in (0,1):
   def des(z,w):return np.c_[np.ones(len(z)),z,f(z*w)] if mode else np.c_[np.ones(len(z)),f(z*w)]
   def loss(w):
    a=des(u,w);return np.mean((a@ls(a,v)-v)**2)
   opt=minimize_scalar(loss,bounds=(.15,4.5),method='bounded');a=des(u,opt.x);coef=ls(a,v);loo=np.sqrt(np.mean(((a@coef-v)/(1-np.sum(a*np.linalg.pinv(a).T,axis=1)[:,None]))**2));res.append((loo,name,mode,opt.x,opt.fun**.5,des(z,opt.x)@coef))
 res.sort(key=lambda a:a[0]);print(key,[a[:5] for a in res[:8]]);return res[0][-1]
g=panels(s['panel_b'],'v');j=panels(s['panel_c'],'v1');k=panels(s['panel_c'],'v2');np.savez('/tmp/train14.npz',z=z,g=g,j=j,k=k,y=y)
