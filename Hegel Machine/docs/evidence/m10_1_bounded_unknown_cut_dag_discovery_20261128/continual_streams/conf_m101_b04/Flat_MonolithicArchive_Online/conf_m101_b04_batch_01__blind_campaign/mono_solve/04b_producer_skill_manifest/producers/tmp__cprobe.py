import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['panel_c']; u=d['u']; n=len(u); rng=np.random.default_rng(3); np.set_printoptions(precision=4,suppress=True,linewidth=120)
def aff(x): return np.c_[np.ones(len(x)),x]
def ls(f,y): return np.linalg.lstsq(f,y,rcond=None)[0]
def poly(x,k): return np.column_stack([np.prod(x[:,p],axis=1) for a in range(k+1) for p in itertools.combinations_with_replacement(range(4),a)])
b=s['panel_b']; B=ls(aff(b['u']),b['v']); print('B',B.T)
for k in ['v1','v2']:
 v=d[k]; print(k,'rms by order and singular values');
 for j in (1,2):
  f=poly(u,j); e=v-f@ls(f,v); print(j,np.linalg.svd(e,compute_uv=False),np.sqrt(np.mean(e**2)))
 print('diag quadratic',np.sqrt(np.mean((v-np.c_[aff(u),u*u]@ls(np.c_[aff(u),u*u],v))**2)))
 for w in [u,aff(u)@B]:
  for name,op in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('cube',lambda a:a**3),('quad',lambda a:a**2),('rat',lambda a:a/(1+a*a))]:
   win=[]
   for base in [aff(u),poly(u,2)]:
    fits=[]
    for f in [.5,1,1.5,2,2.5,3,4]:
     a=np.c_[base,op(f*w)]; ids=np.arange(19); fits.append((np.sqrt(np.mean((v[19:]-a[19:]@ls(a[ids],v[ids]))**2)),f))
    win.append(min(fits))
   print('dict',name,win)
 for name,op in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('sq',lambda a:a*a),('rat',lambda a:a/(1+a*a))]:
  out=[]
  for col in range(4):
   y=v[:,col]; Q=aff(u); N=np.linalg.qr(Q,mode='complete')[0][:,5:]; yy=N.T@y
   def res(p):
    h=N.T@op(u@p[:4]+p[4]); return yy-h*(h@yy)/(h@h+1e-24)
   fits=[least_squares(res,p,max_nfev=130) for p in np.r_[np.c_[np.eye(4)*1.5,np.ones(4)*.2],rng.normal(size=(12,5))]]; f=min(fits,key=lambda f:f.cost); out.append((round(np.linalg.norm(f.fun)/np.sqrt(n),5),f.x.round(3)))
  print('single index',name,out)
