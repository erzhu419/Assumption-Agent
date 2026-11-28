import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; n=len(y)
def aff(x): return np.c_[np.ones(len(x)),x]
def ls(D,y): return np.linalg.lstsq(D,y,rcond=None)[0]
def poly(x,k=2):
 return np.column_stack([np.prod(x[:,ii],axis=1) for j in range(k+1) for ii in itertools.combinations_with_replacement(range(x.shape[1]),j)])
u,v=[s['panel_a'][k] for k in ('u','v')]; z=u.copy()
for j in range(4):
 d=aff(u[:,j:j+1]); w=ls(d,v[:,j]);
 for _ in range(8):
  ids=np.argsort(abs(d@w-v[:,j]))[:36]; w=ls(d[ids],v[ids,j]);
 z[:,j]=u[:,j]*w[1]+w[0]
g=np.std(z,axis=0)/np.std(u,axis=0); a=np.mean(z-u*g,axis=0); x=(tr['x']-a)/g
p=s['panel_b']; W=ls(aff(p['u']),p['v']); b=aff(x)@W
from scipy.special import hyp2f1,erf
from scipy.stats import spearmanr
print('time',t,'telemetry',s['telemetry_vec']); print('data',np.c_[b,y,T[:,-1]][:16].round(4)); print('std y/T',y.std(),T.std(axis=0)); print('correlations',np.corrcoef(np.c_[x,b,y,T[:,-1]].T)[-2:]);
idx=np.arange(52); val=np.arange(52,72); folds=[(idx,val),(np.arange(16,72),np.arange(16)),(np.r_[0:20,40:72],np.arange(20,40))];
def cv(d, target=y,ridge=0):
 return [np.mean(abs(d[j]@np.linalg.solve(d[i].T@d[i]+ridge*np.eye(d.shape[1]),d[i].T@target[i])-target[j])) if ridge else np.mean(abs(d[j]@ls(d[i],target[i])-target[j])) for i,j in folds]
print('y from observed state',cv(aff(T[:,-1:])),cv(poly(T[:,-1:],4)),cv(aff(T)))
for z,name in [(b,'b'),(x,'x')]:
 for k in (1,2,3): print(name,k,cv(poly(z,k)),cv(poly(z,k),T))
 print('linear coeff',ls(aff(z),y)); print('quad coeff',ls(poly(z),y)); print('state coeff',ls(aff(z),T[:,-1]));
 for kind,fn in [('atan',np.arctan),('asinh',np.arcsinh),('erf',erf),('tanh',np.tanh),('softsign',lambda a:a/(1+abs(a))),('slog',lambda a:np.sign(a)*np.log1p(abs(a))),('pow',lambda a:np.sign(a)*abs(a)**1.7),('hyp',lambda a:a*hyp2f1(.5,.7,1.5,-a*a))]:
  co=[]
  for k in (.6,1.,1.7,2.5,3.5,5.):
   h=fn(k*z); co.append((np.mean(cv(poly(h))),k,cv(poly(h)),cv(aff(h),T)))
  print(name,kind,min(co))
 for kind,fn in [('rational',lambda a:a/(1+a*a)),('gauss',lambda a:np.exp(-a*a)),('sin',np.sin),('tanh',np.tanh),('abs',np.abs)]:
  for source in (z,np.column_stack([z[:,i]+sg*z[:,j] for i in range(4) for j in range(i+1,4) for sg in [-1,1]])):
   base=poly(z); scores=[]
   for k in (.5,1.,1.7,2.5,3.5,5.):
    h=fn(k*source); scores.append((np.mean(cv(np.c_[base,h],ridge=.02)),k,cv(np.c_[base,h],ridge=.02)))
   print(name,kind,source.shape[1],min(scores))
