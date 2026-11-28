import sys,pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar,least_squares
sys.path.insert(0,'/holdout_task');import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; w=m._operator(s,d['x']); y=d['y']; T=d['traj']; n=len(y); ids=np.arange(n); ls=m._ls; aff=m._aff
splits=[(ids[:52],ids[52:])];
for v in [*w.T,np.linalg.norm(w,axis=1)]:
 for b in (np.argsort(v)[:12],np.argsort(v)[-12:]):splits.append((np.setdiff1d(ids,b),b))
pairs=list(itertools.combinations_with_replacement(range(4),2))
def F(w,g,kind,raw,pair):
 v=w*g
 if kind=='asinh':z=np.arcsinh(v)
 if kind=='inv3':z=np.sinh(np.arcsinh(v)/3)
 if kind=='inv5':
  z=np.arcsinh(v)
  for i in range(16):z-=(z+z**5-v)/(1+5*z**4)
 if kind=='atanhlin':
  z=np.tanh(v)
  for i in range(20):z-=(z+np.tanh(z)-v)/(2-np.tanh(z)**2)
 if kind=='tanh':z=np.tanh(v)
 if kind=='atan':z=np.arctan(v)
 if kind=='root':z=v/np.sqrt(1+v*v)
 return np.c_[aff(z),*([w] if raw else []),*([z[:,i]*z[:,j] for i,j in pairs] if pair else [])]
rows=[]
for k,target in enumerate((y,T)):
 for kind,raw in itertools.product(('asinh','inv3','inv5','atanhlin','tanh','atan','root'),(False,True)):
  out=[]; gains=[]; inn=[]
  for a,b in splits:
   def err(g):
    H=F(w[a],g,kind,raw,k==0);return np.mean(((H@ls(H,target[a])-target[a])/np.std(target[a],axis=0))**2)
   g=minimize_scalar(err,bounds=(.1,12),method='bounded').x; H=F(w,g,kind,raw,k==0); out.append(np.mean(abs(H[b]@ls(H[a],target[a])-target[b])));gains.append(g);inn.append(err(g)**.5)
  rows.append((k,kind,raw,round(out[0],6),round(max(out),6),round(np.mean(out),6),round(gains[0],4),round(inn[0],6)))
print(*sorted(rows,key=lambda r:(r[0],r[4])),sep='\n')
B=m._features(w,3.43,False);print('state coefficient first/last',ls(B,T)[:,[0,-1]])
t=s['eval_times']; E=lambda p:-np.expm1(-t[:,None]*p); p=least_squares(lambda p:(E(p)@ls(E(p),T.T)-T.T).ravel(),[1.8,.4]).x; print('time rates',p,'telemetry',s['telemetry_vec']); print('state residual per time',np.std(E(p)@ls(E(p),T.T)-T.T,axis=1))
