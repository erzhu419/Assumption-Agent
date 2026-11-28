import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.linalg import solve
from adaptive import coords
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; Z=coords(s,x); n=len(y); rng=np.random.default_rng(735)
folds=[(np.arange(52),np.arange(52,n))];
for _ in range(4):
 i=rng.permutation(n);folds.append((i[:52],i[52:]))

def fit(z,q,y,k,typ,seed=0,quadratic=False):
 rng=np.random.default_rng(seed); n,d=z.shape; Z=np.r_[z,q]; base=np.c_[np.ones(len(Z)),Z];
 if quadratic:base=np.c_[base,np.array([Z[:,i]*Z[:,j] for i in range(d) for j in range(i,d)]).T]
 Q=np.linalg.qr(base[:n])[0]; yy=y-Q@(Q.T@y);
 def calc(p,jac=False):
  H=Z@p.reshape(k,d+1)[:,:d].T+p.reshape(k,d+1)[:,d];
  if typ=='sin':f=np.sin(H);g=np.cos(H)
  if typ=='tanh':f=np.tanh(H);g=1-f*f
  if typ=='rat':f=H/(1+H*H);g=(1-H*H)/(1+H*H)**2
  if typ=='even':f=1/(1+H*H);g=-2*H*f*f
  X=f[:n]-Q@(Q.T@f[:n]); inv=np.linalg.pinv(X,rcond=1e-9); b=inv@yy; r=X@b-yy
  if jac:
   J=(np.c_[Z[:n],np.ones(n)][:,None,:]*g[:n,:,None]).reshape(n,-1); J-=Q@(Q.T@J); J1=J*np.repeat(b,d+1);return J1-X@(inv@J1)-inv.T[:,np.repeat(np.arange(k),d+1)]*(r@J)
  return r,f,b
 best=None
 for j in range(13):
  if j<4: W=np.roll(np.eye(d),j,axis=0)[:k]*1.15; a=rng.normal(0,.15,k)
  else: W=rng.normal(0,.75,(k,d));a=rng.uniform(-1.5,1.5,k)
  opt=least_squares(lambda p:calc(p)[0],np.c_[W,a].ravel(),jac=lambda p:calc(p,True),bounds=(-5,5),max_nfev=95)
  r,f,b=calc(opt.x); score=np.mean(r*r)
  if best is None or score<best[0]:
   pred=f@b+base@np.linalg.lstsq(base[:n],y-f[:n]@b,rcond=None)[0];best=(score,pred[n:],opt.x.reshape(k,d+1),b)
 return best
if __name__=='__main__':
 i,j=folds[0]; z=Z[1]; logs=[]
 for typ in ['sin','tanh','rat','even']:
  for k in [1,2,3,4]:
   for quad in [False,True]:
    sc,p,h,w=fit(z[i],z[j],y[i],k,typ,quadratic=quad); l=(np.mean(abs(p-y[j])),np.sqrt(sc),typ,k,quad); logs.append(l); print(l,np.round(h,3),np.round(w,4),flush=True)
 print('best',sorted(logs)[:12])
 print('range',np.round(np.c_[z.min(0),z.max(0)],3),s['telemetry_vec'])
