import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times']; enc=np.load('/tmp/enc.npz'); q=enc['q']; z=enc['bq']; np.set_printoptions(precision=6,suppress=True,linewidth=140)
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
one=np.ones(len(z)); p=z[:,0]*z[:,2]; r=z[:,1]*z[:,3];
folds=[(np.arange(52),np.arange(52,72))]
for seed in range(12):
 ix=np.random.default_rng(seed).permutation(72);folds.append((ix[:52],ix[52:]))
for name,D in [('bilinear',np.c_[z,p+r]),('bilinear+const',np.c_[one,z,p+r]),('bilinear-separate',np.c_[z,p,r]),('quadratic',np.c_[one,z,*[z[:,i]*z[:,j] for i in range(4) for j in range(i,4)]])]:
 err=np.array([np.mean(abs(D[v]@ls(D[a],y[a])-y[v])) for a,v in folds]);print('readout',name,'coefficients',ls(D[:52],y[:52]),'cv first,mean,worst',err[0],err.mean(),err.max())
for name,D in [('z',z),('const-z',np.c_[one,z]),('q',q),('const-q',np.c_[one,q]),('bilinear-z',np.c_[z,p+r])]:
 for rank in [2,99]:
  errors=[]
  for a,v in folds:
   Q,R=np.linalg.qr(D[a],mode='reduced');U,S,V=np.linalg.svd(Q.T@T[a],full_matrices=False); C=ls(R,(U[:,:rank]*S[:rank])@V[:rank]);errors.append(abs(D[v]@C-T[v]));
  e=np.asarray(errors); print('traj',name,rank,'cv first,mean,worst,worst time',e[0].mean(),e.mean(),e.mean((1,2)).max(),e.mean(1).max()); print('coef first & last',ls(D[:52],T[:52])[:,[0,-1]].T if rank==2 else '')
D=z; C=ls(D[:52],T[:52]);print('trajectory coef matrix',C);print('resid singular',np.linalg.svd(T[:52]-D[:52]@C,compute_uv=False));
for name,basis,starts in [
 ('exponential pair',lambda h:np.array([-np.expm1(-np.exp(h[0])*t),-np.expm1(-np.exp(h[1])*t)]),[[-1,1],[-2,0],[0,1]]),
 ('exponential plus linear',lambda h:np.array([-np.expm1(-np.exp(h[0])*t),t]),[[-1],[0],[1]]),
 ('exponential plus time-weighted exponential',lambda h:np.array([-np.expm1(-np.exp(h[0])*t),t*np.exp(-np.exp(h[1])*t)]),[[-1,1],[1,0],[0,0]]),
 ('exponential plus damped sine',lambda h:np.array([-np.expm1(-np.exp(h[0])*t),np.exp(-np.exp(h[1])*t)*np.sin(np.exp(h[2])*t)]),[[-1,-1,0],[1,-2,-1],[0,0,0]])]:
 def calc(h):
  B=basis(h); W=ls(B.T,C.T).T; return W@B
 def err(h):return (D[:52]@(calc(h)-C)).ravel()
 f=min([least_squares(err,h,bounds=(-5,3)) for h in starts],key=lambda f:np.linalg.norm(f.fun));print(name,np.exp(f.x),np.sqrt(np.mean(f.fun**2)),np.mean(abs(D[52:]@calc(f.x)-T[52:])))
