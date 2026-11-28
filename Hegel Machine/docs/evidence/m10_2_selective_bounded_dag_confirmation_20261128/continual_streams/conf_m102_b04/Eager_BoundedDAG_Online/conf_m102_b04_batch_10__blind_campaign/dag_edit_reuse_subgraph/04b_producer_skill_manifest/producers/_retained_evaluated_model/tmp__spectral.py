import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; rng=np.random.default_rng(7)
def aff(u):return np.c_[np.ones(len(u)),u]
def fit(F,y):return np.linalg.lstsq(F,y,rcond=1e-10)[0]
def search(u,y,name):
 F=aff(u); freq=np.arange(.75,20.01,.15); D=[]
 for d in product((-1,0,1),repeat=4):
  if any(d) and next(j for j in d if j)!=-1:D.append(d)
 W=(np.array(D)[:,None,:]*freq[None,:,None]).reshape(-1,4)
 for z in range(4):
  Q=np.linalg.qr(F)[0]; R=y-Q@(Q.T@y); print(name,z,'rms',np.sqrt(np.mean(R**2,axis=0)).round(4),flush=True)
  ang=u@W.T; B=np.stack([np.sin(ang),np.cos(ang)],axis=2); B-=np.einsum('ij,jkl->ikl',Q,np.einsum('ij,ikl->jkl',Q,B)); bs=B.transpose(1,0,2); p=bs.transpose(0,2,1)@R; cov=bs.transpose(0,2,1)@bs+1e-12*np.eye(2); gain=(p*np.linalg.solve(cov,p)).sum((1,2)); best=[]
  def mat(w):return np.c_[F,np.sin(u@w),np.cos(u@w)]
  def loss(w):
   G=mat(w);return (G@fit(G,y)-y).ravel()
  for i in np.argsort(gain)[-30:][::-1][::3]:
   a=least_squares(loss,W[i],max_nfev=35);best.append((np.sum(a.fun**2),a.x))
  best.sort(key=lambda q:q[0]); w=best[0][1]; G=mat(w); print('w',w.round(4),'coeff',fit(G,y)[-2:].round(3),'gain',np.sum(R*R)-best[0][0],flush=True); F=G
for p,k in [('panel_a','v'),('panel_b','v'),('panel_c','v1')]:
 u=s[p]['u']; v=s[p][k]; keep=np.linalg.norm(v,axis=1)<4 if p=='panel_a' else np.ones(len(u),bool);search(u[keep],v[keep],p)
u=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; U,S,V=np.linalg.svd(z,full_matrices=False); search(u,np.c_[y,U[:,:2]*S[:2]/[3,1]],'heads')
