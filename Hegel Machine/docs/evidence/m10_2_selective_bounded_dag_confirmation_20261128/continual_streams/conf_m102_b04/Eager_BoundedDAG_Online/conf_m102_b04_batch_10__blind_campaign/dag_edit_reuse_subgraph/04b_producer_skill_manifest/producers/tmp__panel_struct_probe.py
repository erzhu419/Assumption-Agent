import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(u):return np.c_[np.ones(len(u)),u]
def ls(F,y):return np.linalg.lstsq(F,y,rcond=1e-10)[0]
def spectrum(u,Y,steps=5):
 F=aff(u); D=np.array([d for d in product((-1,0,1),repeat=4) if any(d) and next(i for i in d if i)>0]); W=(D[:,None]*np.arange(.75,6.1,.11)[None,:,None]).reshape(-1,4); ws=[]
 for z in range(steps):
  Q=np.linalg.qr(F)[0]; R=Y-Q@(Q.T@Y); print('rms',np.sqrt(np.mean(R*R,axis=0)).round(4),end=' ')
  ang=u@W.T; B=np.stack([np.sin(ang),np.cos(ang)],axis=2); B-=np.einsum('ij,jkl->ikl',Q,np.einsum('ij,ikl->jkl',Q,B)); bs=B.transpose(1,0,2); p=bs.transpose(0,2,1)@R; cov=bs.transpose(0,2,1)@bs+1e-12*np.eye(2); gain=(p*np.linalg.solve(cov,p)).sum((1,2));
  def mat(w):
   t=u@w.reshape(-1,4).T;return np.c_[aff(u),np.sin(t),np.cos(t)]
  def loss(w):
   G=mat(w);return (G@ls(G,Y)-Y).ravel()
  w=min((least_squares(loss,np.r_[ws,W[i]],max_nfev=45) for i in np.argsort(gain)[-32:][::-1][::3]),key=lambda f:f.cost).x; ws=w;F=mat(w);print(w.reshape(-1,4).round(2).tolist(),flush=True)
 print('final',np.sqrt(np.mean(loss(ws).reshape(Y.shape)**2,0)).round(5))
def fitmap(u,v):
 def M(u,w):return np.c_[aff(u),np.sin(u*w)]
 def err(w):
  f=M(u,w);return (f@ls(f,v)-v).ravel()
 a=min((least_squares(err,np.full(4,k),bounds=(.4,7),max_nfev=120) for k in (1.5,2.5,3.5,4.5,5.5)),key=lambda a:a.cost); w=a.x; B=ls(M(u,w),v); print('map',w.round(4),B.round(3),'rms',np.sqrt(np.mean(err(w).reshape(v.shape)**2,0)).round(4));return w,B

def predict(u,w,B):return np.c_[aff(u),np.sin(u*w)]@B
if __name__=='__main__':
 for panel,key in [('panel_a','v'),('panel_b','v'),('panel_c','v1')]:
  p=s[panel]; u,v=p['u'],p[key]; keep=np.linalg.norm(v,axis=1)<4 if panel=='panel_a' else np.ones(len(u),bool); print(panel); w,B=fitmap(u[keep],v[keep]);
  if panel=='panel_a': wa,Ba=w,B
  if panel=='panel_c': wc,Bc=w,B
 tr=s['train'];x=tr['x'];q=(x-Ba[0])@np.linalg.inv(Ba[1:5]);
 a=least_squares(lambda q:(predict(q.reshape(x.shape),wa,Ba)-x).ravel(),q.ravel(),jac_sparsity=np.repeat(np.repeat(np.eye(len(q)),4,0),4,1),max_nfev=50);q=a.x.reshape(x.shape);print('inv rms',np.std(a.fun),'max',abs(q).max(0));
 z=tr['traj']; U,S,V=np.linalg.svd(z,full_matrices=False); Y=np.c_[tr['y'],U[:,:2]*S[:2]/[3,.2]]
 for label,r in [('raw',x),('Ainv',q),('C1raw',predict(x,wc,Bc)),('C1Ainv',predict(q,wc,Bc))]:
  print(label,flush=True); spectrum(r,Y,5)
