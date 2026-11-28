import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task'); import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=m._calibrate(s['panel_a'],tr['x']); p=s['panel_b'];u,v=p['u'],p['v']; c=s['panel_c']; I=m._intercept; ls=m._ls
C=lambda x:I(x*np.roll(x,-1,axis=1)); cc=ls(C(c['u']),c['v1']); rng=np.random.default_rng(173)
folds=[(np.arange(52),np.arange(52,72))]; folds.extend((np.setdiff1d(np.arange(72),b),b) for b in np.array_split(rng.permutation(72),4));
for f in [np.linalg.norm(z,axis=1),np.max(np.abs(z),axis=1),*[z[:,j]*sign for j in range(4) for sign in (-1,1)]]:
 b=np.argsort(f)[-16:]; folds.append((np.setdiff1d(np.arange(72),b),b))
def probe(w):
 q=C(w)@cc; cross=z[:,:,None]*q[:,None,:]; a=I(np.c_[z,q,np.trace(cross,axis1=1,axis2=2)]); b=I(np.c_[z,q,cross.reshape(len(z),-1)]); scores=[]
 for j,k in folds:
  y=tr['y']; t=tr['traj']; tm=t[j].mean(0); modes=np.linalg.svd(t[j]-tm,full_matrices=False)[2][:2]; yh=a[k]@ls(a[j],y[j]); th=b[k]@ls(b[j],(t[j]-tm)@modes.T)@modes+tm; e=abs(th-t[k]); scores.append([abs(yh-y[k]).mean(),e.mean(),e.mean(0).max(),e[:,-1].mean()]);
 r=np.array(scores);return np.round(r[0],6),np.round(r[1:5].max(0),6),np.round(r[5:].max(0),6)
print('baseline [target, trajectory, worst time, endpoint]: fixed / random worst / extrapolation worst',probe(m._b_operator(p,z)))
roll=lambda x,k:np.roll(x,k,axis=1)
variants={'none':lambda x,q:[], 'square':lambda x,q:[x*x], 'cos':lambda x,q:[np.cos(x*q)], 'cycpair':lambda x,q:[x*roll(x,-1)],'opppair':lambda x,q:[x[:,:2]*x[:,2:]], 'sinprod':lambda x,q:[np.sin(x*roll(x,-1))], 'triple':lambda x,q:[x*roll(x,-1)*roll(x,-2)], 'two_sin':lambda x,q:[np.sin(1.25*x)],'rational':lambda x,q:[x/(1+.22*x*x)]}
for name,fn in variants.items():
 for shared in (True,False):
  def D(x,q):return I(np.column_stack([x,np.sin(x*q),*fn(x,q)]))
  def resid(q):d=D(u,q);return (d@ls(d,v)-v).ravel()
  fit=least_squares(resid,np.full(1 if shared else 4,2.9),bounds=(1.5,4.5));d=D(u,fit.x);coef=ls(d,v); r=d@coef-v; loo=np.mean((r/np.maximum(1-np.diag(d@np.linalg.pinv(d)),1e-8)[:,None])**2)**.5; print(name,shared,np.round(fit.x,4),'rms/loo',round(np.mean(r*r)**.5,7),round(loo,7),probe(D(z,fit.x)@coef));
  if name=='none' and shared: print('B coefficients',np.round(coef,4),'residual rows',np.round(np.linalg.norm(r,axis=1),5));
