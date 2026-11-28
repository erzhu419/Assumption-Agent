import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; x=tr['x']; T=tr['traj']; y=tr['y']; np.set_printoptions(precision=5,suppress=True,linewidth=130)
def phi(x): return np.c_[np.ones(len(x)),x]
p=s['panel_a']; u=p['u']; v=p['v']; a=phi(u); rng=np.random.default_rng(42)
W=min((np.linalg.lstsq(a[i],v[i],rcond=None)[0] for i in (rng.choice(len(u),6,replace=False) for _ in range(600))),key=lambda w: np.median(np.linalg.norm(a@w-v,axis=1)))
r=np.linalg.norm(a@W-v,axis=1); print('A robust errors',np.sort(r));
for cutoff in (3*np.median(r),):
 W=np.linalg.lstsq(a[r<cutoff],v[r<cutoff],rcond=None)[0]
print('robust A',W,'orthogonality',W[1:].T@W[1:]);
def modes(r): return -np.expm1(-t[:,None]*r)
def err(r):
 b=modes(r); return (b@np.linalg.lstsq(b,T[:52].T,rcond=None)[0]-T[:52].T).ravel()
r=least_squares(err,[1.5,.25],bounds=([.6,.005],[5,.59])).x; b=modes(r); amps=np.linalg.lstsq(b,T.T,rcond=None)[0].T; print('rates',r,'proj mae',np.mean(abs(T-amps@b.T),axis=0)); print('corr y amps',np.corrcoef(np.c_[y,amps].T));
P=np.linalg.lstsq(phi(s['panel_b']['u']),s['panel_b']['v'],rcond=None)[0]
for n,z in [('raw',x),('A',phi(x)@W),('Ainv',(x-W[0])@np.linalg.pinv(W[1:])),('B',phi(x)@P),('AB',phi(phi(x)@W)@P),('BA',phi(phi(x)@P)@W)]:
 for name,F in [('lin',phi(z)),('quad',np.c_[phi(z),*[z[:,j]*z[:,k] for j in range(4) for k in range(j,4)]]),('lin+tanh',np.c_[phi(z),np.tanh(z)]),('lin+sin',np.c_[phi(z),np.sin(z)]),('lin+tanh+sq',np.c_[phi(z),np.tanh(z),z*z])]:
  O=np.c_[y,amps]; pred=F@np.linalg.lstsq(F[:52],O[:52],rcond=None)[0]; print(n,name,np.mean(abs(pred[52:,0]-y[52:])), np.mean(abs(pred[52:,1:]@b.T-T[52:])))
