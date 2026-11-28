from explore import *
from itertools import combinations
from scipy.optimize import minimize_scalar
Y=s['train']['traj']; time=s['eval_times']; rate=s['telemetry_vec'][0];
def err(T):
 return np.sqrt(np.mean((ls(T,Y.T).T@T.T-Y)**2))
for a,b in [(time,1-np.exp(-rate*time)),(1-np.exp(-rate*time),time*np.exp(-rate*time)),(time, np.exp(-rate*time)),(time, np.log1p(rate*time)),(1-np.exp(-rate*time), np.log1p(rate*time)),(time, time**2),(1-np.exp(-rate*time),np.sin(rate*time))]:print('time',err(np.c_[a,b]))
for a in (time,np.log1p(time),np.sqrt(time),1-np.exp(-rate*time)):
 for fn in (lambda k:1-np.exp(-k*time),lambda k: np.log1p(k*time),lambda k:time*np.exp(-k*time)):
  f=minimize_scalar(lambda k:err(np.c_[a,fn(k)]),bounds=(.001,6),method='bounded'); print('rate',f.x,f.fun)
V=np.linalg.svd(Y[:52],full_matrices=False)[2][:2]; H=np.c_[y,Y@V.T]
i,j=map(np.array,zip((0,0),(1,2),(0,3),(2,3))); p=w[:,i]*w[:,j]; q=aug(powr(w,2))@C-C[0]; pq=lambda a,b:(a[:,:,None]*b[:,None,:]).reshape(len(a),-1); print('q-vs-products',np.max(np.abs(q-p@ls(p,q)))); g={'zq':z*q,'wq':w*q,'zqall':pq(z,q),'wqall':pq(w,q),'zpall':pq(z,p),'zp0':z[:,[0]]*p,'zp1':z[:,[1]]*p,'zp2':z[:,[2]]*p,'zp3':z[:,[3]]*p,'c2z':c2fun(z),'c2w':c2fun(w),'c2zw':z*c2fun(w),'c2wz':w*c2fun(z),'z2':z*z,'w2':w*w}; base=aug(np.c_[w,p]);
for names in [(),*((k,) for k in g),*(('zq',k) for k in g if k!='zq')]:
 F=np.column_stack([base]+[g[k] for k in names]); coef=ls(F[:52],H[:52]); R=H-F@coef; print(names,np.linalg.matrix_rank(F),np.abs(R[52:,0]).mean(),np.abs(R[52:,1:]@V).mean(),np.std(R[:52],0));
 if names==('zq',):print('coef_zq',coef)
# Estimate the rank of the cubic readout for each temporal mode.
F=np.c_[base,w*w,powr(w,3)]; K=ls(F[:52],H[:52]); print('cubic correlations',np.corrcoef(K[-20:].T));print('coef cubic',K[-20:]);
