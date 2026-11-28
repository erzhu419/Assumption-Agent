import numpy as np, pickle, importlib.util
from scipy.linalg import svd
spec=importlib.util.spec_from_file_location('a','/tmp/hegel_continual_archive/conf_m102_b03_batch_01_solver.py'); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; n=len(y); z,l,q,r,p,c=a.features(s,s['train']['x']); dot=(q*r).sum(1); cross=(p[:,:,None]*c[:,None,:]).reshape(n,-1); mix=(q[:,:,None]*r[:,None,:]).reshape(n,-1); zp=(z[:,:,None]*p[:,None,:]).reshape(n,-1); zc=(z[:,:,None]*c[:,None,:]).reshape(n,-1)
D={'branch':np.c_[p,c], 'dot':np.c_[p,c,dot], 'diag':np.c_[p,c,q*r], 'cross':np.c_[p,c,cross], 'zbranch':np.c_[z,p,c], 'zdot':np.c_[z,p,c,dot], 'zcross':np.c_[z,p,c,cross], 'z3':np.c_[z,p,c,zp,zc], 'zpoly':np.c_[z,p,c,z**2,z**3], 'Q':p,'R':c,'z':z}
F=[np.arange(52,72),*np.array_split(np.random.default_rng(325).permutation(n),4)]; logs=[]
for j,te in enumerate(F):
 tr=np.setdiff1d(np.arange(n),te); ya,ta=a._predict(s,tr,s['train']['x'][te]); print('archive',j,np.mean(abs(ya-y[te])),np.mean(abs(ta-T[te])),flush=True)
for k,b in D.items():
 b=np.c_[np.ones(n),b]; es=[]
 for te in F:
  tr=np.setdiff1d(np.arange(n),te); r0=np.c_[y,T]; cc=np.linalg.lstsq(b[tr],r0[tr],rcond=1e-10)[0]; er=np.abs(b[te]@cc-r0[te]); es.append([er[:,0].mean(),er[:,1:].mean()]);
 print(k,np.round(es,5).tolist())
U,S,V=svd(T,full_matrices=False); print('trajectory singular vals:',S); print('temporal modes',np.round(V[:4],4)); print('corr y vs temporal modes',np.corrcoef(np.c_[y,U[:,:6]].T)[0]);
for k in ['dot','cross','zcross','zpoly']:
 b=np.c_[np.ones(n),D[k]]; v=np.c_[y,U[:,:4]*S[:4]]; e=v-b@np.linalg.lstsq(b,v,rcond=1e-10)[0]; print(k,'resid std',np.std(e,0),'resid corrs',np.round(np.corrcoef(e.T)[0],3))
