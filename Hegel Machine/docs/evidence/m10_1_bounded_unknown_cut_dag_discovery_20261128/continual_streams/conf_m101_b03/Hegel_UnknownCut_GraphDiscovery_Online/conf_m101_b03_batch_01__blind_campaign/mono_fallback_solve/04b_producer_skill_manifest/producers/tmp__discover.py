import pickle, numpy as np
from scipy.optimize import minimize_scalar
from importlib.machinery import SourceFileLoader
from latent import reverse_a
from operators import poly
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; z=reverse_a(s['panel_a'],tr['x']); prior=SourceFileLoader('prior','/tmp/hegel_continual_archive/conf_m101_b03_batch_00_solver.py').load_module()

def score(A,V):
 B=np.linalg.lstsq(A,V,rcond=None)[0]; R=A@B-V; U=np.linalg.svd(A,full_matrices=False)[0]; h=(U**2).sum(1); return np.mean(abs(R/(1-h[:,None]))),B
p=s['panel_c']; u=p['u']; V=np.c_[p['v1'],p['v2']]; Q=poly(u,2)
for k,v in enumerate(np.split(V,2,axis=1)):
 print('Panel C group',k)
 for A in [Q,np.c_[Q,u**3],np.c_[Q,u**3,u**4],np.c_[Q,u**3,u**4,u**5]]:
  e,B=score(A,v); print('poly',A.shape[1],e,'coef',B[5:].T.round(3))
 for name,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rational',lambda x:x/(1+x*x)),('evenrat',lambda x:1/(1+x*x))]:
  opt=minimize_scalar(lambda w:score(np.c_[Q,f(w*u)],v)[0],bounds=(.15,5),method='bounded'); print(name,opt.x,opt.fun)
for n in [52,56]:
 H,M=prior._time_modes(s['eval_times'],tr['traj'][:n]); Y=np.c_[tr['y'][:n],M]; print('split',n,'rates',-np.log1p(-H[0])/s['eval_times'][0]);
 for tag,x in [('raw',tr['x']),('reverse',z)]:
  for mode in [0,1,3]:
   pred=np.column_stack([prior._regress(x[:n],y,x[n:],mode) for y in Y.T]);print('GP',tag,mode,np.mean(abs(pred[:,0]-tr['y'][n:])),np.mean(abs(pred[:,1:]@H.T-tr['traj'][n:])),flush=True)
 D=poly(z,3); B=np.linalg.lstsq(D[:n],Y,rcond=None)[0]; print('latent cubic coefficients',B.T.round(3))
