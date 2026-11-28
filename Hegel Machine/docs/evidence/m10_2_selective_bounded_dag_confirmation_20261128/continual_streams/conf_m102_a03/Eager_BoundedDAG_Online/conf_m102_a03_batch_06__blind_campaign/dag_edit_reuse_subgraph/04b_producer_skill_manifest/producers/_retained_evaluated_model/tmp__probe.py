import numpy as np,pickle,importlib.util,itertools
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; out=f.run_node({'__input__':s}); tr=s['train']; y=tr['y']; T=tr['traj']; t=s['eval_times'];
def poly(x,p):
 terms=[i for d in range(p+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),d)];return np.stack([np.prod(x[:,i],axis=1) for i in terms],1),terms
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(a,b):return np.mean(abs(a-b))
u=s['panel_b']['u'];v=s['panel_b']['v'];a=f._affine(u); z=out['z']; zt=out['zt']; raw=np.linalg.solve(ls(a,v)[1:].T,(z-ls(a,v)[0]).T).T
for pn in ['panel_b','panel_c']:
 p=s[pn]; U=p['u'];
 for k,V in p.items():
  if k=='u':continue
  print(pn,k,'affine coefficients',ls(f._affine(U),V).round(3));
  for nm,D in [('affine',f._affine(U)),('quad',poly(U,2)[0]),('cycpair',np.c_[f._affine(U),U*np.roll(U,-1,axis=1)]),('sincyc',np.c_[f._affine(U),np.sin(U), U*np.roll(U,-1,axis=1)]),('cube',np.c_[f._affine(U),U**3]),('trig',np.c_[f._affine(U),np.sin(U),np.cos(U)]),('tanh',np.c_[f._affine(U),np.tanh(U)])]:
   e=[]
   for j in range(len(U)):
    m=np.arange(len(U))!=j;e.append(D[j]@ls(D[m],V[m])-V[j])
   print(nm,err(np.array(e),0),np.max(abs(np.array(e))))
print('traj sv',np.linalg.svd(T,compute_uv=False)); print('traj differences sv',np.linalg.svd(np.diff(T,axis=1),compute_uv=False)); print('affine residual sv',np.linalg.svd(T-f._affine(z)@ls(f._affine(z),T),compute_uv=False))
for xn,X in [('raw',raw),('enc',z)]:
 for d in range(1,5):
  D,names=poly(X,d); co=ls(D[:52],np.c_[y,T][:52]); E=abs((D@co-np.c_[y,T])[52:]); print(xn,d,'cv',E[:,0].mean(),E[:,1:].mean(axis=0).round(4), 'train',abs(D[:52]@co-np.c_[y,T][:52]).mean(axis=0).round(4));
  if d==2: print('quad coef',list(zip(names,co[:,0].round(3),co[:,-1].round(3))))
