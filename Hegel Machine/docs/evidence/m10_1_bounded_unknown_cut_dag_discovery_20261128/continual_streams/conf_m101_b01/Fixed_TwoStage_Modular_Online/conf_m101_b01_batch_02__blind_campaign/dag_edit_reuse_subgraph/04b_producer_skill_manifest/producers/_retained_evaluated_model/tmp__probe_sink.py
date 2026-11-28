import importlib.util,pickle,numpy as np,itertools
from scipy.linalg import qr
s=importlib.util.spec_from_file_location('f','/holdout_task/_frozen_nodes/node_00.py'); f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; o=f.run_node({'__input__':d}); z=o['z_tr']; y=o['train_y']; t=o['train_traj']; O=np.column_stack((y,t)); print('z ranges',z.min(0),z.max(0),'sv',np.linalg.svd(np.c_[np.ones(len(z)),z],compute_uv=False)); print('trajectory SV',np.linalg.svd(t,compute_uv=False));print('centered trajectory SV',np.linalg.svd(t-t.mean(0),compute_uv=False));print('joint SV',np.linalg.svd(O-O.mean(0),compute_uv=False));print('Y std',O.std(0)); print('time correlations',np.corrcoef(O.T)[0]);
def mats(x):
 n=len(x); xs=[np.ones(n)]; out={}
 for k in range(1,5):
  xs.extend(np.prod(x[:,p],axis=1) for p in itertools.combinations_with_replacement(range(x.shape[1]),k));out['poly'+str(k)]=np.column_stack(xs)
 for b in [.5,1,2,3]:
  for fn in [np.tanh,np.sin,np.cos]:
   for k in [1,2]:out[f'{fn.__name__}{b}/p{k}']=np.c_[out['poly'+str(k)],fn(b*x),np.cos(b*x) if fn==np.sin else np.empty((n,0))]
 return out
def stats(x,v):
 for key,A in mats(x).items():
  if A.shape[1]>=len(x)-3: continue
  P=np.linalg.pinv(A); E=v-A@(P@v); H=np.sum(A*P.T,axis=1); Q=E/(1-H).reshape((-1,)+(1,)*(v.ndim-1)); print(key,'fit',np.round(np.mean(abs(E),axis=0),5),'loo',np.round(np.mean(abs(Q),axis=0),5))
print('sink'); stats(z,O[:,[0,1,4,8]]); print('panel c v1');stats(d['panel_c']['u'],d['panel_c']['v1']);print('panel c v2');stats(d['panel_c']['u'],d['panel_c']['v2']);print('panel b affine residual',np.linalg.norm(d['panel_b']['v']-f._affine_features(d['panel_b']['u'])@np.linalg.lstsq(f._affine_features(d['panel_b']['u']),d['panel_b']['v'],rcond=None)[0])); print('panel C maps quadratic');x=d['panel_c']['u']; A=mats(x)['poly2'];print(np.round(np.linalg.lstsq(A,d['panel_c']['v1'],rcond=None)[0],3));print('sink quadratic coefficients');print(np.round(np.linalg.lstsq(mats(z)['poly2'],O[:,[0,1,4,8]],rcond=None)[0],3))
