import numpy as np, pickle, runpy,itertools
p='/holdout_task/'
raw=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
d=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']({'__input__':raw})
z,y,T,t=(d[k] for k in ('z_tr','train_y','train_traj','eval_times'))
def poly(x,k):
 return np.stack([np.prod(x[:,i],axis=1) for j in range(k+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),j)],axis=1)
def score(F,Y):
 inv=np.linalg.pinv(F); e=Y-F@(inv@Y); h=np.sum(F*inv.T,axis=1)
 loo=e/np.maximum(1-h,1e-10).reshape((-1,)+(1,)*(Y.ndim-1))
 return np.round([np.mean(abs(e)),np.mean(abs(loo)),np.max(abs(loo))],5)
if __name__=='__main__':
 print('z range',z.min(0),z.max(0),'test',d['z_te'].min(0),d['z_te'].max(0)); print('svd traj',np.linalg.svd(T,compute_uv=False)); print('T examples',T[:5]); print('y traj corr',np.corrcoef(np.column_stack((y,T)).T)[0]); print('y range',y.min(),y.max())
 for k in (1,2,3,4):
  F=poly(z,k); print('poly',k,score(F,y),score(F,T))
 for panel in ('panel_b','panel_c'):
  v=raw[panel]; u=v['u'];
  for key,Y in v.items():
   if key=='u':continue
   print(panel,key,Y.min(0),Y.max(0))
   for resp in (Y,np.arctanh(np.clip(Y,-.99999,.99999)),np.arcsin(np.clip(Y,-1,1))):
    print([score(poly(u,k),resp).tolist() for k in (1,2)])
