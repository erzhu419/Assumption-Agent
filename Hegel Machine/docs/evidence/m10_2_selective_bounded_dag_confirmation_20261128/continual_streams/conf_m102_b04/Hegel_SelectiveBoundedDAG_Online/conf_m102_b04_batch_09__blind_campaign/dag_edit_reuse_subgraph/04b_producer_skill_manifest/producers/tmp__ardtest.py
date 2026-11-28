import sys, pickle, numpy as np
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes'];from node_00 import run_node;from node_01 import representation;from ard import ard
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; r=run_node({'__input__':s});x=np.cbrt(r['cubic_train']);y=s['train']['y'];t=s['train']['traj']; q=representation(s,x);print('latent quantiles',np.quantile(x,[0,.02,.5,.98,1],axis=0));print('x dimension',x.shape);print('closest',np.sort(np.partition(np.linalg.norm(x[:,None]-x[None,:],axis=2),1,axis=1)[:,1])[-8:]);
for groups in [q,[q[0],q[1]],[q[0],q[2],q[3]],q+[s['train']['x']]]:
 for te in [np.arange(0,len(x),4),np.arange(1,len(x),4),np.arange(2,len(x),4),np.arange(3,len(x),4)]:
  tr=np.setdiff1d(np.arange(len(x)),te);z=[a[np.r_[tr,te]] for a in groups]; v=np.linalg.svd(t[tr],full_matrices=False)[2][:2];a=t[tr]@v.T
  p=ard(z,y[tr],detail=te[0]==0);pt=np.c_[ard(z,a[:,0],detail=te[0]==0),ard(z,a[:,1],detail=te[0]==0)]@v
  print(len(groups),te[0],np.mean(abs(p-y[te])),np.mean(abs(pt-t[te])),flush=True)
