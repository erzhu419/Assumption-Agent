import numpy as np,pickle,runpy
n = runpy.run_path('/holdout_task/node_00.py'); f = runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; train=s['train']; ids=np.arange(len(train['x']))
worst=np.zeros(2)
for te in np.array_split(ids,4):
 tr=np.setdiff1d(ids,te); sc=dict(s,train={k:train[k][tr] for k in ('x','y','traj')},test={'x':train['x'][te]}); o=f({'node_00':n['run_node']({'__input__':sc})}); err=[np.mean(abs(o[k]-train[j][te])) for k,j in [('y_hat','y'),('state_pred','traj')]]; worst=np.maximum(worst,err); print('blocked fold',len(te),err)
print('worst blocked fold',worst.tolist())
