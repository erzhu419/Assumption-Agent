import copy,pickle,sys,numpy as np
sys.path.insert(0,'/holdout_task'); import node_00
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_01,node_02
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=len(tr['x'])
for hold in [np.arange(20),np.arange(26,46),np.arange(0,N,4)]:
    keep=np.setdiff1d(np.arange(N),hold); st=dict(s,train={k:tr[k][keep] for k in ('x','y','traj')},test={'x':tr['x'][hold]}); q=node_00.run_node({'__input__':st}); a=node_01.run_node({'__input__':st,'node_00':q}); b=node_02.run_node({'node_01':a}); print('block',hold[0],hold[-1],np.mean(abs(a['y_hat']-tr['y'][hold])),np.mean(abs(b['state_pred']-tr['traj'][hold])))
