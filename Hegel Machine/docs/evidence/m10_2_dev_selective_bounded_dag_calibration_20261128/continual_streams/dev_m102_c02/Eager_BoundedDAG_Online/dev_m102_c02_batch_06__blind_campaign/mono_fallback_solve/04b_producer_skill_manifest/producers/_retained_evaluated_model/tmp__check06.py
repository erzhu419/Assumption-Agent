import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task'); import batch_solver as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]; k,h=s['telemetry_vec'][:2]
D=b._coordinates(s,t['x']); p=b._fit(D,Y[:52],k,h,True); print('public MAEs',np.mean(abs(p[:,0]-Y[52:,0])),np.mean(abs(p[:,1:]-Y[52:,1:])),np.mean(abs(p-Y[52:]),0))
