import sys
sys.path.insert(0,'/holdout_task')
import batch_solver as m
import pickle,numpy as np
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; z=m._calibrate(s['panel_a'],x); fy,fs=m._features(s,x)
print('response',m._ls(fy,tr['y']).round(4)); print('state readout',m._ls(fs,tr['traj'])[:,[0,-1]].round(4)); print('collinearity',np.linalg.cond(fs));
sets=[('required',np.arange(52,72))]
sets.extend((f'random {i}',j[:20]) for i,j in enumerate(np.random.default_rng(432).permutation(np.tile(np.arange(72),(16,1)),axis=1))) if False else None
rng=np.random.default_rng(432)
sets.extend((f'random {i}',rng.permutation(72)[:20]) for i in range(16))
sets.extend((f'edge {k}',np.argsort(z[:,k])[-12:]) for k in range(4)); sets.extend((f'edge low {k}',np.argsort(z[:,k])[:12]) for k in range(4)); sets.append(('norm',np.argsort(np.linalg.norm(z,axis=1))[-16:]))
for label,j in sets:
 i=np.setdiff1d(np.arange(len(x)),j); ss={**s,'train':{**tr,**{k:tr[k][i] for k in ['x','y','traj']}},'test':{'x':x[j]}}; p=m.solve_batch(ss); a=abs(p['y_hat']-tr['y'][j]); b=abs(p['state_pred']-tr['traj'][j]); print(label,np.round([a.mean(),b.mean(),b.mean(axis=0).max(),a.max(),b.max()],5))
