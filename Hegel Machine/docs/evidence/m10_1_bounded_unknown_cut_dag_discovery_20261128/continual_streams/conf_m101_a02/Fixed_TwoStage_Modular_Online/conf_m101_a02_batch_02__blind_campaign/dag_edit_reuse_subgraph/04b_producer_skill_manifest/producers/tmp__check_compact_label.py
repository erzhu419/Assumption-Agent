import pickle, runpy, numpy as np
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; t=s['train']['traj']
z=runpy.run_path(P+'_frozen_nodes/node_00.py')['run_node']({'__input__':s})['z_tr']; old=runpy.run_path(P+'node_01.py'); new=runpy.run_path('/tmp/compact_label.py')
rng=np.random.default_rng(9061); ix=np.arange(len(z)); a=np.column_stack((np.ones(len(z)),z)); groups=[ix[-20:]]+[rng.permutation(ix)[:20] for _ in range(24)] + [np.argsort(abs(v))[-12:] for v in z.T] + [np.argsort(v)[-12:] for v in (*z.T, z[:,2]+z[:,3]-z[:,0]-z[:,1])] + [np.argsort(v)[:12] for v in (*z.T, z[:,2]+z[:,3]-z[:,0]-z[:,1])]; folds=[(np.setdiff1d(ix,g),g) for g in groups]
for name, fn in [('baseline',old['_label'])]+[(str((sh,ti)),lambda z,zt,y,sh=sh,ti=ti:new['_label'](z,zt,y,sh,ti)) for sh in (True,False) for ti in (True,False)]:
    mae=np.array([np.mean(abs(fn(z[tr],z[te],y[tr])-y[te])) for tr,te in folds]); print(name,'train/public/random_mean/random_max/tail_max:',np.round([np.mean(abs(fn(z,z,y)-y)),mae[0],mae[1:25].mean(),mae[1:25].max(),mae[25:].max()],6))
horizons=np.array([np.mean(abs(a[te]@np.linalg.lstsq(a[tr],t[tr],rcond=None)[0]-t[te]),axis=0) for tr,te in folds]); print('worst trajectory horizon:',horizons.max())
