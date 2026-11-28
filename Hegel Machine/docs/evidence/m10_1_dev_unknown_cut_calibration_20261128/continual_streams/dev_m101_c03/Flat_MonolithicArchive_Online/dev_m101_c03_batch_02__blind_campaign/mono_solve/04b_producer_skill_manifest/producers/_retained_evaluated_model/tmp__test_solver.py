import importlib.util, pickle, numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
r = s['train']; A,B,D,C = m._features(s,r['x']); aff=m._offset_column
T=s['eval_times']; Y=r['traj']; y=r['y']; E=aff(np.c_[A,B,D]); S=aff(np.c_[A,B,C]); F=np.c_[E,C]
print('endpoint coefficients',m._fit(E,y))
for label,slow,fast in [('both full',S,F),('slow B only',aff(np.c_[B,C]),F),('simple fast',aff(np.c_[B,C]),E),('fast no intercept B cross',np.c_[B,C],np.c_[A,D])]:
 ba,co,q=m._fit_modes(slow,fast,T,Y); print(label,'rates',q,'coef',co, 'fit MAE', np.mean(abs(np.c_[slow@co[0], fast@co[1]]@ba.T-Y)))
ix=np.arange(len(y)); z=(r['x']-m._calibration(s['panel_a'])[0])/m._calibration(s['panel_a'])[1]; splits=[('ordered',ix)]+[(f'rand{k}',np.random.default_rng(k).permutation(len(y))) for k in range(10)]+[('extreme',np.argsort(np.max(abs(z),axis=1)))]; vals=[]
for name,p in splits:
 i,j=p[:56],p[56:]; v=m.solve_batch(dict(s,train={k:r[k][i] for k in ['x','y','traj']},test={'x':r['x'][j]})); ey=abs(v['y_hat']-y[j]); es=abs(v['state_pred']-Y[j]); score=[ey.mean(),es.mean(),es.mean(0).max(),es[:,-1].mean()]; vals.append(score); print(name,np.round(score,6))
print('max',np.max(vals,axis=0))
