import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=s['train']; x,y,z=(d[k] for k in ('x','y','traj')); t=s['eval_times']; idx=np.arange(len(y)); rng=np.random.default_rng(1)
c=m._coordinates(x,m._panel_maps(s['panel_a'],s['panel_b'])); print('times',t,'maps',m._panel_maps(s['panel_a'],s['panel_b']))
slices=[('designated',idx[52:])]+[(f'random{k}',j) for k,j in enumerate(np.array_split(rng.permutation(len(y)),4))]
for nm,C in [('meas',x),('cal',c)]:
 for k,col in enumerate(C.T):
  order=np.argsort(col); slices.extend([(f'{nm}{k}_low',order[:14]),(f'{nm}{k}_high',order[-14:])])
slices.append(('radial_edge',np.argsort(np.linalg.norm(x-x.mean(0),axis=1))[-14:]))
rows=[]
for name,j in slices:
 i=np.setdiff1d(idx,j); tr={k:d[k][i] for k in ('x','y','traj')}; out=m.solve_batch(dict(s,train=tr,test={'x':x[j]})); np.testing.assert_allclose(out['diagnostics_vec'],[out['y_hat'].mean(),out['y_hat'].std(),out['state_pred'][:,-1].mean(),out['state_pred'][:,-1].std()]);
 G=m._readout(c,False); yn=G[j]@np.linalg.lstsq(G[i],y[i],rcond=None)[0]; zn=m._trajectory(c[i],c[j],z[i],t,False)
 e=abs(out['state_pred']-z[j]); f=abs(zn-z[j]); row=[abs(out['y_hat']-y[j]).mean(), e.mean(), e.mean(0).max(),abs(yn-y[j]).mean(),f.mean(),f.mean(0).max()]; rows.append(row); print(name,*[round(a,6) for a in row]);
 if name=='designated': assert row[0]<=.038
print('mean',np.mean(rows,axis=0),'worst',np.max(rows,axis=0)); print('readout no intercept global',np.linalg.lstsq(m._readout(c,False),y,rcond=None)[0]); print('state affine residual mean',np.mean(z-m._affine(c)@np.linalg.lstsq(m._affine(c),z,rcond=None)[0],axis=0))
